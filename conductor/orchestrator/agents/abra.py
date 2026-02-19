"""
Abra — Home Automation Agent.

Resolves room context from source device, discovers available devices
in that room, and translates natural-language requests into Home Assistant
service calls.

Flow:
  1. Utterance arrives with source_device_id (which Alexa/Echo heard it)
  2. Resolve source_device_id → room/area via device registry
  3. Query device inventory for that room's entities
  4. Interpret intent (hot → fan + thermostat logic, dark → lights, etc.)
  5. Check environmental context (outdoor temp, current HVAC mode, etc.)
  6. Build the appropriate HA service call(s)
  7. Execute via HA REST API

Environmental awareness:
  - If user says "it's hot" and heat is running, check outdoor temp
    before deciding AC vs lower setpoint.
  - If outdoor temp is safe for AC (above AC_MIN_OUTDOOR_TEMP), switch to cool
  - Otherwise just lower the thermostat setpoint
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


class DeviceDomain(str, Enum):
    """Home Assistant device domains we support."""
    FAN = "fan"
    CLIMATE = "climate"
    LIGHT = "light"
    LOCK = "lock"
    SWITCH = "switch"
    COVER = "cover"         # blinds, shades
    SENSOR = "sensor"
    WEATHER = "weather"


@dataclass
class HADevice:
    """A Home Assistant entity."""
    entity_id: str              # e.g. "fan.living_room"
    domain: DeviceDomain        # e.g. DeviceDomain.FAN
    friendly_name: str          # e.g. "Living Room Fan"
    area_id: str                # e.g. "living_room"
    supported_services: list[str] = field(default_factory=list)
    # e.g. ["turn_on", "turn_off", "set_percentage"]


@dataclass
class Room:
    """A room/area with its associated devices and source device mappings."""
    area_id: str                # HA area ID
    name: str                   # Human-readable: "Living Room"
    alexa_device_ids: list[str] = field(default_factory=list)
    # Echo device IDs that live in this room
    devices: list[HADevice] = field(default_factory=list)

    def devices_by_domain(self, domain: DeviceDomain) -> list[HADevice]:
        """Get all devices in this room for a given domain."""
        return [d for d in self.devices if d.domain == domain]

    @property
    def has_fan(self) -> bool:
        return any(d.domain == DeviceDomain.FAN for d in self.devices)

    @property
    def has_climate(self) -> bool:
        return any(d.domain == DeviceDomain.CLIMATE for d in self.devices)

    @property
    def has_light(self) -> bool:
        return any(d.domain == DeviceDomain.LIGHT for d in self.devices)


@dataclass
class HAServiceCall:
    """A structured Home Assistant service call."""
    domain: str                 # "fan", "climate", "light"
    service: str                # "turn_on", "set_temperature", etc.
    entity_id: str              # "fan.living_room"
    data: dict = field(default_factory=dict)
    # e.g. {"percentage": 50} or {"temperature": 72, "hvac_mode": "cool"}

    def to_ha_payload(self) -> dict:
        """Format for HA REST API POST /api/services/{domain}/{service}."""
        payload = {"entity_id": self.entity_id}
        payload.update(self.data)
        return payload


@dataclass
class EnvironmentState:
    """Current environmental readings relevant to HVAC decisions."""
    outdoor_temp_f: float | None = None
    indoor_temp_f: float | None = None
    hvac_mode: str | None = None        # "heat", "cool", "auto", "off"
    hvac_action: str | None = None      # "heating", "cooling", "idle"
    current_setpoint_f: float | None = None


@dataclass
class AbraResult:
    """Result of Abra processing an utterance."""
    success: bool
    service_calls: list[HAServiceCall] = field(default_factory=list)
    room_resolved: str = ""             # Which room was identified
    reasoning: str = ""                 # Why these actions were chosen
    error: str = ""                     # If failed, why
    environment: EnvironmentState | None = None


# ------------------------------------------------------------------
# Device registry — maps source devices to rooms, rooms to HA entities
# ------------------------------------------------------------------


class DeviceRegistry:
    """Maps Alexa device IDs to rooms, and rooms to their HA devices.

    Two modes of operation:
      1. sync_from_ha() — pulls areas + entities from the HA REST API
         at startup and periodically. This is the production path.
      2. build_default() — hardcoded fallback for tests and offline dev.

    Alexa device → room mapping is configured separately (alexa_map)
    because HA doesn't know which Echo is in which room natively.
    """

    def __init__(
        self,
        rooms: list[Room] | None = None,
        alexa_map: dict[str, str] | None = None,
    ) -> None:
        self._rooms: dict[str, Room] = {}
        self._alexa_to_room: dict[str, str] = alexa_map or {}

        if rooms:
            for room in rooms:
                self.register_room(room)

    def register_room(self, room: Room) -> None:
        """Add a room to the registry."""
        self._rooms[room.area_id] = room
        for device_id in room.alexa_device_ids:
            self._alexa_to_room[device_id] = room.area_id

    def resolve_room(self, source_device_id: str) -> Room | None:
        """Look up which room a source device (Alexa Echo) is in."""
        area_id = self._alexa_to_room.get(source_device_id)
        if area_id:
            return self._rooms.get(area_id)
        return None

    def get_room(self, area_id: str) -> Room | None:
        """Get a room by area ID directly."""
        return self._rooms.get(area_id)

    def all_rooms(self) -> list[Room]:
        """Return all registered rooms."""
        return list(self._rooms.values())

    def find_devices(self, area_id: str, domain: DeviceDomain) -> list[HADevice]:
        """Find all devices of a given domain in a room."""
        room = self._rooms.get(area_id)
        if not room:
            return []
        return room.devices_by_domain(domain)

    # ── HA API sync ────────────────────────────────────────────────

    async def sync_from_ha(
        self,
        ha_url: str,
        ha_token: str,
        alexa_map: dict[str, str] | None = None,
    ) -> int:
        """Pull areas and entities from Home Assistant and rebuild the registry.

        Uses HA's REST API:
          - GET /api/states → all entity states with attributes
          - Entity IDs encode domain (fan.xxx, climate.xxx, light.xxx)
          - Attributes include friendly_name
          - Area assignment uses the entity_id prefix convention
            (e.g. fan.living_room → area "living_room") unless HA's
            entity registry provides explicit area_id.

        For the full area → entity mapping, we use the websocket API
        endpoints through the REST wrapper:
          - /api/config/area_registry/list
          - /api/config/entity_registry/list

        Args:
            ha_url: Home Assistant base URL (e.g. http://homeassistant.local:8123)
            ha_token: Long-lived access token
            alexa_map: Optional Alexa device ID → area_id mapping to merge

        Returns:
            Number of entities discovered
        """
        if alexa_map:
            self._alexa_to_room.update(alexa_map)

        headers = {"Authorization": f"Bearer {ha_token}"}
        entity_count = 0

        async with httpx.AsyncClient(
            base_url=ha_url.rstrip("/"), headers=headers, timeout=15,
        ) as client:
            # ── 1. Get areas ────────────────────────────────────────
            areas: dict[str, str] = {}  # area_id → name
            try:
                resp = await client.post(
                    "/api/template",
                    json={"template": "{{ areas() | list }}"},
                )
                resp.raise_for_status()
                # HA returns area IDs as a rendered string list
                import ast
                area_ids = ast.literal_eval(resp.text)
                for aid in area_ids:
                    # Get area name
                    resp2 = await client.post(
                        "/api/template",
                        json={"template": f"{{{{ area_name('{aid}') }}}}"},
                    )
                    areas[aid] = resp2.text.strip()
            except Exception as exc:
                logger.warning("Could not load areas from HA: %s — falling back to entity parsing", exc)

            # ── 2. Get all entity states ────────────────────────────
            try:
                resp = await client.get("/api/states")
                resp.raise_for_status()
                entities = resp.json()
            except Exception as exc:
                logger.error("Failed to get entities from HA: %s", exc)
                return 0

            # ── 3. Get entity → area mapping via template API ───────
            entity_areas: dict[str, str] = {}
            try:
                resp = await client.post(
                    "/api/template",
                    json={"template": (
                        "{% set ns = namespace(result=[]) %}"
                        "{% for state in states %}"
                        "{% set area = area_id(state.entity_id) %}"
                        "{% if area %}"
                        "{% set ns.result = ns.result + [state.entity_id ~ '|' ~ area] %}"
                        "{% endif %}"
                        "{% endfor %}"
                        "{{ ns.result | join('\\n') }}"
                    )},
                )
                resp.raise_for_status()
                for line in resp.text.strip().split("\n"):
                    if "|" in line:
                        eid, area = line.split("|", 1)
                        entity_areas[eid.strip()] = area.strip()
            except Exception as exc:
                logger.warning("Could not load entity→area map: %s — using entity_id parsing", exc)

            # ── 4. Build rooms from discovered data ─────────────────
            # Supported domains for device control
            supported_domains = {d.value for d in DeviceDomain}

            room_devices: dict[str, list[HADevice]] = {}

            for entity in entities:
                eid: str = entity.get("entity_id", "")
                parts = eid.split(".", 1)
                if len(parts) != 2:
                    continue
                domain_str, entity_name = parts

                if domain_str not in supported_domains:
                    continue

                attrs = entity.get("attributes", {})
                friendly = attrs.get("friendly_name", entity_name)

                # Determine area — prefer explicit mapping, fall back to parsing
                area = entity_areas.get(eid)
                if not area:
                    # Parse from entity_id: fan.living_room → living_room
                    # Strip trailing numbers: fan.living_room_2 → living_room
                    area = re.sub(r"_\d+$", "", entity_name)

                # Map supported_features bitmask to service names
                services = _infer_services(domain_str, attrs.get("supported_features", 0))

                device = HADevice(
                    entity_id=eid,
                    domain=DeviceDomain(domain_str),
                    friendly_name=friendly,
                    area_id=area,
                    supported_services=services,
                )

                room_devices.setdefault(area, []).append(device)
                entity_count += 1

            # ── 5. Rebuild rooms ────────────────────────────────────
            self._rooms.clear()
            for area_id, devices in room_devices.items():
                name = areas.get(area_id, area_id.replace("_", " ").title())
                # Find alexa devices mapped to this area
                alexa_ids = [
                    did for did, aid in self._alexa_to_room.items()
                    if aid == area_id
                ]
                room = Room(
                    area_id=area_id,
                    name=name,
                    alexa_device_ids=alexa_ids,
                    devices=devices,
                )
                self._rooms[area_id] = room

            logger.info(
                "DeviceRegistry synced from HA: %d entities in %d rooms",
                entity_count, len(self._rooms),
            )

        return entity_count

    # ── Fallback for tests ─────────────────────────────────────────

    @classmethod
    def build_default(cls) -> DeviceRegistry:
        """Build the default device registry (4 fans + thermostat).

        Represents the real home layout. Alexa device IDs would come
        from the Alexa Skills Kit — using placeholder IDs for now.
        """
        rooms = [
            Room(
                area_id="living_room",
                name="Living Room",
                alexa_device_ids=["echo_living_room"],
                devices=[
                    HADevice(
                        entity_id="fan.living_room",
                        domain=DeviceDomain.FAN,
                        friendly_name="Living Room Fan",
                        area_id="living_room",
                        supported_services=["turn_on", "turn_off", "set_percentage"],
                    ),
                    HADevice(
                        entity_id="climate.main_thermostat",
                        domain=DeviceDomain.CLIMATE,
                        friendly_name="Thermostat",
                        area_id="living_room",
                        supported_services=[
                            "set_temperature", "set_hvac_mode", "turn_on", "turn_off",
                        ],
                    ),
                ],
            ),
            Room(
                area_id="bedroom",
                name="Bedroom",
                alexa_device_ids=["echo_bedroom"],
                devices=[
                    HADevice(
                        entity_id="fan.bedroom",
                        domain=DeviceDomain.FAN,
                        friendly_name="Bedroom Fan",
                        area_id="bedroom",
                        supported_services=["turn_on", "turn_off", "set_percentage"],
                    ),
                ],
            ),
            Room(
                area_id="office",
                name="Office",
                alexa_device_ids=["echo_office"],
                devices=[
                    HADevice(
                        entity_id="fan.office",
                        domain=DeviceDomain.FAN,
                        friendly_name="Office Fan",
                        area_id="office",
                        supported_services=["turn_on", "turn_off", "set_percentage"],
                    ),
                ],
            ),
            Room(
                area_id="kitchen",
                name="Kitchen",
                alexa_device_ids=["echo_kitchen"],
                devices=[
                    HADevice(
                        entity_id="fan.kitchen",
                        domain=DeviceDomain.FAN,
                        friendly_name="Kitchen Fan",
                        area_id="kitchen",
                        supported_services=["turn_on", "turn_off", "set_percentage"],
                    ),
                ],
            ),
        ]
        return cls(rooms=rooms)


# ------------------------------------------------------------------
# Environment reader — gets outdoor temp, HVAC state from HA
# ------------------------------------------------------------------

# Minimum outdoor temp (F) where AC is safe to run
AC_MIN_OUTDOOR_TEMP = 60.0

# Default setpoint adjustment (degrees F)
SETPOINT_STEP = 2.0


class EnvironmentReader:
    """Reads environmental state from Home Assistant sensors."""

    def __init__(self, ha_url: str, ha_token: str) -> None:
        self._ha_url = ha_url.rstrip("/")
        self._ha_token = ha_token
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._ha_url,
                headers={"Authorization": f"Bearer {self._ha_token}"},
                timeout=10,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_state(self, entity_id: str) -> dict:
        """Get the full state of a HA entity."""
        client = await self._ensure_client()
        try:
            resp = await client.get(f"/api/states/{entity_id}")
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("Failed to get state for %s: %s", entity_id, exc)
            return {}

    async def get_environment(
        self,
        climate_entity: str = "climate.main_thermostat",
        outdoor_sensor: str = "sensor.outdoor_temperature",
        indoor_sensor: str = "sensor.indoor_temperature",
    ) -> EnvironmentState:
        """Read current environment from HA sensors."""
        env = EnvironmentState()

        # Get outdoor temp
        outdoor = await self.get_state(outdoor_sensor)
        if outdoor.get("state") and outdoor["state"] != "unavailable":
            try:
                env.outdoor_temp_f = float(outdoor["state"])
            except (ValueError, TypeError):
                pass

        # Get indoor temp
        indoor = await self.get_state(indoor_sensor)
        if indoor.get("state") and indoor["state"] != "unavailable":
            try:
                env.indoor_temp_f = float(indoor["state"])
            except (ValueError, TypeError):
                pass

        # Get thermostat state
        climate = await self.get_state(climate_entity)
        if climate:
            env.hvac_mode = climate.get("state")  # "heat", "cool", "auto", "off"
            attrs = climate.get("attributes", {})
            env.hvac_action = attrs.get("hvac_action")  # "heating", "cooling", "idle"
            env.current_setpoint_f = attrs.get("temperature")

        return env


# ------------------------------------------------------------------
# Intent interpreter — maps natural language to device actions
# ------------------------------------------------------------------


class ComfortIntent(str, Enum):
    """What the user wants comfort-wise."""
    COOL_DOWN = "cool_down"         # "it's hot", "cool it down", "I'm sweating"
    WARM_UP = "warm_up"             # "it's cold", "warm it up", "freezing"
    LIGHTS_ON = "lights_on"
    LIGHTS_OFF = "lights_off"
    LIGHTS_DIM = "lights_dim"
    FAN_ON = "fan_on"
    FAN_OFF = "fan_off"
    SPECIFIC_TEMP = "specific_temp" # "set it to 72"
    LOCK = "lock"
    UNLOCK = "unlock"


# Pattern → comfort intent mapping
# Order matters: more specific patterns first (e.g. "warm it up" before bare "warm")
_COMFORT_PATTERNS: list[tuple[str, ComfortIntent]] = [
    # Warm up (check BEFORE cool_down since "warm" is ambiguous)
    (r"\bwarm\s*(it|the|this)\s*(up)\b", ComfortIntent.WARM_UP),
    (r"\bwarm\s+up\b", ComfortIntent.WARM_UP),
    (r"\b(cold|freezing|chilly|frigid|shivering)\b", ComfortIntent.WARM_UP),
    (r"\bheat\s*(it)?\s*up\b", ComfortIntent.WARM_UP),
    # Cool down (bare "warm" = "it's warm in here" = too hot)
    (r"\b(hot|warm|sweating|stuffy|humid|boiling|roasting|burning\s+up)\b", ComfortIntent.COOL_DOWN),
    (r"\bcool\s*(it|the|this)?\s*(down|off)\b", ComfortIntent.COOL_DOWN),
    (r"\bturn\s+(on\s+)?(the\s+)?(ac|air\s+condition)", ComfortIntent.COOL_DOWN),
    # Fan
    (r"\bfan\s+(on|off)\b", None),  # handled by on/off split below
    (r"\bturn\s+(on\s+)?(the\s+)?fan\b", ComfortIntent.FAN_ON),
    (r"\bturn\s+off\s+(the\s+)?fan\b", ComfortIntent.FAN_OFF),
    # Lights
    (r"\b(lights?|lites?)\s+on\b", ComfortIntent.LIGHTS_ON),
    (r"\b(lights?|lites?)\s+off\b", ComfortIntent.LIGHTS_OFF),
    (r"\bturn\s+(on\s+)?(the\s+)?(lights?|lites?)\b", ComfortIntent.LIGHTS_ON),
    (r"\bturn\s+off\s+(the\s+)?(lights?|lites?)\b", ComfortIntent.LIGHTS_OFF),
    (r"\b(kill|cut)\s+(the\s+)?(lights?|lites?)\b", ComfortIntent.LIGHTS_OFF),
    (r"\bdim\b", ComfortIntent.LIGHTS_DIM),
    (r"\b(dark|too\s+bright)\b", ComfortIntent.LIGHTS_DIM),
    # Temperature
    (r"\bset\s+(it\s+)?to\s+(\d+)\b", ComfortIntent.SPECIFIC_TEMP),
    (r"\b(\d+)\s*degrees?\b", ComfortIntent.SPECIFIC_TEMP),
    # Lock
    (r"\block\b", ComfortIntent.LOCK),
    (r"\bunlock\b", ComfortIntent.UNLOCK),
]


import re


def _infer_services(domain: str, supported_features: int) -> list[str]:
    """Infer available services from HA supported_features bitmask.

    Each domain has its own bitmask. We decode the common ones
    and always include turn_on/turn_off for switchable domains.
    """
    services: list[str] = []

    if domain in ("fan", "light", "switch", "cover"):
        services.extend(["turn_on", "turn_off"])

    if domain == "fan":
        if supported_features & 1:    # SET_SPEED / SET_PERCENTAGE
            services.append("set_percentage")
        if supported_features & 2:    # OSCILLATE
            services.append("oscillate")
        if supported_features & 4:    # DIRECTION
            services.append("set_direction")
        if supported_features & 8:    # PRESET_MODE
            services.append("set_preset_mode")

    elif domain == "climate":
        services.extend(["set_temperature", "set_hvac_mode", "turn_on", "turn_off"])
        if supported_features & 1:    # TARGET_TEMPERATURE
            pass  # already included
        if supported_features & 2:    # TARGET_TEMPERATURE_RANGE
            services.append("set_temperature_range")
        if supported_features & 4:    # TARGET_HUMIDITY
            services.append("set_humidity")
        if supported_features & 8:    # FAN_MODE
            services.append("set_fan_mode")
        if supported_features & 16:   # PRESET_MODE
            services.append("set_preset_mode")
        if supported_features & 32:   # SWING_MODE
            services.append("set_swing_mode")

    elif domain == "light":
        if supported_features & 1:    # BRIGHTNESS
            services.append("set_brightness")
        if supported_features & 2:    # COLOR_TEMP
            services.append("set_color_temp")
        if supported_features & 4:    # EFFECT
            services.append("set_effect")
        if supported_features & 16:   # COLOR
            services.append("set_color")

    elif domain == "cover":
        if supported_features & 1:    # OPEN
            services.append("open_cover")
        if supported_features & 2:    # CLOSE
            services.append("close_cover")
        if supported_features & 4:    # SET_POSITION
            services.append("set_cover_position")

    elif domain == "lock":
        services.extend(["lock", "unlock"])

    elif domain in ("sensor", "weather"):
        pass  # read-only

    return services


def interpret_comfort(text: str) -> tuple[ComfortIntent | None, dict]:
    """Parse a natural-language utterance into a comfort intent + extracted data.

    Returns (intent, data_dict) where data_dict may contain:
      - "target_temp": int if a specific temperature was mentioned
    """
    text_lower = text.lower()
    data: dict = {}

    # Extract temperature if mentioned
    temp_match = re.search(r"\b(\d{2,3})\s*(?:degrees?|°|f)?\b", text_lower)
    if temp_match:
        data["target_temp"] = int(temp_match.group(1))

    # Check patterns
    for pattern_str, intent in _COMFORT_PATTERNS:
        if re.search(pattern_str, text_lower, re.IGNORECASE):
            if intent is not None:
                if intent == ComfortIntent.SPECIFIC_TEMP and "target_temp" not in data:
                    continue  # No actual temp found
                return intent, data

    # Fan on/off disambiguation
    fan_match = re.search(r"\bfan\s+(on|off)\b", text_lower)
    if fan_match:
        if fan_match.group(1) == "on":
            return ComfortIntent.FAN_ON, data
        else:
            return ComfortIntent.FAN_OFF, data

    return None, data


# ------------------------------------------------------------------
# Abra agent — the orchestrator
# ------------------------------------------------------------------


class Abra:
    """Home automation agent that resolves context and executes actions.

    Takes a natural-language utterance + source device ID, figures out
    which room, what devices are available, what the user wants, checks
    environmental conditions, and builds the right HA service calls.
    """

    def __init__(
        self,
        registry: DeviceRegistry | None = None,
        env_reader: EnvironmentReader | None = None,
        ha_url: str = "",
        ha_token: str = "",
    ) -> None:
        self._registry = registry or DeviceRegistry.build_default()
        self._env_reader = env_reader
        self._ha_url = ha_url.rstrip("/") if ha_url else ""
        self._ha_token = ha_token

    async def close(self) -> None:
        if self._env_reader:
            await self._env_reader.close()

    async def handle(
        self,
        utterance: str,
        source_device_id: str = "",
        area_id: str = "",
    ) -> AbraResult:
        """Process a home automation utterance.

        Args:
            utterance: The natural-language text ("It's fucking hot in here!")
            source_device_id: Alexa/Echo device ID that captured the utterance
            area_id: If known, the room area_id directly (overrides device lookup)

        Returns:
            AbraResult with service calls to execute
        """
        # ── 1. Resolve room ────────────────────────────────────────
        room = None
        if area_id:
            room = self._registry.get_room(area_id)
        elif source_device_id:
            room = self._registry.resolve_room(source_device_id)

        if room is None:
            return AbraResult(
                success=False,
                error=(
                    f"Cannot determine room. source_device_id={source_device_id!r}, "
                    f"area_id={area_id!r}. Is the device registered?"
                ),
            )

        logger.info("Abra: resolved room=%s (%s)", room.area_id, room.name)

        # ── 2. Interpret comfort intent ─────────────────────────────
        comfort, data = interpret_comfort(utterance)
        if comfort is None:
            return AbraResult(
                success=False,
                room_resolved=room.name,
                error=f"Could not interpret comfort intent from: {utterance[:200]}",
            )

        logger.info("Abra: comfort_intent=%s data=%s", comfort.value, data)

        # ── 3. Build service calls based on intent ──────────────────
        if comfort == ComfortIntent.COOL_DOWN:
            return await self._handle_cool_down(room, data)
        elif comfort == ComfortIntent.WARM_UP:
            return await self._handle_warm_up(room, data)
        elif comfort == ComfortIntent.FAN_ON:
            return self._handle_fan(room, turn_on=True)
        elif comfort == ComfortIntent.FAN_OFF:
            return self._handle_fan(room, turn_on=False)
        elif comfort == ComfortIntent.SPECIFIC_TEMP:
            return self._handle_set_temp(room, data.get("target_temp", 72))
        elif comfort in (ComfortIntent.LIGHTS_ON, ComfortIntent.LIGHTS_OFF, ComfortIntent.LIGHTS_DIM):
            return self._handle_lights(room, comfort)
        elif comfort in (ComfortIntent.LOCK, ComfortIntent.UNLOCK):
            return self._handle_lock(room, comfort)
        else:
            return AbraResult(
                success=False,
                room_resolved=room.name,
                error=f"Unhandled comfort intent: {comfort.value}",
            )

    # ── Cool down logic ─────────────────────────────────────────────

    async def _handle_cool_down(self, room: Room, data: dict) -> AbraResult:
        """User is hot. Turn on fan immediately. Check HVAC state for thermostat."""
        calls: list[HAServiceCall] = []
        reasoning_parts: list[str] = []

        # Always turn on the fan in this room first (immediate relief)
        fans = room.devices_by_domain(DeviceDomain.FAN)
        if fans:
            fan = fans[0]
            calls.append(HAServiceCall(
                domain="fan",
                service="turn_on",
                entity_id=fan.entity_id,
            ))
            reasoning_parts.append(f"Turning on {fan.friendly_name} for immediate relief")

        # Check thermostat situation
        climate_devices = room.devices_by_domain(DeviceDomain.CLIMATE)
        if not climate_devices:
            # Check other rooms — thermostat might be elsewhere
            for r in self._registry.all_rooms():
                climate_devices = r.devices_by_domain(DeviceDomain.CLIMATE)
                if climate_devices:
                    break

        if climate_devices:
            thermostat = climate_devices[0]
            env = await self._read_environment(thermostat.entity_id)

            if env and env.hvac_mode == "heat" and env.hvac_action == "heating":
                # Heat is running — need to decide: AC or lower setpoint
                if env.outdoor_temp_f is not None and env.outdoor_temp_f >= AC_MIN_OUTDOOR_TEMP:
                    # Outdoor temp is warm enough — switch to cool
                    calls.append(HAServiceCall(
                        domain="climate",
                        service="set_hvac_mode",
                        entity_id=thermostat.entity_id,
                        data={"hvac_mode": "cool"},
                    ))
                    reasoning_parts.append(
                        f"Outdoor temp is {env.outdoor_temp_f}°F (>= {AC_MIN_OUTDOOR_TEMP}°F) — "
                        f"safe to switch to AC"
                    )
                else:
                    # Too cold outside for AC — just lower the setpoint
                    new_setpoint = (env.current_setpoint_f or 72) - SETPOINT_STEP
                    calls.append(HAServiceCall(
                        domain="climate",
                        service="set_temperature",
                        entity_id=thermostat.entity_id,
                        data={"temperature": new_setpoint},
                    ))
                    outdoor_desc = (
                        f"{env.outdoor_temp_f}°F" if env.outdoor_temp_f is not None
                        else "unknown"
                    )
                    reasoning_parts.append(
                        f"Outdoor temp is {outdoor_desc} (< {AC_MIN_OUTDOOR_TEMP}°F) — "
                        f"too cold for AC, lowering setpoint to {new_setpoint}°F"
                    )
            elif env and env.hvac_mode in ("cool", "auto"):
                # Already cooling — lower the setpoint
                new_setpoint = (env.current_setpoint_f or 72) - SETPOINT_STEP
                calls.append(HAServiceCall(
                    domain="climate",
                    service="set_temperature",
                    entity_id=thermostat.entity_id,
                    data={"temperature": new_setpoint},
                ))
                reasoning_parts.append(
                    f"Already in {env.hvac_mode} mode — lowering setpoint to {new_setpoint}°F"
                )
            elif env and env.hvac_mode == "off":
                # HVAC is off — turn on cool mode
                calls.append(HAServiceCall(
                    domain="climate",
                    service="set_hvac_mode",
                    entity_id=thermostat.entity_id,
                    data={"hvac_mode": "cool"},
                ))
                reasoning_parts.append("HVAC was off — switching to cool mode")
            else:
                # No env data — just try to lower setpoint conservatively
                calls.append(HAServiceCall(
                    domain="climate",
                    service="set_temperature",
                    entity_id=thermostat.entity_id,
                    data={"temperature": 70},
                ))
                reasoning_parts.append(
                    "Could not read current HVAC state — setting to 70°F as safe default"
                )

        if not calls:
            return AbraResult(
                success=False,
                room_resolved=room.name,
                error=f"No fan or thermostat found accessible from {room.name}",
            )

        return AbraResult(
            success=True,
            service_calls=calls,
            room_resolved=room.name,
            reasoning=". ".join(reasoning_parts),
        )

    # ── Warm up logic ───────────────────────────────────────────────

    async def _handle_warm_up(self, room: Room, data: dict) -> AbraResult:
        """User is cold. Turn off fan, raise setpoint or switch to heat."""
        calls: list[HAServiceCall] = []
        reasoning_parts: list[str] = []

        # Turn off fan if running (makes you colder)
        fans = room.devices_by_domain(DeviceDomain.FAN)
        if fans:
            calls.append(HAServiceCall(
                domain="fan",
                service="turn_off",
                entity_id=fans[0].entity_id,
            ))
            reasoning_parts.append(f"Turning off {fans[0].friendly_name}")

        # Raise thermostat
        climate_devices = room.devices_by_domain(DeviceDomain.CLIMATE)
        if not climate_devices:
            for r in self._registry.all_rooms():
                climate_devices = r.devices_by_domain(DeviceDomain.CLIMATE)
                if climate_devices:
                    break

        if climate_devices:
            thermostat = climate_devices[0]
            env = await self._read_environment(thermostat.entity_id)

            if env and env.hvac_mode in ("cool", "auto"):
                # Switch to heat
                calls.append(HAServiceCall(
                    domain="climate",
                    service="set_hvac_mode",
                    entity_id=thermostat.entity_id,
                    data={"hvac_mode": "heat"},
                ))
                reasoning_parts.append("Switching HVAC to heat mode")
            elif env and env.hvac_mode == "heat":
                # Already heating — raise setpoint
                new_setpoint = (env.current_setpoint_f or 70) + SETPOINT_STEP
                calls.append(HAServiceCall(
                    domain="climate",
                    service="set_temperature",
                    entity_id=thermostat.entity_id,
                    data={"temperature": new_setpoint},
                ))
                reasoning_parts.append(f"Raising setpoint to {new_setpoint}°F")
            else:
                # Off or unknown — turn on heat
                calls.append(HAServiceCall(
                    domain="climate",
                    service="set_hvac_mode",
                    entity_id=thermostat.entity_id,
                    data={"hvac_mode": "heat"},
                ))
                reasoning_parts.append("Turning on heat")

        return AbraResult(
            success=True,
            service_calls=calls,
            room_resolved=room.name,
            reasoning=". ".join(reasoning_parts),
        )

    # ── Simple device controls ──────────────────────────────────────

    def _handle_fan(self, room: Room, turn_on: bool) -> AbraResult:
        """Turn fan on or off in the given room."""
        fans = room.devices_by_domain(DeviceDomain.FAN)
        if not fans:
            return AbraResult(
                success=False,
                room_resolved=room.name,
                error=f"No fan found in {room.name}",
            )
        service = "turn_on" if turn_on else "turn_off"
        return AbraResult(
            success=True,
            service_calls=[
                HAServiceCall(domain="fan", service=service, entity_id=fans[0].entity_id)
            ],
            room_resolved=room.name,
            reasoning=f"{'Turning on' if turn_on else 'Turning off'} {fans[0].friendly_name}",
        )

    def _handle_set_temp(self, room: Room, target: int) -> AbraResult:
        """Set thermostat to a specific temperature."""
        # Find thermostat — may be in this room or elsewhere
        climate_devices = room.devices_by_domain(DeviceDomain.CLIMATE)
        if not climate_devices:
            for r in self._registry.all_rooms():
                climate_devices = r.devices_by_domain(DeviceDomain.CLIMATE)
                if climate_devices:
                    break
        if not climate_devices:
            return AbraResult(
                success=False,
                room_resolved=room.name,
                error="No thermostat found",
            )
        thermostat = climate_devices[0]
        return AbraResult(
            success=True,
            service_calls=[
                HAServiceCall(
                    domain="climate",
                    service="set_temperature",
                    entity_id=thermostat.entity_id,
                    data={"temperature": target},
                )
            ],
            room_resolved=room.name,
            reasoning=f"Setting {thermostat.friendly_name} to {target}°F",
        )

    def _handle_lights(self, room: Room, intent: ComfortIntent) -> AbraResult:
        """Control lights in the room."""
        lights = room.devices_by_domain(DeviceDomain.LIGHT)
        if not lights:
            return AbraResult(
                success=False,
                room_resolved=room.name,
                error=f"No lights found in {room.name}",
            )
        calls = []
        for light in lights:
            if intent == ComfortIntent.LIGHTS_ON:
                calls.append(HAServiceCall(domain="light", service="turn_on", entity_id=light.entity_id))
            elif intent == ComfortIntent.LIGHTS_OFF:
                calls.append(HAServiceCall(domain="light", service="turn_off", entity_id=light.entity_id))
            elif intent == ComfortIntent.LIGHTS_DIM:
                calls.append(HAServiceCall(
                    domain="light", service="turn_on", entity_id=light.entity_id,
                    data={"brightness_pct": 30},
                ))
        action = {"lights_on": "on", "lights_off": "off", "lights_dim": "dimmed"}[intent.value]
        return AbraResult(
            success=True,
            service_calls=calls,
            room_resolved=room.name,
            reasoning=f"Turning lights {action} in {room.name}",
        )

    def _handle_lock(self, room: Room, intent: ComfortIntent) -> AbraResult:
        """Lock or unlock in the room."""
        locks = room.devices_by_domain(DeviceDomain.LOCK)
        if not locks:
            return AbraResult(
                success=False,
                room_resolved=room.name,
                error=f"No lock found in {room.name}",
            )
        service = "lock" if intent == ComfortIntent.LOCK else "unlock"
        return AbraResult(
            success=True,
            service_calls=[
                HAServiceCall(domain="lock", service=service, entity_id=locks[0].entity_id)
            ],
            room_resolved=room.name,
            reasoning=f"{'Locking' if service == 'lock' else 'Unlocking'} {locks[0].friendly_name}",
        )

    # ── Environment helpers ─────────────────────────────────────────

    async def _read_environment(self, climate_entity: str) -> EnvironmentState | None:
        """Read current environmental state from HA. Returns None if no reader."""
        if self._env_reader is None:
            return None
        try:
            return await self._env_reader.get_environment(climate_entity=climate_entity)
        except Exception as exc:
            logger.warning("Failed to read environment: %s", exc)
            return None

    # ── HA execution ────────────────────────────────────────────────

    async def execute(self, result: AbraResult) -> list[dict]:
        """Actually execute the service calls against Home Assistant.

        Returns list of HA API responses. This is separate from handle()
        so the conductor can inspect/approve before executing.
        """
        if not self._ha_url or not self._ha_token:
            logger.warning("No HA URL/token configured — dry run only")
            return [{"dry_run": True, "call": f"{c.domain}.{c.service}({c.entity_id})"}
                    for c in result.service_calls]

        responses = []
        async with httpx.AsyncClient(
            base_url=self._ha_url,
            headers={"Authorization": f"Bearer {self._ha_token}"},
            timeout=10,
        ) as client:
            for call in result.service_calls:
                try:
                    resp = await client.post(
                        f"/api/services/{call.domain}/{call.service}",
                        json=call.to_ha_payload(),
                    )
                    resp.raise_for_status()
                    responses.append(resp.json())
                except Exception as exc:
                    logger.error(
                        "HA service call failed: %s.%s(%s): %s",
                        call.domain, call.service, call.entity_id, exc,
                    )
                    responses.append({"error": str(exc)})

        return responses
