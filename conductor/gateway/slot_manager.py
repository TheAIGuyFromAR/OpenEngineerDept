"""
Slot Manager — KV cache slot lifecycle for llama-server.

Slot allocation:
  Slot 0: SACRED template slot. Never used for generation.
          Holds the pre-warmed project context KV cache.
  Slots 1-4: Worker slots for actual generation.

KV cache operations talk to llama-server's /slots API:
  GET  /slots                               — list all slots
  POST /slots/{id}?action=save&filename=X   — save KV cache to disk
  POST /slots/{id}?action=restore&filename=X — restore from disk
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

from .config import GatewayConfig

logger = logging.getLogger(__name__)


@dataclass
class SlotMetrics:
    slot_id: int
    operation: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error: str | None = None


class SlotManager:
    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.llama_server_url,
            timeout=config.slot_restore_timeout_seconds,
        )
        self._template_slot = config.template_slot_id
        self._worker_ids = list(config.worker_slot_ids)
        # Track which workers are currently in use
        self._available = asyncio.Queue[int]()
        for wid in self._worker_ids:
            self._available.put_nowait(wid)
        self._metrics: list[SlotMetrics] = []

    # ------------------------------------------------------------------
    # Worker slot acquisition
    # ------------------------------------------------------------------

    async def acquire_workers(self, n: int, timeout: float | None = None) -> list[int]:
        """Acquire N worker slots. Blocks until all are available."""
        if n > len(self._worker_ids):
            raise ValueError(
                f"Requested {n} workers but only {len(self._worker_ids)} exist"
            )
        slots: list[int] = []
        effective_timeout = timeout or self._config.generation_timeout_seconds
        for _ in range(n):
            try:
                slot_id = await asyncio.wait_for(
                    self._available.get(), timeout=effective_timeout
                )
                slots.append(slot_id)
            except asyncio.TimeoutError:
                # Return any already-acquired slots
                for s in slots:
                    self._available.put_nowait(s)
                raise TimeoutError(
                    f"Timed out acquiring worker slots (got {len(slots)}/{n})"
                )
        return slots

    def release_workers(self, slot_ids: list[int]) -> None:
        """Return worker slots to the pool."""
        for sid in slot_ids:
            if sid == self._template_slot:
                logger.error("BUG: attempted to release template slot %d", sid)
                continue
            self._available.put_nowait(sid)

    # ------------------------------------------------------------------
    # KV cache operations
    # ------------------------------------------------------------------

    async def save_template(self, project_id: str) -> SlotMetrics:
        """Save template slot 0 KV cache to disk."""
        filename = f"template-{project_id}"
        return await self._slot_action(self._template_slot, "save", filename)

    async def restore_to_worker(
        self, project_id: str, worker_slot_id: int
    ) -> SlotMetrics:
        """Restore template cache into a worker slot."""
        if worker_slot_id == self._template_slot:
            raise ValueError("Cannot restore into template slot")
        filename = f"template-{project_id}"
        return await self._slot_action(worker_slot_id, "restore", filename)

    async def restore_workers_parallel(
        self, project_id: str, worker_ids: list[int]
    ) -> list[SlotMetrics]:
        """Restore template cache into multiple worker slots concurrently."""
        tasks = [self.restore_to_worker(project_id, wid) for wid in worker_ids]
        return list(await asyncio.gather(*tasks))

    async def warm_template(
        self, project_id: str, layer0_text: str
    ) -> SlotMetrics:
        """
        Warm slot 0 with project context by sending a prompt-only request.
        This computes the KV cache for the prefix, then saves it to disk.
        """
        start = time.monotonic()
        try:
            # Send a minimal completion request to slot 0 to compute prefix KV
            resp = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": "conductor-template",
                    "messages": [{"role": "system", "content": layer0_text}],
                    "max_tokens": 1,  # We only want the prefix computed
                    "id_slot": self._template_slot,
                    "cache_prompt": True,
                },
                timeout=self._config.generation_timeout_seconds,
            )
            resp.raise_for_status()

            # Now save the computed KV cache
            await self.save_template(project_id)

            elapsed = (time.monotonic() - start) * 1000
            metric = SlotMetrics(
                slot_id=self._template_slot,
                operation="warm_template",
                duration_ms=elapsed,
            )
            self._metrics.append(metric)
            return metric
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            metric = SlotMetrics(
                slot_id=self._template_slot,
                operation="warm_template",
                duration_ms=elapsed,
                success=False,
                error=str(exc),
            )
            self._metrics.append(metric)
            raise

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_slots_status(self) -> list[dict]:
        """Query llama-server for current slot states."""
        try:
            resp = await self._client.get("/slots")
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return []

    @property
    def available_worker_count(self) -> int:
        return self._available.qsize()

    def get_metrics(self) -> list[SlotMetrics]:
        return list(self._metrics)

    def clear_metrics(self) -> None:
        self._metrics.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _slot_action(
        self, slot_id: int, action: str, filename: str
    ) -> SlotMetrics:
        start = time.monotonic()
        try:
            resp = await self._client.post(
                f"/slots/{slot_id}",
                params={"action": action, "filename": filename},
            )
            resp.raise_for_status()
            elapsed = (time.monotonic() - start) * 1000
            metric = SlotMetrics(
                slot_id=slot_id,
                operation=f"slot_{action}",
                duration_ms=elapsed,
            )
            self._metrics.append(metric)
            logger.debug("Slot %d %s '%s' in %.1fms", slot_id, action, filename, elapsed)
            return metric
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            metric = SlotMetrics(
                slot_id=slot_id,
                operation=f"slot_{action}",
                duration_ms=elapsed,
                success=False,
                error=str(exc),
            )
            self._metrics.append(metric)
            raise

    async def close(self) -> None:
        await self._client.aclose()
