"""Tests for Ultra Think — tier dispatch, diversity, concurrent generation."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from gateway.config import GatewayConfig
from gateway.slot_manager import SlotManager
from gateway.ultra_think import UltraThink, TIER_N, DIVERSITY_PROFILES


@pytest.fixture
def config():
    return GatewayConfig(
        llama_server_url="http://fake:8080",
        worker_slot_ids=[1, 2, 3, 4],
    )


@pytest.fixture
def slot_manager(config):
    return SlotManager(config)


def _make_completion_response(content: str = "print('hello')", tokens: int = 10):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"completion_tokens": tokens, "prompt_tokens": 50},
        },
    )


# ------------------------------------------------------------------
# Tier configuration
# ------------------------------------------------------------------


class TestTierConfig:
    def test_tier_1_single(self):
        assert TIER_N[1] == 1

    def test_tier_2_triple(self):
        assert TIER_N[2] == 3

    def test_tier_3_five(self):
        assert TIER_N[3] == 5

    def test_diversity_profiles_count(self):
        assert len(DIVERSITY_PROFILES) >= 5

    def test_profiles_have_distinct_temperatures(self):
        temps = [p["temperature"] for p in DIVERSITY_PROFILES[:3]]
        assert len(set(temps)) == 3


# ------------------------------------------------------------------
# Tier 2 concurrent dispatch
# ------------------------------------------------------------------


class TestTier2Dispatch:
    async def test_tier2_generates_3_candidates(self, config, slot_manager):
        ut = UltraThink(config, slot_manager)

        responses = [_make_completion_response(f"candidate_{i}") for i in range(3)]
        call_count = 0

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return resp

        transport = httpx.MockTransport(mock_handler)
        ut._client = httpx.AsyncClient(transport=transport)

        result = await ut.generate(
            task_id="test-1",
            prompt="Write hello world",
            system_prompt="You are a coder",
            tier=2,
        )

        assert result.tier == 2
        assert len(result.candidates) == 3
        assert result.errors == []
        assert result.timing.total_ms > 0

        await ut.close()

    async def test_tier2_uses_different_sampling(self, config, slot_manager):
        """Each candidate should use different sampling parameters."""
        ut = UltraThink(config, slot_manager)
        captured_bodies = []

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            captured_bodies.append(body)
            return _make_completion_response()

        transport = httpx.MockTransport(mock_handler)
        ut._client = httpx.AsyncClient(transport=transport)

        await ut.generate(
            task_id="test-2",
            prompt="task",
            system_prompt="sys",
            tier=2,
        )

        temps = [b["temperature"] for b in captured_bodies]
        assert len(set(temps)) == 3  # All different

        await ut.close()

    async def test_tier2_pins_to_different_slots(self, config, slot_manager):
        """Each generation should be pinned to a different worker slot."""
        ut = UltraThink(config, slot_manager)
        captured_slots = []

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            captured_slots.append(body.get("id_slot"))
            return _make_completion_response()

        transport = httpx.MockTransport(mock_handler)
        ut._client = httpx.AsyncClient(transport=transport)

        await ut.generate(
            task_id="test-3",
            prompt="task",
            system_prompt="sys",
            tier=2,
        )

        assert len(set(captured_slots)) == 3  # All different slots

        await ut.close()


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


class TestErrorHandling:
    async def test_tier4_raises(self, config, slot_manager):
        ut = UltraThink(config, slot_manager)
        with pytest.raises(ValueError, match="Tier 4"):
            await ut.generate(
                task_id="test-4",
                prompt="task",
                system_prompt="sys",
                tier=4,
            )
        await ut.close()

    async def test_partial_failure_returns_successful_candidates(
        self, config, slot_manager
    ):
        """If one generation fails, other results still returned."""
        ut = UltraThink(config, slot_manager)
        call_count = 0

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return httpx.Response(500, json={"error": "internal"})
            return _make_completion_response()

        transport = httpx.MockTransport(mock_handler)
        ut._client = httpx.AsyncClient(transport=transport)

        result = await ut.generate(
            task_id="test-5",
            prompt="task",
            system_prompt="sys",
            tier=2,
        )

        # Should have 2 successes and 1 error
        assert len(result.candidates) == 2
        assert len(result.errors) == 1

        await ut.close()

    async def test_workers_released_after_generation(self, config, slot_manager):
        """Workers must be returned to pool even after errors."""
        ut = UltraThink(config, slot_manager)

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return _make_completion_response()

        transport = httpx.MockTransport(mock_handler)
        ut._client = httpx.AsyncClient(transport=transport)

        initial_available = slot_manager.available_worker_count
        await ut.generate(
            task_id="test-6",
            prompt="task",
            system_prompt="sys",
            tier=2,
        )
        assert slot_manager.available_worker_count == initial_available

        await ut.close()
