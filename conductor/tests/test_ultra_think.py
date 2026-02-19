"""Tests for Ultra Think — tier dispatch, diversity, concurrent generation.

Tests both local (slot-pinned) and API (concurrent requests) paths."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from gateway.config import GatewayConfig
from gateway.providers.base import CompletionResult, InferenceProvider
from gateway.providers.local import LocalProvider
from gateway.slot_manager import SlotManager
from gateway.ultra_think import UltraThink, TIER_N, DIVERSITY_PROFILES


@pytest.fixture
def config():
    return GatewayConfig(
        llama_server_url="http://fake:8080",
        worker_slot_ids=[1, 2, 3, 4],
    )


@pytest.fixture
def api_config():
    return GatewayConfig(
        inference_provider="anthropic",
        inference_api_key="sk-ant-test",
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


class MockProvider(InferenceProvider):
    """Test provider that records calls and returns canned responses."""

    def __init__(self, responses: list[str] | None = None):
        self.calls: list[dict] = []
        self._responses = responses or ["print('hello')"]
        self._call_count = 0
        self._fail_on: set[int] = set()

    def fail_on_call(self, call_index: int) -> None:
        self._fail_on.add(call_index)

    async def chat_completion(
        self,
        *,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: list[str] | None = None,
        extra: dict | None = None,
    ) -> CompletionResult:
        idx = self._call_count
        self._call_count += 1

        self.calls.append({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "extra": extra,
        })

        if idx in self._fail_on:
            raise httpx.HTTPStatusError(
                "Server error",
                request=httpx.Request("POST", "http://fake"),
                response=httpx.Response(500),
            )

        content = self._responses[min(idx, len(self._responses) - 1)]
        return CompletionResult(
            content=content,
            model="test-model",
            usage={"completion_tokens": 10, "prompt_tokens": 50},
            raw_response={},
        )

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    @property
    def supports_slots(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return "test-mock"


class MockAPIProvider(MockProvider):
    """Mock that does NOT support slots — simulates API providers."""

    @property
    def supports_slots(self) -> bool:
        return False

    @property
    def provider_name(self) -> str:
        return "test-mock-api"


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
# Local path: Tier 2 concurrent dispatch (slot-pinned)
# ------------------------------------------------------------------


class TestLocalTier2Dispatch:
    async def test_tier2_generates_3_candidates(self, config, slot_manager):
        provider = MockProvider(["cand_0", "cand_1", "cand_2"])
        ut = UltraThink(config, slot_manager, provider)

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

    async def test_tier2_uses_different_sampling(self, config, slot_manager):
        """Each candidate should use different sampling parameters."""
        provider = MockProvider()
        ut = UltraThink(config, slot_manager, provider)

        await ut.generate(
            task_id="test-2",
            prompt="task",
            system_prompt="sys",
            tier=2,
        )

        temps = [c["temperature"] for c in provider.calls]
        assert len(set(temps)) == 3  # All different

    async def test_tier2_pins_to_different_slots(self, config, slot_manager):
        """Each generation should be pinned to a different worker slot."""
        provider = MockProvider()
        ut = UltraThink(config, slot_manager, provider)

        await ut.generate(
            task_id="test-3",
            prompt="task",
            system_prompt="sys",
            tier=2,
        )

        slots = [c["extra"]["id_slot"] for c in provider.calls]
        assert len(set(slots)) == 3  # All different slots

    async def test_workers_released_after_generation(self, config, slot_manager):
        """Workers must be returned to pool even after errors."""
        provider = MockProvider()
        ut = UltraThink(config, slot_manager, provider)

        initial_available = slot_manager.available_worker_count
        await ut.generate(
            task_id="test-6",
            prompt="task",
            system_prompt="sys",
            tier=2,
        )
        assert slot_manager.available_worker_count == initial_available


# ------------------------------------------------------------------
# API path: Tier 2 concurrent dispatch (no slots)
# ------------------------------------------------------------------


class TestAPITier2Dispatch:
    async def test_api_tier2_generates_3_candidates(self, api_config):
        provider = MockAPIProvider(["api_0", "api_1", "api_2"])
        ut = UltraThink(api_config, None, provider)

        result = await ut.generate(
            task_id="api-1",
            prompt="Write hello world",
            system_prompt="You are a coder",
            tier=2,
        )

        assert result.tier == 3  # tier is set to len(profiles)=3 in _generate_api
        assert len(result.candidates) == 3
        assert result.errors == []
        assert result.timing.slot_restore_ms == 0.0  # No slot restore for API

    async def test_api_no_slot_pinning(self, api_config):
        """API provider should never receive slot IDs."""
        provider = MockAPIProvider()
        ut = UltraThink(api_config, None, provider)

        await ut.generate(
            task_id="api-2",
            prompt="task",
            system_prompt="sys",
            tier=2,
        )

        for call in provider.calls:
            # extra should be None or empty (no id_slot)
            assert call["extra"] is None or "id_slot" not in call.get("extra", {})

    async def test_api_uses_different_sampling(self, api_config):
        provider = MockAPIProvider()
        ut = UltraThink(api_config, None, provider)

        await ut.generate(
            task_id="api-3",
            prompt="task",
            system_prompt="sys",
            tier=2,
        )

        temps = [c["temperature"] for c in provider.calls]
        assert len(set(temps)) == 3


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


class TestErrorHandling:
    async def test_tier4_raises(self, config, slot_manager):
        provider = MockProvider()
        ut = UltraThink(config, slot_manager, provider)
        with pytest.raises(ValueError, match="Tier 4"):
            await ut.generate(
                task_id="test-4",
                prompt="task",
                system_prompt="sys",
                tier=4,
            )

    async def test_partial_failure_returns_successful_candidates(
        self, config, slot_manager
    ):
        """If one generation fails, other results still returned."""
        provider = MockProvider()
        provider.fail_on_call(1)
        ut = UltraThink(config, slot_manager, provider)

        result = await ut.generate(
            task_id="test-5",
            prompt="task",
            system_prompt="sys",
            tier=2,
        )

        assert len(result.candidates) == 2
        assert len(result.errors) == 1

    async def test_workers_released_on_failure(self, config, slot_manager):
        """Workers must be returned to pool even when generations fail."""
        provider = MockProvider()
        provider.fail_on_call(0)
        provider.fail_on_call(1)
        provider.fail_on_call(2)
        ut = UltraThink(config, slot_manager, provider)

        initial = slot_manager.available_worker_count
        result = await ut.generate(
            task_id="test-7",
            prompt="task",
            system_prompt="sys",
            tier=2,
        )
        assert len(result.candidates) == 0
        assert len(result.errors) == 3
        assert slot_manager.available_worker_count == initial

    async def test_api_partial_failure(self, api_config):
        """API path also handles partial failures gracefully."""
        provider = MockAPIProvider()
        provider.fail_on_call(1)
        ut = UltraThink(api_config, None, provider)

        result = await ut.generate(
            task_id="api-err-1",
            prompt="task",
            system_prompt="sys",
            tier=2,
        )

        assert len(result.candidates) == 2
        assert len(result.errors) == 1
