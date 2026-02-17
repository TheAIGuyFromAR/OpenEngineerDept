"""Tests for SlotManager — lifecycle, slot 0 protection, acquire/release."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from gateway.config import GatewayConfig
from gateway.slot_manager import SlotManager


@pytest.fixture
def config():
    return GatewayConfig(
        llama_server_url="http://fake:8080",
        template_slot_id=0,
        worker_slot_ids=[1, 2, 3, 4],
    )


@pytest.fixture
def manager(config):
    return SlotManager(config)


# ------------------------------------------------------------------
# Slot 0 protection
# ------------------------------------------------------------------


class TestSlotZeroProtection:
    async def test_restore_into_template_slot_raises(self, manager):
        """Slot 0 is sacred — cannot be used for restore."""
        with pytest.raises(ValueError, match="Cannot restore into template slot"):
            await manager.restore_to_worker("proj1", worker_slot_id=0)

    async def test_release_template_slot_is_noop(self, manager):
        """Releasing slot 0 should log error but not crash."""
        initial_available = manager.available_worker_count
        manager.release_workers([0])
        # Should NOT add slot 0 to the worker pool
        assert manager.available_worker_count == initial_available


# ------------------------------------------------------------------
# Worker acquisition
# ------------------------------------------------------------------


class TestWorkerAcquisition:
    async def test_acquire_single_worker(self, manager):
        workers = await manager.acquire_workers(1)
        assert len(workers) == 1
        assert workers[0] in [1, 2, 3, 4]
        assert manager.available_worker_count == 3

    async def test_acquire_all_workers(self, manager):
        workers = await manager.acquire_workers(4)
        assert len(workers) == 4
        assert set(workers) == {1, 2, 3, 4}
        assert manager.available_worker_count == 0

    async def test_acquire_too_many_raises(self, manager):
        with pytest.raises(ValueError, match="only 4 exist"):
            await manager.acquire_workers(5)

    async def test_acquire_blocks_then_succeeds(self, manager):
        """Acquiring when pool is empty blocks until release."""
        all_workers = await manager.acquire_workers(4)

        async def delayed_release():
            await asyncio.sleep(0.1)
            manager.release_workers([all_workers[0]])

        asyncio.create_task(delayed_release())
        workers = await manager.acquire_workers(1, timeout=2.0)
        assert len(workers) == 1

        # Cleanup
        manager.release_workers(workers)
        manager.release_workers(all_workers[1:])

    async def test_acquire_timeout(self, manager):
        """Timeout when no workers available."""
        await manager.acquire_workers(4)  # exhaust pool
        with pytest.raises(TimeoutError):
            await manager.acquire_workers(1, timeout=0.1)


# ------------------------------------------------------------------
# Release
# ------------------------------------------------------------------


class TestWorkerRelease:
    async def test_release_returns_to_pool(self, manager):
        workers = await manager.acquire_workers(2)
        assert manager.available_worker_count == 2
        manager.release_workers(workers)
        assert manager.available_worker_count == 4

    async def test_release_allows_reacquire(self, manager):
        w1 = await manager.acquire_workers(4)
        manager.release_workers(w1)
        w2 = await manager.acquire_workers(4)
        assert set(w2) == {1, 2, 3, 4}
        manager.release_workers(w2)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


class TestMetrics:
    async def test_metrics_initially_empty(self, manager):
        assert manager.get_metrics() == []

    async def test_clear_metrics(self, manager):
        manager._metrics.append(None)  # dummy
        manager.clear_metrics()
        assert manager.get_metrics() == []
