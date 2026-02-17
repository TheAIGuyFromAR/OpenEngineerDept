"""Tests for PrefixCacheManager — hash-based invalidation and cache lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig
from gateway.prefix_cache import PrefixCacheManager
from gateway.slot_manager import SlotManager


@pytest.fixture
def tmp_cache_dir(tmp_path):
    return str(tmp_path / "kv-cache")


@pytest.fixture
def config(tmp_cache_dir, tmp_path):
    return GatewayConfig(
        llama_server_url="http://fake:8080",
        kv_cache_dir=tmp_cache_dir,
        metrics_log_path=str(tmp_path / "metrics" / "gateway.jsonl"),
    )


@pytest.fixture
def cache_mgr(config):
    return PrefixCacheManager(config)


@pytest.fixture
def mock_slot_manager():
    mgr = AsyncMock(spec=SlotManager)
    mgr.warm_template = AsyncMock()
    mgr.save_template = AsyncMock()
    return mgr


# ------------------------------------------------------------------
# Content hashing
# ------------------------------------------------------------------


class TestContentHashing:
    def test_same_content_same_hash(self):
        h1 = PrefixCacheManager.compute_hash("hello world")
        h2 = PrefixCacheManager.compute_hash("hello world")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = PrefixCacheManager.compute_hash("hello")
        h2 = PrefixCacheManager.compute_hash("world")
        assert h1 != h2

    def test_hash_is_16_chars(self):
        h = PrefixCacheManager.compute_hash("test")
        assert len(h) == 16


# ------------------------------------------------------------------
# Cache validity
# ------------------------------------------------------------------


class TestCacheValidity:
    def test_no_cache_is_invalid(self, cache_mgr):
        assert not cache_mgr.has_valid_cache("proj1", "some content")

    async def test_valid_after_ensure(self, cache_mgr, mock_slot_manager):
        layer0 = "# Project constraints\nUse Python 3.11+"
        await cache_mgr.ensure_loaded("proj1", layer0, mock_slot_manager)
        assert cache_mgr.has_valid_cache("proj1", layer0)

    async def test_invalid_after_content_change(self, cache_mgr, mock_slot_manager):
        layer0_v1 = "version 1 constraints"
        layer0_v2 = "version 2 constraints"

        await cache_mgr.ensure_loaded("proj1", layer0_v1, mock_slot_manager)
        assert cache_mgr.has_valid_cache("proj1", layer0_v1)
        assert not cache_mgr.has_valid_cache("proj1", layer0_v2)


# ------------------------------------------------------------------
# Cache reuse vs recompute
# ------------------------------------------------------------------


class TestCacheReuse:
    async def test_first_load_returns_recompute(self, cache_mgr, mock_slot_manager):
        result = await cache_mgr.ensure_loaded(
            "proj1", "content", mock_slot_manager
        )
        assert result == "recompute"
        mock_slot_manager.warm_template.assert_called_once()

    async def test_second_load_same_content_returns_hit(
        self, cache_mgr, mock_slot_manager
    ):
        layer0 = "same content"
        await cache_mgr.ensure_loaded("proj1", layer0, mock_slot_manager)
        mock_slot_manager.warm_template.reset_mock()

        result = await cache_mgr.ensure_loaded("proj1", layer0, mock_slot_manager)
        assert result == "hit"
        mock_slot_manager.warm_template.assert_not_called()

    async def test_changed_content_recomputes(self, cache_mgr, mock_slot_manager):
        await cache_mgr.ensure_loaded("proj1", "v1", mock_slot_manager)
        mock_slot_manager.warm_template.reset_mock()

        result = await cache_mgr.ensure_loaded("proj1", "v2", mock_slot_manager)
        assert result == "recompute"
        mock_slot_manager.warm_template.assert_called_once()


# ------------------------------------------------------------------
# Invalidation
# ------------------------------------------------------------------


class TestInvalidation:
    async def test_invalidate_clears_cache(self, cache_mgr, mock_slot_manager):
        await cache_mgr.ensure_loaded("proj1", "content", mock_slot_manager)
        assert cache_mgr.has_valid_cache("proj1", "content")

        cache_mgr.invalidate("proj1")
        assert not cache_mgr.has_valid_cache("proj1", "content")

    def test_invalidate_nonexistent_is_noop(self, cache_mgr):
        cache_mgr.invalidate("nonexistent")  # Should not raise


# ------------------------------------------------------------------
# Stats tracking
# ------------------------------------------------------------------


class TestStats:
    async def test_stats_recorded(self, cache_mgr, mock_slot_manager):
        await cache_mgr.ensure_loaded("proj1", "content", mock_slot_manager)
        stats = cache_mgr.get_stats()
        assert len(stats) == 1
        assert stats[0].action == "recompute"

    async def test_hit_stats_recorded(self, cache_mgr, mock_slot_manager):
        await cache_mgr.ensure_loaded("proj1", "c", mock_slot_manager)
        await cache_mgr.ensure_loaded("proj1", "c", mock_slot_manager)
        stats = cache_mgr.get_stats()
        assert stats[0].action == "recompute"
        assert stats[1].action == "hit"


# ------------------------------------------------------------------
# Metadata persistence
# ------------------------------------------------------------------


class TestMetaPersistence:
    async def test_meta_file_created(self, cache_mgr, mock_slot_manager, config):
        await cache_mgr.ensure_loaded("proj1", "content", mock_slot_manager)
        meta_path = Path(config.kv_cache_dir) / "projects" / "proj1" / "template.meta.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["project_id"] == "proj1"
        assert meta["content_hash"] == PrefixCacheManager.compute_hash("content")
        assert meta["token_count_estimate"] > 0
