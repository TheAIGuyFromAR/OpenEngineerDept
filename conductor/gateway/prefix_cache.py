"""
Prefix Cache Manager — Per-project KV cache persistence on NVMe.

Layout:
  kv-cache/projects/{project_id}/
  ├── template.bin           # current KV cache (managed by llama-server)
  ├── template.meta.json     # {hash, token_count, timestamp}
  └── history/
      └── template-{ts}.bin  # rollback copies

Invalidation: hash Layer 0 text.  Match → restore.  Mismatch → recompute.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .config import GatewayConfig
from .slot_manager import SlotManager

logger = logging.getLogger(__name__)


@dataclass
class CacheMeta:
    content_hash: str
    token_count_estimate: int
    created_at: float
    project_id: str


@dataclass
class CacheStats:
    project_id: str
    action: str  # "hit" | "miss" | "recompute" | "save"
    duration_ms: float
    timestamp: float


class PrefixCacheManager:
    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._cache_dir = Path(config.kv_cache_dir)
        self._stats: list[CacheStats] = []
        self._metrics_path = Path(config.metrics_log_path).parent / "cache_stats.jsonl"

    def _project_dir(self, project_id: str) -> Path:
        return self._cache_dir / "projects" / project_id

    def _meta_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "template.meta.json"

    @staticmethod
    def compute_hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def has_valid_cache(self, project_id: str, layer0_text: str) -> bool:
        """Check if on-disk cache matches current Layer 0 content."""
        meta = self._load_meta(project_id)
        if meta is None:
            return False
        return meta.content_hash == self.compute_hash(layer0_text)

    async def ensure_loaded(
        self,
        project_id: str,
        layer0_text: str,
        slot_manager: SlotManager,
    ) -> str:
        """
        Ensure the project's prefix cache is ready.
        Returns "hit" if restored from disk, "recompute" if freshly computed.
        """
        content_hash = self.compute_hash(layer0_text)
        meta = self._load_meta(project_id)

        if meta is not None and meta.content_hash == content_hash:
            # Cache hit — template file already on disk, llama-server can restore
            start = time.monotonic()
            logger.info("Prefix cache HIT for project %s", project_id)
            self._record_stat(project_id, "hit", (time.monotonic() - start) * 1000)
            return "hit"

        # Cache miss — need to recompute
        start = time.monotonic()
        logger.info("Prefix cache MISS for project %s — recomputing", project_id)

        # Archive old cache if it exists
        if meta is not None:
            self._archive_old(project_id)

        # Warm the template slot with new content
        await slot_manager.warm_template(project_id, layer0_text)

        # Save metadata
        token_estimate = len(layer0_text) // 4  # rough estimate
        self._save_meta(
            project_id,
            CacheMeta(
                content_hash=content_hash,
                token_count_estimate=token_estimate,
                created_at=time.time(),
                project_id=project_id,
            ),
        )

        elapsed = (time.monotonic() - start) * 1000
        self._record_stat(project_id, "recompute", elapsed)
        logger.info("Prefix recomputed in %.0fms (~%d tokens)", elapsed, token_estimate)
        return "recompute"

    def invalidate(self, project_id: str) -> None:
        """Force cache invalidation for a project."""
        meta_path = self._meta_path(project_id)
        if meta_path.exists():
            self._archive_old(project_id)
            meta_path.unlink(missing_ok=True)
            logger.info("Cache invalidated for project %s", project_id)

    def get_stats(self) -> list[CacheStats]:
        return list(self._stats)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_meta(self, project_id: str) -> CacheMeta | None:
        meta_path = self._meta_path(project_id)
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text())
            return CacheMeta(**data)
        except (json.JSONDecodeError, TypeError, KeyError):
            return None

    def _save_meta(self, project_id: str, meta: CacheMeta) -> None:
        project_dir = self._project_dir(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._meta_path(project_id)
        meta_path.write_text(
            json.dumps(
                {
                    "content_hash": meta.content_hash,
                    "token_count_estimate": meta.token_count_estimate,
                    "created_at": meta.created_at,
                    "project_id": meta.project_id,
                },
                indent=2,
            )
        )

    def _archive_old(self, project_id: str) -> None:
        """Move current template files to history/ for rollback."""
        project_dir = self._project_dir(project_id)
        history_dir = project_dir / "history"
        history_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time())
        for suffix in (".bin", ".meta.json"):
            src = project_dir / f"template{suffix}"
            if src.exists():
                dst = history_dir / f"template-{ts}{suffix}"
                src.rename(dst)

    def _record_stat(self, project_id: str, action: str, duration_ms: float) -> None:
        stat = CacheStats(
            project_id=project_id,
            action=action,
            duration_ms=duration_ms,
            timestamp=time.time(),
        )
        self._stats.append(stat)

        # Append to JSONL
        try:
            self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._metrics_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "project_id": stat.project_id,
                            "action": stat.action,
                            "duration_ms": stat.duration_ms,
                            "timestamp": stat.timestamp,
                        }
                    )
                    + "\n"
                )
        except OSError:
            pass
