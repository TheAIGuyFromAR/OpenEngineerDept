"""
Vault Sync Adapters — Keep the local vault folder in sync with remote.

The Conductor doesn't need Obsidian running. It just needs the vault
folder contents to match what's on the user's machine. These adapters
handle the sync transport:

  "local"     — no sync, vault is already local (default)
  "git"       — git pull before read, git add+commit+push after write
  "syncthing" — Syncthing handles it; we just wait for inotify

The ObsidianWatcher calls sync_before_read() before scanning inbox
and sync_after_write() after writing completed/failed results.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class VaultSyncAdapter(ABC):
    """Base class for vault synchronization."""

    @abstractmethod
    async def sync_before_read(self) -> None:
        """Pull remote changes before reading inbox."""

    @abstractmethod
    async def sync_after_write(self) -> None:
        """Push local changes after writing results."""

    @abstractmethod
    async def check_health(self) -> dict:
        """Return sync health status."""


class LocalSync(VaultSyncAdapter):
    """No-op sync — vault is already on the local filesystem."""

    async def sync_before_read(self) -> None:
        pass

    async def sync_after_write(self) -> None:
        pass

    async def check_health(self) -> dict:
        return {"adapter": "local", "status": "ok"}


class GitSync(VaultSyncAdapter):
    """
    Sync via git (works with obsidian-git plugin).

    Assumes the vault directory is a git repo with a configured remote.
    Pull before reading, commit+push after writing.
    """

    def __init__(self, vault_path: str, remote: str = "origin", branch: str = "main") -> None:
        self._vault = vault_path
        self._remote = remote
        self._branch = branch

    async def sync_before_read(self) -> None:
        """git pull to get latest vault state."""
        result = await self._run(f"git pull {self._remote} {self._branch} --no-rebase")
        if result.returncode != 0:
            logger.warning("git pull failed: %s", result.stderr)

    async def sync_after_write(self) -> None:
        """Stage conductor/ changes, commit, push."""
        # Only stage conductor output folders — don't touch user's vault files
        await self._run("git add conductor/completed/ conductor/failed/")
        result = await self._run(
            'git diff --cached --quiet || git commit -m "conductor: task results"'
        )
        if result.returncode == 0:
            push = await self._run(f"git push {self._remote} {self._branch}")
            if push.returncode != 0:
                logger.warning("git push failed: %s", push.stderr)

    async def check_health(self) -> dict:
        result = await self._run("git status --short")
        return {
            "adapter": "git",
            "status": "ok" if result.returncode == 0 else "error",
            "remote": self._remote,
            "branch": self._branch,
            "dirty_files": len(result.stdout.strip().splitlines()) if result.stdout else 0,
        }

    async def _run(self, cmd: str) -> asyncio.subprocess.Process:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._vault,
        )
        stdout, stderr = await proc.communicate()
        proc.stdout = stdout.decode(errors="replace") if stdout else ""
        proc.stderr = stderr.decode(errors="replace") if stderr else ""
        return proc


class SyncthingSync(VaultSyncAdapter):
    """
    Syncthing-based sync — Syncthing daemon handles bidirectional sync.

    The conductor just needs to:
    - Wait a beat after detecting a new file (Syncthing may still be writing)
    - Optionally check Syncthing API for folder sync status
    """

    def __init__(
        self,
        vault_path: str,
        syncthing_api: str = "http://localhost:8384",
        api_key: str = "",
        folder_id: str = "",
        settle_seconds: float = 1.0,
    ) -> None:
        self._vault = vault_path
        self._api_url = syncthing_api
        self._api_key = api_key
        self._folder_id = folder_id
        self._settle = settle_seconds

    async def sync_before_read(self) -> None:
        """Wait for Syncthing to settle — files may still be syncing."""
        await asyncio.sleep(self._settle)

    async def sync_after_write(self) -> None:
        """Syncthing picks up filesystem changes automatically. Just wait for propagation."""
        await asyncio.sleep(self._settle)

    async def check_health(self) -> dict:
        """Check Syncthing API for folder status (if configured)."""
        if not self._api_key or not self._folder_id:
            return {"adapter": "syncthing", "status": "ok", "api_configured": False}

        try:
            import httpx

            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{self._api_url}/rest/db/status",
                    params={"folder": self._folder_id},
                    headers={"X-API-Key": self._api_key},
                )
                resp.raise_for_status()
                data = resp.json()
                return {
                    "adapter": "syncthing",
                    "status": "ok",
                    "api_configured": True,
                    "state": data.get("state", "unknown"),
                    "need_files": data.get("needFiles", 0),
                    "global_files": data.get("globalFiles", 0),
                }
        except Exception as exc:
            return {
                "adapter": "syncthing",
                "status": "degraded",
                "error": str(exc),
            }


def create_sync_adapter(
    mode: str,
    vault_path: str,
    **kwargs,
) -> VaultSyncAdapter:
    """Factory for sync adapters based on config."""
    if mode == "local":
        return LocalSync()
    elif mode == "git":
        return GitSync(
            vault_path,
            remote=kwargs.get("git_remote", "origin"),
            branch=kwargs.get("git_branch", "main"),
        )
    elif mode == "syncthing":
        return SyncthingSync(
            vault_path,
            syncthing_api=kwargs.get("syncthing_api", "http://localhost:8384"),
            api_key=kwargs.get("syncthing_api_key", ""),
            folder_id=kwargs.get("syncthing_folder_id", ""),
            settle_seconds=kwargs.get("syncthing_settle_seconds", 1.0),
        )
    else:
        raise ValueError(f"Unknown vault sync mode: {mode!r}. Use: local, git, syncthing")
