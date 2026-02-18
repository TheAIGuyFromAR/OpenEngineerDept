"""
Vault Sync Adapters — Keep the local vault folder in sync with remote.

The Conductor doesn't need Obsidian running. It just needs the vault
folder contents to match what's on the user's machine. These adapters
handle the sync transport:

  "local"     — no sync, vault is already local (default)
  "git"       — git pull before read, git add+commit+push after write
  "syncthing" — Syncthing handles it; we just wait for inotify
  "couchdb"   — Direct CouchDB access for Obsidian LiveSync vaults.
                Reads/writes documents via CouchDB HTTP API.
                Syncs CouchDB documents ↔ local vault folder.

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


class CouchDBSync(VaultSyncAdapter):
    """
    CouchDB-based sync for Obsidian LiveSync vaults.

    Talks directly to the CouchDB instance that Obsidian LiveSync uses.
    On sync_before_read: pulls changed documents from CouchDB into the
    local vault folder so the filesystem watcher can find them.
    On sync_after_write: pushes local file changes back to CouchDB so
    other Obsidian instances see them via LiveSync replication.
    """

    def __init__(
        self,
        vault_path: str,
        couchdb_url: str = "http://localhost:5984",
        couchdb_database: str = "obsidian",
        couchdb_username: str = "",
        couchdb_password: str = "",
        conductor_prefix: str = "conductor/",
    ) -> None:
        self._vault = Path(vault_path)
        self._prefix = conductor_prefix
        self._client: CouchDBClient | None = None
        self._couchdb_url = couchdb_url
        self._couchdb_database = couchdb_database
        self._couchdb_username = couchdb_username
        self._couchdb_password = couchdb_password

    async def _ensure_client(self) -> CouchDBClient:
        if self._client is None:
            from .couchdb_client import CouchDBClient
            self._client = CouchDBClient(
                url=self._couchdb_url,
                database=self._couchdb_database,
                username=self._couchdb_username,
                password=self._couchdb_password,
            )
        return self._client

    async def sync_before_read(self) -> None:
        """
        Pull conductor/* documents from CouchDB → local vault folder.
        This materializes CouchDB documents as local .md files so the
        filesystem-based ObsidianWatcher can find them.
        """
        client = await self._ensure_client()
        inbox_prefix = f"{self._prefix}inbox/"

        try:
            # List all docs under conductor/inbox/
            paths = await client.list_files(inbox_prefix)
            for path in paths:
                if not path.endswith(".md"):
                    continue

                # Read document from CouchDB and reassemble from chunks
                doc = await client.read_file(path)
                if doc is None:
                    continue

                # Write to local filesystem
                local_path = self._vault / path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Only write if CouchDB version is newer
                if local_path.exists():
                    local_mtime_ms = int(local_path.stat().st_mtime * 1000)
                    if doc.mtime <= local_mtime_ms:
                        continue

                local_path.write_text(doc.content, encoding="utf-8")
                logger.debug("CouchDB → local: %s", path)

        except Exception as exc:
            logger.warning("CouchDB sync_before_read failed: %s", exc)

    async def sync_after_write(self) -> None:
        """
        Push conductor/completed/* and conductor/failed/* back to CouchDB.
        This makes results visible to Obsidian instances via LiveSync.
        """
        client = await self._ensure_client()

        for subdir in ["completed", "failed"]:
            local_dir = self._vault / self._prefix / subdir
            if not local_dir.exists():
                continue

            for local_file in local_dir.glob("*.md"):
                path = f"{self._prefix}{subdir}/{local_file.name}"
                content = local_file.read_text(encoding="utf-8")

                try:
                    await client.write_file(path, content)
                    logger.debug("local → CouchDB: %s", path)
                except Exception as exc:
                    logger.warning("Failed to push %s to CouchDB: %s", path, exc)

        # Also clean up inbox in CouchDB (mark deleted for processed tasks)
        inbox_dir = self._vault / self._prefix / "inbox"
        if inbox_dir.exists():
            # List CouchDB inbox docs
            try:
                couch_inbox = await client.list_files(f"{self._prefix}inbox/")
                local_inbox = {f.name for f in inbox_dir.glob("*.md")}

                for couch_path in couch_inbox:
                    filename = couch_path.rsplit("/", 1)[-1]
                    if filename not in local_inbox:
                        # File was processed (moved to completed/failed locally)
                        # Mark as deleted in CouchDB
                        await client.delete_file(couch_path)
                        logger.debug("CouchDB deleted: %s", couch_path)
            except Exception as exc:
                logger.warning("CouchDB inbox cleanup failed: %s", exc)

    async def check_health(self) -> dict:
        """Check CouchDB connectivity."""
        try:
            client = await self._ensure_client()
            info = await client.check_connection()
            return {"adapter": "couchdb", **info}
        except Exception as exc:
            return {"adapter": "couchdb", "status": "error", "error": str(exc)}


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
    elif mode == "couchdb":
        return CouchDBSync(
            vault_path,
            couchdb_url=kwargs.get("couchdb_url", "http://localhost:5984"),
            couchdb_database=kwargs.get("couchdb_database", "obsidian"),
            couchdb_username=kwargs.get("couchdb_username", ""),
            couchdb_password=kwargs.get("couchdb_password", ""),
            conductor_prefix=kwargs.get("couchdb_conductor_prefix", "conductor/"),
        )
    else:
        raise ValueError(
            f"Unknown vault sync mode: {mode!r}. Use: local, git, syncthing, couchdb"
        )
