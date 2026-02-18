"""
Obsidian Interface — Filesystem watcher for task ingestion.

Watches:
  {vault}/conductor/inbox/      — new .md files = tasks
  {vault}/conductor/completed/  — results written here
  {vault}/conductor/failed/     — errors written here

Also watches Layer 0 constraints file for changes,
triggering prefix cache invalidation.

Sync: The conductor doesn't need Obsidian running. It watches the vault
folder on the local filesystem. A VaultSyncAdapter keeps the folder
in sync with the user's machine via git, Syncthing, or nothing (local).
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Callable, Awaitable

from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
from watchdog.observers import Observer

from .vault_sync import VaultSyncAdapter, LocalSync

logger = logging.getLogger(__name__)


class ObsidianWatcher:
    def __init__(
        self,
        vault_path: str,
        layer0_path: str | None = None,
        on_new_task: Callable[[str, str], Awaitable[None]] | None = None,
        on_constraints_changed: Callable[[], Awaitable[None]] | None = None,
        sync_adapter: VaultSyncAdapter | None = None,
    ) -> None:
        self._vault = Path(vault_path)
        self._inbox = self._vault / "conductor" / "inbox"
        self._completed = self._vault / "conductor" / "completed"
        self._failed = self._vault / "conductor" / "failed"
        self._layer0_path = Path(layer0_path) if layer0_path else None

        self._on_new_task = on_new_task
        self._on_constraints_changed = on_constraints_changed
        self._sync = sync_adapter or LocalSync()
        self._observer: Observer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Ensure directories exist
        for d in [self._inbox, self._completed, self._failed]:
            d.mkdir(parents=True, exist_ok=True)

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the filesystem watcher."""
        self._loop = loop
        self._observer = Observer()

        # Watch inbox for new tasks
        handler = _InboxHandler(self._inbox, self._dispatch_task)
        self._observer.schedule(handler, str(self._inbox), recursive=False)

        # Watch constraints file for changes
        if self._layer0_path and self._layer0_path.exists():
            constraints_handler = _ConstraintsHandler(
                self._layer0_path, self._dispatch_constraints_change
            )
            self._observer.schedule(
                constraints_handler,
                str(self._layer0_path.parent),
                recursive=False,
            )

        self._observer.start()
        logger.info("Obsidian watcher started — inbox: %s", self._inbox)

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info("Obsidian watcher stopped")

    async def write_completed(self, task_filename: str, result: str) -> Path:
        """Move task to completed/ with result appended, then sync."""
        src = self._inbox / task_filename
        dst = self._completed / task_filename

        content = ""
        if src.exists():
            content = src.read_text(encoding="utf-8")
            src.unlink()

        content += f"\n\n---\n## Result\n{result}\n"
        dst.write_text(content, encoding="utf-8")
        logger.info("Completed: %s", task_filename)
        await self._sync.sync_after_write()
        return dst

    async def write_failed(self, task_filename: str, error: str) -> Path:
        """Move task to failed/ with error appended, then sync."""
        src = self._inbox / task_filename
        dst = self._failed / task_filename

        content = ""
        if src.exists():
            content = src.read_text(encoding="utf-8")
            src.unlink()

        content += f"\n\n---\n## Error\n{error}\n"
        dst.write_text(content, encoding="utf-8")
        logger.info("Failed: %s", task_filename)
        await self._sync.sync_after_write()
        return dst

    async def list_pending(self) -> list[tuple[str, str]]:
        """Sync, then list pending tasks in inbox. Returns (filename, content) pairs."""
        await self._sync.sync_before_read()
        tasks = []
        for f in sorted(self._inbox.glob("*.md")):
            content = f.read_text(encoding="utf-8")
            tasks.append((f.name, content))
        return tasks

    async def sync_health(self) -> dict:
        """Check vault sync adapter health."""
        return await self._sync.check_health()

    def _dispatch_task(self, filename: str, content: str) -> None:
        """Called by watchdog handler when a new task file appears."""
        if self._loop and self._on_new_task:
            asyncio.run_coroutine_threadsafe(
                self._on_new_task(filename, content), self._loop
            )

    def _dispatch_constraints_change(self) -> None:
        """Called when the constraints file changes."""
        if self._loop and self._on_constraints_changed:
            asyncio.run_coroutine_threadsafe(
                self._on_constraints_changed(), self._loop
            )


class _InboxHandler(FileSystemEventHandler):
    def __init__(
        self,
        inbox_dir: Path,
        callback: Callable[[str, str], None],
    ) -> None:
        self._inbox = inbox_dir
        self._callback = callback
        self._debounce: dict[str, float] = {}

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix != ".md":
            return

        # Debounce — some editors write multiple events
        now = time.monotonic()
        if path.name in self._debounce and now - self._debounce[path.name] < 1.0:
            return
        self._debounce[path.name] = now

        try:
            content = path.read_text(encoding="utf-8")
            logger.info("New task detected: %s", path.name)
            self._callback(path.name, content)
        except Exception as exc:
            logger.error("Failed to read task %s: %s", path.name, exc)


class _ConstraintsHandler(FileSystemEventHandler):
    def __init__(
        self,
        constraints_path: Path,
        callback: Callable[[], None],
    ) -> None:
        self._path = constraints_path
        self._callback = callback

    def on_modified(self, event: FileModifiedEvent) -> None:
        if Path(event.src_path).resolve() == self._path.resolve():
            logger.info("Constraints file modified — triggering cache invalidation")
            self._callback()
