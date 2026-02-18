"""
File Operations Tool — Read, write, and patch files.

The Conductor orchestrator uses this to apply code candidates.
All operations are logged for audit trail.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FileOpResult:
    success: bool
    path: str
    operation: str
    message: str = ""
    diff: str = ""


class FileOps:
    def __init__(self, project_dir: str) -> None:
        self._root = Path(project_dir).resolve()

    def _resolve(self, path: str) -> Path:
        """Resolve a path relative to project root, preventing escapes."""
        resolved = (self._root / path).resolve()
        if not str(resolved).startswith(str(self._root)):
            raise ValueError(f"Path escapes project root: {path}")
        return resolved

    def read(self, path: str) -> FileOpResult:
        """Read a file's contents."""
        try:
            full = self._resolve(path)
            if not full.exists():
                return FileOpResult(False, path, "read", f"File not found: {path}")
            content = full.read_text(encoding="utf-8")
            return FileOpResult(True, path, "read", message=content)
        except Exception as exc:
            return FileOpResult(False, path, "read", str(exc))

    def write(self, path: str, content: str) -> FileOpResult:
        """Write content to a file (create or overwrite)."""
        try:
            full = self._resolve(path)
            full.parent.mkdir(parents=True, exist_ok=True)
            old_content = full.read_text(encoding="utf-8") if full.exists() else ""
            full.write_text(content, encoding="utf-8")

            diff = "".join(
                difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )
            logger.info("Wrote %s (%d bytes)", path, len(content))
            return FileOpResult(True, path, "write", diff=diff)
        except Exception as exc:
            return FileOpResult(False, path, "write", str(exc))

    def patch(self, path: str, unified_diff: str) -> FileOpResult:
        """Apply a unified diff to an existing file."""
        try:
            full = self._resolve(path)
            if not full.exists():
                return FileOpResult(False, path, "patch", f"File not found: {path}")

            original = full.read_text(encoding="utf-8").splitlines(keepends=True)

            # Parse and apply the diff
            patched = self._apply_diff(original, unified_diff)
            full.write_text("".join(patched), encoding="utf-8")

            logger.info("Patched %s", path)
            return FileOpResult(True, path, "patch", diff=unified_diff)
        except Exception as exc:
            return FileOpResult(False, path, "patch", str(exc))

    def list_dir(self, path: str = ".") -> FileOpResult:
        """List directory contents."""
        try:
            full = self._resolve(path)
            if not full.is_dir():
                return FileOpResult(False, path, "list", f"Not a directory: {path}")
            entries = sorted(
                f"{'d' if e.is_dir() else 'f'} {e.name}"
                for e in full.iterdir()
                if not e.name.startswith(".")
            )
            return FileOpResult(True, path, "list", message="\n".join(entries))
        except Exception as exc:
            return FileOpResult(False, path, "list", str(exc))

    def exists(self, path: str) -> bool:
        try:
            return self._resolve(path).exists()
        except ValueError:
            return False

    @staticmethod
    def _apply_diff(original: list[str], diff_text: str) -> list[str]:
        """
        Apply a unified diff to original lines.

        Phase 0: Line-by-line hunk application. Falls back to returning
        original unchanged if anything goes wrong. The conductor can
        always fall back to full file replacement via write().
        """
        try:
            result = list(original)
            offset = 0  # tracks line shifts from prior hunks

            in_hunk = False
            hunk_start = 0
            hunk_removes: list[int] = []
            hunk_adds: list[tuple[int, str]] = []
            pos = 0

            for raw_line in diff_text.splitlines(keepends=True):
                if raw_line.startswith("@@"):
                    # Apply previous hunk
                    if hunk_removes or hunk_adds:
                        result, offset = _apply_hunk(
                            result, hunk_removes, hunk_adds, offset
                        )
                    # Parse: @@ -start,count +start,count @@
                    parts = raw_line.split()
                    old_spec = parts[1]  # e.g. "-10,5"
                    hunk_start = int(old_spec.split(",")[0].lstrip("-")) - 1
                    in_hunk = True
                    pos = hunk_start
                    hunk_removes = []
                    hunk_adds = []
                elif in_hunk:
                    if raw_line.startswith("-"):
                        hunk_removes.append(pos)
                        pos += 1
                    elif raw_line.startswith("+"):
                        hunk_adds.append((pos, raw_line[1:]))
                    elif raw_line.startswith(" "):
                        pos += 1

            # Apply final hunk
            if hunk_removes or hunk_adds:
                result, offset = _apply_hunk(result, hunk_removes, hunk_adds, offset)

            return result
        except Exception:
            return original


def _apply_hunk(
    lines: list[str],
    removes: list[int],
    adds: list[tuple[int, str]],
    offset: int,
) -> tuple[list[str], int]:
    """Apply a single diff hunk, returning updated lines and new offset."""
    # Remove lines (in reverse so indices stay valid)
    for idx in sorted(removes, reverse=True):
        actual = idx + offset
        if 0 <= actual < len(lines):
            lines.pop(actual)

    # Insert added lines
    insert_base = (min(removes) if removes else (adds[0][0] if adds else 0)) + offset
    insert_base -= len(removes)  # adjust for removals
    for i, (_, text) in enumerate(adds):
        lines.insert(insert_base + i, text)

    offset += len(adds) - len(removes)
    return lines, offset
