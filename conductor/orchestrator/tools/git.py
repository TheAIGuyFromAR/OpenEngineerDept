"""
Git Tool — Version control operations.

Wraps git commands for the Conductor orchestrator.
All operations run in the project directory.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass

from .shell import Shell

logger = logging.getLogger(__name__)


@dataclass
class GitResult:
    success: bool
    operation: str
    message: str


class Git:
    def __init__(self, project_dir: str) -> None:
        self._shell = Shell(project_dir, timeout=30)

    async def status(self) -> GitResult:
        result = await self._shell.run("git status --short")
        return GitResult(result.success, "status", result.stdout)

    async def diff(self, staged: bool = False) -> GitResult:
        cmd = "git diff --cached" if staged else "git diff"
        result = await self._shell.run(cmd)
        return GitResult(result.success, "diff", result.stdout)

    async def add(self, paths: list[str]) -> GitResult:
        if not paths:
            return GitResult(False, "add", "No paths specified")
        # Quote paths for safety
        quoted = " ".join(f'"{p}"' for p in paths)
        result = await self._shell.run(f"git add {quoted}")
        return GitResult(result.success, "add", result.stdout or result.stderr)

    async def commit(self, message: str) -> GitResult:
        # Use stdin for message to avoid shell escaping issues
        result = await self._shell.run(f"git commit -m {_shell_quote(message)}")
        return GitResult(result.success, "commit", result.stdout or result.stderr)

    async def current_branch(self) -> GitResult:
        result = await self._shell.run("git rev-parse --abbrev-ref HEAD")
        return GitResult(result.success, "branch", result.stdout.strip())

    async def log(self, n: int = 5) -> GitResult:
        result = await self._shell.run(f"git log --oneline -n {n}")
        return GitResult(result.success, "log", result.stdout)

    async def stash(self) -> GitResult:
        result = await self._shell.run("git stash")
        return GitResult(result.success, "stash", result.stdout or result.stderr)

    async def stash_pop(self) -> GitResult:
        result = await self._shell.run("git stash pop")
        return GitResult(result.success, "stash_pop", result.stdout or result.stderr)


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell use."""
    return shlex.quote(s)
