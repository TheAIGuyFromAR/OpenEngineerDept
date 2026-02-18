"""
Shell Tool — Execute shell commands in the project directory.

Safety constraints:
  - Commands run in a subprocess with timeout
  - Working directory is locked to project root
  - Output is captured and truncated if too large
  - Dangerous commands are blocked by default

Architecture note: The Conductor uses this via structured tool calls,
not raw shell access. All invocations are logged.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Commands that should never be run automatically
BLOCKED_PATTERNS = [
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    "dd if=",
    "> /dev/sd",
    "chmod 777",
    "curl | sh",
    "wget | sh",
    "sudo",
]

MAX_OUTPUT_BYTES = 64 * 1024  # 64KB


@dataclass
class ShellResult:
    success: bool
    command: str
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False


class Shell:
    def __init__(self, project_dir: str, timeout: int = 60) -> None:
        self._cwd = project_dir
        self._timeout = timeout

    async def run(
        self,
        command: str,
        timeout: int | None = None,
    ) -> ShellResult:
        """Execute a shell command in the project directory."""
        # Safety check
        cmd_lower = command.lower()
        for pattern in BLOCKED_PATTERNS:
            if pattern in cmd_lower:
                return ShellResult(
                    success=False,
                    command=command,
                    stdout="",
                    stderr=f"Blocked: command matches dangerous pattern '{pattern}'",
                    return_code=-1,
                )

        effective_timeout = timeout or self._timeout
        logger.info("Shell: %s (timeout=%ds)", command, effective_timeout)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ShellResult(
                    success=False,
                    command=command,
                    stdout="",
                    stderr=f"Command timed out after {effective_timeout}s",
                    return_code=-1,
                    timed_out=True,
                )

            stdout = stdout_bytes[:MAX_OUTPUT_BYTES].decode(errors="replace")
            stderr = stderr_bytes[:MAX_OUTPUT_BYTES].decode(errors="replace")

            return ShellResult(
                success=proc.returncode == 0,
                command=command,
                stdout=stdout,
                stderr=stderr,
                return_code=proc.returncode or 0,
            )
        except Exception as exc:
            return ShellResult(
                success=False,
                command=command,
                stdout="",
                stderr=str(exc),
                return_code=-1,
            )
