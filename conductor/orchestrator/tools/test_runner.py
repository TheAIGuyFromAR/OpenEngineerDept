"""
Test Runner Tool — Execute project tests and parse results.

Supports common test frameworks by convention:
  - Python: pytest
  - Node: npm test / npx jest
  - Go: go test
  - Rust: cargo test

Phase 0: Auto-detect from project files, run, parse exit code + output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .shell import Shell

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    success: bool
    framework: str
    output: str
    tests_passed: int
    tests_failed: int
    tests_total: int


class TestRunner:
    def __init__(self, project_dir: str, timeout: int = 120) -> None:
        self._project_dir = project_dir
        self._shell = Shell(project_dir, timeout=timeout)

    async def run(self, command: str | None = None) -> TestResult:
        """Run tests. Auto-detect framework if no command given."""
        if command:
            return await self._run_command(command, "custom")

        framework, cmd = self._detect_framework()
        if not cmd:
            return TestResult(
                success=False,
                framework="unknown",
                output="No test framework detected",
                tests_passed=0,
                tests_failed=0,
                tests_total=0,
            )
        return await self._run_command(cmd, framework)

    async def _run_command(self, command: str, framework: str) -> TestResult:
        logger.info("Running tests: %s (%s)", command, framework)
        result = await self._shell.run(command)

        output = result.stdout + ("\n" + result.stderr if result.stderr else "")
        passed, failed, total = self._parse_counts(output, framework)

        return TestResult(
            success=result.success,
            framework=framework,
            output=output,
            tests_passed=passed,
            tests_failed=failed,
            tests_total=total,
        )

    def _detect_framework(self) -> tuple[str, str]:
        """Detect test framework from project files."""
        root = Path(self._project_dir)

        # Python
        if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists():
            return "pytest", "python -m pytest -v --tb=short"

        if (root / "setup.py").exists() or (root / "setup.cfg").exists():
            return "pytest", "python -m pytest -v --tb=short"

        # Node
        if (root / "package.json").exists():
            return "node", "npm test"

        # Go
        if (root / "go.mod").exists():
            return "go", "go test ./..."

        # Rust
        if (root / "Cargo.toml").exists():
            return "rust", "cargo test"

        # Makefile
        if (root / "Makefile").exists():
            return "make", "make test"

        return "", ""

    @staticmethod
    def _parse_counts(output: str, framework: str) -> tuple[int, int, int]:
        """Extract pass/fail/total counts from test output. Best effort."""
        passed = 0
        failed = 0

        if framework == "pytest":
            # Look for "X passed, Y failed" pattern
            for line in output.splitlines():
                if "passed" in line or "failed" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            try:
                                passed = int(parts[i - 1])
                            except ValueError:
                                pass
                        if part == "failed" and i > 0:
                            try:
                                failed = int(parts[i - 1])
                            except ValueError:
                                pass

        total = passed + failed
        return passed, failed, total
