"""
Layer 3 — Changelog (append-only JSONL).

Every completed task gets a structured entry. This is the audit trail
and the training signal for the difficulty estimator.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ChangelogEntry:
    task_id: str
    project_id: str
    description: str
    tier_used: int
    candidates_generated: int
    accepted_candidate_idx: int
    reviewer_score: float
    test_passed: bool
    files_modified: list[str] = field(default_factory=list)
    retries: int = 0
    human_accepted: bool | None = None
    timestamp: float = field(default_factory=time.time)


class Changelog:
    def __init__(self, log_path: str = "./data/changelog.jsonl") -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: ChangelogEntry) -> None:
        """Append a changelog entry to the JSONL file."""
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def read_recent(self, n: int = 20) -> list[ChangelogEntry]:
        """Read the N most recent entries."""
        if not self._path.exists():
            return []
        entries: list[ChangelogEntry] = []
        lines = self._path.read_text().strip().splitlines()
        for line in lines[-n:]:
            try:
                data = json.loads(line)
                entries.append(ChangelogEntry(**data))
            except (json.JSONDecodeError, TypeError):
                continue
        return entries

    def acceptance_rate(self, last_n: int = 50) -> float:
        """Calculate first-attempt acceptance rate over recent tasks."""
        entries = self.read_recent(last_n)
        if not entries:
            return 0.0
        first_attempts = [e for e in entries if e.retries == 0]
        if not first_attempts:
            return 0.0
        passed = sum(1 for e in first_attempts if e.test_passed)
        return passed / len(first_attempts)

    def build_prompt_section(self) -> str:
        """Build a brief summary for prompt context (optional inclusion)."""
        recent = self.read_recent(5)
        if not recent:
            return ""
        parts = ["=== RECENT CHANGELOG ==="]
        for e in recent:
            status = "PASS" if e.test_passed else "FAIL"
            parts.append(
                f"- [{e.task_id}] {e.description[:60]} "
                f"tier={e.tier_used} score={e.reviewer_score:.1f} {status}"
            )
        parts.append("=== END CHANGELOG ===")
        return "\n".join(parts)
