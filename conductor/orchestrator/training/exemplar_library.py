"""
Exemplar Library — Curated high-quality examples for few-shot prompting.

Stores successful task completions that scored above threshold,
organized by task type. Used to build few-shot examples in prompts.

Phase 0: Simple file-based storage. No semantic search yet.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Exemplar:
    task_type: str  # e.g. "bugfix", "feature", "refactor", "test"
    description: str
    prompt: str
    solution: str
    reviewer_score: float
    project_id: str
    tags: list[str]


class ExemplarLibrary:
    def __init__(self, library_dir: str = "./data/exemplars") -> None:
        self._dir = Path(library_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def add(self, exemplar: Exemplar) -> None:
        """Add an exemplar to the library."""
        file_path = self._dir / f"{exemplar.task_type}.jsonl"
        entry = {
            "task_type": exemplar.task_type,
            "description": exemplar.description,
            "prompt": exemplar.prompt,
            "solution": exemplar.solution,
            "reviewer_score": exemplar.reviewer_score,
            "project_id": exemplar.project_id,
            "tags": exemplar.tags,
        }
        with open(file_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("Added exemplar: %s (score=%.1f)", exemplar.task_type, exemplar.reviewer_score)

    def find(
        self,
        task_type: str,
        n: int = 3,
        min_score: float = 7.0,
    ) -> list[Exemplar]:
        """Find top N exemplars for a task type, sorted by score."""
        file_path = self._dir / f"{task_type}.jsonl"
        if not file_path.exists():
            return []

        candidates: list[Exemplar] = []
        for line in file_path.read_text().strip().splitlines():
            try:
                data = json.loads(line)
                if data.get("reviewer_score", 0) >= min_score:
                    candidates.append(Exemplar(**data))
            except (json.JSONDecodeError, TypeError):
                continue

        candidates.sort(key=lambda e: e.reviewer_score, reverse=True)
        return candidates[:n]

    def build_few_shot_section(
        self,
        task_type: str,
        n: int = 2,
    ) -> str:
        """Build a few-shot prompt section from exemplars."""
        exemplars = self.find(task_type, n=n)
        if not exemplars:
            return ""

        parts = ["=== EXEMPLARS ==="]
        for i, ex in enumerate(exemplars):
            parts.append(f"\n### Example {i + 1}: {ex.description[:80]}")
            parts.append(f"Prompt: {ex.prompt[:200]}")
            parts.append(f"Solution:\n```\n{ex.solution[:500]}\n```")
        parts.append("=== END EXEMPLARS ===")
        return "\n".join(parts)

    def list_types(self) -> list[str]:
        """List available exemplar types."""
        return [f.stem for f in self._dir.glob("*.jsonl")]
