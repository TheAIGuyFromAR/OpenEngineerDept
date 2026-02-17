"""
Training Data Collector — Records operational traces for future fine-tuning.

Per Ultra Think cycle, records:
  - Prompt + context hash
  - All candidate outputs
  - Reviewer scores for each candidate
  - Test outcomes
  - Accepted candidate index
  - Human acceptance status (when available)

Storage: append-only JSONL. One file per project per day.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class CandidateRecord:
    content: str
    sampling_params: dict
    reviewer_score: float
    tokens_generated: int


@dataclass
class TrainingRow:
    task_id: str
    subtask_id: str
    project_id: str
    timestamp: float = field(default_factory=time.time)
    prompt_hash: str = ""
    context_hash: str = ""
    tier: int = 2
    candidates: list[CandidateRecord] = field(default_factory=list)
    test_passed: bool = False
    test_output_summary: str = ""
    accepted_candidate_idx: int = -1
    human_accepted: bool | None = None
    retry_count: int = 0


class DataCollector:
    def __init__(self, data_dir: str = "./data/training") -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def record(self, row: TrainingRow) -> Path:
        """Append a training row to today's JSONL file."""
        date_str = datetime.utcfromtimestamp(row.timestamp).strftime("%Y-%m-%d")
        file_path = self._dir / f"{row.project_id}-{date_str}.jsonl"

        entry = {
            "task_id": row.task_id,
            "subtask_id": row.subtask_id,
            "project_id": row.project_id,
            "timestamp": row.timestamp,
            "prompt_hash": row.prompt_hash,
            "context_hash": row.context_hash,
            "tier": row.tier,
            "candidates": [asdict(c) for c in row.candidates],
            "test_passed": row.test_passed,
            "test_output_summary": row.test_output_summary,
            "accepted_candidate_idx": row.accepted_candidate_idx,
            "human_accepted": row.human_accepted,
            "retry_count": row.retry_count,
        }

        with open(file_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return file_path

    def read_project_data(
        self, project_id: str, last_n_days: int = 7
    ) -> list[dict]:
        """Read recent training data for a project."""
        rows: list[dict] = []
        files = sorted(self._dir.glob(f"{project_id}-*.jsonl"))
        for f in files[-last_n_days:]:
            for line in f.read_text().strip().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def compute_stats(self, project_id: str) -> dict:
        """Compute training data statistics."""
        rows = self.read_project_data(project_id)
        if not rows:
            return {"total_rows": 0}

        total = len(rows)
        tests_passed = sum(1 for r in rows if r.get("test_passed"))
        avg_candidates = (
            sum(len(r.get("candidates", [])) for r in rows) / total
        )
        tier_dist = {}
        for r in rows:
            t = r.get("tier", 0)
            tier_dist[t] = tier_dist.get(t, 0) + 1

        return {
            "total_rows": total,
            "test_pass_rate": tests_passed / total if total else 0,
            "avg_candidates_per_task": round(avg_candidates, 1),
            "tier_distribution": tier_dist,
        }

    @staticmethod
    def hash_content(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]
