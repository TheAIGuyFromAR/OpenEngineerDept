"""
Reviewer Agent — Scores and selects the best candidate.

Evaluates each code candidate on:
  - Correctness (does it address the subtask?)
  - Code quality (readability, patterns, naming)
  - Safety (no obvious bugs, injection, etc.)
  - Completeness (handles edge cases mentioned in subtask)

Phase 0: Uses a single LLM call to score all candidates.
Future: Could use structured rubrics, AST analysis, etc.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ReviewScore:
    candidate_idx: int
    correctness: float  # 0-10
    quality: float  # 0-10
    safety: float  # 0-10
    completeness: float  # 0-10
    overall: float  # 0-10
    feedback: str


@dataclass
class ReviewResult:
    subtask_id: str
    scores: list[ReviewScore]
    selected_idx: int
    selected_score: float
    feedback_summary: str


REVIEWER_SYSTEM_PROMPT = """\
You are a code reviewer. Given a subtask description and one or more candidate \
implementations, score each on these dimensions (0-10):

1. correctness: Does it correctly implement the subtask?
2. quality: Is the code clean, readable, well-structured?
3. safety: Are there security issues, bugs, or bad practices?
4. completeness: Does it handle edge cases and requirements?
5. overall: Weighted average (correctness 40%, quality 20%, safety 20%, completeness 20%)

Respond in this exact JSON format:
{
  "scores": [
    {
      "candidate_idx": 0,
      "correctness": 8.0,
      "quality": 7.5,
      "safety": 9.0,
      "completeness": 7.0,
      "overall": 7.9,
      "feedback": "brief explanation"
    }
  ],
  "selected_idx": 0,
  "feedback_summary": "why this candidate was selected"
}
"""


class Reviewer:
    def __init__(self, gateway_url: str, accept_threshold: float = 7.0) -> None:
        self._gateway_url = gateway_url
        self._accept_threshold = accept_threshold
        self._client = httpx.AsyncClient(
            base_url=gateway_url,
            timeout=120,
        )

    async def review(
        self,
        *,
        subtask_id: str,
        subtask_description: str,
        candidates: list[str],
        context: str = "",
    ) -> ReviewResult:
        """Score candidates and select the best one."""
        # Build the review prompt
        parts = [f"## Subtask\n{subtask_description}\n"]
        for i, candidate in enumerate(candidates):
            parts.append(f"## Candidate {i}\n```\n{candidate}\n```\n")

        prompt = "\n".join(parts)
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "system", "content": REVIEWER_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": prompt})

        resp = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": "conductor",
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.3,  # Low temp for consistent scoring
            },
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._parse_review(subtask_id, content, len(candidates))

    def _parse_review(
        self, subtask_id: str, raw: str, n_candidates: int
    ) -> ReviewResult:
        """Parse LLM review output into structured scores."""
        try:
            cleaned = raw.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0]

            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse reviewer output, using default scores")
            # Fallback: give all candidates a middling score, pick first
            scores = [
                ReviewScore(
                    candidate_idx=i,
                    correctness=5.0,
                    quality=5.0,
                    safety=5.0,
                    completeness=5.0,
                    overall=5.0,
                    feedback="(unparseable review)",
                )
                for i in range(n_candidates)
            ]
            return ReviewResult(
                subtask_id=subtask_id,
                scores=scores,
                selected_idx=0,
                selected_score=5.0,
                feedback_summary="Review parse failed — defaulting to candidate 0",
            )

        scores: list[ReviewScore] = []
        for s in parsed.get("scores", []):
            scores.append(
                ReviewScore(
                    candidate_idx=s.get("candidate_idx", 0),
                    correctness=s.get("correctness", 5.0),
                    quality=s.get("quality", 5.0),
                    safety=s.get("safety", 5.0),
                    completeness=s.get("completeness", 5.0),
                    overall=s.get("overall", 5.0),
                    feedback=s.get("feedback", ""),
                )
            )

        selected_idx = parsed.get("selected_idx", 0)
        selected_score = 5.0
        if scores and 0 <= selected_idx < len(scores):
            selected_score = scores[selected_idx].overall

        return ReviewResult(
            subtask_id=subtask_id,
            scores=scores,
            selected_idx=selected_idx,
            selected_score=selected_score,
            feedback_summary=parsed.get("feedback_summary", ""),
        )

    @property
    def accept_threshold(self) -> float:
        return self._accept_threshold

    async def close(self) -> None:
        await self._client.aclose()
