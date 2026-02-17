"""
Coder Agent — Generates code via Ultra Think.

Takes a subtask + context and produces candidate implementations
through the gateway's Ultra Think endpoint. Does not apply changes
itself — returns candidates for the Reviewer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class CodeCandidate:
    content: str
    slot_id: int
    sampling_params: dict
    tokens_generated: int
    generation_time_ms: float
    tokens_per_second: float


@dataclass
class CoderResult:
    subtask_id: str
    candidates: list[CodeCandidate]
    errors: list[str] = field(default_factory=list)
    total_ms: float = 0.0


CODER_SYSTEM_PROMPT = """\
You are an expert software engineer. Given a subtask description and project \
context, produce a complete implementation.

Rules:
- Output ONLY the code changes needed
- Use unified diff format when modifying existing files
- For new files, output the complete file content with a header: `=== NEW FILE: path/to/file.py ===`
- Include minimal, necessary comments
- Follow the project's existing patterns and conventions
- Do not include explanations outside of code comments
"""


class Coder:
    def __init__(self, gateway_url: str) -> None:
        self._gateway_url = gateway_url
        self._client = httpx.AsyncClient(
            base_url=gateway_url,
            timeout=300,
        )

    async def generate(
        self,
        *,
        subtask_id: str,
        subtask_description: str,
        context: str,
        tier: int = 2,
        project_id: str | None = None,
        max_tokens: int | None = None,
    ) -> CoderResult:
        """Generate candidate implementations via Ultra Think."""

        prompt = f"## Subtask\n{subtask_description}"

        resp = await self._client.post(
            "/v1/ultra-think",
            json={
                "task_id": subtask_id,
                "prompt": prompt,
                "system_prompt": f"{context}\n\n{CODER_SYSTEM_PROMPT}",
                "tier": tier,
                "max_tokens": max_tokens,
                "project_id": project_id,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        candidates = [
            CodeCandidate(
                content=c["content"],
                slot_id=c["slot_id"],
                sampling_params=c["sampling_params"],
                tokens_generated=c["tokens_generated"],
                generation_time_ms=c["generation_time_ms"],
                tokens_per_second=c["tokens_per_second"],
            )
            for c in data.get("candidates", [])
        ]

        return CoderResult(
            subtask_id=subtask_id,
            candidates=candidates,
            errors=data.get("errors", []),
            total_ms=data.get("timing", {}).get("total_ms", 0.0),
        )

    async def close(self) -> None:
        await self._client.aclose()
