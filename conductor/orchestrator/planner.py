"""
Planner Agent — Task decomposition and tier estimation.

Takes a high-level task and decomposes it into ordered subtasks,
each annotated with an estimated difficulty tier.

Tier heuristics (Phase 0):
  Tier 1: Single-file, <20 lines, well-defined
  Tier 2: Multi-file or ambiguous logic
  Tier 3: Architectural change, cross-cutting concern
  Tier 4: Requires human guidance or decomposition
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class Subtask:
    subtask_id: str
    description: str
    tier: int = 2
    dependencies: list[str] = field(default_factory=list)
    files_likely: list[str] = field(default_factory=list)


@dataclass
class Plan:
    task_id: str
    original_request: str
    subtasks: list[Subtask]
    summary: str


PLANNER_SYSTEM_PROMPT = """\
You are a senior software architect. Given a coding task, decompose it into \
ordered subtasks. For each subtask, provide:
1. A clear description of what to implement
2. Estimated difficulty tier (1-4)
3. Files likely to be modified
4. Dependencies on other subtasks (by ID)

Tier guidelines:
- Tier 1: Single file, <20 lines, well-defined change
- Tier 2: Multi-file or requires understanding existing patterns
- Tier 3: Architectural change, new abstractions, cross-cutting
- Tier 4: Needs human guidance or further decomposition

Respond in this exact JSON format:
{
  "summary": "brief plan summary",
  "subtasks": [
    {
      "description": "what to do",
      "tier": 2,
      "files_likely": ["path/to/file.py"],
      "dependencies": []
    }
  ]
}
"""


class Planner:
    def __init__(self, gateway_url: str) -> None:
        self._gateway_url = gateway_url
        self._client = httpx.AsyncClient(
            base_url=gateway_url,
            timeout=120,
        )

    async def decompose(
        self,
        task_id: str,
        task_text: str,
        project_context: str = "",
    ) -> Plan:
        """Decompose a task into subtasks via the gateway."""
        messages = []
        if project_context:
            messages.append({"role": "system", "content": project_context})
        messages.append({"role": "system", "content": PLANNER_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": task_text})

        resp = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": "conductor",
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._parse_plan(task_id, task_text, content)

    def _parse_plan(self, task_id: str, task_text: str, raw: str) -> Plan:
        """Parse LLM output into a structured Plan."""
        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            cleaned = raw.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0]

            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            # Fallback: treat entire task as a single Tier 2 subtask
            logger.warning("Failed to parse planner output, using single-subtask fallback")
            return Plan(
                task_id=task_id,
                original_request=task_text,
                subtasks=[
                    Subtask(
                        subtask_id=f"{task_id}-1",
                        description=task_text,
                        tier=2,
                    )
                ],
                summary=task_text[:100],
            )

        subtasks: list[Subtask] = []
        for i, st in enumerate(parsed.get("subtasks", [])):
            subtasks.append(
                Subtask(
                    subtask_id=f"{task_id}-{i + 1}",
                    description=st.get("description", ""),
                    tier=st.get("tier", 2),
                    dependencies=st.get("dependencies", []),
                    files_likely=st.get("files_likely", []),
                )
            )

        if not subtasks:
            subtasks = [
                Subtask(
                    subtask_id=f"{task_id}-1",
                    description=task_text,
                    tier=2,
                )
            ]

        return Plan(
            task_id=task_id,
            original_request=task_text,
            subtasks=subtasks,
            summary=parsed.get("summary", task_text[:100]),
        )

    async def close(self) -> None:
        await self._client.aclose()
