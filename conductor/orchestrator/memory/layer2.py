"""
Layer 2 — Compressed History (Phase 0 Stub).

In later phases, this will hold compressed summaries of previous tasks
for long-range context. In Phase 0, it returns an empty section.
"""

from __future__ import annotations


class Layer2:
    """Stub for compressed history. Returns empty in Phase 0."""

    def __init__(self) -> None:
        self._entries: list[str] = []

    def add_summary(self, task_id: str, summary: str) -> None:
        """Store a compressed summary of a completed task."""
        self._entries.append(f"[{task_id}] {summary}")

    @property
    def token_estimate(self) -> int:
        return len(self.build_prompt_section()) // 4

    def build_prompt_section(self) -> str:
        if not self._entries:
            return ""
        parts = ["=== COMPRESSED HISTORY ==="]
        for entry in self._entries[-10:]:  # Keep last 10
            parts.append(f"- {entry}")
        parts.append("=== END HISTORY ===")
        return "\n".join(parts)

    def clear(self) -> None:
        self._entries.clear()
