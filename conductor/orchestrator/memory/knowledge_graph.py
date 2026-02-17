"""
Layer 4 — Knowledge Graph (Phase 0 Stub).

Returns empty responses. In later phases, this will hold
project structure, dependency graphs, and learned patterns.
"""

from __future__ import annotations


class KnowledgeGraph:
    """Phase 0 stub. Always returns empty."""

    def query(self, query: str) -> str:
        return ""

    def update(self, entity: str, relation: str, target: str) -> None:
        pass

    @property
    def token_estimate(self) -> int:
        return 0

    def build_prompt_section(self) -> str:
        return ""
