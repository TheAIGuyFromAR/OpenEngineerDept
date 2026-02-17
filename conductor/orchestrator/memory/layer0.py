"""
Layer 0 — Pinned Constraints.

Read from a markdown file. Never summarized, never compressed.
Always included verbatim at the top of every prompt.
Content hash for prefix cache invalidation.

Target size: 1,000-4,000 tokens.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class Layer0:
    def __init__(self, constraints_path: str) -> None:
        self._path = Path(constraints_path)
        self._content: str = ""
        self._hash: str = ""
        self.reload()

    def reload(self) -> None:
        """Reload constraints from disk."""
        if self._path.exists():
            self._content = self._path.read_text(encoding="utf-8")
        else:
            self._content = ""
        self._hash = hashlib.sha256(self._content.encode()).hexdigest()[:16]

    @property
    def content(self) -> str:
        return self._content

    @property
    def content_hash(self) -> str:
        return self._hash

    @property
    def token_estimate(self) -> int:
        return len(self._content) // 4

    def build_prompt_section(self) -> str:
        if not self._content:
            return ""
        return (
            "=== PROJECT CONSTRAINTS (always follow these) ===\n"
            f"{self._content}\n"
            "=== END CONSTRAINTS ===\n"
        )
