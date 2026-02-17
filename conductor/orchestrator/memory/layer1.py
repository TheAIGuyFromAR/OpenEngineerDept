"""
Layer 1 — Working Memory.

Current task context. Resets with each new top-level task.

Contains:
  - The original user request
  - The Planner's decomposition
  - Results of completed subtasks (diffs, test outputs)
  - Accumulated reviewer feedback
  - Current subtask being worked on

If working memory exceeds max_tokens, compress completed subtask
results to summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubtaskRecord:
    subtask_id: str
    description: str
    status: str = "pending"  # pending | in_progress | completed | failed
    candidate_summary: str = ""
    test_output: str = ""
    reviewer_feedback: str = ""
    files_modified: list[str] = field(default_factory=list)


class Layer1:
    def __init__(self, max_tokens: int = 8000) -> None:
        self._max_tokens = max_tokens
        self._task: str = ""
        self._plan_summary: str = ""
        self._subtasks: list[SubtaskRecord] = []
        self._current_subtask_idx: int = -1
        self._feedback_history: list[str] = []

    def start_task(self, task_text: str, plan_summary: str = "") -> None:
        """Begin a new top-level task. Clears all working memory."""
        self._task = task_text
        self._plan_summary = plan_summary
        self._subtasks.clear()
        self._current_subtask_idx = -1
        self._feedback_history.clear()

    def add_subtask(self, subtask_id: str, description: str) -> None:
        self._subtasks.append(SubtaskRecord(subtask_id=subtask_id, description=description))

    def start_subtask(self, subtask_id: str) -> None:
        for i, st in enumerate(self._subtasks):
            if st.subtask_id == subtask_id:
                st.status = "in_progress"
                self._current_subtask_idx = i
                return

    def complete_subtask(
        self,
        subtask_id: str,
        candidate_summary: str,
        test_output: str = "",
        files_modified: list[str] | None = None,
    ) -> None:
        for st in self._subtasks:
            if st.subtask_id == subtask_id:
                st.status = "completed"
                st.candidate_summary = candidate_summary
                st.test_output = test_output
                st.files_modified = files_modified or []
                break
        self._maybe_compress()

    def fail_subtask(self, subtask_id: str, reason: str) -> None:
        for st in self._subtasks:
            if st.subtask_id == subtask_id:
                st.status = "failed"
                st.reviewer_feedback = reason
                break

    def add_feedback(self, feedback: str) -> None:
        """Add reviewer feedback for the current subtask attempt."""
        self._feedback_history.append(feedback)
        if self._current_subtask_idx >= 0:
            st = self._subtasks[self._current_subtask_idx]
            st.reviewer_feedback = feedback

    def clear(self) -> None:
        self._task = ""
        self._plan_summary = ""
        self._subtasks.clear()
        self._current_subtask_idx = -1
        self._feedback_history.clear()

    @property
    def token_estimate(self) -> int:
        return len(self.build_prompt_section()) // 4

    def build_prompt_section(self) -> str:
        if not self._task:
            return ""

        parts: list[str] = []
        parts.append("=== WORKING MEMORY ===")
        parts.append(f"Task: {self._task}")

        if self._plan_summary:
            parts.append(f"\nPlan: {self._plan_summary}")

        # Completed subtasks
        completed = [s for s in self._subtasks if s.status == "completed"]
        if completed:
            parts.append("\n## Completed subtasks")
            for st in completed:
                parts.append(f"- [{st.subtask_id}] {st.description}")
                if st.candidate_summary:
                    parts.append(f"  Result: {st.candidate_summary[:200]}")
                if st.files_modified:
                    parts.append(f"  Modified: {', '.join(st.files_modified)}")

        # Current subtask
        if 0 <= self._current_subtask_idx < len(self._subtasks):
            current = self._subtasks[self._current_subtask_idx]
            parts.append(f"\n## Current subtask: {current.description}")

        # Feedback from previous attempts
        if self._feedback_history:
            parts.append("\n## Feedback from previous attempts")
            for fb in self._feedback_history[-3:]:  # Last 3 feedbacks
                parts.append(f"- {fb}")

        parts.append("=== END WORKING MEMORY ===")
        return "\n".join(parts)

    def _maybe_compress(self) -> None:
        """If working memory exceeds threshold, compress completed subtask summaries."""
        if self.token_estimate <= self._max_tokens:
            return

        # Truncate completed subtask summaries
        for st in self._subtasks:
            if st.status == "completed" and len(st.candidate_summary) > 100:
                st.candidate_summary = st.candidate_summary[:100] + "..."
            if st.test_output and len(st.test_output) > 50:
                st.test_output = st.test_output[:50] + "..."

        # Trim old feedback
        if len(self._feedback_history) > 3:
            self._feedback_history = self._feedback_history[-3:]
