"""Golden dataset evaluator — runs the orchestrator pipeline against sampled
(task, phrasing) pairs and scores decision quality.

Scoring dimensions per prompt:
  1. Plan decomposition — subtask count, description keyword coverage
  2. Tier estimation — accuracy vs expected difficulty
  3. File targeting — did the planner identify the right files?
  4. Output content — required keywords present, forbidden absent

Process:
  1. Sample N (task, phrasing) pairs from the matrix
  2. For each: send phrasing through planner + coder
  3. Score against the task's expected outcome
  4. Save results as JSON (includes human_rating fields for calibration)
  5. Render report showing scores by category AND by phrasing style

The phrasing-style breakdown is the key insight: it reveals whether
the system handles vague/frustrated/non-technical requests as well as
specific/technical ones.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .golden import (
    DecisionScore,
    EvalResult,
    ExpectedSubtask,
    GoldenEvaluation,
    GoldenTask,
    Phrasing,
    GOLDEN_TASKS,
    sample_prompts,
)

logger = logging.getLogger(__name__)


async def _chat(
    client: httpx.AsyncClient,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    """Send chat completion, return content string."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


# ------------------------------------------------------------------
# Individual scoring functions
# ------------------------------------------------------------------


def score_plan_decomposition(
    actual_subtasks: list[dict],
    expected: list[ExpectedSubtask],
    n_min: int,
    n_max: int,
) -> DecisionScore:
    """Score: did the planner produce the right number and kind of subtasks?"""
    n = len(actual_subtasks)

    # Count check
    if n < n_min:
        count_score = 0.0
        detail = f"Too few subtasks: {n} < {n_min}"
    elif n > n_max:
        count_score = max(0.0, 1.0 - (n - n_max) * 0.2)
        detail = f"Too many subtasks: {n} > {n_max}"
    else:
        count_score = 1.0
        detail = f"Subtask count {n} in range [{n_min}, {n_max}]"

    # Keyword coverage: do actual subtask descriptions mention expected keywords?
    keyword_hits = 0
    keyword_total = 0
    for exp in expected:
        for kw in exp.description_keywords:
            keyword_total += 1
            for actual in actual_subtasks:
                desc = actual.get("description", "").lower()
                if kw.lower() in desc:
                    keyword_hits += 1
                    break

    keyword_score = (keyword_hits / keyword_total) if keyword_total > 0 else 1.0
    combined = count_score * 0.4 + keyword_score * 0.6

    return DecisionScore(
        checkpoint="plan_decomposition",
        score=round(combined, 3),
        max_score=1.0,
        details=f"{detail}; keywords {keyword_hits}/{keyword_total}",
    )


def score_tier_estimates(
    actual_subtasks: list[dict],
    expected: list[ExpectedSubtask],
) -> DecisionScore:
    """Score: were tier estimates accurate?"""
    if not expected or not actual_subtasks:
        return DecisionScore(
            checkpoint="tier_estimation",
            score=0.5,
            max_score=1.0,
            details="No subtasks to compare",
        )

    correct = 0
    close = 0
    total = min(len(actual_subtasks), len(expected))

    for i in range(total):
        actual_tier = actual_subtasks[i].get("tier", 2)
        exp = expected[i]
        diff = abs(actual_tier - exp.expected_tier)
        if diff == 0:
            correct += 1
        elif diff <= exp.tier_tolerance:
            close += 1

    score = (correct * 1.0 + close * 0.5) / total if total > 0 else 0

    return DecisionScore(
        checkpoint="tier_estimation",
        score=round(score, 3),
        max_score=1.0,
        details=f"Exact: {correct}/{total}, within tolerance: {close}/{total}",
    )


def score_file_targeting(
    actual_subtasks: list[dict],
    expected: list[ExpectedSubtask],
) -> DecisionScore:
    """Score: did the planner identify the right files?"""
    if not expected:
        return DecisionScore(
            checkpoint="file_targeting", score=1.0, max_score=1.0,
            details="No file expectations",
        )

    expected_files = set()
    for exp in expected:
        expected_files.update(exp.expected_files)

    if not expected_files:
        return DecisionScore(
            checkpoint="file_targeting", score=1.0, max_score=1.0,
            details="No specific files expected",
        )

    actual_files = set()
    for st in actual_subtasks:
        actual_files.update(st.get("files_likely", []))

    if not actual_files:
        return DecisionScore(
            checkpoint="file_targeting", score=0.0, max_score=1.0,
            details=f"No files predicted (expected {expected_files})",
        )

    hits = expected_files & actual_files
    precision = len(hits) / len(actual_files)
    recall = len(hits) / len(expected_files)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return DecisionScore(
        checkpoint="file_targeting",
        score=round(f1, 3),
        max_score=1.0,
        details=f"F1={f1:.2f} (precision={precision:.2f}, recall={recall:.2f})",
    )


def score_output_content(
    code_output: str,
    must_contain: list[str],
    must_not_contain: list[str],
) -> DecisionScore:
    """Score: does the generated code contain required keywords?"""
    lower_output = code_output.lower()

    hits = sum(1 for kw in must_contain if kw.lower() in lower_output)
    contain_score = (hits / len(must_contain)) if must_contain else 1.0

    violations = sum(1 for kw in must_not_contain if kw.lower() in lower_output)
    avoid_score = 1.0 - (violations / len(must_not_contain)) if must_not_contain else 1.0

    combined = contain_score * 0.7 + avoid_score * 0.3

    return DecisionScore(
        checkpoint="output_content",
        score=round(combined, 3),
        max_score=1.0,
        details=f"Contains {hits}/{len(must_contain)}, violations {violations}/{len(must_not_contain)}",
    )


# ------------------------------------------------------------------
# Prompts (match orchestrator agent prompts exactly)
# ------------------------------------------------------------------

_PLANNER_PROMPT = """\
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

_CODER_PROMPT = """\
You are an expert software engineer. Given a subtask description and project \
context, produce a complete implementation.

Rules:
- Output ONLY the code changes needed
- For new files, output the complete file content with a header: `=== NEW FILE: path/to/file.py ===`
- Include minimal, necessary comments
- Follow the project's existing patterns and conventions
- Do not include explanations outside of code comments
"""


def _parse_plan_json(raw: str) -> dict:
    """Extract plan JSON from LLM response (handles markdown fences)."""
    cleaned = raw.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0]
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        return {"summary": "", "subtasks": []}


# ------------------------------------------------------------------
# Single prompt evaluation
# ------------------------------------------------------------------


async def evaluate_prompt(
    client: httpx.AsyncClient,
    task: GoldenTask,
    phrasing: Phrasing,
) -> EvalResult:
    """Evaluate one (task, phrasing) pair against the live gateway."""
    result = EvalResult(
        task_id=task.id,
        task_name=task.name,
        category=task.category.value,
        phrasing_style=phrasing.style.value,
        phrasing_text=phrasing.text,
    )

    try:
        # Build context from the task's reference files
        context_parts = []
        for path, content in task.context_files.items():
            context_parts.append(f"=== {path} ===\n{content}")
        context = "\n\n".join(context_parts) if context_parts else ""

        # ── Step 1: Run the planner with THIS phrasing ───────────
        plan_messages = []
        if context:
            plan_messages.append({"role": "system", "content": f"Project files:\n{context}"})
        plan_messages.append({"role": "system", "content": _PLANNER_PROMPT})
        plan_messages.append({"role": "user", "content": phrasing.text})

        plan_raw = await _chat(client, plan_messages, max_tokens=2048, temperature=0.7)
        plan_data = _parse_plan_json(plan_raw)
        actual_subtasks = plan_data.get("subtasks", [])

        # Score plan
        result.scores.append(
            score_plan_decomposition(
                actual_subtasks,
                task.expected.subtasks,
                task.expected.n_subtasks_min,
                task.expected.n_subtasks_max,
            )
        )

        # Score tiers
        result.scores.append(
            score_tier_estimates(actual_subtasks, task.expected.subtasks)
        )

        # Score file targeting
        result.scores.append(
            score_file_targeting(actual_subtasks, task.expected.subtasks)
        )

        # ── Step 2: Run the coder on first subtask ───────────────
        first_subtask = (
            actual_subtasks[0].get("description", phrasing.text)
            if actual_subtasks else phrasing.text
        )

        code_messages = [
            {"role": "system", "content": f"{context}\n\n{_CODER_PROMPT}" if context else _CODER_PROMPT},
            {"role": "user", "content": f"## Subtask\n{first_subtask}"},
        ]

        code_output = await _chat(client, code_messages, max_tokens=2048, temperature=0.3)

        # Score output
        result.scores.append(
            score_output_content(
                code_output,
                task.expected.output_must_contain,
                task.expected.output_must_not_contain,
            )
        )

    except Exception as exc:
        result.error = str(exc)[:200]
        logger.error("Golden eval %s/%s failed: %s", task.id, phrasing.style.value, exc)

    result.compute_overall()
    return result


# ------------------------------------------------------------------
# Main collection function
# ------------------------------------------------------------------


async def collect_golden(
    client: httpx.AsyncClient,
    n_samples: int = 0,
    seed: int = 42,
    tasks: list[GoldenTask] | None = None,
) -> GoldenEvaluation:
    """Run golden evaluation on sampled (task, phrasing) pairs.

    Args:
        client: httpx client pointed at the gateway
        n_samples: Number of pairs to sample (0 = all pairs)
        seed: Random seed for reproducible sampling
        tasks: Override task list (default: GOLDEN_TASKS)
    """
    task_list = tasks or GOLDEN_TASKS
    evaluation = GoldenEvaluation()

    if n_samples > 0:
        pairs = sample_prompts(task_list, n=n_samples, seed=seed)
    else:
        pairs = [(t, p) for t in task_list for p in t.phrasings]

    for task, phrasing in pairs:
        print(f"    [{task.id}] {phrasing.style.value:15s} ...", end="", flush=True)
        result = await evaluate_prompt(client, task, phrasing)
        evaluation.results.append(result)
        icon = "PASS" if result.passed else "FAIL"
        print(f" [{icon}] {result.overall_score:.1%}")

    evaluation.compute_summary()
    return evaluation


# ------------------------------------------------------------------
# Human rating file I/O
# ------------------------------------------------------------------


def save_for_rating(evaluation: GoldenEvaluation, path: str | Path) -> None:
    """Save evaluation results as JSON for human rating.

    The output file contains all results with empty human_rating fields.
    A human can fill in ratings (1-5) and notes, then load them back.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(evaluation.to_json())
    print(f"Saved {len(evaluation.results)} results to {p}")
    print(f"Edit the 'human_rating' (1-5) and 'human_notes' fields, then reload.")


def load_with_ratings(path: str | Path) -> GoldenEvaluation:
    """Load evaluation results that may include human ratings."""
    from dataclasses import fields

    p = Path(path)
    data = json.loads(p.read_text())

    evaluation = GoldenEvaluation()
    for r_data in data.get("results", []):
        result = EvalResult(
            task_id=r_data["task_id"],
            task_name=r_data["task_name"],
            category=r_data["category"],
            phrasing_style=r_data["phrasing_style"],
            phrasing_text=r_data["phrasing_text"],
            overall_score=r_data.get("overall_score", 0.0),
            passed=r_data.get("passed", False),
            error=r_data.get("error", ""),
            human_rating=r_data.get("human_rating"),
            human_notes=r_data.get("human_notes", ""),
        )
        for s_data in r_data.get("scores", []):
            result.scores.append(DecisionScore(**s_data))
        evaluation.results.append(result)

    evaluation.compute_summary()
    return evaluation
