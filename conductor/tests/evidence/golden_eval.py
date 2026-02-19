"""Golden dataset evaluator — uses Langfuse for storage, annotation, and
bootstrapping of orchestrator decision quality data.

Architecture:
  - Golden tasks + phrasings are stored as a Langfuse Dataset
  - Each evaluation run creates a Langfuse Dataset Run with traces
  - Automated scores (plan, tier, files, output) attached to each trace
  - Human annotations added directly in Langfuse UI on each trace
  - Every real task processed by the orchestrator ALSO adds to the dataset
    (bootstrapping) — human reviews those traces to grow the golden set

Bootstrapping flow:
  1. Start with seed golden tasks (3 tasks x 12 phrasings = 36 prompts)
  2. Run evaluation → automated scores + traces in Langfuse
  3. Every real orchestrator task automatically creates a dataset item
  4. Human annotates a sample in Langfuse UI (score 1-5, notes)
  5. Annotated items become new golden references
  6. Repeat — dataset grows organically with real usage

Fallback: If Langfuse is unavailable, runs in local-only mode.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .golden import (
    Annotation,
    AnnotationPair,
    DecisionScore,
    EvalResult,
    ExpectedSubtask,
    GoldenEvaluation,
    GoldenTask,
    Phrasing,
    GOLDEN_TASKS,
    build_agent_annotation,
    compute_annotation_agreement,
    sample_prompts,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Langfuse client
# ------------------------------------------------------------------


def _get_langfuse():
    """Get Langfuse client, or None if unavailable."""
    try:
        from langfuse import Langfuse
        client = Langfuse()
        client.auth_check()
        return client
    except ImportError:
        logger.info("Langfuse SDK not installed — using local fallback")
        return None
    except Exception as exc:
        logger.warning("Langfuse not reachable — using local fallback: %s", exc)
        return None


# ------------------------------------------------------------------
# Dataset management
# ------------------------------------------------------------------


def upload_golden_dataset(
    tasks: list[GoldenTask] | None = None,
    dataset_name: str = "golden-tasks",
) -> bool:
    """Upload seed golden tasks to Langfuse as a Dataset.

    Each dataset item = one (task, phrasing) pair.
    Returns True if uploaded, False if Langfuse unavailable.
    """
    lf = _get_langfuse()
    if lf is None:
        return False

    task_list = tasks or GOLDEN_TASKS
    lf.create_dataset(name=dataset_name)

    item_count = 0
    for task in task_list:
        for phrasing in task.phrasings:
            item_id = f"{task.id}_{phrasing.style.value}"
            lf.create_dataset_item(
                dataset_name=dataset_name,
                id=item_id,
                input={
                    "task_id": task.id,
                    "task_name": task.name,
                    "phrasing_style": phrasing.style.value,
                    "phrasing_text": phrasing.text,
                    "context_files": task.context_files,
                },
                expected_output={
                    "n_subtasks_min": task.expected.n_subtasks_min,
                    "n_subtasks_max": task.expected.n_subtasks_max,
                    "subtasks": [
                        {
                            "description_keywords": s.description_keywords,
                            "expected_tier": s.expected_tier,
                            "expected_files": s.expected_files,
                        }
                        for s in task.expected.subtasks
                    ],
                    "output_must_contain": task.expected.output_must_contain,
                    "output_must_not_contain": task.expected.output_must_not_contain,
                    "min_review_score": task.expected.min_review_score,
                },
                metadata={
                    "category": task.category.value,
                    "difficulty": task.difficulty,
                    "source": "seed",  # vs "bootstrapped" for real tasks
                },
            )
            item_count += 1

    lf.flush()
    logger.info("Uploaded %d seed items to Langfuse dataset '%s'", item_count, dataset_name)
    return True


def bootstrap_from_task(
    *,
    task_id: str,
    task_text: str,
    plan_raw: str,
    code_output: str,
    review_score: float,
    files_modified: list[str],
    test_passed: bool,
    context_files: dict[str, str] | None = None,
    agent_annotation: Annotation | None = None,
    dataset_name: str = "golden-tasks",
) -> bool:
    """Add a real orchestrator task execution to the golden dataset.

    Called by the Conductor after processing a task. Creates a Langfuse
    dataset item from the actual execution, ready for human annotation.

    The agent_annotation captures the system's self-assessment of the output.
    The human later reviews this in Langfuse UI and adds a "user_annotation"
    score with structured JSON in the comment field:
      {"rating": 0.8, "rationale": "...", "strengths": [...], "weaknesses": [...],
       "intent_correct": true, "plan_correct": true, "output_correct": false, "tags": [...]}

    Agreement between agent and user annotations is computed by pull_human_ratings().
    """
    lf = _get_langfuse()
    if lf is None:
        return False

    try:
        item_id = f"bootstrap_{task_id}"

        # Build agent annotation from review score if not provided
        if agent_annotation is None:
            agent_annotation = Annotation(
                rating=review_score / 10.0,  # Normalize from 0-10 to 0-1
                rationale=f"Automated review score: {review_score}/10. Tests {'passed' if test_passed else 'failed'}.",
                strengths=["tests_passed"] if test_passed else [],
                weaknesses=[] if test_passed else ["tests_failed"],
                tags=["bootstrapped"],
            )

        ann_dict = {
            "rating": agent_annotation.rating,
            "rationale": agent_annotation.rationale,
            "strengths": agent_annotation.strengths,
            "weaknesses": agent_annotation.weaknesses,
            "plan_correct": agent_annotation.plan_correct,
            "output_correct": agent_annotation.output_correct,
            "tier_correct": agent_annotation.tier_correct,
            "tags": agent_annotation.tags,
        }

        lf.create_dataset_item(
            dataset_name=dataset_name,
            id=item_id,
            input={
                "task_id": task_id,
                "task_name": task_text[:80],
                "phrasing_style": "real_task",
                "phrasing_text": task_text,
                "context_files": context_files or {},
            },
            expected_output={
                "plan_raw": plan_raw[:2000],
                "code_output": code_output[:5000],
                "review_score": review_score,
                "files_modified": files_modified,
                "test_passed": test_passed,
                "agent_annotation": ann_dict,
            },
            metadata={
                "source": "bootstrapped",
                "review_score": review_score,
                "test_passed": test_passed,
                "awaiting_human_review": True,
                "agent_rating": agent_annotation.rating,
            },
        )
        lf.flush()
        return True
    except Exception as exc:
        logger.debug("Bootstrap to Langfuse failed: %s", exc)
        return False


# ------------------------------------------------------------------
# Scoring functions
# ------------------------------------------------------------------


async def _chat(
    client: httpx.AsyncClient,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def score_plan_decomposition(
    actual_subtasks: list[dict], expected: list[ExpectedSubtask], n_min: int, n_max: int,
) -> DecisionScore:
    n = len(actual_subtasks)
    if n < n_min:
        count_score, detail = 0.0, f"Too few subtasks: {n} < {n_min}"
    elif n > n_max:
        count_score, detail = max(0.0, 1.0 - (n - n_max) * 0.2), f"Too many subtasks: {n} > {n_max}"
    else:
        count_score, detail = 1.0, f"Subtask count {n} in [{n_min}, {n_max}]"

    keyword_hits = keyword_total = 0
    for exp in expected:
        for kw in exp.description_keywords:
            keyword_total += 1
            if any(kw.lower() in a.get("description", "").lower() for a in actual_subtasks):
                keyword_hits += 1
    keyword_score = (keyword_hits / keyword_total) if keyword_total > 0 else 1.0

    return DecisionScore(
        checkpoint="plan_decomposition", score=round(count_score * 0.4 + keyword_score * 0.6, 3),
        max_score=1.0, details=f"{detail}; keywords {keyword_hits}/{keyword_total}",
    )


def score_tier_estimates(actual_subtasks: list[dict], expected: list[ExpectedSubtask]) -> DecisionScore:
    if not expected or not actual_subtasks:
        return DecisionScore(checkpoint="tier_estimation", score=0.5, max_score=1.0, details="No subtasks to compare")
    correct = close = 0
    total = min(len(actual_subtasks), len(expected))
    for i in range(total):
        diff = abs(actual_subtasks[i].get("tier", 2) - expected[i].expected_tier)
        if diff == 0: correct += 1
        elif diff <= expected[i].tier_tolerance: close += 1
    score = (correct + close * 0.5) / total if total > 0 else 0
    return DecisionScore(checkpoint="tier_estimation", score=round(score, 3), max_score=1.0, details=f"Exact: {correct}/{total}, close: {close}/{total}")


def score_file_targeting(actual_subtasks: list[dict], expected: list[ExpectedSubtask]) -> DecisionScore:
    expected_files = {f for exp in expected for f in exp.expected_files}
    if not expected_files:
        return DecisionScore(checkpoint="file_targeting", score=1.0, max_score=1.0, details="No specific files expected")
    actual_files = {f for st in actual_subtasks for f in st.get("files_likely", [])}
    if not actual_files:
        return DecisionScore(checkpoint="file_targeting", score=0.0, max_score=1.0, details=f"No files predicted")
    hits = expected_files & actual_files
    p, r = len(hits) / len(actual_files), len(hits) / len(expected_files)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0
    return DecisionScore(checkpoint="file_targeting", score=round(f1, 3), max_score=1.0, details=f"F1={f1:.2f}")


def score_output_content(code_output: str, must_contain: list[str], must_not_contain: list[str]) -> DecisionScore:
    lower = code_output.lower()
    hits = sum(1 for kw in must_contain if kw.lower() in lower)
    violations = sum(1 for kw in must_not_contain if kw.lower() in lower)
    contain = (hits / len(must_contain)) if must_contain else 1.0
    avoid = 1.0 - (violations / len(must_not_contain)) if must_not_contain else 1.0
    return DecisionScore(checkpoint="output_content", score=round(contain * 0.7 + avoid * 0.3, 3), max_score=1.0, details=f"Contains {hits}/{len(must_contain)}, violations {violations}/{len(must_not_contain)}")


def score_intent_routing(
    actual_intent: str,
    actual_agent: str,
    actual_confidence: float,
    actual_denied: bool,
    expected: "ExpectedOutcome",
) -> DecisionScore:
    """Score the intent router's decision against expected outcome.

    Checks:
      - Did the router pick the right intent?
      - Did it route to the right agent?
      - Should it have denied/clarified?
      - Is confidence appropriate?
    """
    from .golden import ExpectedOutcome  # avoid circular at module level

    points = 0.0
    max_points = 0.0
    details_parts = []

    # Intent match
    if expected.expected_intent:
        max_points += 1.0
        if actual_intent == expected.expected_intent:
            points += 1.0
            details_parts.append(f"intent={actual_intent} correct")
        else:
            details_parts.append(f"intent={actual_intent} expected={expected.expected_intent}")

    # Agent match
    if expected.expected_agent:
        max_points += 1.0
        if actual_agent == expected.expected_agent:
            points += 1.0
            details_parts.append(f"agent={actual_agent} correct")
        else:
            details_parts.append(f"agent={actual_agent} expected={expected.expected_agent}")

    # Deny check
    if expected.should_deny:
        max_points += 1.0
        if actual_denied:
            points += 1.0
            details_parts.append("correctly denied")
        else:
            details_parts.append("should have denied but didn't")
    elif actual_denied:
        max_points += 1.0
        details_parts.append("incorrectly denied (false positive)")

    # Clarify check
    if expected.should_clarify:
        max_points += 1.0
        if actual_intent == "unclear":
            points += 1.0
            details_parts.append("correctly asked for clarification")
        else:
            details_parts.append("should have asked for clarification")

    score = points / max_points if max_points > 0 else 1.0
    return DecisionScore(
        checkpoint="intent_routing",
        score=round(score, 3),
        max_score=1.0,
        details="; ".join(details_parts),
    )


# ------------------------------------------------------------------
# Prompts
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
    {"description": "what to do", "tier": 2, "files_likely": ["path/file.py"], "dependencies": []}
  ]
}
"""

_CODER_PROMPT = """\
You are an expert software engineer. Given a subtask description and project \
context, produce a complete implementation.

Rules:
- Output ONLY the code changes needed
- For new files: `=== NEW FILE: path/to/file.py ===` header
- Include minimal, necessary comments
- Follow existing patterns and conventions
"""


def _parse_plan_json(raw: str) -> dict:
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
# Evaluation
# ------------------------------------------------------------------


async def evaluate_prompt(
    client: httpx.AsyncClient,
    task: GoldenTask,
    phrasing: Phrasing,
    *,
    lf_client=None,
    run_name: str = "",
    dataset_name: str = "golden-tasks",
) -> EvalResult:
    """Evaluate one (task, phrasing) pair. Traces to Langfuse if available."""
    result = EvalResult(
        task_id=task.id, task_name=task.name, category=task.category.value,
        phrasing_style=phrasing.style.value, phrasing_text=phrasing.text,
    )

    trace = None
    if lf_client:
        trace = lf_client.trace(
            name=f"golden-eval-{task.id}-{phrasing.style.value}",
            metadata={
                "task_id": task.id, "task_name": task.name,
                "category": task.category.value, "difficulty": task.difficulty,
                "phrasing_style": phrasing.style.value, "run_name": run_name,
            },
            input={"phrasing": phrasing.text, "context_files": list(task.context_files.keys())},
        )

    try:
        context_parts = [f"=== {p} ===\n{c}" for p, c in task.context_files.items()]
        context = "\n\n".join(context_parts) if context_parts else ""

        # Planner
        plan_msgs = []
        if context:
            plan_msgs.append({"role": "system", "content": f"Project files:\n{context}"})
        plan_msgs.append({"role": "system", "content": _PLANNER_PROMPT})
        plan_msgs.append({"role": "user", "content": phrasing.text})

        plan_raw = await _chat(client, plan_msgs, max_tokens=2048, temperature=0.7)
        plan_data = _parse_plan_json(plan_raw)
        actual_subtasks = plan_data.get("subtasks", [])

        if trace:
            trace.generation(name="planner", input=plan_msgs, output=plan_raw,
                             metadata={"subtask_count": len(actual_subtasks)})

        result.scores.append(score_plan_decomposition(actual_subtasks, task.expected.subtasks, task.expected.n_subtasks_min, task.expected.n_subtasks_max))
        result.scores.append(score_tier_estimates(actual_subtasks, task.expected.subtasks))
        result.scores.append(score_file_targeting(actual_subtasks, task.expected.subtasks))

        # Coder
        first = actual_subtasks[0].get("description", phrasing.text) if actual_subtasks else phrasing.text
        code_msgs = [
            {"role": "system", "content": f"{context}\n\n{_CODER_PROMPT}" if context else _CODER_PROMPT},
            {"role": "user", "content": f"## Subtask\n{first}"},
        ]
        code_output = await _chat(client, code_msgs, max_tokens=2048, temperature=0.3)

        if trace:
            trace.generation(name="coder", input=code_msgs, output=code_output)

        result.scores.append(score_output_content(code_output, task.expected.output_must_contain, task.expected.output_must_not_contain))

        # Attach scores to Langfuse
        if lf_client and trace:
            for s in result.scores:
                lf_client.score(trace_id=trace.id, name=s.checkpoint, value=s.score, comment=s.details)
            try:
                item_id = f"{task.id}_{phrasing.style.value}"
                trace.link_dataset_item(dataset_name=dataset_name, dataset_item_id=item_id, run_name=run_name)
            except Exception as exc:
                logger.debug("Could not link dataset item: %s", exc)

    except Exception as exc:
        result.error = str(exc)[:200]
        logger.error("Golden eval %s/%s failed: %s", task.id, phrasing.style.value, exc)

    result.compute_overall()

    # Build agent self-annotation (store to "memory")
    agent_ann = build_agent_annotation(
        overall_score=result.overall_score,
        scores=result.scores,
    )
    result.annotations = AnnotationPair(agent=agent_ann)

    if lf_client and trace:
        lf_client.score(trace_id=trace.id, name="overall", value=result.overall_score)
        # Store agent annotation as structured Langfuse score
        lf_client.score(
            trace_id=trace.id,
            name="agent_annotation",
            value=agent_ann.rating,
            comment=json.dumps({
                "rationale": agent_ann.rationale,
                "strengths": agent_ann.strengths,
                "weaknesses": agent_ann.weaknesses,
                "plan_correct": agent_ann.plan_correct,
                "output_correct": agent_ann.output_correct,
                "tier_correct": agent_ann.tier_correct,
                "tags": agent_ann.tags,
            }),
        )
        trace.update(output={
            "overall_score": result.overall_score,
            "passed": result.passed,
            "agent_annotation": {
                "rating": agent_ann.rating,
                "rationale": agent_ann.rationale,
                "strengths": agent_ann.strengths,
                "weaknesses": agent_ann.weaknesses,
                "tags": agent_ann.tags,
            },
        })

    return result


# ------------------------------------------------------------------
# Main collector
# ------------------------------------------------------------------


async def collect_golden(
    client: httpx.AsyncClient,
    n_samples: int = 0,
    seed: int = 42,
    tasks: list[GoldenTask] | None = None,
    dataset_name: str = "golden-tasks",
) -> GoldenEvaluation:
    """Run golden evaluation. Traces to Langfuse; human annotates in Langfuse UI.

    Every real task also bootstraps the dataset via bootstrap_from_task().
    """
    lf = _get_langfuse()
    task_list = tasks or GOLDEN_TASKS
    evaluation = GoldenEvaluation()
    run_name = f"golden-eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    if lf:
        print(f"  Langfuse connected — run: '{run_name}'")
        print(f"  Annotate: Langfuse UI → Datasets → {dataset_name} → {run_name}")
        upload_golden_dataset(task_list, dataset_name=dataset_name)
    else:
        print("  Langfuse unavailable — local-only mode")

    pairs = sample_prompts(task_list, n=n_samples, seed=seed) if n_samples > 0 else [(t, p) for t in task_list for p in t.phrasings]

    for task, phrasing in pairs:
        print(f"    [{task.id}] {phrasing.style.value:15s} ...", end="", flush=True)
        result = await evaluate_prompt(client, task, phrasing, lf_client=lf, run_name=run_name, dataset_name=dataset_name)
        evaluation.results.append(result)
        icon = "PASS" if result.passed else "FAIL"
        print(f" [{icon}] {result.overall_score:.1%}")

    if lf:
        lf.flush()

    evaluation.compute_summary()
    return evaluation


# ------------------------------------------------------------------
# Pull human ratings back from Langfuse
# ------------------------------------------------------------------


def _parse_annotation_comment(comment: str) -> dict:
    """Parse structured annotation from Langfuse score comment JSON."""
    if not comment:
        return {}
    try:
        return json.loads(comment)
    except (json.JSONDecodeError, TypeError):
        return {"rationale": comment}


def _build_annotation_from_langfuse(value: float, comment: str) -> Annotation:
    """Reconstruct an Annotation from a Langfuse score."""
    data = _parse_annotation_comment(comment)
    return Annotation(
        rating=value,
        rationale=data.get("rationale", comment or ""),
        strengths=data.get("strengths", []),
        weaknesses=data.get("weaknesses", []),
        intent_correct=data.get("intent_correct"),
        plan_correct=data.get("plan_correct"),
        output_correct=data.get("output_correct"),
        tier_correct=data.get("tier_correct"),
        tags=data.get("tags", []),
    )


def pull_human_ratings(
    dataset_name: str = "golden-tasks",
    run_name: str = "",
    embed_fn=None,
) -> GoldenEvaluation | None:
    """Pull evaluation + human annotations from Langfuse.

    After running collect_golden() or bootstrapping real tasks, humans
    annotate traces in the Langfuse UI. This pulls those ratings back,
    reconstructs the annotation pairs, and computes agreement scores.

    Langfuse score naming convention:
      - "agent_annotation": agent self-assessment (auto-generated)
      - "user_annotation": human assessment (added in Langfuse UI)
      - "human_rating": legacy numeric rating (1-5)
      - "overall": automated overall score

    If embed_fn is provided, it's used to compute semantic similarity
    between agent and user annotation rationales (cosine similarity).
    Signature: (text: str) -> list[float]
    """
    lf = _get_langfuse()
    if lf is None:
        return None

    evaluation = GoldenEvaluation()

    try:
        dataset = lf.get_dataset(name=dataset_name)
        if not dataset:
            return None

        runs = dataset.runs
        if run_name:
            runs = [r for r in runs if r.name == run_name]
        if not runs:
            return None

        run = runs[-1]
        for item_run in run.dataset_run_items:
            trace = item_run.trace
            if not trace:
                continue

            metadata = trace.metadata or {}
            result = EvalResult(
                task_id=metadata.get("task_id", ""),
                task_name=metadata.get("task_name", ""),
                category=metadata.get("category", ""),
                phrasing_style=metadata.get("phrasing_style", ""),
                phrasing_text=trace.input.get("phrasing", "") if trace.input else "",
            )

            agent_ann = None
            user_ann = None

            try:
                scores = lf.client.score.get_by_trace(trace.id)
                for s in scores:
                    if s.name == "agent_annotation":
                        agent_ann = _build_annotation_from_langfuse(s.value, s.comment or "")
                    elif s.name == "user_annotation":
                        user_ann = _build_annotation_from_langfuse(s.value, s.comment or "")
                    elif s.name == "human_rating":
                        result.human_rating = s.value
                        result.human_notes = s.comment or ""
                        # Also build a user annotation from legacy rating
                        if user_ann is None:
                            user_ann = Annotation(
                                rating=s.value / 5.0,
                                rationale=s.comment or "",
                            )
                    elif s.name == "overall":
                        result.overall_score = s.value
                    else:
                        result.scores.append(DecisionScore(
                            checkpoint=s.name, score=s.value, max_score=1.0, details=s.comment or "",
                        ))
            except Exception as exc:
                logger.debug("Error fetching scores for trace %s: %s", trace.id, exc)

            # Compute annotation agreement if both sides exist
            if agent_ann and user_ann:
                result.annotations = compute_annotation_agreement(
                    agent_ann, user_ann, embed_fn=embed_fn,
                )
            elif agent_ann:
                result.annotations = AnnotationPair(agent=agent_ann)
            elif user_ann:
                result.annotations = AnnotationPair(user=user_ann)

            result.passed = result.overall_score >= 0.7
            evaluation.results.append(result)

    except Exception as exc:
        logger.error("Error pulling from Langfuse: %s", exc)
        return None

    evaluation.compute_summary()
    return evaluation
