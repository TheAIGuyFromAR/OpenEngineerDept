"""Golden dataset — task x phrasing matrix for evaluating orchestrator decision quality.

Structure:
  - GoldenTask: one canonical task (what to accomplish)
  - Phrasing: one way to express that task (how the user asks)
  - Each task has ~12 phrasings covering different communication styles:
    terse, verbose, vague, specific, imperative, question-form, etc.
  - Expected outcomes are attached to the TASK (not the phrasing) since
    the result should be the same regardless of how you ask

Evaluation flow:
  1. Pick a random sample of (task, phrasing) pairs
  2. Run each through the orchestrator pipeline
  3. Score decisions at each checkpoint against expected outcomes
  4. Human rates a subset to calibrate automated scoring
  5. Compute agreement between automated and human scores

Starting with 3 tasks (verification set). Full suite targets 100 tasks x 12 phrasings = 1,200 prompts.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


# ------------------------------------------------------------------
# Categories
# ------------------------------------------------------------------


class TaskCategory(str, Enum):
    CODE_BUGFIX = "code_bugfix"
    CODE_FEATURE = "code_feature"
    CODE_REFACTOR = "code_refactor"
    CODE_TEST = "code_test"
    AUTOMATION_FILE_OPS = "automation_file_ops"
    AUTOMATION_CONFIG = "automation_config"
    AUTOMATION_MULTI_STEP = "automation_multi_step"
    ANALYSIS_REVIEW = "analysis_review"
    ANALYSIS_ARCHITECTURE = "analysis_architecture"


class PhrasingStyle(str, Enum):
    """How the user communicates the task."""
    TERSE = "terse"                   # Minimal words: "fix pagination bug"
    VERBOSE = "verbose"               # Full context paragraph
    VAGUE = "vague"                   # Imprecise: "the page thing is broken"
    SPECIFIC = "specific"             # Exact file, line, variable names
    IMPERATIVE = "imperative"         # Direct command: "Change X to Y"
    QUESTION = "question"             # "Can you fix...?" / "How would you..."
    FRUSTRATED = "frustrated"         # "This STILL doesn't work, page 1 shows wrong items"
    TECHNICAL = "technical"           # Uses precise jargon
    NON_TECHNICAL = "non_technical"   # Describes symptoms, not causes
    MULTI_PART = "multi_part"         # Lists multiple things at once
    CONTEXT_HEAVY = "context_heavy"   # Lots of background before the actual ask
    MINIMAL_CONTEXT = "minimal_context"  # Assumes you know the codebase


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class Phrasing:
    """One way to express a task."""
    style: PhrasingStyle
    text: str


@dataclass
class ExpectedSubtask:
    """What a correct decomposition should produce."""
    description_keywords: list[str]     # Keywords that MUST appear in subtask description
    expected_tier: int                  # Correct tier (1-3)
    tier_tolerance: int = 1             # Acceptable deviation
    expected_files: list[str] = field(default_factory=list)
    change_type: str = ""               # "create", "modify", "delete"


@dataclass
class ExpectedOutcome:
    """What the orchestrator SHOULD decide — attached to the task, not the phrasing."""
    n_subtasks_min: int = 1
    n_subtasks_max: int = 1
    subtasks: list[ExpectedSubtask] = field(default_factory=list)
    output_must_contain: list[str] = field(default_factory=list)
    output_must_not_contain: list[str] = field(default_factory=list)
    min_review_score: float = 7.0
    max_retries_expected: int = 1


@dataclass
class GoldenTask:
    """One canonical task with multiple phrasings and expected outcomes."""
    id: str
    category: TaskCategory
    name: str                           # Human-readable name
    difficulty: str                     # "easy", "medium", "hard"
    phrasings: list[Phrasing]           # ~12 ways to ask for this
    context_files: dict[str, str]       # path → file content (project context)
    expected: ExpectedOutcome           # What correct looks like


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------


@dataclass
class DecisionScore:
    """Score for a single decision checkpoint."""
    checkpoint: str
    score: float          # 0.0 - 1.0
    max_score: float      # 1.0
    details: str = ""


@dataclass
class EvalResult:
    """Result for one (task, phrasing) evaluation."""
    task_id: str
    task_name: str
    category: str
    phrasing_style: str
    phrasing_text: str
    scores: list[DecisionScore] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = False
    error: str = ""
    # Human rating fields (filled in during calibration)
    human_rating: Optional[float] = None     # 1-5 scale
    human_notes: str = ""

    def compute_overall(self) -> None:
        if not self.scores:
            self.overall_score = 0.0
            return
        total = sum(s.score for s in self.scores)
        max_total = sum(s.max_score for s in self.scores)
        self.overall_score = round(total / max_total, 3) if max_total > 0 else 0.0
        self.passed = self.overall_score >= 0.7


@dataclass
class GoldenEvaluation:
    """Complete evaluation across sampled (task, phrasing) pairs."""
    results: list[EvalResult] = field(default_factory=list)
    overall_score: float = 0.0
    pass_rate: float = 0.0
    category_scores: dict[str, float] = field(default_factory=dict)
    phrasing_scores: dict[str, float] = field(default_factory=dict)
    human_agreement: float = 0.0   # Correlation between automated and human scores

    def compute_summary(self) -> None:
        if not self.results:
            return

        self.overall_score = round(
            sum(r.overall_score for r in self.results) / len(self.results), 3
        )
        self.pass_rate = round(
            sum(1 for r in self.results if r.passed) / len(self.results) * 100, 1
        )

        # Per-category
        by_cat: dict[str, list[float]] = {}
        for r in self.results:
            by_cat.setdefault(r.category, []).append(r.overall_score)
        self.category_scores = {
            cat: round(sum(s) / len(s), 3) for cat, s in by_cat.items()
        }

        # Per-phrasing style
        by_style: dict[str, list[float]] = {}
        for r in self.results:
            by_style.setdefault(r.phrasing_style, []).append(r.overall_score)
        self.phrasing_scores = {
            style: round(sum(s) / len(s), 3) for style, s in by_style.items()
        }

        # Human agreement (if any human ratings exist)
        rated = [(r.overall_score, r.human_rating) for r in self.results if r.human_rating is not None]
        if len(rated) >= 3:
            auto_scores = [r[0] for r in rated]
            human_scores = [r[1] / 5.0 for r in rated]  # Normalize to 0-1
            # Simple correlation (Pearson)
            n = len(rated)
            mean_a = sum(auto_scores) / n
            mean_h = sum(human_scores) / n
            cov = sum((a - mean_a) * (h - mean_h) for a, h in zip(auto_scores, human_scores))
            std_a = (sum((a - mean_a) ** 2 for a in auto_scores)) ** 0.5
            std_h = (sum((h - mean_h) ** 2 for h in human_scores)) ** 0.5
            if std_a > 0 and std_h > 0:
                self.human_agreement = round(cov / (std_a * std_h), 3)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def render_report(self) -> str:
        lines: list[str] = []
        w = lines.append

        w("=" * 68)
        w("  GOLDEN DATASET EVALUATION REPORT")
        w("=" * 68)
        w(f"  Prompts evaluated: {len(self.results)}")
        w(f"  Overall score:     {self.overall_score:.1%}")
        w(f"  Pass rate:         {self.pass_rate:.0f}%")
        if self.human_agreement:
            w(f"  Human agreement:   {self.human_agreement:.2f} (Pearson r)")
        w("")

        # By category
        w("── By Category ─────────────────────────────────────────")
        for cat, score in sorted(self.category_scores.items()):
            icon = "PASS" if score >= 0.7 else "FAIL"
            w(f"  [{icon}] {cat:30s}  {score:.1%}")
        w("")

        # By phrasing style — this is the key insight
        w("── By Phrasing Style ───────────────────────────────────")
        for style, score in sorted(self.phrasing_scores.items(), key=lambda x: -x[1]):
            icon = "PASS" if score >= 0.7 else "FAIL"
            w(f"  [{icon}] {style:20s}  {score:.1%}")
        w("")

        # Individual results
        w("── Individual Results ──────────────────────────────────")
        for r in self.results:
            icon = "PASS" if r.passed else "FAIL"
            human = f"  human={r.human_rating:.0f}/5" if r.human_rating else ""
            w(f"  [{icon}] {r.task_name[:30]:30s}  {r.phrasing_style:15s}  {r.overall_score:.1%}{human}")
            if r.error:
                w(f"         ERROR: {r.error[:60]}")
            for s in r.scores:
                mark = "ok" if s.score >= s.max_score * 0.7 else "!!"
                w(f"         [{mark}] {s.checkpoint:22s}  {s.score:.2f}/{s.max_score:.2f}  {s.details}")
        w("")

        w("=" * 68)
        verdict = "PASS" if self.pass_rate >= 70 else "FAIL"
        w(f"  VERDICT: {verdict}")
        w("=" * 68)

        return "\n".join(lines)


# ------------------------------------------------------------------
# Sampling
# ------------------------------------------------------------------


def sample_prompts(
    tasks: list[GoldenTask],
    n: int = 50,
    seed: int | None = None,
) -> list[tuple[GoldenTask, Phrasing]]:
    """Sample n (task, phrasing) pairs, stratified by category and style.

    Ensures coverage: tries to include at least one phrasing per task
    and at least one task per category before filling randomly.
    """
    rng = random.Random(seed)

    # Build all possible pairs
    all_pairs = []
    for task in tasks:
        for phrasing in task.phrasings:
            all_pairs.append((task, phrasing))

    if n >= len(all_pairs):
        return all_pairs

    # Ensure at least one per task
    selected: list[tuple[GoldenTask, Phrasing]] = []
    for task in tasks:
        if task.phrasings:
            selected.append((task, rng.choice(task.phrasings)))

    # Fill remaining slots randomly (no duplicates)
    selected_set = {(t.id, p.style.value) for t, p in selected}
    remaining = [(t, p) for t, p in all_pairs if (t.id, p.style.value) not in selected_set]
    rng.shuffle(remaining)

    while len(selected) < n and remaining:
        selected.append(remaining.pop())

    return selected[:n]


# ------------------------------------------------------------------
# Golden Task Suite — First 3 (verification set)
# ------------------------------------------------------------------


GOLDEN_TASKS: list[GoldenTask] = [

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TASK 1: Fix off-by-one bug in pagination
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    GoldenTask(
        id="T001",
        category=TaskCategory.CODE_BUGFIX,
        name="Fix off-by-one in pagination",
        difficulty="easy",
        context_files={
            "utils/pagination.py": (
                "def paginate(items: list, page: int, page_size: int = 10) -> list:\n"
                '    """Return a page of items. Pages are 1-indexed."""\n'
                "    start = page * page_size  # BUG: should be (page - 1) * page_size\n"
                "    end = start + page_size\n"
                "    return items[start:end]\n"
            ),
            "tests/test_pagination.py": (
                "from utils.pagination import paginate\n\n"
                "def test_first_page():\n"
                "    items = list(range(25))\n"
                "    result = paginate(items, page=1, page_size=10)\n"
                "    assert result == list(range(10))\n\n"
                "def test_last_page():\n"
                "    items = list(range(25))\n"
                "    result = paginate(items, page=3, page_size=10)\n"
                "    assert result == [20, 21, 22, 23, 24]\n"
            ),
        },
        expected=ExpectedOutcome(
            n_subtasks_min=1,
            n_subtasks_max=1,
            subtasks=[
                ExpectedSubtask(
                    description_keywords=["pagination", "fix", "page", "off-by-one"],
                    expected_tier=1,
                    expected_files=["utils/pagination.py"],
                    change_type="modify",
                ),
            ],
            output_must_contain=["(page - 1)"],
            output_must_not_contain=["page * page_size"],
            min_review_score=8.0,
            max_retries_expected=0,
        ),
        phrasings=[
            Phrasing(
                style=PhrasingStyle.TERSE,
                text="fix pagination bug in utils/pagination.py",
            ),
            Phrasing(
                style=PhrasingStyle.VERBOSE,
                text=(
                    "I've been debugging an issue with our pagination utility. When a user "
                    "requests page 1 with a page_size of 10 on a list that has 25 items, "
                    "they should get items 0-9 (the first 10). But instead they're getting "
                    "items 10-19, which is actually page 2's data. The function is in "
                    "utils/pagination.py. The existing tests in tests/test_pagination.py "
                    "document the expected behavior — they're currently failing."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.VAGUE,
                text="the page thing is broken, page 1 shows the wrong items",
            ),
            Phrasing(
                style=PhrasingStyle.SPECIFIC,
                text=(
                    "In utils/pagination.py line 3, change `start = page * page_size` "
                    "to `start = (page - 1) * page_size`. The pages are 1-indexed but "
                    "the slice is 0-indexed."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.IMPERATIVE,
                text="Fix the slice calculation in paginate(). It's using page * page_size but pages start at 1.",
            ),
            Phrasing(
                style=PhrasingStyle.QUESTION,
                text="Can you figure out why paginate() returns wrong results for page 1? The tests are failing.",
            ),
            Phrasing(
                style=PhrasingStyle.FRUSTRATED,
                text=(
                    "The pagination is STILL broken. I told you pages are 1-indexed! "
                    "Page 1 should be the FIRST page, not the second. Look at the test failures."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.TECHNICAL,
                text=(
                    "Off-by-one error in paginate() — the start index uses 1-indexed page "
                    "number directly as a 0-indexed multiplier. Need to subtract 1 from page "
                    "before multiplying by page_size for the slice offset."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.NON_TECHNICAL,
                text=(
                    "When I look at the first page of results, it's showing me the second "
                    "page's data. Everything is shifted by one page. The last page is empty "
                    "too. Something is off with the page numbering."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.MULTI_PART,
                text=(
                    "1. Fix the pagination bug in utils/pagination.py\n"
                    "2. Make sure all existing tests pass after the fix\n"
                    "3. The issue is that page 1 returns page 2's results"
                ),
            ),
            Phrasing(
                style=PhrasingStyle.CONTEXT_HEAVY,
                text=(
                    "We have a REST API that uses cursor-based pagination for large datasets. "
                    "The frontend sends page=1&page_size=10 for the initial load. Our "
                    "pagination utility in utils/pagination.py converts these to list slices. "
                    "We recently switched from 0-indexed to 1-indexed pages to match the API "
                    "convention, but forgot to update the slice math. The function now skips "
                    "the first page_size items because it multiplies page directly instead of "
                    "(page-1). Please fix this."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.MINIMAL_CONTEXT,
                text="paginate() is off by one. page=1 should start at index 0.",
            ),
        ],
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TASK 2: Add retry decorator with exponential backoff
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    GoldenTask(
        id="T002",
        category=TaskCategory.CODE_FEATURE,
        name="Add retry decorator with exponential backoff",
        difficulty="medium",
        context_files={},
        expected=ExpectedOutcome(
            n_subtasks_min=1,
            n_subtasks_max=3,
            subtasks=[
                ExpectedSubtask(
                    description_keywords=["retry", "decorator", "backoff"],
                    expected_tier=2,
                    expected_files=["utils/retry.py"],
                    change_type="create",
                ),
            ],
            output_must_contain=["retry", "def", "sleep", "except"],
            output_must_not_contain=[],
            min_review_score=7.0,
            max_retries_expected=1,
        ),
        phrasings=[
            Phrasing(
                style=PhrasingStyle.TERSE,
                text="create a retry decorator with exponential backoff",
            ),
            Phrasing(
                style=PhrasingStyle.VERBOSE,
                text=(
                    "We need a reusable retry mechanism for our HTTP client calls and "
                    "database operations. Please create a Python decorator called @retry "
                    "that wraps a function and automatically retries it when it raises an "
                    "exception. It should use exponential backoff — first retry after 1 second, "
                    "then 2 seconds, then 4 seconds, etc. Configuration should include "
                    "max_retries (default 3), base_delay (default 1.0 seconds), and which "
                    "exception types to catch (default: all exceptions). Put it in utils/retry.py "
                    "with type hints and a docstring. Also write tests."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.VAGUE,
                text="we need something to retry failed operations automatically with increasing delays",
            ),
            Phrasing(
                style=PhrasingStyle.SPECIFIC,
                text=(
                    "Create utils/retry.py with a @retry(max_retries=3, base_delay=1.0, "
                    "exceptions=(Exception,)) decorator. Use time.sleep(base_delay * 2**attempt) "
                    "for backoff. Re-raise the last exception after all retries are exhausted. "
                    "Include type annotations for Callable and ParamSpec."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.IMPERATIVE,
                text=(
                    "Build a @retry decorator. Exponential backoff. Max 3 retries by default. "
                    "Configurable exception types. Put it in utils/retry.py. Add tests."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.QUESTION,
                text=(
                    "How would you implement a retry decorator with exponential backoff in Python? "
                    "I need it to handle transient failures in our API calls. Can you create one "
                    "and add it to our utils package?"
                ),
            ),
            Phrasing(
                style=PhrasingStyle.FRUSTRATED,
                text=(
                    "Our API calls keep failing randomly and we're not handling retries AT ALL. "
                    "We need a retry decorator ASAP. Exponential backoff, configurable retries, "
                    "the whole thing. I'm tired of these transient failures crashing the pipeline."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.TECHNICAL,
                text=(
                    "Implement a parametric decorator @retry(max_retries, base_delay, exceptions) "
                    "with exponential backoff (delay = base_delay * 2^attempt). Should preserve "
                    "the wrapped function's signature via functools.wraps. Handle the edge case "
                    "where max_retries=0 means no retry. Raise the final exception after exhaustion."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.NON_TECHNICAL,
                text=(
                    "Sometimes our network requests fail temporarily and we just need to try again. "
                    "Can you make something that automatically tries again when something fails, "
                    "but waits a little longer each time before trying? Maybe 3 tries total, "
                    "with longer waits between each."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.MULTI_PART,
                text=(
                    "I need a retry utility:\n"
                    "- Create a @retry decorator in utils/retry.py\n"
                    "- Exponential backoff (1s, 2s, 4s, ...)\n"
                    "- Configurable: max retries, base delay, exception types\n"
                    "- Write tests in tests/test_retry.py\n"
                    "- Include type hints"
                ),
            ),
            Phrasing(
                style=PhrasingStyle.CONTEXT_HEAVY,
                text=(
                    "Our system makes a lot of calls to external services — payment gateways, "
                    "email providers, third-party APIs. These occasionally fail with transient "
                    "errors (timeouts, 429s, connection resets). Right now each caller implements "
                    "its own retry logic (or doesn't), which is inconsistent and error-prone. "
                    "We need a centralized retry decorator that can be applied to any function. "
                    "Exponential backoff is important to avoid thundering herd. Default to 3 "
                    "retries with 1-second base delay. Please create this in utils/retry.py."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.MINIMAL_CONTEXT,
                text="@retry decorator. Exp backoff. utils/retry.py.",
            ),
        ],
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TASK 3: Security review of authentication module
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    GoldenTask(
        id="T003",
        category=TaskCategory.ANALYSIS_REVIEW,
        name="Security review of auth module",
        difficulty="hard",
        context_files={
            "auth/login.py": (
                "import hashlib\nimport sqlite3\n\n"
                "def login(username: str, password: str) -> bool:\n"
                '    conn = sqlite3.connect("users.db")\n'
                "    hashed = hashlib.md5(password.encode()).hexdigest()\n"
                '    query = f"SELECT * FROM users WHERE username=\'{username}\' AND password=\'{hashed}\'"\n'
                "    result = conn.execute(query).fetchone()\n"
                "    return result is not None\n\n"
                "def reset_password(email: str, new_password: str) -> None:\n"
                '    conn = sqlite3.connect("users.db")\n'
                '    conn.execute(f"UPDATE users SET password=\'{new_password}\' WHERE email=\'{email}\'")\n'
                "    conn.commit()\n"
            ),
        },
        expected=ExpectedOutcome(
            n_subtasks_min=1,
            n_subtasks_max=3,
            subtasks=[
                ExpectedSubtask(
                    description_keywords=["security", "auth", "vulnerab"],
                    expected_tier=2,
                    expected_files=["auth/login.py"],
                    change_type="modify",
                ),
            ],
            output_must_contain=["injection", "parameterized", "md5"],
            output_must_not_contain=[],
            min_review_score=7.0,
            max_retries_expected=1,
        ),
        phrasings=[
            Phrasing(
                style=PhrasingStyle.TERSE,
                text="security review auth/login.py and fix the issues",
            ),
            Phrasing(
                style=PhrasingStyle.VERBOSE,
                text=(
                    "I need a thorough security review of our authentication module at "
                    "auth/login.py. This code handles user login and password reset. "
                    "Please identify every security vulnerability you can find, classify "
                    "each by severity (critical, high, medium, low), explain why each is "
                    "dangerous, and then write the fixed code. Don't just describe the "
                    "problems — actually fix them."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.VAGUE,
                text="the login code looks sketchy, can you make it more secure?",
            ),
            Phrasing(
                style=PhrasingStyle.SPECIFIC,
                text=(
                    "auth/login.py has at least 3 critical vulnerabilities: SQL injection "
                    "via f-string queries, MD5 for password hashing (unsalted), and storing "
                    "plaintext in reset_password. Fix all three: use parameterized queries, "
                    "switch to bcrypt with salt, and hash the new password in reset_password."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.IMPERATIVE,
                text=(
                    "Fix the security holes in auth/login.py. Replace f-strings with "
                    "parameterized queries. Replace MD5 with bcrypt. Hash passwords in "
                    "reset_password too."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.QUESTION,
                text=(
                    "Is our login code in auth/login.py secure? I'm worried about SQL "
                    "injection and the password hashing. Can you audit it and fix any "
                    "problems you find?"
                ),
            ),
            Phrasing(
                style=PhrasingStyle.FRUSTRATED,
                text=(
                    "A pentester just found SQL injection in our login page. This is "
                    "CRITICAL — we're using f-strings for SQL queries and MD5 for "
                    "passwords! Fix auth/login.py before this goes live."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.TECHNICAL,
                text=(
                    "Audit auth/login.py for OWASP Top 10 violations. I can see at least "
                    "A03:2021-Injection (SQL injection via string interpolation) and "
                    "A02:2021-Cryptographic Failures (MD5, no salt, no KDF). Fix with "
                    "parameterized queries and bcrypt/argon2 respectively. Check for "
                    "A07:2021 issues too."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.NON_TECHNICAL,
                text=(
                    "I hired a security consultant and they said our login system has "
                    "problems. Something about the way we check passwords and look up "
                    "users in the database isn't safe. Can you look at auth/login.py "
                    "and make it more secure?"
                ),
            ),
            Phrasing(
                style=PhrasingStyle.MULTI_PART,
                text=(
                    "Security audit needed for auth/login.py:\n"
                    "1. Find all vulnerabilities\n"
                    "2. Classify severity (critical/high/medium/low)\n"
                    "3. Fix each vulnerability in the code\n"
                    "4. Make sure both login() and reset_password() are secure"
                ),
            ),
            Phrasing(
                style=PhrasingStyle.CONTEXT_HEAVY,
                text=(
                    "We're preparing for SOC 2 compliance and our auditor flagged the "
                    "authentication module as a risk area. The code in auth/login.py "
                    "was written early in the project when security wasn't a priority. "
                    "We use SQLite for the user database and the code does direct SQL "
                    "queries for login verification and password reset. The password "
                    "hashing looks like it might be using an older algorithm. We need "
                    "this fixed before the audit next month. Please review it and "
                    "implement all necessary security improvements."
                ),
            ),
            Phrasing(
                style=PhrasingStyle.MINIMAL_CONTEXT,
                text="audit and fix auth/login.py — SQL injection + weak hashing",
            ),
        ],
    ),
]
