"""Evidence collectors — run real tasks against the live gateway and capture metrics.

Each collector is an async function that takes an httpx.AsyncClient pointed
at the gateway and returns a typed evidence dataclass.
"""

from __future__ import annotations

import asyncio
import subprocess
import shutil
import textwrap
import time

import httpx

from .models import (
    ConsistencyEvidence,
    ContextEvidence,
    CorrectnessCase,
    CorrectnessEvidence,
    LatencyStats,
    ReasoningCase,
    ReasoningEvidence,
    ResourceEvidence,
    ThroughputEvidence,
)


# ── Helpers ────────────────────────────────────────────────────────


async def _chat(
    client: httpx.AsyncClient,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> tuple[str, dict, float]:
    """Send chat completion, return (content, usage, elapsed_ms)."""
    start = time.monotonic()
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    elapsed_ms = (time.monotonic() - start) * 1000
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return content, usage, elapsed_ms


def _extract_python(text: str) -> str:
    """Extract Python code from markdown fences or raw response."""
    # Try to extract from ```python ... ``` block
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) > 2:
            return parts[1].strip()
    # If no fences, return as-is (might be raw code)
    return text.strip()


# ── Sandboxed code execution ─────────────────────────────────────
#
# LLM-generated code is UNTRUSTED. Before executing it, we:
# 1. AST-parse to block dangerous constructs (imports, builtins, dunders)
# 2. Execute in subprocess with timeout, no stdin, captured output
#
# This is defense-in-depth, not a true sandbox. For production, use
# Docker with --network=none or a WASM runtime.

import ast

_BLOCKED_NAMES = frozenset({
    "exec", "eval", "compile", "open", "__import__",
    "breakpoint", "exit", "quit", "input",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr",
})

_BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "httpx",
    "ctypes", "importlib", "signal", "multiprocessing",
    "threading", "pickle", "shelve", "tempfile",
    "webbrowser", "code", "codeop", "pty",
})


class UnsafeCodeError(Exception):
    """Raised when LLM-generated code contains disallowed constructs."""


def _validate_code_safety(code: str) -> None:
    """Static analysis: reject code that uses dangerous constructs."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCodeError(f"Syntax error in generated code: {exc}") from exc

    for node in ast.walk(tree):
        # Block dangerous imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                root_module = node.module.split(".")[0]
                if root_module in _BLOCKED_MODULES:
                    raise UnsafeCodeError(f"Blocked import: {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    if root_module in _BLOCKED_MODULES:
                        raise UnsafeCodeError(f"Blocked import: {alias.name}")

        # Block dangerous built-in calls
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_NAMES:
                raise UnsafeCodeError(f"Blocked function call: {func.id}")
            if isinstance(func, ast.Attribute) and func.attr in _BLOCKED_NAMES:
                raise UnsafeCodeError(f"Blocked method call: {func.attr}")

        # Block dangerous dunder access
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                safe_dunders = {"__init__", "__str__", "__repr__", "__len__",
                                "__eq__", "__lt__", "__gt__", "__hash__",
                                "__iter__", "__next__", "__contains__",
                                "__getitem__", "__setitem__", "__bool__"}
                if node.attr not in safe_dunders:
                    raise UnsafeCodeError(f"Blocked dunder access: {node.attr}")


def _run_code_sandboxed(code: str, timeout: int = 10) -> subprocess.CompletedProcess:
    """Validate and execute code with safety checks + resource limits."""
    _validate_code_safety(code)
    return subprocess.run(
        ["python3", "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


# ── Throughput + Latency ──────────────────────────────────────────


async def collect_throughput(
    client: httpx.AsyncClient,
    n_requests: int = 5,
) -> ThroughputEvidence:
    """Fire N requests sequentially, measure tok/s and latency distribution."""
    latencies: list[float] = []
    total_prompt_tok = 0
    total_gen_tok = 0
    total_prompt_ms = 0
    total_gen_ms = 0

    prompt = "Write a Python function that reverses a linked list iteratively. Just the function."

    for _ in range(n_requests):
        content, usage, elapsed_ms = await _chat(
            client,
            messages=[
                {"role": "system", "content": "You are a concise Python coder."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
            temperature=0.5,
        )
        latencies.append(elapsed_ms)

        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        gen_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))

        total_prompt_tok += prompt_tokens
        total_gen_tok += gen_tokens

        # Rough split: assume prompt takes ~10% of time, generation ~90%
        if prompt_tokens > 0 and gen_tokens > 0:
            ratio = gen_tokens / (prompt_tokens + gen_tokens)
            total_gen_ms += elapsed_ms * ratio
            total_prompt_ms += elapsed_ms * (1 - ratio)
        else:
            total_gen_ms += elapsed_ms

    prompt_tps = (total_prompt_tok / (total_prompt_ms / 1000)) if total_prompt_ms > 0 else 0
    gen_tps = (total_gen_tok / (total_gen_ms / 1000)) if total_gen_ms > 0 else 0

    return ThroughputEvidence(
        prompt_tok_per_sec=round(prompt_tps, 1),
        generation_tok_per_sec=round(gen_tps, 1),
        total_requests=n_requests,
        latency=LatencyStats.from_samples(latencies),
    )


# ── Code Correctness ─────────────────────────────────────────────

# Each case: (name, prompt, test_code_template)
# test_code_template gets the generated code prepended, then runs assertions.
CORRECTNESS_CASES = [
    (
        "two_sum",
        "Write a Python function `two_sum(nums: list[int], target: int) -> list[int]` that returns the indices of two numbers that add up to target. Assume exactly one solution exists. Return ONLY the function, no examples or explanation.",
        textwrap.dedent("""\
            assert sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]
            assert sorted(two_sum([3, 2, 4], 6)) == [1, 2]
            assert sorted(two_sum([3, 3], 6)) == [0, 1]
        """),
    ),
    (
        "is_palindrome",
        "Write a Python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome, ignoring case and non-alphanumeric characters. Return ONLY the function.",
        textwrap.dedent("""\
            assert is_palindrome("A man, a plan, a canal: Panama") is True
            assert is_palindrome("race a car") is False
            assert is_palindrome("") is True
            assert is_palindrome("a") is True
        """),
    ),
    (
        "fizzbuzz",
        "Write a Python function `fizzbuzz(n: int) -> list[str]` that returns FizzBuzz for 1 to n. Return ONLY the function.",
        textwrap.dedent("""\
            result = fizzbuzz(15)
            assert result[0] == "1"
            assert result[2] == "Fizz"
            assert result[4] == "Buzz"
            assert result[14] == "FizzBuzz"
            assert len(result) == 15
        """),
    ),
    (
        "flatten_list",
        "Write a Python function `flatten(lst: list) -> list` that recursively flattens a nested list. Return ONLY the function.",
        textwrap.dedent("""\
            assert flatten([1, [2, [3, 4], 5], 6]) == [1, 2, 3, 4, 5, 6]
            assert flatten([]) == []
            assert flatten([1, 2, 3]) == [1, 2, 3]
            assert flatten([[[[1]]]]) == [1]
        """),
    ),
    (
        "binary_search",
        "Write a Python function `binary_search(arr: list[int], target: int) -> int` that returns the index of target in a sorted array, or -1 if not found. Return ONLY the function.",
        textwrap.dedent("""\
            assert binary_search([1, 3, 5, 7, 9], 5) == 2
            assert binary_search([1, 3, 5, 7, 9], 6) == -1
            assert binary_search([], 1) == -1
            assert binary_search([1], 1) == 0
        """),
    ),
    (
        "max_subarray",
        "Write a Python function `max_subarray_sum(nums: list[int]) -> int` that finds the contiguous subarray with the largest sum (Kadane's algorithm). Return ONLY the function.",
        textwrap.dedent("""\
            assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
            assert max_subarray_sum([1]) == 1
            assert max_subarray_sum([-1, -2, -3]) == -1
            assert max_subarray_sum([5, 4, -1, 7, 8]) == 23
        """),
    ),
]


async def collect_correctness(client: httpx.AsyncClient) -> CorrectnessEvidence:
    """Generate code for known problems, execute it, verify assertions pass."""
    cases: list[CorrectnessCase] = []
    passed = 0
    failed = 0
    errors = 0

    for name, prompt, test_code in CORRECTNESS_CASES:
        case = CorrectnessCase(name=name, prompt=prompt)
        try:
            content, usage, elapsed = await _chat(
                client,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Python expert. Output ONLY the function definition. "
                        "No markdown, no explanation, no examples. Just the def ... block.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.2,
            )
            case.generated_code = _extract_python(content)
            case.generation_ms = elapsed

            # Execute the generated code + test assertions (sandboxed)
            full_code = case.generated_code + "\n\n" + test_code
            try:
                result = _run_code_sandboxed(full_code, timeout=10)
                case.executed = True
                if result.returncode == 0:
                    case.passed = True
                    passed += 1
                else:
                    case.passed = False
                    case.error = (result.stderr or result.stdout).strip()[:200]
                    failed += 1
            except UnsafeCodeError as exc:
                case.executed = False
                case.error = f"BLOCKED (unsafe code): {exc}"
                errors += 1
            except subprocess.TimeoutExpired:
                case.executed = True
                case.error = "Execution timed out (10s)"
                errors += 1

        except Exception as exc:
            case.error = str(exc)[:200]
            errors += 1

        cases.append(case)

    return CorrectnessEvidence(
        cases=cases,
        total=len(cases),
        passed=passed,
        failed=failed,
        errors=errors,
    )


# ── Reasoning Quality ────────────────────────────────────────────

REASONING_CASES = [
    ReasoningCase(
        name="exponential_recursion",
        prompt="What is wrong with this code and how would you fix it?\ndef fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\nprint(fib(50))",
        expected_keywords=["exponential", "slow", "memo", "cache", "O(2", "recursion", "dynamic"],
    ),
    ReasoningCase(
        name="sql_injection",
        prompt='Is this Python code safe?\nimport sqlite3\ndef get_user(name):\n    conn = sqlite3.connect("db.sqlite")\n    return conn.execute(f"SELECT * FROM users WHERE name = \'{name}\'").fetchone()',
        expected_keywords=["injection", "sql", "parameterized", "sanitize", "placeholder", "unsafe"],
    ),
    ReasoningCase(
        name="race_condition",
        prompt="What's the bug in this code?\nimport threading\ncounter = 0\ndef increment():\n    global counter\n    for _ in range(100000):\n        counter += 1\nthreads = [threading.Thread(target=increment) for _ in range(4)]\nfor t in threads: t.start()\nfor t in threads: t.join()\nprint(counter)",
        expected_keywords=["race", "thread", "lock", "atomic", "mutex", "concurrent", "GIL"],
    ),
    ReasoningCase(
        name="memory_leak",
        prompt="What's problematic about this pattern?\nclass EventBus:\n    _listeners = []\n    @classmethod\n    def subscribe(cls, callback):\n        cls._listeners.append(callback)\n    @classmethod\n    def emit(cls, event):\n        for cb in cls._listeners:\n            cb(event)",
        expected_keywords=["leak", "memory", "unsubscribe", "remove", "grow", "accumulate", "never cleared", "class variable", "shared"],
    ),
    ReasoningCase(
        name="floating_point",
        prompt="Why does this assertion fail? `assert 0.1 + 0.2 == 0.3`",
        expected_keywords=["float", "precision", "IEEE", "binary", "representation", "decimal", "approx", "isclose"],
    ),
]


async def collect_reasoning(client: httpx.AsyncClient) -> ReasoningEvidence:
    """Ask known-answer reasoning questions, check response contains key insights."""
    cases: list[ReasoningCase] = []
    passed = 0

    for template in REASONING_CASES:
        case = ReasoningCase(
            name=template.name,
            prompt=template.prompt,
            expected_keywords=template.expected_keywords,
        )

        try:
            content, _, elapsed = await _chat(
                client,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert software engineer. Be specific about the problem.",
                    },
                    {"role": "user", "content": case.prompt},
                ],
                max_tokens=512,
                temperature=0.3,
            )

            case.response = content
            case.generation_ms = elapsed

            # Check which keywords appear in the response
            lower_content = content.lower()
            for kw in case.expected_keywords:
                if kw.lower() in lower_content:
                    case.keywords_found.append(kw)
                else:
                    case.keywords_missing.append(kw)

            # Pass if at least 2 keywords found (model must demonstrate depth, not just mention)
            case.passed = len(case.keywords_found) >= 2
            if case.passed:
                passed += 1

        except Exception as exc:
            case.response = f"ERROR: {exc}"

        cases.append(case)

    return ReasoningEvidence(
        cases=cases,
        total=len(cases),
        passed=passed,
    )


# ── Context Window Stress ────────────────────────────────────────


async def collect_context(
    client: httpx.AsyncClient,
    claimed_ctx: int = 32768,
) -> ContextEvidence:
    """Test whether the model can actually use its claimed context window.

    Injects a hidden fact at the beginning of a long prompt and asks
    the model to retrieve it at the end.
    """
    evidence = ContextEvidence(claimed_ctx_size=claimed_ctx)

    # Test sizes: 1k, 2k, 4k, 8k, 16k, 32k (or up to claimed)
    test_sizes = [s for s in [1024, 2048, 4096, 8192, 16384, 32768] if s <= claimed_ctx]
    evidence.tested_sizes = test_sizes

    secret = "The launch code is ALPHA-7749-ZEBRA."
    padding_line = "This is line {n} of the document. It discusses various topics including software architecture, deployment strategies, and testing methodologies. "

    for size in test_sizes:
        # Build a prompt that's approximately `size` tokens (rough: 1 token ≈ 4 chars)
        target_chars = size * 3  # conservative estimate
        lines = [f"IMPORTANT FACT: {secret}\n"]
        n = 1
        while len("\n".join(lines)) < target_chars:
            lines.append(padding_line.format(n=n))
            n += 1
        lines.append("\nQUESTION: What is the launch code mentioned at the beginning of this document? Reply with ONLY the code, nothing else.")

        full_prompt = "\n".join(lines)

        try:
            content, _, elapsed = await _chat(
                client,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=64,
                temperature=0.1,
            )

            if "ALPHA-7749-ZEBRA" in content:
                evidence.passed_sizes.append(size)
                evidence.effective_ctx_size = size
            else:
                evidence.failed_size = size
                break  # No point testing larger sizes

        except Exception:
            evidence.failed_size = size
            break

    return evidence


# ── Consistency ──────────────────────────────────────────────────


async def collect_consistency(
    client: httpx.AsyncClient,
    n_runs: int = 5,
) -> ConsistencyEvidence:
    """Run the same prompt N times, measure variance in outputs."""
    prompt = "Write a Python function that checks if a number is prime. Return ONLY the function."
    outputs: list[str] = []
    all_correct = True

    for _ in range(n_runs):
        content, _, _ = await _chat(
            client,
            messages=[
                {"role": "system", "content": "Output ONLY the function. No markdown, no explanation."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
            temperature=0.3,
        )
        code = _extract_python(content)
        outputs.append(code)

        # Check correctness (sandboxed)
        test_code = code + textwrap.dedent("""
            assert is_prime(2) is True
            assert is_prime(3) is True
            assert is_prime(4) is False
            assert is_prime(17) is True
            assert is_prime(1) is False
        """)
        try:
            result = _run_code_sandboxed(test_code, timeout=5)
            if result.returncode != 0:
                all_correct = False
        except (UnsafeCodeError, subprocess.TimeoutExpired):
            all_correct = False

    lengths = [len(o) for o in outputs]
    from statistics import mean as _mean, stdev as _stdev

    return ConsistencyEvidence(
        prompt=prompt,
        n_runs=n_runs,
        outputs=outputs,
        unique_outputs=len(set(outputs)),
        mean_length=round(_mean(lengths), 1) if lengths else 0,
        length_stdev=round(_stdev(lengths), 1) if len(lengths) >= 2 else 0,
        all_correct=all_correct,
    )


# ── Resource Usage ───────────────────────────────────────────────

# Cost per 1M tokens (approximate, as of mid-2025)
API_COSTS: dict[str, tuple[float, float]] = {
    "anthropic": (3.0, 15.0),      # input, output per 1M tokens
    "openai": (2.5, 10.0),         # gpt-4o
    "openrouter": (1.0, 5.0),      # varies widely, rough average
}


async def collect_resources(
    client: httpx.AsyncClient,
    provider: str,
    total_prompt_tokens: int = 0,
    total_completion_tokens: int = 0,
    model_path: str = "",
) -> ResourceEvidence:
    """Collect resource usage — VRAM/RAM for local, cost for API."""
    evidence = ResourceEvidence(provider=provider)

    if provider == "local":
        # Try to get VRAM usage from nvidia-smi
        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    vram_values = [float(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
                    evidence.vram_used_mb = sum(vram_values)
            except Exception:
                pass

        # RAM usage from /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) / 1024  # KB to MB
                    elif line.startswith("MemAvailable:"):
                        available = int(line.split()[1]) / 1024
                evidence.ram_used_mb = round(total - available, 0)
        except Exception:
            pass

        # Model file size
        if model_path:
            try:
                from pathlib import Path

                p = Path(model_path)
                if p.exists():
                    evidence.model_file_size_mb = round(p.stat().st_size / (1024 * 1024), 0)
            except Exception:
                pass

    else:
        # API cost estimation
        evidence.total_prompt_tokens = total_prompt_tokens
        evidence.total_completion_tokens = total_completion_tokens
        costs = API_COSTS.get(provider, (2.0, 10.0))
        cost = (total_prompt_tokens * costs[0] + total_completion_tokens * costs[1]) / 1_000_000
        evidence.estimated_cost_usd = round(cost, 6)

    return evidence
