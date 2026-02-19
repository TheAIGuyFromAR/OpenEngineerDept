"""Evidence data models — structured results from each evidence collector."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median, stdev


@dataclass
class LatencyStats:
    """Percentile-based latency statistics."""

    count: int = 0
    min_ms: float = 0
    max_ms: float = 0
    mean_ms: float = 0
    median_ms: float = 0
    p95_ms: float = 0
    p99_ms: float = 0
    stdev_ms: float = 0

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> LatencyStats:
        if not samples_ms:
            return cls()
        s = sorted(samples_ms)
        n = len(s)
        return cls(
            count=n,
            min_ms=round(s[0], 1),
            max_ms=round(s[-1], 1),
            mean_ms=round(mean(s), 1),
            median_ms=round(median(s), 1),
            p95_ms=round(s[int(n * 0.95)] if n >= 2 else s[-1], 1),
            p99_ms=round(s[int(n * 0.99)] if n >= 2 else s[-1], 1),
            stdev_ms=round(stdev(s), 1) if n >= 2 else 0,
        )


@dataclass
class ThroughputEvidence:
    """Tokens-per-second measurement."""

    prompt_tok_per_sec: float = 0
    generation_tok_per_sec: float = 0
    total_requests: int = 0
    latency: LatencyStats = field(default_factory=LatencyStats)


@dataclass
class CorrectnessCase:
    """A single code-correctness test case."""

    name: str
    prompt: str
    generated_code: str = ""
    executed: bool = False
    passed: bool = False
    error: str = ""
    generation_ms: float = 0


@dataclass
class CorrectnessEvidence:
    """Evidence that generated code actually works."""

    cases: list[CorrectnessCase] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0  # couldn't even execute

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0


@dataclass
class ReasoningCase:
    """A single reasoning/quality test case."""

    name: str
    prompt: str
    expected_keywords: list[str]
    response: str = ""
    keywords_found: list[str] = field(default_factory=list)
    keywords_missing: list[str] = field(default_factory=list)
    passed: bool = False
    generation_ms: float = 0


@dataclass
class ReasoningEvidence:
    """Evidence of reasoning quality on known-answer problems."""

    cases: list[ReasoningCase] = field(default_factory=list)
    total: int = 0
    passed: int = 0

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0


@dataclass
class ContextEvidence:
    """Evidence of context window utilization."""

    claimed_ctx_size: int = 0
    tested_sizes: list[int] = field(default_factory=list)
    passed_sizes: list[int] = field(default_factory=list)
    failed_size: int = 0  # first size that failed
    effective_ctx_size: int = 0  # largest that worked


@dataclass
class ConsistencyEvidence:
    """Evidence of output stability across identical prompts."""

    prompt: str = ""
    n_runs: int = 0
    outputs: list[str] = field(default_factory=list)
    unique_outputs: int = 0
    mean_length: float = 0
    length_stdev: float = 0
    all_correct: bool = False


@dataclass
class ResourceEvidence:
    """Resource usage during inference."""

    # Local
    vram_used_mb: float = 0
    ram_used_mb: float = 0
    model_file_size_mb: float = 0
    # API
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    estimated_cost_usd: float = 0
    provider: str = ""


@dataclass
class DeploymentEvidence:
    """Complete evidence report for a deployment."""

    timestamp: str = ""
    provider: str = ""
    model: str = ""
    hardware_summary: str = ""

    throughput: ThroughputEvidence = field(default_factory=ThroughputEvidence)
    correctness: CorrectnessEvidence = field(default_factory=CorrectnessEvidence)
    reasoning: ReasoningEvidence = field(default_factory=ReasoningEvidence)
    context: ContextEvidence = field(default_factory=ContextEvidence)
    consistency: ConsistencyEvidence = field(default_factory=ConsistencyEvidence)
    resources: ResourceEvidence = field(default_factory=ResourceEvidence)

    verdict: str = ""  # "PASS", "DEGRADED", "FAIL"
    verdict_reasons: list[str] = field(default_factory=list)

    def compute_verdict(self) -> None:
        """Determine overall verdict from evidence."""
        reasons: list[str] = []
        severity = "PASS"

        # Throughput
        if self.throughput.generation_tok_per_sec < 5:
            reasons.append(
                f"Generation speed critically slow: {self.throughput.generation_tok_per_sec:.1f} tok/s"
            )
            severity = "FAIL"
        elif self.throughput.generation_tok_per_sec < 15:
            reasons.append(
                f"Generation speed below comfort: {self.throughput.generation_tok_per_sec:.1f} tok/s"
            )
            if severity != "FAIL":
                severity = "DEGRADED"

        # Latency
        if self.throughput.latency.p95_ms > 60_000:
            reasons.append(f"p95 latency > 60s: {self.throughput.latency.p95_ms:.0f}ms")
            severity = "FAIL"
        elif self.throughput.latency.p95_ms > 30_000:
            reasons.append(f"p95 latency > 30s: {self.throughput.latency.p95_ms:.0f}ms")
            if severity != "FAIL":
                severity = "DEGRADED"

        # Correctness
        if self.correctness.total > 0:
            if self.correctness.pass_rate < 50:
                reasons.append(
                    f"Code correctness critically low: {self.correctness.pass_rate:.0f}%"
                )
                severity = "FAIL"
            elif self.correctness.pass_rate < 80:
                reasons.append(
                    f"Code correctness below threshold: {self.correctness.pass_rate:.0f}%"
                )
                if severity != "FAIL":
                    severity = "DEGRADED"

        # Reasoning
        if self.reasoning.total > 0 and self.reasoning.pass_rate < 60:
            reasons.append(
                f"Reasoning quality low: {self.reasoning.pass_rate:.0f}%"
            )
            if severity != "FAIL":
                severity = "DEGRADED"

        # Context
        if self.context.claimed_ctx_size > 0 and self.context.effective_ctx_size > 0:
            ratio = self.context.effective_ctx_size / self.context.claimed_ctx_size
            if ratio < 0.5:
                reasons.append(
                    f"Context utilization poor: {self.context.effective_ctx_size}"
                    f" of {self.context.claimed_ctx_size} claimed"
                )
                if severity != "FAIL":
                    severity = "DEGRADED"

        if not reasons:
            reasons.append("All evidence within acceptable thresholds")

        self.verdict = severity
        self.verdict_reasons = reasons

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())

    def render_report(self) -> str:
        """Human-readable evidence report."""
        lines: list[str] = []
        w = lines.append

        w("=" * 64)
        w("  CONDUCTOR DEPLOYMENT EVIDENCE REPORT")
        w("=" * 64)
        w(f"  Timestamp:  {self.timestamp}")
        w(f"  Provider:   {self.provider}")
        w(f"  Model:      {self.model}")
        w(f"  Hardware:   {self.hardware_summary}")
        w("")

        # Throughput
        w("── Throughput ──────────────────────────────────────")
        t = self.throughput
        w(f"  Prompt processing:    {t.prompt_tok_per_sec:>8.1f} tok/s")
        w(f"  Generation:           {t.generation_tok_per_sec:>8.1f} tok/s")
        w(f"  Total requests:       {t.total_requests:>8d}")
        if t.latency.count > 0:
            w(f"  Latency p50:          {t.latency.median_ms:>8.1f} ms")
            w(f"  Latency p95:          {t.latency.p95_ms:>8.1f} ms")
            w(f"  Latency p99:          {t.latency.p99_ms:>8.1f} ms")
        w("")

        # Correctness
        w("── Code Correctness ────────────────────────────────")
        c = self.correctness
        w(f"  Pass rate:   {c.pass_rate:>5.1f}%  ({c.passed}/{c.total})")
        for case in c.cases:
            icon = "PASS" if case.passed else ("ERR " if case.error else "FAIL")
            w(f"    [{icon}] {case.name} ({case.generation_ms:.0f}ms)")
            if case.error:
                w(f"           {case.error[:80]}")
        w("")

        # Reasoning
        w("── Reasoning Quality ───────────────────────────────")
        r = self.reasoning
        w(f"  Pass rate:   {r.pass_rate:>5.1f}%  ({r.passed}/{r.total})")
        for case in r.cases:
            icon = "PASS" if case.passed else "FAIL"
            w(f"    [{icon}] {case.name}")
            if case.keywords_missing:
                w(f"           Missing: {', '.join(case.keywords_missing)}")
        w("")

        # Context
        w("── Context Window ──────────────────────────────────")
        cx = self.context
        w(f"  Claimed:    {cx.claimed_ctx_size:>8d} tokens")
        w(f"  Effective:  {cx.effective_ctx_size:>8d} tokens")
        if cx.failed_size > 0:
            w(f"  Failed at:  {cx.failed_size:>8d} tokens")
        w("")

        # Consistency
        w("── Consistency ─────────────────────────────────────")
        cs = self.consistency
        if cs.n_runs > 0:
            w(f"  Runs:           {cs.n_runs}")
            w(f"  Unique outputs: {cs.unique_outputs}/{cs.n_runs}")
            w(f"  Mean length:    {cs.mean_length:.0f} chars")
            w(f"  Length stdev:   {cs.length_stdev:.0f} chars")
            w(f"  All correct:    {'Yes' if cs.all_correct else 'No'}")
        w("")

        # Resources
        w("── Resource Usage ──────────────────────────────────")
        rs = self.resources
        if rs.provider == "local":
            w(f"  VRAM used:       {rs.vram_used_mb:>8.0f} MB")
            w(f"  RAM used:        {rs.ram_used_mb:>8.0f} MB")
            w(f"  Model file:      {rs.model_file_size_mb:>8.0f} MB")
        else:
            w(f"  Prompt tokens:   {rs.total_prompt_tokens:>8d}")
            w(f"  Completion tokens:{rs.total_completion_tokens:>7d}")
            w(f"  Estimated cost:  ${rs.estimated_cost_usd:>7.4f}")
        w("")

        # Verdict
        w("=" * 64)
        verdict_color = {
            "PASS": "PASS",
            "DEGRADED": "DEGRADED",
            "FAIL": "FAIL",
        }
        w(f"  VERDICT: {self.verdict}")
        for reason in self.verdict_reasons:
            w(f"    - {reason}")
        w("=" * 64)

        return "\n".join(lines)
