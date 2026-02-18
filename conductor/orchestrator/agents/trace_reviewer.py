"""
Trace Reviewer Agent — Cron-able analysis of Langfuse traces and conversations.

Periodically reviews:
  1. Langfuse traces — generation quality, latency patterns, failure modes
  2. Conversation storage — task completion rates, retry patterns, escalations
  3. Training data — prompt effectiveness, score distributions

Produces structured improvement recommendations written to:
  - {vault}/conductor/reviews/  (for human review in Obsidian)
  - Langfuse annotations        (for machine-readable feedback loops)

Usage:
  python -m orchestrator.agents.trace_reviewer --config projects/example/conductor.yaml
  python -m orchestrator.agents.trace_reviewer --since 24h --output reviews/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ReviewFinding:
    category: str        # "prompt", "latency", "quality", "failure", "pattern"
    severity: str        # "info", "warning", "action"
    title: str
    description: str
    evidence: list[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class TraceReviewReport:
    period_start: str
    period_end: str
    traces_analyzed: int
    conversations_analyzed: int
    findings: list[ReviewFinding]
    summary: str


class TraceReviewer:
    """Analyzes Langfuse traces and conversation logs for improvement opportunities."""

    def __init__(
        self,
        training_data_dir: str = "./data/training",
        metrics_dir: str = "./data/metrics",
        output_dir: str = "./data/reviews",
    ) -> None:
        self._training_dir = Path(training_data_dir)
        self._metrics_dir = Path(metrics_dir)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._langfuse = None

    def _get_langfuse(self):
        """Lazy-init Langfuse client for reading traces."""
        if self._langfuse is not None:
            return self._langfuse
        try:
            from langfuse import Langfuse

            client = Langfuse()
            client.auth_check()
            self._langfuse = client
            return client
        except Exception as exc:
            logger.info("Langfuse not available for trace review: %s", exc)
            return None

    async def review(
        self,
        since: timedelta = timedelta(hours=24),
    ) -> TraceReviewReport:
        """Run a full review cycle."""
        now = datetime.now(timezone.utc)
        period_start = now - since
        findings: list[ReviewFinding] = []

        # Phase 1: Review Langfuse traces
        trace_count = 0
        lf = self._get_langfuse()
        if lf is not None:
            trace_findings, trace_count = self._review_langfuse_traces(
                lf, period_start, now
            )
            findings.extend(trace_findings)

        # Phase 2: Review training data / conversation logs
        conv_count = 0
        conv_findings, conv_count = self._review_training_data(period_start, now)
        findings.extend(conv_findings)

        # Phase 3: Review metrics
        metric_findings = self._review_metrics(period_start, now)
        findings.extend(metric_findings)

        # Generate summary
        action_count = sum(1 for f in findings if f.severity == "action")
        warning_count = sum(1 for f in findings if f.severity == "warning")
        summary = (
            f"Reviewed {trace_count} traces and {conv_count} conversations. "
            f"Found {action_count} action items and {warning_count} warnings."
        )

        report = TraceReviewReport(
            period_start=period_start.isoformat(),
            period_end=now.isoformat(),
            traces_analyzed=trace_count,
            conversations_analyzed=conv_count,
            findings=findings,
            summary=summary,
        )

        # Write report
        self._write_report(report)

        return report

    def _review_langfuse_traces(
        self, lf, start: datetime, end: datetime
    ) -> tuple[list[ReviewFinding], int]:
        """Analyze Langfuse traces for patterns."""
        findings: list[ReviewFinding] = []
        trace_count = 0

        try:
            # Fetch recent traces
            traces = lf.client.trace.list(
                page=1,
                limit=100,
            )

            if not traces.data:
                return findings, 0

            # Analyze generation latencies
            latencies = []
            failures = []
            scores_by_dim = {"correctness": [], "quality": [], "safety": [], "completeness": []}

            for trace in traces.data:
                trace_count += 1

                # Check for errors
                if trace.metadata and trace.metadata.get("error_count", 0) > 0:
                    failures.append(trace)

                # Collect scores
                try:
                    trace_scores = lf.client.score.get_by_trace(trace.id)
                    for score in trace_scores:
                        if score.name in scores_by_dim:
                            scores_by_dim[score.name].append(score.value)
                except Exception:
                    pass

            # Finding: high failure rate
            if trace_count > 0 and len(failures) / trace_count > 0.2:
                findings.append(ReviewFinding(
                    category="failure",
                    severity="action",
                    title="High generation failure rate",
                    description=f"{len(failures)}/{trace_count} traces had errors ({len(failures)/trace_count:.0%})",
                    evidence=[f"trace:{t.id}" for t in failures[:5]],
                    recommendation="Review failing traces in Langfuse UI. Common causes: timeout, OOM, slot contention.",
                ))

            # Finding: low scores in specific dimensions
            for dim, values in scores_by_dim.items():
                if values:
                    avg = sum(values) / len(values)
                    if avg < 6.0:
                        findings.append(ReviewFinding(
                            category="quality",
                            severity="warning",
                            title=f"Low average {dim} score: {avg:.1f}",
                            description=f"Average {dim} score across {len(values)} reviews is {avg:.1f}/10",
                            evidence=[f"min={min(values):.1f}", f"max={max(values):.1f}"],
                            recommendation=f"Review the {dim} dimension in reviewer prompts. Consider adjusting prompt template 'reviewer.score'.",
                        ))

        except Exception as exc:
            logger.warning("Langfuse trace review failed: %s", exc)
            findings.append(ReviewFinding(
                category="failure",
                severity="info",
                title="Could not fetch Langfuse traces",
                description=str(exc),
            ))

        return findings, trace_count

    def _review_training_data(
        self, start: datetime, end: datetime
    ) -> tuple[list[ReviewFinding], int]:
        """Analyze training data JSONL for patterns."""
        findings: list[ReviewFinding] = []
        conv_count = 0

        if not self._training_dir.exists():
            return findings, 0

        total_tasks = 0
        accepted_tasks = 0
        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        retry_counts = []
        escalations = 0

        for jsonl_file in self._training_dir.glob("**/*.jsonl"):
            try:
                for line in jsonl_file.read_text(encoding="utf-8").strip().splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Filter by time if timestamp present
                    ts = row.get("timestamp", "")
                    if ts:
                        try:
                            row_time = datetime.fromisoformat(ts)
                            if row_time < start:
                                continue
                        except (ValueError, TypeError):
                            pass

                    total_tasks += 1
                    conv_count += 1

                    # Track acceptance
                    if row.get("accepted", False):
                        accepted_tasks += 1

                    # Track tiers
                    tier = row.get("tier", 2)
                    if tier in tier_counts:
                        tier_counts[tier] += 1

                    # Track retries
                    retries = row.get("retry_count", 0)
                    retry_counts.append(retries)

                    # Track escalations
                    if row.get("escalated", False):
                        escalations += 1

            except Exception as exc:
                logger.debug("Error reading %s: %s", jsonl_file, exc)

        if total_tasks == 0:
            return findings, 0

        # Finding: acceptance rate
        accept_rate = accepted_tasks / total_tasks
        if accept_rate < 0.3:
            findings.append(ReviewFinding(
                category="quality",
                severity="action",
                title=f"Low first-attempt acceptance rate: {accept_rate:.0%}",
                description=f"Only {accepted_tasks}/{total_tasks} tasks accepted on first attempt",
                recommendation="Review prompts for planner.decompose and coder.generate. Consider adding more examples to exemplar library.",
            ))
        elif accept_rate >= 0.5:
            findings.append(ReviewFinding(
                category="quality",
                severity="info",
                title=f"Good acceptance rate: {accept_rate:.0%}",
                description=f"{accepted_tasks}/{total_tasks} tasks accepted",
            ))

        # Finding: high retry rate
        if retry_counts:
            avg_retries = sum(retry_counts) / len(retry_counts)
            if avg_retries > 1.5:
                findings.append(ReviewFinding(
                    category="pattern",
                    severity="warning",
                    title=f"High average retry count: {avg_retries:.1f}",
                    description=f"Tasks require an average of {avg_retries:.1f} retries before acceptance or escalation",
                    recommendation="Analyze retry patterns — are retries fixing the same type of issue? Consider adding those patterns to the coder prompt.",
                ))

        # Finding: tier distribution
        if tier_counts[4] > 0:
            findings.append(ReviewFinding(
                category="pattern",
                severity="info",
                title=f"Tier 4 escalations: {tier_counts[4]}",
                description=f"Tier distribution: T1={tier_counts[1]}, T2={tier_counts[2]}, T3={tier_counts[3]}, T4={tier_counts[4]}",
                recommendation="Review Tier 4 tasks — can any be decomposed into Tier 2/3 with better planner prompts?",
            ))

        # Finding: escalation rate
        if escalations > 0:
            esc_rate = escalations / total_tasks
            findings.append(ReviewFinding(
                category="failure",
                severity="warning" if esc_rate < 0.1 else "action",
                title=f"Escalation rate: {esc_rate:.0%} ({escalations} tasks)",
                description=f"{escalations} out of {total_tasks} tasks required human escalation",
                recommendation="Review escalated tasks to identify common blockers. Update constraints.md with discovered patterns.",
            ))

        return findings, conv_count

    def _review_metrics(self, start: datetime, end: datetime) -> list[ReviewFinding]:
        """Analyze gateway metrics for performance patterns."""
        findings: list[ReviewFinding] = []

        metrics_file = self._metrics_dir / "gateway.jsonl"
        if not metrics_file.exists():
            return findings

        latencies = []
        cache_hits = 0
        cache_misses = 0
        tok_per_sec_values = []

        try:
            for line in metrics_file.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "generation_time_ms" in row:
                    latencies.append(row["generation_time_ms"])
                if "tokens_per_second" in row:
                    tok_per_sec_values.append(row["tokens_per_second"])
                if "cache_action" in row:
                    if row["cache_action"] == "hit":
                        cache_hits += 1
                    else:
                        cache_misses += 1

        except Exception as exc:
            logger.debug("Error reading metrics: %s", exc)
            return findings

        # Finding: cache hit rate
        total_cache = cache_hits + cache_misses
        if total_cache > 0:
            hit_rate = cache_hits / total_cache
            if hit_rate < 0.5:
                findings.append(ReviewFinding(
                    category="latency",
                    severity="warning",
                    title=f"Low cache hit rate: {hit_rate:.0%}",
                    description=f"{cache_hits}/{total_cache} cache hits",
                    recommendation="Check if Layer 0 constraints are changing frequently. Stable constraints improve cache reuse.",
                ))

        # Finding: throughput
        if tok_per_sec_values:
            avg_tps = sum(tok_per_sec_values) / len(tok_per_sec_values)
            if avg_tps < 10:
                findings.append(ReviewFinding(
                    category="latency",
                    severity="warning",
                    title=f"Low throughput: {avg_tps:.1f} tok/s average",
                    description=f"Average generation speed across {len(tok_per_sec_values)} calls",
                    recommendation="Consider a smaller quantization or reducing context size. Run detect-hardware.sh to re-evaluate.",
                ))

        return findings

    def _write_report(self, report: TraceReviewReport) -> None:
        """Write report as both JSON and Markdown."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        # JSON (machine-readable)
        json_path = self._output_dir / f"review-{timestamp}.json"
        json_path.write_text(
            json.dumps(
                {
                    "period_start": report.period_start,
                    "period_end": report.period_end,
                    "traces_analyzed": report.traces_analyzed,
                    "conversations_analyzed": report.conversations_analyzed,
                    "summary": report.summary,
                    "findings": [
                        {
                            "category": f.category,
                            "severity": f.severity,
                            "title": f.title,
                            "description": f.description,
                            "evidence": f.evidence,
                            "recommendation": f.recommendation,
                        }
                        for f in report.findings
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Markdown (human-readable, for Obsidian)
        md_path = self._output_dir / f"review-{timestamp}.md"
        lines = [
            f"# Conductor Review — {timestamp}",
            "",
            f"**Period:** {report.period_start} → {report.period_end}",
            f"**Traces analyzed:** {report.traces_analyzed}",
            f"**Conversations analyzed:** {report.conversations_analyzed}",
            "",
            f"## Summary",
            "",
            report.summary,
            "",
        ]

        # Group findings by severity
        for severity in ["action", "warning", "info"]:
            severity_findings = [f for f in report.findings if f.severity == severity]
            if not severity_findings:
                continue

            icon = {"action": "!!!", "warning": "!!", "info": "i"}[severity]
            lines.append(f"## {severity.title()} Items [{icon}]")
            lines.append("")

            for finding in severity_findings:
                lines.append(f"### [{finding.category}] {finding.title}")
                lines.append("")
                lines.append(finding.description)
                if finding.evidence:
                    lines.append("")
                    lines.append("**Evidence:**")
                    for e in finding.evidence:
                        lines.append(f"- `{e}`")
                if finding.recommendation:
                    lines.append("")
                    lines.append(f"**Recommendation:** {finding.recommendation}")
                lines.append("")

        md_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Review report written: %s", md_path)


async def main():
    parser = argparse.ArgumentParser(description="Conductor Trace Reviewer")
    parser.add_argument(
        "--since",
        default="24h",
        help="Review period (e.g. '24h', '7d', '1h')",
    )
    parser.add_argument(
        "--training-dir",
        default="./data/training",
        help="Training data directory",
    )
    parser.add_argument(
        "--metrics-dir",
        default="./data/metrics",
        help="Metrics directory",
    )
    parser.add_argument(
        "--output",
        default="./data/reviews",
        help="Output directory for review reports",
    )
    args = parser.parse_args()

    # Parse duration
    duration_str = args.since
    if duration_str.endswith("h"):
        since = timedelta(hours=int(duration_str[:-1]))
    elif duration_str.endswith("d"):
        since = timedelta(days=int(duration_str[:-1]))
    elif duration_str.endswith("m"):
        since = timedelta(minutes=int(duration_str[:-1]))
    else:
        since = timedelta(hours=24)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    reviewer = TraceReviewer(
        training_data_dir=args.training_dir,
        metrics_dir=args.metrics_dir,
        output_dir=args.output,
    )

    report = await reviewer.review(since=since)
    print(f"\n{report.summary}")
    print(f"  Findings: {len(report.findings)}")

    actions = [f for f in report.findings if f.severity == "action"]
    if actions:
        print(f"\n  Action items:")
        for a in actions:
            print(f"    - [{a.category}] {a.title}")

    print(f"\n  Report: {args.output}/")


if __name__ == "__main__":
    asyncio.run(main())
