"""Evidence runner — orchestrates all collectors and produces the deployment report.

Usage:
    python3 -m tests.evidence.runner --gateway http://localhost:9090 --provider local
    python3 -m tests.evidence.runner --gateway http://localhost:9090 --provider anthropic

Output:
    - Human-readable report to stdout
    - JSON evidence file to data/evidence/deployment-<timestamp>.json
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timezone

import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.evidence.collectors import (
    collect_consistency,
    collect_context,
    collect_correctness,
    collect_reasoning,
    collect_resources,
    collect_throughput,
)
from tests.evidence.models import DeploymentEvidence


async def run_evidence_collection(
    gateway_url: str,
    provider: str,
    model: str = "",
    hardware_summary: str = "",
    claimed_ctx: int = 32768,
    model_path: str = "",
    output_dir: str = "./data/evidence",
) -> DeploymentEvidence:
    """Run all evidence collectors and produce a report."""

    evidence = DeploymentEvidence(
        timestamp=datetime.now(timezone.utc).isoformat(),
        provider=provider,
        model=model,
        hardware_summary=hardware_summary,
    )

    client = httpx.AsyncClient(
        base_url=gateway_url,
        timeout=120,
    )

    total_prompt_tokens = 0
    total_completion_tokens = 0

    try:
        # ── 1. Throughput + Latency ───────────────────────────────
        print("\n[evidence] Collecting throughput + latency (5 requests)...")
        evidence.throughput = await collect_throughput(client, n_requests=5)
        print(
            f"           Generation: {evidence.throughput.generation_tok_per_sec:.1f} tok/s, "
            f"p50={evidence.throughput.latency.median_ms:.0f}ms, "
            f"p95={evidence.throughput.latency.p95_ms:.0f}ms"
        )

        # ── 2. Code Correctness ───────────────────────────────────
        print("\n[evidence] Collecting code correctness (6 problems)...")
        evidence.correctness = await collect_correctness(client)
        print(
            f"           Pass rate: {evidence.correctness.pass_rate:.0f}% "
            f"({evidence.correctness.passed}/{evidence.correctness.total})"
        )
        for c in evidence.correctness.cases:
            icon = "PASS" if c.passed else ("ERR " if c.error else "FAIL")
            print(f"             [{icon}] {c.name} ({c.generation_ms:.0f}ms)")

        # ── 3. Reasoning Quality ──────────────────────────────────
        print("\n[evidence] Collecting reasoning quality (5 problems)...")
        evidence.reasoning = await collect_reasoning(client)
        print(
            f"           Pass rate: {evidence.reasoning.pass_rate:.0f}% "
            f"({evidence.reasoning.passed}/{evidence.reasoning.total})"
        )
        for r in evidence.reasoning.cases:
            icon = "PASS" if r.passed else "FAIL"
            print(f"             [{icon}] {r.name}")

        # ── 4. Context Window ─────────────────────────────────────
        print(f"\n[evidence] Stress-testing context window (claimed: {claimed_ctx})...")
        evidence.context = await collect_context(client, claimed_ctx=claimed_ctx)
        print(f"           Effective: {evidence.context.effective_ctx_size} tokens")
        if evidence.context.failed_size:
            print(f"           Failed at: {evidence.context.failed_size} tokens")

        # ── 5. Consistency ────────────────────────────────────────
        print("\n[evidence] Collecting consistency (5 identical runs)...")
        evidence.consistency = await collect_consistency(client, n_runs=5)
        print(
            f"           Unique outputs: {evidence.consistency.unique_outputs}/5, "
            f"all correct: {evidence.consistency.all_correct}"
        )

        # ── 6. Resource Usage ─────────────────────────────────────
        print("\n[evidence] Collecting resource usage...")
        evidence.resources = await collect_resources(
            client,
            provider=provider,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            model_path=model_path,
        )

    finally:
        await client.aclose()

    # ── Compute verdict ───────────────────────────────────────────
    evidence.compute_verdict()

    # ── Save report ───────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = f"{output_dir}/deployment-{ts}.json"
    evidence.save(json_path)

    # Print report
    report = evidence.render_report()
    print("\n")
    print(report)
    print(f"\n[evidence] JSON saved to: {json_path}")

    return evidence


def main():
    parser = argparse.ArgumentParser(description="Conductor deployment evidence collection")
    parser.add_argument("--gateway", default="http://localhost:9090", help="Gateway URL")
    parser.add_argument("--provider", default="local", help="Inference provider")
    parser.add_argument("--model", default="", help="Model name")
    parser.add_argument("--hardware", default="", help="Hardware summary string")
    parser.add_argument("--ctx-size", type=int, default=32768, help="Claimed context size")
    parser.add_argument("--model-path", default="", help="Path to model file (local only)")
    parser.add_argument("--output-dir", default="./data/evidence", help="Output directory")
    args = parser.parse_args()

    evidence = asyncio.run(
        run_evidence_collection(
            gateway_url=args.gateway,
            provider=args.provider,
            model=args.model,
            hardware_summary=args.hardware,
            claimed_ctx=args.ctx_size,
            model_path=args.model_path,
            output_dir=args.output_dir,
        )
    )

    # Exit code based on verdict
    if evidence.verdict == "FAIL":
        sys.exit(1)
    elif evidence.verdict == "DEGRADED":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
