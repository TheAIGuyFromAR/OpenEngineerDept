"""
Langfuse tracing integration for the Conductor Gateway.

Wraps inference calls with Langfuse traces for observability:
- Every /v1/chat/completions call → Langfuse generation span
- Every /v1/ultra-think call → Langfuse trace with N generation spans
- Prompt cache hits/misses → Langfuse events
- Slot lifecycle → Langfuse spans

The tracer is optional — if Langfuse is unreachable or not configured,
all methods are no-ops. Inference never blocks on telemetry.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Langfuse client — lazy-initialized, None if unavailable
_langfuse = None
_initialized = False


def _get_langfuse():
    """Lazy-init the Langfuse client. Returns None if not configured."""
    global _langfuse, _initialized
    if _initialized:
        return _langfuse
    _initialized = True

    try:
        from langfuse import Langfuse

        client = Langfuse()
        # Quick connectivity check
        client.auth_check()
        _langfuse = client
        logger.info("Langfuse tracing enabled: %s", client.base_url)
    except ImportError:
        logger.info("Langfuse SDK not installed — tracing disabled")
    except Exception as exc:
        logger.warning("Langfuse not reachable — tracing disabled: %s", exc)

    return _langfuse


class LangfuseTracer:
    """Gateway-level Langfuse tracer. Safe to call even when Langfuse is down."""

    def trace_generation(
        self,
        *,
        trace_id: str,
        name: str,
        model: str = "qwen3-coder-next",
        prompt: list[dict] | None = None,
        completion: str = "",
        usage: dict | None = None,
        metadata: dict | None = None,
        slot_id: int | None = None,
        tier: int | None = None,
        candidate_idx: int | None = None,
        latency_ms: float | None = None,
        cache_status: str | None = None,
    ) -> None:
        """Record a single LLM generation call."""
        lf = _get_langfuse()
        if lf is None:
            return

        try:
            trace = lf.trace(
                id=trace_id,
                name=name,
                metadata={
                    **(metadata or {}),
                    "slot_id": slot_id,
                    "tier": tier,
                    "cache_status": cache_status,
                },
            )

            input_messages = prompt or []
            prompt_tokens = usage.get("prompt_tokens", 0) if usage else 0
            completion_tokens = usage.get("completion_tokens", 0) if usage else 0

            trace.generation(
                name=f"generation{'_' + str(candidate_idx) if candidate_idx is not None else ''}",
                model=model,
                input=input_messages,
                output=completion,
                usage={
                    "input": prompt_tokens,
                    "output": completion_tokens,
                    "total": prompt_tokens + completion_tokens,
                },
                metadata={
                    "slot_id": slot_id,
                    "candidate_idx": candidate_idx,
                    "latency_ms": latency_ms,
                },
            )
        except Exception as exc:
            logger.debug("Langfuse trace_generation failed: %s", exc)

    def trace_ultra_think(
        self,
        *,
        task_id: str,
        tier: int,
        candidates: list[dict],
        errors: list[str],
        total_ms: float,
        metadata: dict | None = None,
    ) -> None:
        """Record an Ultra Think multi-candidate generation."""
        lf = _get_langfuse()
        if lf is None:
            return

        try:
            trace = lf.trace(
                name=f"ultra-think-t{tier}",
                metadata={
                    **(metadata or {}),
                    "task_id": task_id,
                    "tier": tier,
                    "candidate_count": len(candidates),
                    "error_count": len(errors),
                    "total_ms": total_ms,
                },
            )

            for i, candidate in enumerate(candidates):
                trace.generation(
                    name=f"candidate_{i}",
                    model="qwen3-coder-next",
                    output=candidate.get("content", ""),
                    usage={
                        "input": candidate.get("prompt_tokens", 0),
                        "output": candidate.get("completion_tokens", 0),
                    },
                    metadata={
                        "slot_id": candidate.get("slot_id"),
                        "profile": candidate.get("profile_name"),
                    },
                )

            for error in errors:
                trace.event(name="generation_error", metadata={"error": error})

        except Exception as exc:
            logger.debug("Langfuse trace_ultra_think failed: %s", exc)

    def trace_cache_event(
        self,
        *,
        project_id: str,
        action: str,
        content_hash: str,
        latency_ms: float | None = None,
    ) -> None:
        """Record a prefix cache hit/miss/recompute event."""
        lf = _get_langfuse()
        if lf is None:
            return

        try:
            trace = lf.trace(
                name=f"cache-{action}",
                metadata={
                    "project_id": project_id,
                    "action": action,
                    "content_hash": content_hash,
                    "latency_ms": latency_ms,
                },
            )
        except Exception as exc:
            logger.debug("Langfuse trace_cache_event failed: %s", exc)

    def trace_review(
        self,
        *,
        task_id: str,
        scores: dict,
        selected_idx: int,
        accepted: bool,
        metadata: dict | None = None,
    ) -> None:
        """Record a reviewer scoring event."""
        lf = _get_langfuse()
        if lf is None:
            return

        try:
            trace = lf.trace(
                name="review",
                metadata={
                    **(metadata or {}),
                    "task_id": task_id,
                    "selected_idx": selected_idx,
                    "accepted": accepted,
                },
            )

            # Record each score dimension as a Langfuse score
            for dimension, value in scores.items():
                lf.score(
                    trace_id=trace.id,
                    name=dimension,
                    value=value,
                )

        except Exception as exc:
            logger.debug("Langfuse trace_review failed: %s", exc)

    def flush(self) -> None:
        """Flush pending events. Call before shutdown."""
        lf = _get_langfuse()
        if lf is not None:
            try:
                lf.flush()
            except Exception as exc:
                logger.debug("Langfuse flush failed: %s", exc)


# Module-level singleton
tracer = LangfuseTracer()
