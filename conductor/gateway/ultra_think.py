"""
Ultra Think — Parallel generation orchestrator.

Generates N diverse completions for the same task using varied sampling
parameters and system prompt suffixes, then returns all candidates for
the Reviewer to evaluate.

Works with both local (llama-server, slot-pinned) and API providers
(concurrent HTTP requests, no slot management).

Tier defaults:
  Tier 1: N=1 (single shot)
  Tier 2: N=3 (parallel diverse)
  Tier 3: N=5 (full ensemble, may queue if only 4 workers)
  Tier 4: decompose or escalate (not handled here)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .config import GatewayConfig
from .providers import InferenceProvider
from .slot_manager import SlotManager

logger = logging.getLogger(__name__)


# Sampling diversity profiles
DIVERSITY_PROFILES = [
    {
        "label": "conservative",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 30,
        "suffix": "Prioritize readability and maintainability.",
    },
    {
        "label": "standard",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "suffix": "Balance clarity with efficiency.",
    },
    {
        "label": "exploratory",
        "temperature": 1.2,
        "top_p": 0.98,
        "top_k": 50,
        "suffix": "Consider unconventional approaches.",
    },
    {
        "label": "creative",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "presence_penalty": 0.3,
        "suffix": "Emphasize simplicity and minimal code.",
    },
    {
        "label": "focused",
        "temperature": 0.8,
        "top_p": 0.85,
        "top_k": 20,
        "suffix": "Focus on robustness and edge cases.",
    },
]

TIER_N = {1: 1, 2: 3, 3: 5}


@dataclass
class CandidateCompletion:
    slot_id: int
    content: str
    sampling_params: dict
    system_prompt_variant: str
    tokens_generated: int
    generation_time_ms: float
    tokens_per_second: float


@dataclass
class UltraThinkTiming:
    slot_restore_ms: float
    parallel_generation_ms: float
    total_ms: float
    prefix_tokens_cached: int
    suffix_tokens_per_candidate: list[int]


@dataclass
class UltraThinkResult:
    task_id: str
    tier: int
    candidates: list[CandidateCompletion]
    timing: UltraThinkTiming
    errors: list[str] = field(default_factory=list)


class UltraThink:
    def __init__(
        self,
        config: GatewayConfig,
        slot_manager: SlotManager | None,
        provider: InferenceProvider,
    ) -> None:
        self._config = config
        self._slots = slot_manager
        self._provider = provider

    async def generate(
        self,
        *,
        task_id: str,
        prompt: str,
        system_prompt: str,
        tier: int,
        max_tokens: int | None = None,
        project_id: str | None = None,
    ) -> UltraThinkResult:
        """Run the full Ultra Think pipeline for a task."""
        total_start = time.monotonic()
        max_tok = max_tokens or self._config.default_max_tokens

        if tier >= 4:
            raise ValueError("Tier 4 requires decomposition — not handled by Ultra Think")

        n = TIER_N.get(tier, 3)
        profiles = DIVERSITY_PROFILES[:n]

        if self._provider.supports_slots and self._slots is not None:
            return await self._generate_local(
                task_id=task_id,
                prompt=prompt,
                system_prompt=system_prompt,
                profiles=profiles,
                max_tokens=max_tok,
                project_id=project_id,
                total_start=total_start,
            )
        else:
            return await self._generate_api(
                task_id=task_id,
                prompt=prompt,
                system_prompt=system_prompt,
                profiles=profiles,
                max_tokens=max_tok,
                total_start=total_start,
            )

    async def _generate_local(
        self,
        *,
        task_id: str,
        prompt: str,
        system_prompt: str,
        profiles: list[dict],
        max_tokens: int,
        project_id: str | None,
        total_start: float,
    ) -> UltraThinkResult:
        """Local llama-server path: slot pinning + KV cache restore."""
        n = len(profiles)
        workers = await self._slots.acquire_workers(n)
        try:
            # Restore template cache into all workers (parallel)
            restore_start = time.monotonic()
            if project_id:
                await self._slots.restore_workers_parallel(project_id, workers)
            restore_ms = (time.monotonic() - restore_start) * 1000

            # Fire all generations concurrently
            gen_start = time.monotonic()
            tasks = [
                self._generate_one(
                    slot_id=workers[i],
                    prompt=prompt,
                    system_prompt=system_prompt,
                    profile=profiles[i],
                    max_tokens=max_tokens,
                )
                for i in range(n)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            gen_ms = (time.monotonic() - gen_start) * 1000

            return self._collect_results(
                task_id=task_id,
                tier=len(profiles),
                results=results,
                workers=workers,
                restore_ms=restore_ms,
                gen_ms=gen_ms,
                total_start=total_start,
            )
        finally:
            self._slots.release_workers(workers)

    async def _generate_api(
        self,
        *,
        task_id: str,
        prompt: str,
        system_prompt: str,
        profiles: list[dict],
        max_tokens: int,
        total_start: float,
    ) -> UltraThinkResult:
        """API provider path: concurrent requests, no slot management."""
        n = len(profiles)

        gen_start = time.monotonic()
        tasks = [
            self._generate_one(
                slot_id=-1,  # no slot for API providers
                prompt=prompt,
                system_prompt=system_prompt,
                profile=profiles[i],
                max_tokens=max_tokens,
            )
            for i in range(n)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        gen_ms = (time.monotonic() - gen_start) * 1000

        return self._collect_results(
            task_id=task_id,
            tier=n,
            results=results,
            workers=[-1] * n,
            restore_ms=0.0,
            gen_ms=gen_ms,
            total_start=total_start,
        )

    def _collect_results(
        self,
        *,
        task_id: str,
        tier: int,
        results: list,
        workers: list[int],
        restore_ms: float,
        gen_ms: float,
        total_start: float,
    ) -> UltraThinkResult:
        candidates: list[CandidateCompletion] = []
        errors: list[str] = []
        suffix_tokens: list[int] = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Gen {i} (slot {workers[i]}): {result}")
                suffix_tokens.append(0)
            else:
                candidates.append(result)
                suffix_tokens.append(result.tokens_generated)

        total_ms = (time.monotonic() - total_start) * 1000

        return UltraThinkResult(
            task_id=task_id,
            tier=tier,
            candidates=candidates,
            timing=UltraThinkTiming(
                slot_restore_ms=restore_ms,
                parallel_generation_ms=gen_ms,
                total_ms=total_ms,
                prefix_tokens_cached=0,
                suffix_tokens_per_candidate=suffix_tokens,
            ),
            errors=errors,
        )

    async def _generate_one(
        self,
        *,
        slot_id: int,
        prompt: str,
        system_prompt: str,
        profile: dict,
        max_tokens: int,
    ) -> CandidateCompletion:
        """Send a single generation request via the provider."""
        start = time.monotonic()

        full_system = f"{system_prompt}\n\n{profile['suffix']}"

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]

        extra: dict = {}
        if self._provider.supports_slots and slot_id >= 0:
            extra["id_slot"] = slot_id
            extra["cache_prompt"] = True

        result = await self._provider.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=profile["temperature"],
            top_p=profile["top_p"],
            top_k=profile.get("top_k", 40),
            extra=extra if extra else None,
        )

        elapsed_ms = (time.monotonic() - start) * 1000
        completion_tokens = result.usage.get("completion_tokens", 0)
        tok_per_sec = (completion_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

        return CandidateCompletion(
            slot_id=slot_id,
            content=result.content,
            sampling_params={
                k: v
                for k, v in profile.items()
                if k not in ("label", "suffix")
            },
            system_prompt_variant=profile["suffix"],
            tokens_generated=completion_tokens,
            generation_time_ms=elapsed_ms,
            tokens_per_second=round(tok_per_sec, 1),
        )

    async def close(self) -> None:
        # Provider lifecycle is managed by the gateway server, not by us
        pass
