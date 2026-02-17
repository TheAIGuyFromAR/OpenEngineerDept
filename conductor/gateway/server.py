"""
Inference Gateway — FastAPI server.

Sits between the Conductor orchestrator and the llama-server.
Manages slot orchestration, prefix caching, and Ultra Think parallel generation.

Endpoints:
  POST /v1/chat/completions   — OpenAI-compatible proxy
  POST /v1/ultra-think        — Parallel diverse generation
  POST /v1/project/load       — Load project context into template slot
  POST /v1/project/save       — Persist template KV cache to disk
  GET  /v1/slots/status       — Current slot utilization
  GET  /v1/metrics            — Throughput and cache stats
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import GatewayConfig
from .prefix_cache import PrefixCacheManager
from .slot_manager import SlotManager
from .ultra_think import UltraThink, UltraThinkResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatCompletionRequest(BaseModel):
    model: str = "conductor"
    messages: list[dict]
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    stop: list[str] | None = None
    # Gateway extension: pin to a specific slot
    id_slot: int | None = None


class UltraThinkRequest(BaseModel):
    task_id: str
    prompt: str
    system_prompt: str = ""
    tier: int = 2
    max_tokens: int | None = None
    project_id: str | None = None


class ProjectLoadRequest(BaseModel):
    project_id: str
    layer0_text: str


class ProjectSaveRequest(BaseModel):
    project_id: str


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

config = GatewayConfig()
slot_manager: SlotManager
prefix_cache: PrefixCacheManager
ultra_think: UltraThink
llama_client: httpx.AsyncClient
metrics_path: Path


@asynccontextmanager
async def lifespan(app: FastAPI):
    global slot_manager, prefix_cache, ultra_think, llama_client, metrics_path

    slot_manager = SlotManager(config)
    prefix_cache = PrefixCacheManager(config)
    ultra_think = UltraThink(config, slot_manager)
    llama_client = httpx.AsyncClient(
        base_url=config.llama_server_url,
        timeout=config.generation_timeout_seconds,
    )
    metrics_path = Path(config.metrics_log_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Gateway started — llama-server at %s", config.llama_server_url)
    yield

    await ultra_think.close()
    await slot_manager.close()
    await llama_client.aclose()
    logger.info("Gateway shut down")


app = FastAPI(title="Conductor Inference Gateway", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible endpoint with transparent slot management."""
    start = time.monotonic()

    # Acquire a single worker slot if not explicitly pinned
    slot_id = req.id_slot
    acquired = False
    if slot_id is None:
        workers = await slot_manager.acquire_workers(1)
        slot_id = workers[0]
        acquired = True

    try:
        body = {
            "model": req.model,
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "top_k": req.top_k,
            "id_slot": slot_id,
            "cache_prompt": True,
        }
        if req.stop:
            body["stop"] = req.stop

        resp = await llama_client.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()

        elapsed_ms = (time.monotonic() - start) * 1000
        _log_metric("chat_completion", elapsed_ms, data.get("usage", {}))
        return data

    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    finally:
        if acquired:
            slot_manager.release_workers([slot_id])


@app.post("/v1/ultra-think")
async def ultra_think_endpoint(req: UltraThinkRequest) -> dict:
    """Parallel diverse generation for Ultra Think pipeline."""
    if req.tier >= 4:
        raise HTTPException(status_code=400, detail="Tier 4 requires decomposition")

    result = await ultra_think.generate(
        task_id=req.task_id,
        prompt=req.prompt,
        system_prompt=req.system_prompt,
        tier=req.tier,
        max_tokens=req.max_tokens,
        project_id=req.project_id,
    )

    _log_metric(
        "ultra_think",
        result.timing.total_ms,
        {
            "tier": result.tier,
            "candidates": len(result.candidates),
            "errors": len(result.errors),
        },
    )

    return _serialize_ultra_result(result)


@app.post("/v1/project/load")
async def project_load(req: ProjectLoadRequest):
    """Load project context into template slot KV cache."""
    action = await prefix_cache.ensure_loaded(
        project_id=req.project_id,
        layer0_text=req.layer0_text,
        slot_manager=slot_manager,
    )
    return {"project_id": req.project_id, "action": action}


@app.post("/v1/project/save")
async def project_save(req: ProjectSaveRequest):
    """Persist current template slot KV cache to NVMe."""
    metric = await slot_manager.save_template(req.project_id)
    return {"project_id": req.project_id, "duration_ms": metric.duration_ms}


@app.get("/v1/slots/status")
async def slots_status():
    """Current slot utilization."""
    raw = await slot_manager.get_slots_status()
    return {
        "template_slot": config.template_slot_id,
        "worker_slots": config.worker_slot_ids,
        "available_workers": slot_manager.available_worker_count,
        "slots": raw,
    }


@app.get("/v1/metrics")
async def metrics():
    """Aggregated metrics."""
    slot_metrics = slot_manager.get_metrics()
    cache_stats = prefix_cache.get_stats()
    return {
        "slot_operations": len(slot_metrics),
        "cache_hits": sum(1 for s in cache_stats if s.action == "hit"),
        "cache_misses": sum(1 for s in cache_stats if s.action in ("miss", "recompute")),
        "recent_slot_ops": [
            {
                "slot_id": m.slot_id,
                "operation": m.operation,
                "duration_ms": round(m.duration_ms, 1),
                "success": m.success,
            }
            for m in slot_metrics[-20:]
        ],
    }


@app.get("/health")
async def health():
    """Health check — also pings llama-server."""
    try:
        resp = await llama_client.get("/health", timeout=5)
        engine_ok = resp.status_code == 200
    except Exception:
        engine_ok = False

    return {
        "gateway": "ok",
        "engine": "ok" if engine_ok else "unreachable",
        "available_workers": slot_manager.available_worker_count,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_ultra_result(result: UltraThinkResult) -> dict:
    return {
        "task_id": result.task_id,
        "tier": result.tier,
        "candidates": [
            {
                "slot_id": c.slot_id,
                "content": c.content,
                "sampling_params": c.sampling_params,
                "system_prompt_variant": c.system_prompt_variant,
                "tokens_generated": c.tokens_generated,
                "generation_time_ms": round(c.generation_time_ms, 1),
                "tokens_per_second": c.tokens_per_second,
            }
            for c in result.candidates
        ],
        "timing": {
            "slot_restore_ms": round(result.timing.slot_restore_ms, 1),
            "parallel_generation_ms": round(result.timing.parallel_generation_ms, 1),
            "total_ms": round(result.timing.total_ms, 1),
            "prefix_tokens_cached": result.timing.prefix_tokens_cached,
            "suffix_tokens_per_candidate": result.timing.suffix_tokens_per_candidate,
        },
        "errors": result.errors,
    }


def _log_metric(endpoint: str, duration_ms: float, extra: dict) -> None:
    try:
        entry = {
            "endpoint": endpoint,
            "duration_ms": round(duration_ms, 1),
            "timestamp": time.time(),
            **extra,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass
