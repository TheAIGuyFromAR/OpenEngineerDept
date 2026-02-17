"""Gateway configuration via environment variables or defaults."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class GatewayConfig(BaseSettings):
    llama_server_url: str = "http://localhost:8080"
    template_slot_id: int = 0
    worker_slot_ids: list[int] = [1, 2, 3, 4]
    kv_cache_dir: str = "./data/kv-cache"
    tier2_candidates: int = 3
    tier3_candidates: int = 5
    default_max_tokens: int = 4096
    generation_timeout_seconds: int = 300
    slot_restore_timeout_seconds: int = 30
    metrics_log_path: str = "./data/metrics/gateway.jsonl"

    model_config = {"env_prefix": "CONDUCTOR_"}
