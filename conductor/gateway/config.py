"""Gateway configuration via environment variables or defaults."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class GatewayConfig(BaseSettings):
    # Inference provider: "local", "anthropic", "openai", "openrouter"
    inference_provider: str = "local"
    inference_api_key: str = ""
    inference_api_base: str = ""
    inference_model: str = ""
    inference_max_tokens: int = 4096

    # Local llama-server settings (only used when inference_provider="local")
    llama_server_url: str = "http://localhost:8080"
    template_slot_id: int = 0
    worker_slot_ids: list[int] = [1, 2, 3, 4]
    kv_cache_dir: str = "./data/kv-cache"
    slot_restore_timeout_seconds: int = 30

    # Shared settings
    tier2_candidates: int = 3
    tier3_candidates: int = 5
    default_max_tokens: int = 4096
    generation_timeout_seconds: int = 300
    metrics_log_path: str = "./data/metrics/gateway.jsonl"

    @property
    def is_local(self) -> bool:
        return self.inference_provider == "local"

    @property
    def resolved_api_base(self) -> str:
        """Return the API base URL for the configured provider."""
        if self.inference_api_base:
            return self.inference_api_base
        defaults = {
            "anthropic": "https://api.anthropic.com",
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }
        return defaults.get(self.inference_provider, "")

    @property
    def resolved_model(self) -> str:
        """Return the model name, with sane defaults per provider."""
        if self.inference_model:
            return self.inference_model
        defaults = {
            "anthropic": "claude-sonnet-4-5-20250929",
            "openai": "gpt-4o",
            "openrouter": "qwen/qwen3-coder",
            "local": "conductor",
        }
        return defaults.get(self.inference_provider, "conductor")

    model_config = {"env_prefix": "CONDUCTOR_"}
