"""OpenAI-compatible API provider — works for OpenAI, OpenRouter, and any
provider that implements the /v1/chat/completions endpoint."""

from __future__ import annotations

import httpx

from ..config import GatewayConfig
from .base import CompletionResult, InferenceProvider


class OpenAICompatProvider(InferenceProvider):
    """Sends requests to any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._base_url = config.resolved_api_base.rstrip("/")
        self._model = config.resolved_model
        self._client = httpx.AsyncClient(
            timeout=config.generation_timeout_seconds,
            headers={
                "Authorization": f"Bearer {config.inference_api_key}",
                "Content-Type": "application/json",
            },
        )

    async def chat_completion(
        self,
        *,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: list[str] | None = None,
        extra: dict | None = None,
    ) -> CompletionResult:
        body: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            body["stop"] = stop
        # top_k is not standard OpenAI but some providers support it
        # Don't pass slot pinning or llama-server-specific params

        resp = await self._client.post(
            f"{self._base_url}/chat/completions", json=body
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return CompletionResult(
            content=content,
            model=data.get("model", self._model),
            usage=data.get("usage", {}),
            raw_response=data,
        )

    async def health_check(self) -> bool:
        # API providers are "healthy" if we have a key configured
        return bool(self._config.inference_api_key)

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def supports_slots(self) -> bool:
        return False

    @property
    def provider_name(self) -> str:
        return f"openai-compat ({self._config.inference_provider})"
