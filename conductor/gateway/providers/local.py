"""Local llama-server inference provider."""

from __future__ import annotations

import httpx

from ..config import GatewayConfig
from .base import CompletionResult, InferenceProvider


class LocalProvider(InferenceProvider):
    """Talks to a local llama-server instance via its OpenAI-compatible API."""

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.llama_server_url,
            timeout=config.generation_timeout_seconds,
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
            "model": "conductor",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "cache_prompt": True,
        }
        if stop:
            body["stop"] = stop
        # Pass through slot pinning and other llama-server-specific params
        if extra:
            body.update(extra)

        resp = await self._client.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return CompletionResult(
            content=content,
            model=data.get("model", "conductor"),
            usage=data.get("usage", {}),
            raw_response=data,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def supports_slots(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return "local (llama-server)"
