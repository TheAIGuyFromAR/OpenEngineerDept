"""Anthropic Claude API provider — uses the Messages API directly."""

from __future__ import annotations

import httpx

from ..config import GatewayConfig
from .base import CompletionResult, InferenceProvider


class AnthropicProvider(InferenceProvider):
    """Sends requests to the Anthropic Messages API."""

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._base_url = config.resolved_api_base.rstrip("/")
        self._model = config.resolved_model
        self._client = httpx.AsyncClient(
            timeout=config.generation_timeout_seconds,
            headers={
                "x-api-key": config.inference_api_key,
                "anthropic-version": "2023-06-01",
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
        # Anthropic Messages API: system prompt is a top-level field, not a message
        system_text = ""
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            else:
                api_messages.append(msg)

        body: dict = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        if system_text:
            body["system"] = system_text
        if stop:
            body["stop_sequences"] = stop

        resp = await self._client.post(
            f"{self._base_url}/v1/messages", json=body
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract text from Anthropic response format
        content_blocks = data.get("content", [])
        content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = data.get("usage", {})
        return CompletionResult(
            content=content,
            model=data.get("model", self._model),
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0)
                + usage.get("output_tokens", 0),
            },
            raw_response=data,
        )

    async def health_check(self) -> bool:
        return bool(self._config.inference_api_key)

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def supports_slots(self) -> bool:
        return False

    @property
    def provider_name(self) -> str:
        return f"anthropic ({self._model})"
