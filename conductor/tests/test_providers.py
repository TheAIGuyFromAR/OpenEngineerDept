"""Tests for inference provider abstraction — factory, local, anthropic, openai_compat."""

from __future__ import annotations

import json

import httpx
import pytest

from gateway.config import GatewayConfig
from gateway.providers import create_provider
from gateway.providers.base import CompletionResult, InferenceProvider
from gateway.providers.local import LocalProvider
from gateway.providers.anthropic import AnthropicProvider
from gateway.providers.openai_compat import OpenAICompatProvider


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


class TestFactory:
    def test_local_provider(self):
        config = GatewayConfig(inference_provider="local")
        provider = create_provider(config)
        assert isinstance(provider, LocalProvider)
        assert provider.supports_slots is True

    def test_anthropic_provider(self):
        config = GatewayConfig(
            inference_provider="anthropic",
            inference_api_key="sk-ant-test",
        )
        provider = create_provider(config)
        assert isinstance(provider, AnthropicProvider)
        assert provider.supports_slots is False

    def test_openai_provider(self):
        config = GatewayConfig(
            inference_provider="openai",
            inference_api_key="sk-test",
        )
        provider = create_provider(config)
        assert isinstance(provider, OpenAICompatProvider)

    def test_openrouter_provider(self):
        config = GatewayConfig(
            inference_provider="openrouter",
            inference_api_key="sk-or-test",
        )
        provider = create_provider(config)
        assert isinstance(provider, OpenAICompatProvider)

    def test_unknown_provider_raises(self):
        config = GatewayConfig(inference_provider="nonsense")
        with pytest.raises(ValueError, match="Unknown inference_provider"):
            create_provider(config)

    def test_anthropic_without_key_raises(self):
        config = GatewayConfig(inference_provider="anthropic", inference_api_key="")
        with pytest.raises(ValueError, match="inference_api_key is required"):
            create_provider(config)

    def test_openai_without_key_raises(self):
        config = GatewayConfig(inference_provider="openai", inference_api_key="")
        with pytest.raises(ValueError, match="inference_api_key is required"):
            create_provider(config)


# ------------------------------------------------------------------
# Config defaults
# ------------------------------------------------------------------


class TestConfigDefaults:
    def test_resolved_api_base_anthropic(self):
        config = GatewayConfig(inference_provider="anthropic")
        assert config.resolved_api_base == "https://api.anthropic.com"

    def test_resolved_api_base_openai(self):
        config = GatewayConfig(inference_provider="openai")
        assert config.resolved_api_base == "https://api.openai.com/v1"

    def test_resolved_api_base_openrouter(self):
        config = GatewayConfig(inference_provider="openrouter")
        assert config.resolved_api_base == "https://openrouter.ai/api/v1"

    def test_custom_api_base_overrides_default(self):
        config = GatewayConfig(
            inference_provider="openai",
            inference_api_base="https://my-azure.openai.azure.com/v1",
        )
        assert config.resolved_api_base == "https://my-azure.openai.azure.com/v1"

    def test_resolved_model_anthropic(self):
        config = GatewayConfig(inference_provider="anthropic")
        assert "claude" in config.resolved_model

    def test_resolved_model_openai(self):
        config = GatewayConfig(inference_provider="openai")
        assert config.resolved_model == "gpt-4o"

    def test_custom_model_overrides_default(self):
        config = GatewayConfig(
            inference_provider="openai",
            inference_model="gpt-4-turbo",
        )
        assert config.resolved_model == "gpt-4-turbo"

    def test_is_local(self):
        assert GatewayConfig(inference_provider="local").is_local is True
        assert GatewayConfig(inference_provider="anthropic").is_local is False


# ------------------------------------------------------------------
# LocalProvider (mock HTTP transport)
# ------------------------------------------------------------------


class TestLocalProvider:
    @pytest.fixture
    def config(self):
        return GatewayConfig(
            llama_server_url="http://fake:8080",
            inference_provider="local",
        )

    async def test_chat_completion(self, config):
        provider = LocalProvider(config)

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["cache_prompt"] is True
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "hello world"}}],
                "model": "conductor",
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            })

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        result = await provider.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result.content == "hello world"
        assert result.usage["completion_tokens"] == 5
        await provider.close()

    async def test_slot_pinning_extra(self, config):
        """Extra params like id_slot are forwarded to llama-server."""
        provider = LocalProvider(config)
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
            })

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        await provider.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            extra={"id_slot": 2, "cache_prompt": True},
        )

        assert captured["id_slot"] == 2
        assert captured["cache_prompt"] is True
        await provider.close()

    async def test_health_check_ok(self, config):
        provider = LocalProvider(config)
        provider._client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"status": "ok"})
            )
        )
        assert await provider.health_check() is True
        await provider.close()

    async def test_health_check_fail(self, config):
        provider = LocalProvider(config)
        provider._client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(503, json={"status": "loading"})
            )
        )
        assert await provider.health_check() is False
        await provider.close()

    def test_supports_slots(self, config):
        provider = LocalProvider(config)
        assert provider.supports_slots is True

    def test_provider_name(self, config):
        provider = LocalProvider(config)
        assert "local" in provider.provider_name


# ------------------------------------------------------------------
# AnthropicProvider (mock HTTP transport)
# ------------------------------------------------------------------


class TestAnthropicProvider:
    @pytest.fixture
    def config(self):
        return GatewayConfig(
            inference_provider="anthropic",
            inference_api_key="sk-ant-test-key",
            inference_model="claude-sonnet-4-5-20250929",
        )

    async def test_chat_completion(self, config):
        provider = AnthropicProvider(config)

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            # System prompt should be extracted to top-level
            assert "system" in body
            assert body["system"] == "You are helpful"
            # Messages should NOT contain system role
            for msg in body["messages"]:
                assert msg["role"] != "system"
            # Auth header
            assert request.headers["x-api-key"] == "sk-ant-test-key"
            assert "anthropic-version" in request.headers

            return httpx.Response(200, json={
                "content": [{"type": "text", "text": "Hello from Claude"}],
                "model": "claude-sonnet-4-5-20250929",
                "usage": {"input_tokens": 20, "output_tokens": 5},
            })

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        result = await provider.chat_completion(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ],
        )

        assert result.content == "Hello from Claude"
        assert result.usage["prompt_tokens"] == 20
        assert result.usage["completion_tokens"] == 5
        await provider.close()

    async def test_stop_sequences_mapped(self, config):
        """stop parameter should map to stop_sequences for Anthropic."""
        provider = AnthropicProvider(config)
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={
                "content": [{"type": "text", "text": "done"}],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            })

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        await provider.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            stop=["```", "\n\n"],
        )

        assert captured["stop_sequences"] == ["```", "\n\n"]
        assert "stop" not in captured
        await provider.close()

    async def test_health_with_key(self, config):
        provider = AnthropicProvider(config)
        assert await provider.health_check() is True
        await provider.close()

    async def test_health_without_key(self):
        config = GatewayConfig(
            inference_provider="anthropic",
            inference_api_key="",
        )
        provider = AnthropicProvider(config)
        assert await provider.health_check() is False
        await provider.close()

    def test_supports_slots(self, config):
        provider = AnthropicProvider(config)
        assert provider.supports_slots is False


# ------------------------------------------------------------------
# OpenAICompatProvider (mock HTTP transport)
# ------------------------------------------------------------------


class TestOpenAICompatProvider:
    @pytest.fixture
    def config(self):
        return GatewayConfig(
            inference_provider="openai",
            inference_api_key="sk-test-key",
            inference_api_base="https://api.openai.com/v1",
            inference_model="gpt-4o",
        )

    @pytest.fixture
    def openrouter_config(self):
        return GatewayConfig(
            inference_provider="openrouter",
            inference_api_key="sk-or-test",
            inference_api_base="https://openrouter.ai/api/v1",
            inference_model="qwen/qwen3-coder",
        )

    async def test_chat_completion(self, config):
        provider = OpenAICompatProvider(config)

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["model"] == "gpt-4o"
            assert request.headers["authorization"] == "Bearer sk-test-key"
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "Hello from GPT"}}],
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 15, "completion_tokens": 4},
            })

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        result = await provider.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result.content == "Hello from GPT"
        await provider.close()

    async def test_no_slot_params_forwarded(self, config):
        """OpenAI-compat provider should NOT forward slot IDs."""
        provider = OpenAICompatProvider(config)
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
            })

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        # Even if extra is passed, provider should not forward it
        await provider.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            extra={"id_slot": 2},  # This should be ignored
        )

        assert "id_slot" not in captured
        await provider.close()

    async def test_openrouter_model(self, openrouter_config):
        provider = OpenAICompatProvider(openrouter_config)
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
            })

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        await provider.chat_completion(
            messages=[{"role": "user", "content": "test"}],
        )

        assert captured["model"] == "qwen/qwen3-coder"
        await provider.close()

    def test_supports_slots(self, config):
        provider = OpenAICompatProvider(config)
        assert provider.supports_slots is False
