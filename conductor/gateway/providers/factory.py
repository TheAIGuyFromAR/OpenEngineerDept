"""Factory to instantiate the correct inference provider from config."""

from __future__ import annotations

import logging

from ..config import GatewayConfig
from .base import InferenceProvider

logger = logging.getLogger(__name__)


def create_provider(config: GatewayConfig) -> InferenceProvider:
    """Create an inference provider based on the gateway config."""
    provider_type = config.inference_provider.lower()

    if provider_type == "local":
        from .local import LocalProvider

        logger.info("Using local llama-server at %s", config.llama_server_url)
        return LocalProvider(config)

    if provider_type == "anthropic":
        if not config.inference_api_key:
            raise ValueError(
                "inference_api_key is required for the Anthropic provider. "
                "Set CONDUCTOR_INFERENCE_API_KEY or add inference_api_key to conductor.yaml"
            )
        from .anthropic import AnthropicProvider

        logger.info("Using Anthropic API — model: %s", config.resolved_model)
        return AnthropicProvider(config)

    if provider_type in ("openai", "openrouter"):
        if not config.inference_api_key:
            raise ValueError(
                f"inference_api_key is required for the {provider_type} provider. "
                f"Set CONDUCTOR_INFERENCE_API_KEY or add inference_api_key to conductor.yaml"
            )
        from .openai_compat import OpenAICompatProvider

        logger.info(
            "Using %s API — model: %s, base: %s",
            provider_type,
            config.resolved_model,
            config.resolved_api_base,
        )
        return OpenAICompatProvider(config)

    raise ValueError(
        f"Unknown inference_provider: {provider_type!r}. "
        f"Valid options: local, anthropic, openai, openrouter"
    )
