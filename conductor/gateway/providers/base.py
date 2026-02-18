"""Abstract base for inference providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class CompletionResult:
    """Standardized completion result regardless of provider."""

    content: str
    model: str
    usage: dict = field(default_factory=dict)
    raw_response: dict = field(default_factory=dict)


class InferenceProvider(ABC):
    """Common interface for local and cloud inference backends."""

    @abstractmethod
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
        """Send a chat completion request."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and ready."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""

    @property
    @abstractmethod
    def supports_slots(self) -> bool:
        """Whether this provider supports llama-server slot pinning."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name for logging."""
