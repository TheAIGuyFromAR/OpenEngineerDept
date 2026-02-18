"""Inference providers — abstracts local llama-server vs. cloud API backends."""

from .base import InferenceProvider, CompletionResult
from .factory import create_provider

__all__ = ["InferenceProvider", "CompletionResult", "create_provider"]
