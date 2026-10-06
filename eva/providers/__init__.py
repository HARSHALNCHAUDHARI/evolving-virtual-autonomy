"""EVA model provider abstraction and intelligence foundation.

Defines the contract that all model providers must implement,
ensuring EVA is never hard-coded to a single AI vendor.
Provides model routing, typed request/response models, and deterministic
mock implementations.
"""

from __future__ import annotations

from eva.providers.base import ModelProvider
from eva.providers.mock import MockProvider
from eva.providers.models import (
    ChatMessage,
    ModelMetadata,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ProviderError,
    ProviderNotFoundError,
    RouteRequest,
    StructuredOutputError,
    TaskType,
)
from eva.providers.router import ModelRouter

__all__ = [
    "ChatMessage",
    "MockProvider",
    "ModelMetadata",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "ModelRouter",
    "ModelUsage",
    "ProviderError",
    "ProviderNotFoundError",
    "RouteRequest",
    "StructuredOutputError",
    "TaskType",
]
