"""EVA model provider abstraction and intelligence foundation.

Defines the contract that all model providers must implement,
ensuring EVA is never hard-coded to a single AI vendor.
Provides model capability specifications, central model registry,
model routing, typed request/response models, and deterministic
mock implementations.
"""

from __future__ import annotations

from eva.providers.base import ModelProvider
from eva.providers.gemini import GeminiProvider
from eva.providers.mock import MockProvider
from eva.providers.models import (
    CandidateEvaluation,
    ChatMessage,
    ContextLimits,
    DuplicateModelError,
    ModelCapability,
    ModelCost,
    ModelMetadata,
    ModelNotFoundError,
    ModelRequest,
    ModelResponse,
    ModelSpec,
    ModelUsage,
    NoEligibleModelError,
    ProviderError,
    ProviderNotFoundError,
    RouteRequest,
    RoutingDecision,
    StructuredOutputError,
    TaskType,
)
from eva.providers.registry import ModelRegistry
from eva.providers.router import ModelRouter

__all__ = [
    "CandidateEvaluation",
    "ChatMessage",
    "ContextLimits",
    "DuplicateModelError",
    "GeminiProvider",
    "MockProvider",
    "ModelCapability",
    "ModelCost",
    "ModelMetadata",
    "ModelNotFoundError",
    "ModelProvider",
    "ModelRegistry",
    "ModelRequest",
    "ModelResponse",
    "ModelRouter",
    "ModelSpec",
    "ModelUsage",
    "NoEligibleModelError",
    "ProviderError",
    "ProviderNotFoundError",
    "RouteRequest",
    "RoutingDecision",
    "StructuredOutputError",
    "TaskType",
]
