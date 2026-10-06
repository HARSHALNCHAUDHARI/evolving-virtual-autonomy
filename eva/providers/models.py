"""Typed request and response models for model providers.

Provider-neutral data models for interacting with language models.
These models do not bind to any external vendor SDK.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base exception for provider-related errors."""


class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider is not registered or found."""


class ModelNotFoundError(ProviderError):
    """Raised when a requested model is not found in the registry."""


class DuplicateModelError(ProviderError):
    """Raised when attempting to register a model that already exists."""


class StructuredOutputError(ProviderError):
    """Raised when structured output cannot be generated or validated."""


class NoEligibleModelError(ProviderError):
    """Raised when no model satisfies the required capabilities or routing criteria."""


# ---------------------------------------------------------------------------
# Model Capabilities & Specification
# ---------------------------------------------------------------------------


class ModelCapability(str, Enum):
    """Capabilities supported by language models.

    Extensible definition covering text, reasoning, multimodality,
    structured output, tool calling, streaming, and context sizes.
    """

    TEXT = "text"
    REASONING = "reasoning"
    VISION = "vision"
    AUDIO = "audio"
    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CALLING = "tool_calling"
    STREAMING = "streaming"
    LONG_CONTEXT = "long_context"


class ContextLimits(BaseModel):
    """Context window and token limits for a model."""

    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    context_window: int | None = None


class ModelCost(BaseModel):
    """Cost estimates per million tokens."""

    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    currency: str = "USD"


class ModelSpec(BaseModel):
    """Specification of an AI model's identity, capabilities, and parameters."""

    provider: str
    model: str
    capabilities: set[ModelCapability | str] = Field(default_factory=set)
    context_limits: ContextLimits | None = None
    latency_tier: str | None = None
    cost: ModelCost | None = None
    priority: int = 0
    available: bool = True
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    metadata: dict[str, Any] = Field(default_factory=dict)

    def has_capability(self, capability: ModelCapability | str) -> bool:
        """Check whether this model provides a specific capability."""
        if isinstance(capability, ModelCapability):
            return capability in self.capabilities or capability.value in self.capabilities
        val = capability.lower()
        try:
            enum_val = ModelCapability(val)
            if enum_val in self.capabilities:
                return True
        except ValueError:
            pass
        return val in self.capabilities

    @property
    def model_id(self) -> str:
        """Composite identifier for the model."""
        return f"{self.provider.lower()}:{self.model.lower()}"


# ---------------------------------------------------------------------------
# Task Types & Routing Criteria
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """Types of tasks requiring model intelligence."""

    CONVERSATION = "conversation"
    UNDERSTANDING = "understanding"
    REASONING = "reasoning"
    STRUCTURED_OUTPUT = "structured_output"
    GENERAL = "general"


class RouteRequest(BaseModel):
    """Criteria for selecting an AI model provider."""

    task_type: TaskType = TaskType.GENERAL
    provider: str | None = None
    model: str | None = None
    capability: str | None = None
    required_capabilities: set[ModelCapability | str] = Field(default_factory=set)
    preferred_provider: str | None = None
    preferred_model: str | None = None
    min_priority: int = 0
    latency_preference: str | None = None  # "low", "medium", "any"
    cost_preference: str | None = None     # "free", "low", "any"
    privacy_preference: str | None = None  # "local_only", "any"
    fallback_allowed: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_route_request(self) -> RouteRequest:
        """Synchronize shorthand fields into capability-aware selection criteria."""
        if self.capability and self.capability not in self.required_capabilities:
            self.required_capabilities = set(self.required_capabilities) | {self.capability}

        if self.provider and not self.preferred_provider:
            self.preferred_provider = self.provider

        if self.model and not self.preferred_model:
            self.preferred_model = self.model

        # Infer required capabilities from explicit task types if not specified
        if self.task_type == TaskType.STRUCTURED_OUTPUT:
            self.required_capabilities = set(self.required_capabilities) | {
                ModelCapability.STRUCTURED_OUTPUT
            }
        elif self.task_type == TaskType.REASONING:
            self.required_capabilities = set(self.required_capabilities) | {
                ModelCapability.REASONING
            }

        return self


class CandidateEvaluation(BaseModel):
    """Detailed evaluation of a single model candidate during routing."""

    model_id: str
    provider: str
    model: str
    model_spec: ModelSpec
    eligible: bool = True
    elimination_reason: str | None = None
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    """Explainable result of a model routing decision."""

    selected_model: ModelSpec | None = None
    provider_name: str = ""
    model_name: str = ""
    success: bool = True
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    candidates_evaluated: list[CandidateEvaluation] = Field(default_factory=list)
    fallback_used: bool = False
    fallback_from: str | None = None
    request: RouteRequest | None = None
    error: str | None = None

    def explain(self) -> str:
        """Return a human-readable explanation of why the model was selected."""
        if not self.success or not self.selected_model:
            return f"Routing failed: {self.error or 'No eligible model found.'}"

        lines = [f"Selected {self.selected_model.provider}:{self.selected_model.model} because:"]
        for r in self.reasons:
            lines.append(f"- {r}")
        if self.fallback_used and self.fallback_from:
            lines.append(f"- (Fallback from primary {self.fallback_from})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Message & Request Models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a conversation sequence."""

    role: str = "user"
    content: str


class ModelRequest(BaseModel):
    """Provider-neutral model generation request."""

    prompt: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    model: str | None = None
    provider: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] = Field(default_factory=list)
    system_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(default_factory=lambda: str(uuid4()))

    def get_prompt_text(self) -> str:
        """Extract prompt text from prompt field or messages."""
        if self.prompt is not None and self.prompt != "":
            return self.prompt
        if self.messages:
            # Return content of the latest message, or all user messages
            return self.messages[-1].content
        return ""


# ---------------------------------------------------------------------------
# Usage & Metadata Models
# ---------------------------------------------------------------------------


class ModelUsage(BaseModel):
    """Token usage metrics for a model interaction."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.total_tokens == 0 and (self.prompt_tokens > 0 or self.completion_tokens > 0):
            self.total_tokens = self.prompt_tokens + self.completion_tokens


class ModelMetadata(BaseModel):
    """Metadata associated with a model response."""

    latency_ms: float | None = None
    finish_reason: str | None = "stop"
    raw: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Response Model
# ---------------------------------------------------------------------------

T = TypeVar("T")


class ModelResponse(BaseModel):
    """Provider-neutral model generation response."""

    content: str = ""
    model: str = ""
    provider: str = ""
    structured_output: Any | None = None
    usage: ModelUsage = Field(default_factory=ModelUsage)
    metadata: ModelMetadata = Field(default_factory=ModelMetadata)
    finish_reason: str = "stop"
    request_id: str | None = None
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    success: bool = True
    error: str | None = None
