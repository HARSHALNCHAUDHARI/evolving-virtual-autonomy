"""Typed request and response models for model providers.

Provider-neutral data models for interacting with language models.
These models do not bind to any external vendor SDK.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base exception for provider-related errors."""


class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider is not registered or found."""


class StructuredOutputError(ProviderError):
    """Raised when structured output cannot be generated or validated."""


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
    metadata: dict[str, Any] = Field(default_factory=dict)


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
