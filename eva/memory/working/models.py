"""Typed models and exceptions for EVA's Working Memory layer.

Defines the WorkingMemoryEntry data model and specialized working memory
exceptions according to docs/MEMORY_SPEC.md.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from eva.memory.models import MemoryError


def _utcnow() -> datetime:
    """Return the current time in UTC with timezone awareness."""
    return datetime.now(timezone.utc)


def _ensure_timezone(dt: datetime | None) -> datetime | None:
    """Ensure a datetime object is timezone-aware, defaulting to UTC."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WorkingMemoryError(MemoryError):
    """Base exception for working memory subsystem operations."""


class WorkingMemoryNotFoundError(WorkingMemoryError):
    """Raised when a requested working memory entry is not found."""


class WorkingMemorySerializationError(WorkingMemoryError):
    """Raised when working memory values cannot be safely serialized or deserialized."""


class WorkingMemoryConnectionError(WorkingMemoryError):
    """Raised when connecting to or communicating with Redis fails."""


class WorkingMemoryConfigurationError(WorkingMemoryError):
    """Raised when Redis or working memory configuration is missing or malformed."""


# ---------------------------------------------------------------------------
# Working Memory Entry Model
# ---------------------------------------------------------------------------


class WorkingMemoryEntry(BaseModel):
    """Representation of an isolated, temporary working memory entry.

    Working memory holds volatile turn scratchpads, active session state,
    and temporary task contexts governed by TTL expiration.
    """

    session_id: str = Field(..., min_length=1, description="Unique session boundary identifier")
    key: str = Field(..., min_length=1, description="Working memory key name")
    value: Any = Field(..., description="JSON-compatible payload value")
    namespace: str = Field(default="default", min_length=1, description="Context/domain namespace")
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    expires_at: datetime | None = None
    ttl_seconds: int | None = Field(default=None, description="Time-to-live in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "key", "namespace", mode="before")
    @classmethod
    def _strip_strings(cls, v: Any) -> Any:
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                raise ValueError("Field cannot be empty or whitespace only.")
            return stripped
        return v

    @field_validator("created_at", "updated_at", "expires_at", mode="after")
    @classmethod
    def _validate_tz(cls, v: datetime | None) -> datetime | None:
        return _ensure_timezone(v)

    @model_validator(mode="after")
    def _sync_ttl_and_expiration(self) -> WorkingMemoryEntry:
        # If ttl_seconds is provided but expires_at is not, calculate expires_at
        if self.ttl_seconds is not None and self.expires_at is None:
            if self.ttl_seconds > 0:
                self.expires_at = self.created_at + timedelta(seconds=self.ttl_seconds)
            else:
                self.expires_at = self.created_at

        # If expires_at is provided but ttl_seconds is not, calculate ttl_seconds
        elif self.expires_at is not None and self.ttl_seconds is None:
            delta = (self.expires_at - self.created_at).total_seconds()
            self.ttl_seconds = max(0, int(delta))

        return self

    def is_expired(self, now: datetime | None = None) -> bool:
        """Check whether this entry has exceeded its expiration timestamp."""
        if self.expires_at is None:
            return False
        current_time = _ensure_timezone(now) or _utcnow()
        return current_time >= self.expires_at
