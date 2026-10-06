"""Data models and value objects for EVA's Memory subsystem.

Defines the canonical conceptual memory record, provenance representations,
typed query filters, metadata containers, and memory exceptions in accordance
with docs/MEMORY_SPEC.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


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


class MemoryError(Exception):
    """Base exception for all memory subsystem errors."""


class MemoryNotFoundError(MemoryError):
    """Raised when a requested memory record is not found."""


class DuplicateMemoryError(MemoryError):
    """Raised when attempting to save a record with an already existing ID."""


class MemoryUpdateError(MemoryError):
    """Raised when an update operation on a memory record fails."""


class ImmutableMemoryError(MemoryUpdateError):
    """Raised when attempting to modify an immutable or protected memory record."""


class DatabaseError(MemoryError):
    """Base exception for database and persistence errors."""


class DatabaseConnectionError(DatabaseError):
    """Raised when unable to establish or maintain a database connection."""


class DatabaseConfigurationError(DatabaseError):
    """Raised when database configuration is missing, invalid, or malformed."""


class DatabaseMigrationError(DatabaseError):
    """Raised when database migration application or verification fails."""


class MemoryRetrievalError(MemoryError):
    """Raised when memory retrieval operations fail."""


class InvalidMemoryQueryError(MemoryRetrievalError):
    """Raised when query parameters, filters, or boundaries are invalid."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MemoryType(str, Enum):
    """Categorical classification of memory records.

    Matches the cognitive hierarchy defined in docs/MEMORY_SPEC.md:
    Working, Session, Episodic, Semantic, Procedural, Preference, and Project Context.
    """

    WORKING = "working"
    SESSION = "session"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    PREFERENCE = "preference"
    PROJECT_CONTEXT = "project_context"

    # Alias to preserve compatibility with the "project" naming in MEMORY_SPEC.md
    PROJECT = PROJECT_CONTEXT

    @classmethod
    def _missing_(cls, value: object) -> MemoryType | None:
        if isinstance(value, str):
            val = value.lower()
            if val in ("project", "project_context"):
                return cls.PROJECT_CONTEXT
        return None


class MemorySource(str, Enum):
    """Origin and provenance source of a memory record.

    Preserves the epistemic distinction between direct/observed telemetry
    and inferred hypotheses as specified in docs/MEMORY_SPEC.md.
    """

    USER_STATEMENT = "user_statement"       # Direct explicit user instruction/assertion
    EVA_OBSERVATION = "eva_observation"     # Sensor/telemetry observation
    TOOL_RESULT = "tool_result"             # Direct output of verified tool invocation
    MODEL_INFERENCE = "model_inference"     # LLM deduction/summary (hypothetical/inferred)
    EXTERNAL_DOCUMENT = "external_document" # Document, file, or web page content
    SYSTEM_EVENT = "system_event"           # Runtime lifecycle signal

    @property
    def is_inferred(self) -> bool:
        """Return True if this source represents an inferred hypothesis."""
        return self == MemorySource.MODEL_INFERENCE

    @property
    def is_observed(self) -> bool:
        """Return True if this source represents direct observed facts or telemetry."""
        return not self.is_inferred


# ---------------------------------------------------------------------------
# Metadata & Provenance Models
# ---------------------------------------------------------------------------


class ProvenanceRecord(BaseModel):
    """Complete causal and epistemic lineage of a memory record.

    Preserves the distinction between observed/direct information
    and inferred hypotheses without performing automatic inference generation.
    """

    source_type: MemorySource
    source_id: str | None = None
    model_id: str | None = None
    raw_quote: str | None = None
    derived_from: list[UUID] = Field(default_factory=list)
    is_inferred: bool | None = None
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("created_at", mode="after")
    @classmethod
    def _validate_created_at_tz(cls, v: datetime) -> datetime:
        return _ensure_timezone(v) or _utcnow()

    def model_post_init(self, __context: Any) -> None:
        if self.is_inferred is None:
            self.is_inferred = self.source_type.is_inferred or bool(self.derived_from)

    @property
    def is_observed(self) -> bool:
        """Return True if the provenance indicates direct observed facts."""
        return not bool(self.is_inferred)


class EmbeddingReference(BaseModel):
    """Pointer and metadata for external dense vector representations."""

    model: str
    dimensions: int = Field(gt=0)
    vector_id: str | None = None
    version: int = Field(default=1, ge=1)


class AccessMetadata(BaseModel):
    """Access telemetry and recency decay metadata."""

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    last_accessed_at: datetime | None = None
    access_count: int = Field(default=0, ge=0)
    decay_rate: float = Field(default=0.01, ge=0.0)

    @field_validator("created_at", "updated_at", "last_accessed_at", mode="after")
    @classmethod
    def _validate_tz(cls, v: datetime | None) -> datetime | None:
        return _ensure_timezone(v)


class RetentionMetadata(BaseModel):
    """Lifecycle, retention tier, and immutability controls."""

    expires_at: datetime | None = None
    is_immutable: bool = False
    is_archived: bool = False
    retention_tier: str = "active"

    @field_validator("expires_at", mode="after")
    @classmethod
    def _validate_expires_at_tz(cls, v: datetime | None) -> datetime | None:
        return _ensure_timezone(v)


# ---------------------------------------------------------------------------
# Canonical Memory Record
# ---------------------------------------------------------------------------


class MemoryRecord(BaseModel):
    """Canonical data model representing a persistent or transient memory unit."""

    memory_id: UUID = Field(default_factory=uuid4)
    memory_type: MemoryType
    content: str
    structured_payload: dict[str, Any] = Field(default_factory=dict)
    provenance: ProvenanceRecord
    source: MemorySource | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    project_id: str | None = None
    session_id: str | None = None
    embedding_ref: EmbeddingReference | None = None
    access_metadata: AccessMetadata = Field(default_factory=AccessMetadata)
    retention_metadata: RetentionMetadata = Field(default_factory=RetentionMetadata)

    @field_validator("created_at", "updated_at", mode="after")
    @classmethod
    def _validate_record_tz(cls, v: datetime) -> datetime:
        return _ensure_timezone(v) or _utcnow()

    @model_validator(mode="after")
    def _sync_provenance_and_metadata(self) -> MemoryRecord:
        # Default source from provenance if not explicitly provided
        if self.source is None:
            self.source = self.provenance.source_type

        # Synchronize access_metadata timestamps if they are at defaults
        if self.access_metadata.created_at != self.created_at:
            self.access_metadata.created_at = self.created_at
        if self.access_metadata.updated_at != self.updated_at:
            self.access_metadata.updated_at = self.updated_at

        return self


# ---------------------------------------------------------------------------
# Query Model
# ---------------------------------------------------------------------------


class MemoryQuery(BaseModel):
    """Query criteria for retrieving and searching memory records.

    Supports keyword/text matching, metadata filtering, temporal filtering,
    contextual isolation, and pagination parameters.
    """

    query_text: str | None = None
    memory_types: list[MemoryType] | None = None
    project_id: str | None = None
    include_global: bool = True
    session_id: str | None = None
    sources: list[MemorySource] | None = None
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    min_importance: float | None = Field(default=None, ge=0.0, le=1.0)
    created_after: datetime | None = None
    created_before: datetime | None = None
    updated_after: datetime | None = None
    updated_before: datetime | None = None
    is_archived: bool | None = False
    limit: int | None = Field(default=None, gt=0)
    offset: int = Field(default=0, ge=0)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "created_after",
        "created_before",
        "updated_after",
        "updated_before",
        mode="after",
    )
    @classmethod
    def _validate_query_tz(cls, v: datetime | None) -> datetime | None:
        return _ensure_timezone(v)

    @model_validator(mode="before")
    @classmethod
    def _map_query_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "query_text" not in data:
                if "text" in data:
                    data["query_text"] = data.pop("text")
                elif "query" in data:
                    data["query_text"] = data.pop("query")
        return data

    @model_validator(mode="after")
    def _validate_boundaries(self) -> MemoryQuery:
        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_after > self.created_before
        ):
            raise InvalidMemoryQueryError(
                f"created_after ({self.created_after}) cannot be later than created_before ({self.created_before})"
            )
        if (
            self.updated_after is not None
            and self.updated_before is not None
            and self.updated_after > self.updated_before
        ):
            raise InvalidMemoryQueryError(
                f"updated_after ({self.updated_after}) cannot be later than updated_before ({self.updated_before})"
            )
        return self

    @property
    def text(self) -> str | None:
        return self.query_text

    @property
    def query(self) -> str | None:
        return self.query_text
