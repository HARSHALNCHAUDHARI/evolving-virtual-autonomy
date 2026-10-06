"""Unit tests for EVA memory data models, enums, and value objects."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from eva.memory.models import (
    AccessMetadata,
    EmbeddingReference,
    MemoryQuery,
    MemoryRecord,
    MemorySource,
    MemoryType,
    ProvenanceRecord,
    RetentionMetadata,
)


def test_memory_type_enum() -> None:
    """Verify all 7 memory categories and project aliases."""
    expected_categories = {
        "working",
        "session",
        "episodic",
        "semantic",
        "procedural",
        "preference",
        "project_context",
    }
    values = {m.value for m in MemoryType}
    assert values == expected_categories

    # Verify all 7 members exist
    assert MemoryType.WORKING == "working"
    assert MemoryType.SESSION == "session"
    assert MemoryType.EPISODIC == "episodic"
    assert MemoryType.SEMANTIC == "semantic"
    assert MemoryType.PROCEDURAL == "procedural"
    assert MemoryType.PREFERENCE == "preference"
    assert MemoryType.PROJECT_CONTEXT == "project_context"

    # Verify PROJECT alias and missing value handling
    assert MemoryType.PROJECT == MemoryType.PROJECT_CONTEXT
    assert MemoryType("project") == MemoryType.PROJECT_CONTEXT
    assert MemoryType("project_context") == MemoryType.PROJECT_CONTEXT


def test_memory_source_epistemic_distinction() -> None:
    """Verify observed vs inferred epistemic properties on MemorySource."""
    # Direct / observed sources
    for source in [
        MemorySource.USER_STATEMENT,
        MemorySource.EVA_OBSERVATION,
        MemorySource.TOOL_RESULT,
        MemorySource.EXTERNAL_DOCUMENT,
        MemorySource.SYSTEM_EVENT,
    ]:
        assert source.is_observed is True
        assert source.is_inferred is False

    # Inferred source
    assert MemorySource.MODEL_INFERENCE.is_inferred is True
    assert MemorySource.MODEL_INFERENCE.is_observed is False


def test_provenance_record_defaults_and_inference() -> None:
    """Test ProvenanceRecord initialization, timezone awareness, and inferred logic."""
    # Direct user statement without derivation
    prov_direct = ProvenanceRecord(source_type=MemorySource.USER_STATEMENT)
    assert prov_direct.is_inferred is False
    assert prov_direct.is_observed is True
    assert prov_direct.created_at.tzinfo is not None

    # Model inference source defaults to inferred
    prov_inferred = ProvenanceRecord(source_type=MemorySource.MODEL_INFERENCE)
    assert prov_inferred.is_inferred is True
    assert prov_inferred.is_observed is False

    # Derived from parents defaults to inferred even if source was observation
    parent_id = uuid4()
    prov_derived = ProvenanceRecord(
        source_type=MemorySource.EVA_OBSERVATION,
        derived_from=[parent_id],
    )
    assert prov_derived.is_inferred is True
    assert prov_derived.is_observed is False

    # Naive timestamp converted to timezone-aware UTC
    naive_dt = datetime(2026, 1, 1, 12, 0, 0)
    prov_tz = ProvenanceRecord(
        source_type=MemorySource.USER_STATEMENT,
        created_at=naive_dt,
    )
    assert prov_tz.created_at.tzinfo == timezone.utc


def test_metadata_containers() -> None:
    """Test EmbeddingReference, AccessMetadata, and RetentionMetadata."""
    # EmbeddingReference
    emb = EmbeddingReference(model="text-embedding-3-small", dimensions=1536)
    assert emb.model == "text-embedding-3-small"
    assert emb.dimensions == 1536
    assert emb.version == 1

    with pytest.raises(ValidationError):
        EmbeddingReference(model="test", dimensions=0)

    # AccessMetadata
    access = AccessMetadata()
    assert access.access_count == 0
    assert access.decay_rate == 0.01
    assert access.created_at.tzinfo is not None
    assert access.updated_at.tzinfo is not None

    # RetentionMetadata
    retention = RetentionMetadata()
    assert retention.is_immutable is False
    assert retention.is_archived is False
    assert retention.retention_tier == "active"
    assert retention.expires_at is None


def test_memory_record_creation_and_defaults() -> None:
    """Test MemoryRecord initialization with minimal arguments and default values."""
    prov = ProvenanceRecord(
        source_type=MemorySource.USER_STATEMENT,
        raw_quote="I prefer Python over TypeScript",
    )
    record = MemoryRecord(
        memory_type=MemoryType.PREFERENCE,
        content="User prefers Python over TypeScript",
        provenance=prov,
    )

    assert isinstance(record.memory_id, UUID)
    assert record.memory_type == MemoryType.PREFERENCE
    assert record.content == "User prefers Python over TypeScript"
    assert record.source == MemorySource.USER_STATEMENT
    assert record.confidence == 1.0
    assert record.importance == 0.5
    assert record.relevance == 0.0
    assert record.project_id is None
    assert record.session_id is None
    assert record.structured_payload == {}
    assert record.created_at.tzinfo is not None
    assert record.updated_at.tzinfo is not None
    assert record.access_metadata.created_at == record.created_at
    assert record.access_metadata.updated_at == record.updated_at


def test_memory_record_validation_boundaries() -> None:
    """Verify confidence, importance, and relevance validation boundaries [0.0, 1.0]."""
    prov = ProvenanceRecord(source_type=MemorySource.USER_STATEMENT)

    # Valid boundaries
    r_min = MemoryRecord(
        memory_type=MemoryType.WORKING,
        content="min test",
        provenance=prov,
        confidence=0.0,
        importance=0.0,
        relevance=0.0,
    )
    assert r_min.confidence == 0.0

    r_max = MemoryRecord(
        memory_type=MemoryType.WORKING,
        content="max test",
        provenance=prov,
        confidence=1.0,
        importance=1.0,
        relevance=1.0,
    )
    assert r_max.confidence == 1.0

    # Invalid confidence
    with pytest.raises(ValidationError):
        MemoryRecord(
            memory_type=MemoryType.WORKING,
            content="test",
            provenance=prov,
            confidence=-0.01,
        )
    with pytest.raises(ValidationError):
        MemoryRecord(
            memory_type=MemoryType.WORKING,
            content="test",
            provenance=prov,
            confidence=1.01,
        )

    # Invalid importance
    with pytest.raises(ValidationError):
        MemoryRecord(
            memory_type=MemoryType.WORKING,
            content="test",
            provenance=prov,
            importance=-0.5,
        )
    with pytest.raises(ValidationError):
        MemoryRecord(
            memory_type=MemoryType.WORKING,
            content="test",
            provenance=prov,
            importance=1.5,
        )

    # Invalid relevance
    with pytest.raises(ValidationError):
        MemoryRecord(
            memory_type=MemoryType.WORKING,
            content="test",
            provenance=prov,
            relevance=1.1,
        )


def test_memory_record_mutable_default_safety() -> None:
    """Ensure mutating default structures in one record does not affect another."""
    prov1 = ProvenanceRecord(source_type=MemorySource.USER_STATEMENT)
    r1 = MemoryRecord(
        memory_type=MemoryType.WORKING,
        content="record 1",
        provenance=prov1,
    )
    prov2 = ProvenanceRecord(source_type=MemorySource.USER_STATEMENT)
    r2 = MemoryRecord(
        memory_type=MemoryType.WORKING,
        content="record 2",
        provenance=prov2,
    )

    r1.structured_payload["key"] = "value1"
    assert "key" not in r2.structured_payload

    r1.provenance.derived_from.append(uuid4())
    assert len(r2.provenance.derived_from) == 0


def test_memory_record_timezone_preservation() -> None:
    """Verify timezone awareness is enforced for naive datetimes."""
    prov = ProvenanceRecord(source_type=MemorySource.USER_STATEMENT)
    naive_dt = datetime(2026, 6, 1, 8, 30, 0)
    record = MemoryRecord(
        memory_type=MemoryType.EPISODIC,
        content="naive dt test",
        provenance=prov,
        created_at=naive_dt,
        updated_at=naive_dt,
    )
    assert record.created_at.tzinfo == timezone.utc
    assert record.updated_at.tzinfo == timezone.utc


def test_memory_query_validation_and_filters() -> None:
    """Test MemoryQuery instantiation, filters, and boundaries."""
    query = MemoryQuery(
        memory_types=[MemoryType.PREFERENCE, MemoryType.PROJECT_CONTEXT],
        project_id="eva",
        include_global=True,
        min_confidence=0.8,
        min_importance=0.5,
        limit=10,
    )

    assert query.memory_types == [MemoryType.PREFERENCE, MemoryType.PROJECT_CONTEXT]
    assert query.project_id == "eva"
    assert query.include_global is True
    assert query.min_confidence == 0.8
    assert query.min_importance == 0.5
    assert query.limit == 10
    assert query.is_archived is False

    # Boundary validation on query
    with pytest.raises(ValidationError):
        MemoryQuery(min_confidence=1.5)
    with pytest.raises(ValidationError):
        MemoryQuery(min_importance=-0.1)
    with pytest.raises(ValidationError):
        MemoryQuery(limit=0)
