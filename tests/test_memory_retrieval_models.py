"""Unit tests for memory retrieval data models and query validation."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from eva.memory.models import (
    InvalidMemoryQueryError,
    MemoryQuery,
    MemoryRecord,
    MemorySource,
    MemoryType,
    ProvenanceRecord,
)
from eva.memory.retrieval.models import (
    MatchExplanation,
    MemorySearchResponse,
    MemorySearchResult,
)


def test_memory_query_text_and_offset_defaults() -> None:
    q = MemoryQuery()
    assert q.query_text is None
    assert q.offset == 0
    assert q.metadata_filters == {}
    assert q.limit is None


def test_memory_query_aliases() -> None:
    q1 = MemoryQuery(text="autonomous agent")  # type: ignore[call-arg]
    assert q1.query_text == "autonomous agent"
    assert q1.text == "autonomous agent"
    assert q1.query == "autonomous agent"

    q2 = MemoryQuery(query="system prompt")  # type: ignore[call-arg]
    assert q2.query_text == "system prompt"
    assert q2.query == "system prompt"


def test_memory_query_offset_validation() -> None:
    q = MemoryQuery(offset=5)
    assert q.offset == 5

    with pytest.raises(ValidationError):
        MemoryQuery(offset=-1)


def test_memory_query_temporal_validation() -> None:
    now = datetime.now(timezone.utc)
    earlier = datetime(2025, 1, 1, tzinfo=timezone.utc)
    later = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Valid: earlier <= later
    q_valid = MemoryQuery(created_after=earlier, created_before=later)
    assert q_valid.created_after == earlier
    assert q_valid.created_before == later

    # Invalid: created_after > created_before
    with pytest.raises(InvalidMemoryQueryError, match="created_after .* cannot be later than created_before"):
        MemoryQuery(created_after=later, created_before=earlier)

    # Invalid: updated_after > updated_before
    with pytest.raises(InvalidMemoryQueryError, match="updated_after .* cannot be later than updated_before"):
        MemoryQuery(updated_after=later, updated_before=earlier)


def test_match_explanation_and_result() -> None:
    record = MemoryRecord(
        memory_type=MemoryType.SEMANTIC,
        content="EVA uses Python 3.14",
        provenance=ProvenanceRecord(source_type=MemorySource.USER_STATEMENT),
    )
    explanation = MatchExplanation(
        exact_phrase_matched=True,
        matched_tokens=["python", "3", "14"],
        token_overlap_ratio=1.0,
        text_score=1.0,
        importance_score=0.5,
        confidence_score=1.0,
        recency_score=0.99,
        reasons=["Exact phrase matched"],
    )
    result = MemorySearchResult(
        memory=record,
        score=0.85,
        explanation=explanation,
    )

    assert result.score == 0.85
    assert result.match_reasons == ["Exact phrase matched"]
    assert result.explanation.exact_phrase_matched is True
    assert result.explanation.token_overlap_ratio == 1.0


def test_memory_search_response() -> None:
    record = MemoryRecord(
        memory_type=MemoryType.EPISODIC,
        content="Session log",
        provenance=ProvenanceRecord(source_type=MemorySource.EVA_OBSERVATION),
    )
    res = MemorySearchResult(memory=record, score=0.7)
    query = MemoryQuery(query_text="session")
    response = MemorySearchResponse(
        results=[res],
        total_candidates=1,
        query=query,
    )

    assert len(response) == 1
    assert response.total_candidates == 1
    assert list(response)[0].score == 0.7
    assert response[0].memory.content == "Session log"
