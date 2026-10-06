"""Unit tests for text normalization, tokenization, recency decay, and deterministic scoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from eva.memory.models import (
    AccessMetadata,
    MemoryQuery,
    MemoryRecord,
    MemorySource,
    MemoryType,
    ProvenanceRecord,
    RetentionMetadata,
)
from eva.memory.retrieval.models import MemorySearchResult
from eva.memory.retrieval.ranking import (
    compute_recency_score,
    normalize_text,
    ranking_sort_key,
    score_memory_record,
    tokenize,
)


def test_normalize_text() -> None:
    assert normalize_text("   Hello   World!  ") == "hello world!"
    assert normalize_text("UPPERCASE  \n\t Text") == "uppercase text"
    assert normalize_text("") == ""


def test_tokenize() -> None:
    tokens = tokenize("FastAPI and Python-3.14 are great tools!")
    assert tokens == ["fastapi", "and", "python-3", "14", "are", "great", "tools"]
    assert tokenize("") == []


def test_compute_recency_score() -> None:
    now = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)

    # Immediately updated (0 days ago) -> 1.0
    s0 = compute_recency_score(now, now=now)
    assert pytest.approx(s0, rel=1e-3) == 1.0

    # 10 days ago with default decay rate 0.01 -> exp(-0.1) ~ 0.9048
    s10 = compute_recency_score(now - timedelta(days=10), now=now)
    assert pytest.approx(s10, rel=1e-3) == 0.9048

    # Immutable memories do not decay regardless of age
    s_imm = compute_recency_score(now - timedelta(days=365), is_immutable=True, now=now)
    assert s_imm == 1.0


def test_score_memory_record_exact_phrase() -> None:
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        memory_type=MemoryType.SEMANTIC,
        content="PostgreSQL database stores long term memory records.",
        importance=0.8,
        confidence=1.0,
        provenance=ProvenanceRecord(source_type=MemorySource.USER_STATEMENT),
        updated_at=now,
    )
    query = MemoryQuery(query_text="postgresql database")

    result = score_memory_record(rec, query, now=now)
    assert result is not None
    assert result.explanation.exact_phrase_matched is True
    assert result.explanation.token_overlap_ratio == 1.0
    assert result.score > 0.8  # high score due to exact match + high importance + recency


def test_score_memory_record_partial_tokens() -> None:
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        memory_type=MemoryType.SEMANTIC,
        content="We prefer SQLite for simple embedded development.",
        importance=0.5,
        confidence=0.9,
        provenance=ProvenanceRecord(source_type=MemorySource.USER_STATEMENT),
        updated_at=now,
    )
    # 1 of 2 tokens matches ("sqlite")
    query = MemoryQuery(query_text="sqlite postgresql")

    result = score_memory_record(rec, query, now=now)
    assert result is not None
    assert result.explanation.exact_phrase_matched is False
    assert result.explanation.token_overlap_ratio == 0.5
    assert "sqlite" in result.explanation.matched_tokens


def test_score_memory_record_no_match_returns_none() -> None:
    rec = MemoryRecord(
        memory_type=MemoryType.PREFERENCE,
        content="User prefers dark mode in code editor.",
        provenance=ProvenanceRecord(source_type=MemorySource.USER_STATEMENT),
    )
    query = MemoryQuery(query_text="kubernetes cluster deployment")

    result = score_memory_record(rec, query)
    assert result is None


def test_score_memory_record_metadata_only() -> None:
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        memory_type=MemoryType.PREFERENCE,
        content="User likes concise responses.",
        importance=0.9,
        confidence=1.0,
        provenance=ProvenanceRecord(source_type=MemorySource.USER_STATEMENT),
        updated_at=now,
    )
    # No query_text provided
    query = MemoryQuery(memory_types=[MemoryType.PREFERENCE])

    result = score_memory_record(rec, query, now=now)
    assert result is not None
    assert result.explanation.text_score == 0.0
    assert "Matched metadata filters" in result.explanation.reasons
    # Score = 0.50 * 0.9 + 0.30 * 1.0 + 0.20 * 1.0 = 0.45 + 0.30 + 0.20 = 0.95
    assert pytest.approx(result.score, rel=1e-3) == 0.95


def test_ranking_sort_key_tie_breakers() -> None:
    now = datetime.now(timezone.utc)
    earlier = now - timedelta(hours=1)

    r1 = MemoryRecord(
        memory_id=UUID("00000000-0000-0000-0000-000000000001"),
        memory_type=MemoryType.SEMANTIC,
        content="A",
        importance=0.8,
        provenance=ProvenanceRecord(source_type=MemorySource.USER_STATEMENT),
        created_at=now,
        updated_at=now,
    )
    r2 = MemoryRecord(
        memory_id=UUID("00000000-0000-0000-0000-000000000002"),
        memory_type=MemoryType.SEMANTIC,
        content="B",
        importance=0.5,
        provenance=ProvenanceRecord(source_type=MemorySource.USER_STATEMENT),
        created_at=now,
        updated_at=now,
    )

    res1 = MemorySearchResult(memory=r1, score=0.8)
    res2 = MemorySearchResult(memory=r2, score=0.8)

    # Both have score 0.8, but r1 has higher importance (0.8 > 0.5)
    sorted_res = sorted([res2, res1], key=ranking_sort_key)
    assert sorted_res[0].memory.memory_id == r1.memory_id

    # Same score and importance, different updated_at
    r3 = MemoryRecord(
        memory_id=UUID("00000000-0000-0000-0000-000000000003"),
        memory_type=MemoryType.SEMANTIC,
        content="C",
        importance=0.8,
        provenance=ProvenanceRecord(source_type=MemorySource.USER_STATEMENT),
        created_at=earlier,
        updated_at=earlier,
    )
    res3 = MemorySearchResult(memory=r3, score=0.8)

    sorted_res2 = sorted([res3, res1], key=ranking_sort_key)
    assert sorted_res2[0].memory.memory_id == r1.memory_id  # newer updated_at first
