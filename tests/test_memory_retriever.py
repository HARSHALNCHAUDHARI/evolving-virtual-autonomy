"""Comprehensive offline unit tests for DeterministicMemoryRetriever."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from eva.memory.in_memory import InMemoryMemoryRepository
from eva.memory.models import (
    InvalidMemoryQueryError,
    MemoryQuery,
    MemoryRecord,
    MemoryRetrievalError,
    MemorySource,
    MemoryType,
    ProvenanceRecord,
)
from eva.memory.retrieval import (
    DeterministicMemoryRetriever,
    MemoryRetriever,
)


@pytest.fixture
def repository() -> InMemoryMemoryRepository:
    return InMemoryMemoryRepository()


@pytest.fixture
def retriever(repository: InMemoryMemoryRepository) -> DeterministicMemoryRetriever:
    return DeterministicMemoryRetriever(repository=repository)


def _make_record(
    content: str,
    *,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    project_id: str | None = None,
    session_id: str | None = None,
    importance: float = 0.5,
    confidence: float = 1.0,
    source: MemorySource = MemorySource.USER_STATEMENT,
    payload: dict | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    c_time = created_at or now
    u_time = updated_at or c_time
    return MemoryRecord(
        memory_type=memory_type,
        content=content,
        project_id=project_id,
        session_id=session_id,
        importance=importance,
        confidence=confidence,
        structured_payload=payload or {},
        provenance=ProvenanceRecord(source_type=source),
        created_at=c_time,
        updated_at=u_time,
    )


def test_retriever_protocol_conformance(retriever: DeterministicMemoryRetriever) -> None:
    assert isinstance(retriever, MemoryRetriever)


@pytest.mark.asyncio
async def test_exact_phrase_ranks_above_partial_tokens(
    repository: InMemoryMemoryRepository,
    retriever: DeterministicMemoryRetriever,
) -> None:
    now = datetime.now(timezone.utc)
    # Record A has exact phrase "database connection pool"
    r_exact = _make_record(
        "Always reuse the database connection pool across worker tasks.",
        importance=0.5,
        updated_at=now,
    )
    # Record B has tokens but separated / not exact phrase
    r_partial = _make_record(
        "A pool of threads connects to the database.",
        importance=0.5,
        updated_at=now,
    )
    await repository.save(r_exact)
    await repository.save(r_partial)

    query = MemoryQuery(query_text="database connection pool")
    results = await retriever.search(query)

    assert len(results) == 2
    assert results[0].memory.memory_id == r_exact.memory_id
    assert results[0].explanation.exact_phrase_matched is True
    assert results[1].explanation.exact_phrase_matched is False
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_case_and_whitespace_normalization(
    repository: InMemoryMemoryRepository,
    retriever: DeterministicMemoryRetriever,
) -> None:
    r = _make_record("Use PostgreSQL for durable relational storage.")
    await repository.save(r)

    # Query with messy whitespace and uppercase
    query = MemoryQuery(query_text="   POSTGRESQL    durable   storage  ")
    results = await retriever.search(query)

    assert len(results) == 1
    assert results[0].memory.memory_id == r.memory_id
    assert "postgresql" in results[0].explanation.matched_tokens


@pytest.mark.asyncio
async def test_no_match_returns_empty_list(
    repository: InMemoryMemoryRepository,
    retriever: DeterministicMemoryRetriever,
) -> None:
    await repository.save(_make_record("Frontend uses TailwindCSS and Vite."))

    query = MemoryQuery(query_text="quantum computing physics")
    results = await retriever.search(query)
    assert len(results) == 0

    count = await retriever.count(query)
    assert count == 0


@pytest.mark.asyncio
async def test_memory_type_filtering(
    repository: InMemoryMemoryRepository,
    retriever: DeterministicMemoryRetriever,
) -> None:
    r_pref = _make_record("Prefers vim keybindings", memory_type=MemoryType.PREFERENCE)
    r_sem = _make_record("EVA architecture overview", memory_type=MemoryType.SEMANTIC)
    await repository.save(r_pref)
    await repository.save(r_sem)

    query = MemoryQuery(memory_types=[MemoryType.PREFERENCE])
    results = await retriever.search(query)

    assert len(results) == 1
    assert results[0].memory.memory_id == r_pref.memory_id


@pytest.mark.asyncio
async def test_project_isolation(
    repository: InMemoryMemoryRepository,
    retriever: DeterministicMemoryRetriever,
) -> None:
    r_eva = _make_record("EVA roadmap milestone 1E", project_id="eva")
    r_global = _make_record("User email is user@example.com", project_id=None)
    r_portal = _make_record("Portal backend runs on port 8080", project_id="portal")

    await repository.save(r_eva)
    await repository.save(r_global)
    await repository.save(r_portal)

    # 1. Query within "eva" project with include_global=True (default)
    q_eva = MemoryQuery(project_id="eva", include_global=True)
    res_eva = await retriever.search(q_eva)
    eva_ids = {r.memory.memory_id for r in res_eva}

    assert r_eva.memory_id in eva_ids
    assert r_global.memory_id in eva_ids
    # "portal" memory must NEVER be visible
    assert r_portal.memory_id not in eva_ids

    # 2. Strict project query without global memories
    q_strict = MemoryQuery(project_id="eva", include_global=False)
    res_strict = await retriever.search(q_strict)
    strict_ids = {r.memory.memory_id for r in res_strict}

    assert strict_ids == {r_eva.memory_id}


@pytest.mark.asyncio
async def test_session_isolation(
    repository: InMemoryMemoryRepository,
    retriever: DeterministicMemoryRetriever,
) -> None:
    r_s1 = _make_record("User asked for test script", session_id="sess-001")
    r_s2 = _make_record("User asked for deployment docs", session_id="sess-002")
    await repository.save(r_s1)
    await repository.save(r_s2)

    query = MemoryQuery(session_id="sess-001")
    results = await retriever.search(query)

    assert len(results) == 1
    assert results[0].memory.memory_id == r_s1.memory_id


@pytest.mark.asyncio
async def test_provenance_and_confidence_filtering(
    repository: InMemoryMemoryRepository,
    retriever: DeterministicMemoryRetriever,
) -> None:
    r_high = _make_record(
        "Verified system fact",
        confidence=0.95,
        source=MemorySource.TOOL_RESULT,
    )
    r_low = _make_record(
        "Unverified rumor",
        confidence=0.40,
        source=MemorySource.MODEL_INFERENCE,
    )
    await repository.save(r_high)
    await repository.save(r_low)

    query = MemoryQuery(
        sources=[MemorySource.TOOL_RESULT],
        min_confidence=0.8,
    )
    results = await retriever.search(query)

    assert len(results) == 1
    assert results[0].memory.memory_id == r_high.memory_id


@pytest.mark.asyncio
async def test_importance_filtering(
    repository: InMemoryMemoryRepository,
    retriever: DeterministicMemoryRetriever,
) -> None:
    r_vital = _make_record("Critical safety invariant", importance=0.95)
    r_trivial = _make_record("Temporary scratchpad note", importance=0.10)
    await repository.save(r_vital)
    await repository.save(r_trivial)

    query = MemoryQuery(min_importance=0.8)
    results = await retriever.search(query)

    assert len(results) == 1
    assert results[0].memory.memory_id == r_vital.memory_id


@pytest.mark.asyncio
async def test_temporal_filtering(
    repository: InMemoryMemoryRepository,
    retriever: DeterministicMemoryRetriever,
) -> None:
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 2, 1, tzinfo=timezone.utc)
    t3 = datetime(2026, 3, 1, tzinfo=timezone.utc)

    r1 = _make_record("Memory Jan", created_at=t1, updated_at=t1)
    r2 = _make_record("Memory Feb", created_at=t2, updated_at=t2)
    r3 = _make_record("Memory Mar", created_at=t3, updated_at=t3)

    await repository.save(r1)
    await repository.save(r2)
    await repository.save(r3)

    # Range [Jan 15, Feb 15]
    query = MemoryQuery(
        created_after=datetime(2026, 1, 15, tzinfo=timezone.utc),
        created_before=datetime(2026, 2, 15, tzinfo=timezone.utc),
    )
    results = await retriever.search(query)

    assert len(results) == 1
    assert results[0].memory.memory_id == r2.memory_id


@pytest.mark.asyncio
async def test_metadata_payload_filtering(
    repository: InMemoryMemoryRepository,
    retriever: DeterministicMemoryRetriever,
) -> None:
    r1 = _make_record("Config A", payload={"env": "prod", "cluster": "us-east"})
    r2 = _make_record("Config B", payload={"env": "dev", "cluster": "us-east"})
    await repository.save(r1)
    await repository.save(r2)

    query = MemoryQuery(metadata_filters={"env": "prod"})
    results = await retriever.search(query)

    assert len(results) == 1
    assert results[0].memory.memory_id == r1.memory_id


@pytest.mark.asyncio
async def test_pagination_limit_and_offset(
    repository: InMemoryMemoryRepository,
    retriever: DeterministicMemoryRetriever,
) -> None:
    for i in range(10):
        await repository.save(_make_record(f"Record item number {i}", importance=i / 10.0))

    query = MemoryQuery(query_text="record item", limit=3, offset=2)
    results = await retriever.search(query)

    assert len(results) == 3
    # Total count across candidates
    total = await retriever.count(query)
    assert total == 10


@pytest.mark.asyncio
async def test_search_with_response_container(
    repository: InMemoryMemoryRepository,
    retriever: DeterministicMemoryRetriever,
) -> None:
    await repository.save(_make_record("Memory entry"))

    query = MemoryQuery(query_text="entry")
    resp = await retriever.search_with_response(query)

    assert resp.total_candidates == 1
    assert len(resp.results) == 1
    assert resp.query == query


@pytest.mark.asyncio
async def test_invalid_query_parameters_raise_error(
    retriever: DeterministicMemoryRetriever,
) -> None:
    query_bad_offset = MemoryQuery()
    query_bad_offset.offset = -5
    with pytest.raises(InvalidMemoryQueryError):
        await retriever.search(query_bad_offset)

    query_bad_limit = MemoryQuery()
    query_bad_limit.limit = -1
    with pytest.raises(InvalidMemoryQueryError):
        await retriever.search(query_bad_limit)


@pytest.mark.asyncio
async def test_repository_failure_wrapped_in_memory_retrieval_error() -> None:
    broken_repo = AsyncMock()
    broken_repo.query.side_effect = RuntimeError("Database connection timed out")

    bad_retriever = DeterministicMemoryRetriever(repository=broken_repo)

    with pytest.raises(MemoryRetrievalError, match="Failed to retrieve memory candidates"):
        await bad_retriever.search(MemoryQuery(query_text="test"))
