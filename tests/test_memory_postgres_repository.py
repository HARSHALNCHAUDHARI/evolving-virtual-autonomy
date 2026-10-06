"""Unit tests for PostgreSQLMemoryRepository using mocked asyncpg connection pool."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import asyncpg
import pytest

from eva.memory.models import (
    DuplicateMemoryError,
    ImmutableMemoryError,
    MemoryNotFoundError,
    MemoryQuery,
    MemoryRecord,
    MemorySource,
    MemoryType,
    ProvenanceRecord,
    RetentionMetadata,
)
from eva.memory.postgres.repository import (
    PostgreSQLMemoryRepository,
    _deserialize_row,
)
from eva.memory.repository import MemoryRepository


def _make_sample_record(
    content: str = "Test persistent memory",
    memory_type: MemoryType = MemoryType.WORKING,
    is_immutable: bool = False,
    is_archived: bool = False,
) -> MemoryRecord:
    prov = ProvenanceRecord(source_type=MemorySource.USER_STATEMENT)
    return MemoryRecord(
        memory_type=memory_type,
        content=content,
        provenance=prov,
        retention_metadata=RetentionMetadata(
            is_immutable=is_immutable,
            is_archived=is_archived,
        ),
    )


def _create_mock_pool_and_conn() -> tuple[MagicMock, AsyncMock]:
    """Create a mock asyncpg.Pool and asyncpg.Connection wired together."""
    mock_conn = AsyncMock()

    # Wire transaction context manager
    @asynccontextmanager
    async def _tx():
        yield

    mock_conn.transaction = MagicMock(side_effect=_tx)

    mock_pool = MagicMock(spec=asyncpg.Pool)

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    mock_pool.acquire = MagicMock(side_effect=_acquire)

    return mock_pool, mock_conn


def test_repository_protocol_conformance() -> None:
    """Verify that PostgreSQLMemoryRepository satisfies the MemoryRepository protocol."""
    mock_pool, _ = _create_mock_pool_and_conn()
    repo = PostgreSQLMemoryRepository(pool=mock_pool)
    assert isinstance(repo, MemoryRepository)


def test_row_deserialization() -> None:
    """Test transforming raw PostgreSQL row data into a typed MemoryRecord."""
    rec_id = uuid4()
    now = datetime(2026, 5, 1, 14, 0, 0, tzinfo=timezone.utc)
    raw_row = {
        "memory_id": rec_id,
        "memory_type": "preference",
        "content": "User prefers dark mode",
        "structured_payload": '{"theme": "dark"}',
        "source": "user_statement",
        "confidence": 0.98,
        "importance": 0.85,
        "relevance": 0.5,
        "project_id": "eva",
        "session_id": "sess-123",
        "provenance": '{"source_type": "user_statement", "created_at": "2026-05-01T14:00:00Z"}',
        "embedding_ref": None,
        "access_metadata": '{"access_count": 3, "created_at": "2026-05-01T14:00:00Z", "updated_at": "2026-05-01T14:00:00Z"}',
        "retention_metadata": '{"is_immutable": true, "is_archived": false}',
        "created_at": now,
        "updated_at": now,
    }

    record = _deserialize_row(raw_row)
    assert record.memory_id == rec_id
    assert record.memory_type == MemoryType.PREFERENCE
    assert record.content == "User prefers dark mode"
    assert record.structured_payload == {"theme": "dark"}
    assert record.source == MemorySource.USER_STATEMENT
    assert record.confidence == 0.98
    assert record.importance == 0.85
    assert record.project_id == "eva"
    assert record.session_id == "sess-123"
    assert record.retention_metadata.is_immutable is True
    assert record.created_at == now
    assert record.updated_at == now


@pytest.mark.asyncio
async def test_save_success() -> None:
    """Verify save executes an INSERT statement with serialized parameters."""
    mock_pool, mock_conn = _create_mock_pool_and_conn()
    repo = PostgreSQLMemoryRepository(pool=mock_pool)
    record = _make_sample_record(content="Persistent note")

    saved = await repo.save(record)
    assert saved.memory_id == record.memory_id
    assert saved.content == "Persistent note"

    assert mock_conn.execute.called
    call_args = mock_conn.execute.call_args[0]
    sql = call_args[0]
    assert "INSERT INTO memory_records" in sql
    assert call_args[1] == record.memory_id
    assert call_args[2] == record.memory_type.value


@pytest.mark.asyncio
async def test_save_duplicate_id_raises() -> None:
    """Duplicate key violations in PostgreSQL must raise DuplicateMemoryError."""
    mock_pool, mock_conn = _create_mock_pool_and_conn()
    mock_conn.execute.side_effect = asyncpg.UniqueViolationError("Duplicate key")

    repo = PostgreSQLMemoryRepository(pool=mock_pool)
    record = _make_sample_record()

    with pytest.raises(DuplicateMemoryError) as exc_info:
        await repo.save(record)
    assert str(record.memory_id) in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_existing_and_nonexistent() -> None:
    """Verify get retrieves matching records or returns None."""
    mock_pool, mock_conn = _create_mock_pool_and_conn()
    repo = PostgreSQLMemoryRepository(pool=mock_pool)
    test_id = uuid4()
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # When record exists
    mock_conn.fetchrow.return_value = {
        "memory_id": test_id,
        "memory_type": "working",
        "content": "Active scratchpad",
        "structured_payload": {},
        "source": "user_statement",
        "confidence": 1.0,
        "importance": 0.5,
        "relevance": 0.0,
        "project_id": None,
        "session_id": None,
        "provenance": {"source_type": "user_statement"},
        "embedding_ref": None,
        "access_metadata": {},
        "retention_metadata": {},
        "created_at": now,
        "updated_at": now,
    }

    result = await repo.get(test_id)
    assert result is not None
    assert result.memory_id == test_id
    assert result.content == "Active scratchpad"

    # When record does not exist
    mock_conn.fetchrow.return_value = None
    assert await repo.get(uuid4()) is None


@pytest.mark.asyncio
async def test_update_existing_and_immutability() -> None:
    """Verify update semantics, created_at preservation, and immutable checks."""
    mock_pool, mock_conn = _create_mock_pool_and_conn()
    repo = PostgreSQLMemoryRepository(pool=mock_pool)

    orig_id = uuid4()
    orig_created = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Mock existing row in database
    mock_conn.fetchrow.return_value = {
        "memory_id": orig_id,
        "memory_type": "working",
        "content": "Before update",
        "structured_payload": {},
        "source": "user_statement",
        "confidence": 1.0,
        "importance": 0.5,
        "relevance": 0.0,
        "project_id": None,
        "session_id": None,
        "provenance": {"source_type": "user_statement"},
        "embedding_ref": None,
        "access_metadata": {},
        "retention_metadata": {"is_immutable": False},
        "created_at": orig_created,
        "updated_at": orig_created,
    }

    update_candidate = _make_sample_record(content="After update")
    update_candidate.memory_id = orig_id
    # Attempt to tamper created_at
    update_candidate.created_at = datetime(2035, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    updated = await repo.update(update_candidate)
    assert updated.memory_id == orig_id
    assert updated.content == "After update"
    assert updated.created_at == orig_created
    assert updated.updated_at > orig_created

    # Test update on missing record raises MemoryNotFoundError
    mock_conn.fetchrow.return_value = None
    with pytest.raises(MemoryNotFoundError):
        await repo.update(update_candidate)

    # Test update on immutable record raises ImmutableMemoryError
    mock_conn.fetchrow.return_value = {
        "memory_id": orig_id,
        "memory_type": "preference",
        "content": "Safety directive",
        "structured_payload": {},
        "source": "user_statement",
        "confidence": 1.0,
        "importance": 1.0,
        "relevance": 0.0,
        "project_id": None,
        "session_id": None,
        "provenance": {"source_type": "user_statement"},
        "embedding_ref": None,
        "access_metadata": {},
        "retention_metadata": {"is_immutable": True},
        "created_at": orig_created,
        "updated_at": orig_created,
    }
    with pytest.raises(ImmutableMemoryError):
        await repo.update(update_candidate)


@pytest.mark.asyncio
async def test_delete_existing_and_missing() -> None:
    """Verify delete operation and missing error handling."""
    mock_pool, mock_conn = _create_mock_pool_and_conn()
    repo = PostgreSQLMemoryRepository(pool=mock_pool)
    target_id = uuid4()

    # Successfully deleted
    mock_conn.execute.return_value = "DELETE 1"
    assert await repo.delete(target_id) is True

    # Missing record without raise
    mock_conn.execute.return_value = "DELETE 0"
    assert await repo.delete(target_id, raise_if_missing=False) is False

    # Missing record with raise_if_missing=True
    with pytest.raises(MemoryNotFoundError):
        await repo.delete(target_id, raise_if_missing=True)


@pytest.mark.asyncio
async def test_query_parameterized_sql_generation() -> None:
    """Verify query compiles all filters into parameterized SQL clauses."""
    mock_pool, mock_conn = _create_mock_pool_and_conn()
    mock_conn.fetch.return_value = []
    repo = PostgreSQLMemoryRepository(pool=mock_pool)

    t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

    query = MemoryQuery(
        memory_types=[MemoryType.PREFERENCE, MemoryType.PROJECT_CONTEXT],
        project_id="eva_project",
        include_global=True,
        session_id="session_alpha",
        sources=[MemorySource.USER_STATEMENT],
        min_confidence=0.85,
        min_importance=0.6,
        created_after=t0,
        created_before=t1,
        is_archived=False,
        limit=5,
    )

    await repo.query(query)

    assert mock_conn.fetch.called
    call_args = mock_conn.fetch.call_args[0]
    sql = call_args[0]
    params = call_args[1:]

    # SQL clauses check
    assert "memory_type = ANY($1::varchar[])" in sql
    assert "(project_id = $2 OR project_id IS NULL)" in sql
    assert "session_id = $3" in sql
    assert "source = ANY($4::varchar[])" in sql
    assert "confidence >= $5" in sql
    assert "importance >= $6" in sql
    assert "created_at >= $7" in sql
    assert "created_at <= $8" in sql
    assert "((retention_metadata->>'is_archived')::boolean = $9)" in sql
    assert "ORDER BY created_at DESC, memory_id ASC" in sql
    assert "LIMIT $10" in sql

    # Parameters check
    assert params[0] == ["preference", "project_context"]
    assert params[1] == "eva_project"
    assert params[2] == "session_alpha"
    assert params[3] == ["user_statement"]
    assert params[4] == 0.85
    assert params[5] == 0.6
    assert params[6] == t0
    assert params[7] == t1
    assert params[8] is False
    assert params[9] == 5
