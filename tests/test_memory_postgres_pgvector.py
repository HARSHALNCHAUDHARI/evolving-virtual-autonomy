"""Tests for PostgreSQL pgvector schema migrations and vector query generation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import asyncpg
import pytest

from eva.memory.models import (
    MemoryQuery,
    MemoryRecord,
    MemorySource,
    MemoryType,
    ProvenanceRecord,
)
from eva.memory.postgres.migration import (
    MIGRATIONS_DOWN,
    MIGRATIONS_UP,
    run_migrations_down,
    run_migrations_up,
)
from eva.memory.postgres.repository import PostgreSQLMemoryRepository, _deserialize_row


def _create_mock_pool_and_conn() -> tuple[MagicMock, AsyncMock]:
    """Create a mock asyncpg.Pool and asyncpg.Connection wired together."""
    mock_conn = AsyncMock()

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


def test_migration_files_exist_and_valid() -> None:
    """Migration files for 002 pgvector extension and embedding column exist and are structured properly."""
    migration_dir = Path(__file__).parent.parent / "migrations"
    up_path = migration_dir / "002_add_pgvector_embeddings.sql"
    down_path = migration_dir / "002_add_pgvector_embeddings_down.sql"

    assert up_path.is_file(), "002_add_pgvector_embeddings.sql must exist"
    assert down_path.is_file(), "002_add_pgvector_embeddings_down.sql must exist"

    up_sql = up_path.read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector;" in up_sql
    assert "ADD COLUMN IF NOT EXISTS embedding vector" in up_sql
    # Preserves existing records safely
    assert "ALTER TABLE memory_records" in up_sql

    down_sql = down_path.read_text(encoding="utf-8")
    assert "DROP COLUMN IF EXISTS embedding" in down_sql

    # Check migration steps registry
    assert MIGRATIONS_UP == [
        "001_create_memory_records.sql",
        "002_add_pgvector_embeddings.sql",
    ]
    assert MIGRATIONS_DOWN == [
        "002_add_pgvector_embeddings_down.sql",
        "001_create_memory_records_down.sql",
    ]


def test_deserialize_row_with_embedding() -> None:
    """_deserialize_row correctly unpacks string and list embedding representations."""
    mem_id = uuid4()
    now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)
    mock_row = {
        "memory_id": mem_id,
        "memory_type": "episodic",
        "content": "Test content with embedding vector.",
        "structured_payload": "{}",
        "source": "system_event",
        "confidence": 0.9,
        "importance": 0.5,
        "relevance": 0.0,
        "project_id": "proj-123",
        "session_id": "sess-456",
        "provenance": json.dumps({"source_type": "system_event"}),
        "embedding_ref": json.dumps({"model": "test-emb", "dimensions": 3, "version": 1}),
        "embedding": "[0.15, -0.42, 0.98]",
        "access_metadata": "{}",
        "retention_metadata": "{}",
        "created_at": now,
        "updated_at": now,
    }

    record = _deserialize_row(mock_row)
    assert record.memory_id == mem_id
    assert record.embedding is not None
    assert len(record.embedding) == 3
    assert record.embedding == pytest.approx([0.15, -0.42, 0.98])
    assert record.embedding_ref is not None
    assert record.embedding_ref.model == "test-emb"


@pytest.mark.asyncio
async def test_postgres_repository_search_vector_sql_generation() -> None:
    """PostgreSQLMemoryRepository.search_vector constructs parameterized SQL with cosine distance."""
    mock_pool, mock_conn = _create_mock_pool_and_conn()

    mem_id = uuid4()
    now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)
    mock_row = {
        "memory_id": mem_id,
        "memory_type": "semantic",
        "content": "Vector query target match.",
        "structured_payload": "{}",
        "source": "model_inference",
        "confidence": 1.0,
        "importance": 0.8,
        "relevance": 0.0,
        "project_id": "project-omega",
        "session_id": None,
        "provenance": json.dumps({"source_type": "model_inference"}),
        "embedding_ref": None,
        "embedding": "[0.1, 0.2, 0.3]",
        "access_metadata": "{}",
        "retention_metadata": "{}",
        "created_at": now,
        "updated_at": now,
        "similarity": 0.875,
    }
    mock_conn.fetch.return_value = [mock_row]

    repo = PostgreSQLMemoryRepository(pool=mock_pool)

    query = MemoryQuery(
        query_text="omega",
        project_id="project-omega",
        min_importance=0.5,
        limit=5,
    )
    query_vector = [0.1, 0.2, 0.3]

    results = await repo.search_vector(
        query_vector=query_vector,
        query=query,
        limit=5,
        similarity_threshold=0.5,
    )

    assert len(results) == 1
    matched_record, sim = results[0]
    assert matched_record.memory_id == mem_id
    assert sim == 0.875

    # Verify SQL and parameters sent to fetch
    mock_conn.fetch.assert_called_once()
    sql_arg = mock_conn.fetch.call_args[0][0]
    params_args = mock_conn.fetch.call_args[0][1:]

    assert "SELECT *, (1 - (embedding <=> $1::vector)) AS similarity" in sql_arg
    assert "embedding IS NOT NULL" in sql_arg
    assert "embedding <=> $1::vector ASC" in sql_arg
    assert params_args[0] == "[0.1,0.2,0.3]"
