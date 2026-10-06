"""Optional live integration tests for PostgreSQLMemoryRepository.

These tests are executed only when a live PostgreSQL instance is available
via EVA_POSTGRES_DSN or default localhost. If PostgreSQL is unreachable,
the tests skip gracefully.
"""

import os
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg
import pytest

from eva.memory.models import (
    MemoryQuery,
    MemoryRecord,
    MemorySource,
    MemoryType,
    ProvenanceRecord,
    RetentionMetadata,
)
from eva.memory.postgres.config import PostgresConfig
from eva.memory.postgres.connection import PostgresPoolManager
from eva.memory.postgres.migration import run_migrations_down, run_migrations_up
from eva.memory.postgres.repository import PostgreSQLMemoryRepository

# pytest marker for integration
pytestmark = pytest.mark.integration


async def _get_live_pool() -> asyncpg.Pool | None:
    """Attempt to establish a live connection pool for testing."""
    config = PostgresConfig.from_env()
    try:
        pool = await asyncpg.create_pool(
            dsn=config.effective_dsn,
            min_size=1,
            max_size=3,
            timeout=1.5,
        )
        # Verify connection
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return pool
    except Exception:
        return None


@pytest.mark.asyncio
async def test_live_postgres_crud_and_query() -> None:
    """Execute live CRUD and query operations against PostgreSQL if available."""
    pool = await _get_live_pool()
    if pool is None:
        pytest.skip("Live PostgreSQL database is not reachable. Skipping integration test.")

    try:
        # Run migrations UP
        await run_migrations_up(pool)

        repo = PostgreSQLMemoryRepository(pool=pool)

        # 1. Save record
        prov = ProvenanceRecord(source_type=MemorySource.USER_STATEMENT)
        record = MemoryRecord(
            memory_type=MemoryType.PREFERENCE,
            content="User prefers Python asyncpg",
            provenance=prov,
            project_id="integration_test",
            retention_metadata=RetentionMetadata(is_immutable=False),
        )

        saved = await repo.save(record)
        assert saved.memory_id == record.memory_id

        # 2. Get record
        retrieved = await repo.get(record.memory_id)
        assert retrieved is not None
        assert retrieved.content == "User prefers Python asyncpg"

        # 3. Update record
        retrieved.content = "User prefers Python asyncpg 0.31"
        updated = await repo.update(retrieved)
        assert updated.content == "User prefers Python asyncpg 0.31"
        assert updated.created_at == record.created_at

        # 4. Query record
        query = MemoryQuery(
            project_id="integration_test",
            memory_types=[MemoryType.PREFERENCE],
        )
        results = await repo.query(query)
        assert len(results) >= 1
        assert any(r.memory_id == record.memory_id for r in results)

        # 5. Delete record
        deleted = await repo.delete(record.memory_id)
        assert deleted is True
        assert await repo.get(record.memory_id) is None

    finally:
        # Clean up
        try:
            await run_migrations_down(pool)
        except Exception:
            pass
        await pool.close()
