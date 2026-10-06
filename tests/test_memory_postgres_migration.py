"""Unit tests for PostgreSQL memory schema migration files and runner."""

import pytest

from eva.memory.models import DatabaseMigrationError
from eva.memory.postgres.migration import get_migration_sql


def test_migration_up_file_contents() -> None:
    """Verify the UP migration file structure, required columns, and indexes."""
    sql = get_migration_sql("001_create_memory_records.sql")

    # Table creation and primary key
    assert "CREATE TABLE IF NOT EXISTS memory_records" in sql
    assert "memory_id UUID PRIMARY KEY" in sql

    # Core required columns
    assert "memory_type VARCHAR(64) NOT NULL" in sql
    assert "content TEXT NOT NULL" in sql
    assert "structured_payload JSONB NOT NULL" in sql
    assert "source VARCHAR(64) NOT NULL" in sql
    assert "confidence DOUBLE PRECISION NOT NULL" in sql
    assert "importance DOUBLE PRECISION NOT NULL" in sql
    assert "relevance DOUBLE PRECISION NOT NULL" in sql
    assert "project_id VARCHAR(255)" in sql
    assert "session_id VARCHAR(255)" in sql
    assert "provenance JSONB NOT NULL" in sql
    assert "embedding_ref JSONB" in sql
    assert "access_metadata JSONB NOT NULL" in sql
    assert "retention_metadata JSONB NOT NULL" in sql
    assert "created_at TIMESTAMPTZ NOT NULL" in sql
    assert "updated_at TIMESTAMPTZ NOT NULL" in sql

    # Indexes
    assert "CREATE INDEX IF NOT EXISTS idx_memory_records_type" in sql
    assert "CREATE INDEX IF NOT EXISTS idx_memory_records_project" in sql
    assert "CREATE INDEX IF NOT EXISTS idx_memory_records_session" in sql
    assert "CREATE INDEX IF NOT EXISTS idx_memory_records_source" in sql
    assert "CREATE INDEX IF NOT EXISTS idx_memory_records_created_at" in sql
    assert "CREATE INDEX IF NOT EXISTS idx_memory_records_updated_at" in sql

    # Strict constraint: pgvector must NOT be present in Milestone 1E.3
    assert "vector" not in sql.lower()
    assert "pgvector" not in sql.lower()


def test_migration_down_file_contents() -> None:
    """Verify the DOWN migration file drops the table cleanly."""
    sql = get_migration_sql("001_create_memory_records_down.sql")
    assert "DROP TABLE IF EXISTS memory_records CASCADE" in sql


def test_missing_migration_raises_error() -> None:
    """Requesting an invalid migration filename raises DatabaseMigrationError."""
    with pytest.raises(DatabaseMigrationError) as exc_info:
        get_migration_sql("nonexistent_migration.sql")
    assert "not found" in str(exc_info.value).lower()
