"""PostgreSQL schema migration runner for the EVA memory subsystem.

Provides programmatic schema execution for 001_create_memory_records.sql (UP)
and 001_create_memory_records_down.sql (DOWN).
"""

from __future__ import annotations

import logging
from pathlib import Path

import asyncpg

from eva.memory.models import DatabaseMigrationError

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "migrations"


MIGRATIONS_UP = [
    "001_create_memory_records.sql",
    "002_add_pgvector_embeddings.sql",
]

MIGRATIONS_DOWN = [
    "002_add_pgvector_embeddings_down.sql",
    "001_create_memory_records_down.sql",
]


def get_migration_sql(filename: str) -> str:
    """Read migration SQL script from the migrations directory."""
    filepath = MIGRATIONS_DIR / filename
    if not filepath.exists():
        raise DatabaseMigrationError(f"Migration file not found: {filepath}")
    return filepath.read_text(encoding="utf-8")


async def run_migrations_up(target: asyncpg.Connection | asyncpg.Pool) -> None:
    """Execute all UP migrations in chronological order."""
    for filename in MIGRATIONS_UP:
        sql = get_migration_sql(filename)
        try:
            if isinstance(target, asyncpg.Pool):
                async with target.acquire() as conn:
                    await conn.execute(sql)
            else:
                await target.execute(sql)
            logger.info("Successfully executed migration UP: %s", filename)
        except Exception as exc:
            logger.error("Failed to execute UP migration '%s': %s", filename, exc)
            raise DatabaseMigrationError(f"Failed to execute UP migration '{filename}': {exc}") from exc


async def run_migrations_down(target: asyncpg.Connection | asyncpg.Pool) -> None:
    """Execute all DOWN migrations in reverse order."""
    for filename in MIGRATIONS_DOWN:
        sql = get_migration_sql(filename)
        try:
            if isinstance(target, asyncpg.Pool):
                async with target.acquire() as conn:
                    await conn.execute(sql)
            else:
                await target.execute(sql)
            logger.info("Successfully executed migration DOWN: %s", filename)
        except Exception as exc:
            logger.error("Failed to execute DOWN migration '%s': %s", filename, exc)
            raise DatabaseMigrationError(f"Failed to execute DOWN migration '{filename}': {exc}") from exc


async def is_table_created(target: asyncpg.Connection | asyncpg.Pool) -> bool:
    """Check if the memory_records table exists in the target database."""
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = 'memory_records'
        );
    """
    try:
        if isinstance(target, asyncpg.Pool):
            async with target.acquire() as conn:
                res = await conn.fetchval(query)
        else:
            res = await target.fetchval(query)
        return bool(res)
    except Exception:
        return False
