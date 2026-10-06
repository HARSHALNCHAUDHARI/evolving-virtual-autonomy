"""PostgreSQL persistent implementation of MemoryRepository.

Provides transactional persistence, parameterized query building, and contract
compatibility with InMemoryMemoryRepository for the EVA memory subsystem.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

import asyncpg

from eva.memory.models import (
    AccessMetadata,
    DuplicateMemoryError,
    EmbeddingReference,
    ImmutableMemoryError,
    MemoryNotFoundError,
    MemoryQuery,
    MemoryRecord,
    MemorySource,
    MemoryType,
    ProvenanceRecord,
    RetentionMetadata,
    _ensure_timezone,
    _utcnow,
)
from eva.memory.postgres.config import PostgresConfig
from eva.memory.postgres.connection import PostgresPoolManager
from eva.memory.repository import MemoryRepository

logger = logging.getLogger(__name__)


def _deserialize_row(row: asyncpg.Record | dict[str, Any]) -> MemoryRecord:
    """Transform a PostgreSQL query row into a typed MemoryRecord."""
    r = dict(row)

    payload = r.get("structured_payload")
    if isinstance(payload, str):
        payload = json.loads(payload)
    elif payload is None:
        payload = {}

    prov_raw = r.get("provenance")
    if isinstance(prov_raw, str):
        prov_raw = json.loads(prov_raw)
    provenance = ProvenanceRecord.model_validate(prov_raw)

    emb_raw = r.get("embedding_ref")
    if isinstance(emb_raw, str):
        emb_raw = json.loads(emb_raw)
    embedding_ref = EmbeddingReference.model_validate(emb_raw) if emb_raw else None

    access_raw = r.get("access_metadata")
    if isinstance(access_raw, str):
        access_raw = json.loads(access_raw)
    access_metadata = (
        AccessMetadata.model_validate(access_raw) if access_raw else AccessMetadata()
    )

    ret_raw = r.get("retention_metadata")
    if isinstance(ret_raw, str):
        ret_raw = json.loads(ret_raw)
    retention_metadata = (
        RetentionMetadata.model_validate(ret_raw) if ret_raw else RetentionMetadata()
    )

    created_at = _ensure_timezone(r.get("created_at")) or _utcnow()
    updated_at = _ensure_timezone(r.get("updated_at")) or _utcnow()

    memory_id = r["memory_id"]
    if not isinstance(memory_id, UUID):
        memory_id = UUID(str(memory_id))

    return MemoryRecord(
        memory_id=memory_id,
        memory_type=MemoryType(r["memory_type"]),
        content=r["content"],
        structured_payload=payload,
        source=MemorySource(r["source"]) if r.get("source") else None,
        provenance=provenance,
        created_at=created_at,
        updated_at=updated_at,
        confidence=float(r["confidence"]),
        importance=float(r["importance"]),
        relevance=float(r["relevance"]),
        project_id=r.get("project_id"),
        session_id=r.get("session_id"),
        embedding_ref=embedding_ref,
        access_metadata=access_metadata,
        retention_metadata=retention_metadata,
    )


class PostgreSQLMemoryRepository(MemoryRepository):
    """PostgreSQL implementation of the MemoryRepository contract.

    Stores memory records in PostgreSQL using parameterized queries and JSONB
    metadata containers. Preserves full behavioural compatibility with
    InMemoryMemoryRepository.
    """

    def __init__(
        self,
        pool: asyncpg.Pool | PostgresPoolManager | None = None,
        config: PostgresConfig | None = None,
    ) -> None:
        self._pool: asyncpg.Pool | None = None
        self._pool_manager: PostgresPoolManager | None = None

        if isinstance(pool, asyncpg.Pool):
            self._pool = pool
        elif isinstance(pool, PostgresPoolManager):
            self._pool_manager = pool
        else:
            self._pool_manager = PostgresPoolManager(config=config or PostgresConfig())

    async def _get_pool(self) -> asyncpg.Pool:
        """Retrieve or initialize the active asyncpg.Pool."""
        if self._pool is not None:
            return self._pool
        if self._pool_manager is not None:
            self._pool = await self._pool_manager.start()
            return self._pool
        raise RuntimeError("PostgreSQLMemoryRepository has neither pool nor pool manager.")

    async def start(self) -> None:
        """Start the backing pool manager if configured."""
        await self._get_pool()

    async def stop(self) -> None:
        """Stop the backing pool manager if configured."""
        if self._pool_manager is not None:
            await self._pool_manager.stop()
            self._pool = None

    async def save(self, memory: MemoryRecord) -> MemoryRecord:
        """Persist a new memory record into PostgreSQL.

        Args:
            memory: The MemoryRecord to store.

        Returns:
            A deep copy of the stored MemoryRecord.

        Raises:
            DuplicateMemoryError: If a record with memory.memory_id already exists.
        """
        pool = await self._get_pool()
        source_val = (memory.source or memory.provenance.source_type).value

        query = """
            INSERT INTO memory_records (
                memory_id, memory_type, content, structured_payload, source,
                confidence, importance, relevance, project_id, session_id,
                provenance, embedding_ref, access_metadata, retention_metadata,
                created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4::jsonb, $5,
                $6, $7, $8, $9, $10,
                $11::jsonb, $12::jsonb, $13::jsonb, $14::jsonb,
                $15, $16
            )
        """

        emb_json = (
            json.dumps(memory.embedding_ref.model_dump(mode="json"))
            if memory.embedding_ref
            else None
        )

        params = (
            memory.memory_id,
            memory.memory_type.value,
            memory.content,
            json.dumps(memory.structured_payload),
            source_val,
            memory.confidence,
            memory.importance,
            memory.relevance,
            memory.project_id,
            memory.session_id,
            json.dumps(memory.provenance.model_dump(mode="json")),
            emb_json,
            json.dumps(memory.access_metadata.model_dump(mode="json")),
            json.dumps(memory.retention_metadata.model_dump(mode="json")),
            memory.created_at,
            memory.updated_at,
        )

        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(query, *params)
            return memory.model_copy(deep=True)
        except asyncpg.UniqueViolationError as exc:
            raise DuplicateMemoryError(
                f"Memory record with ID '{memory.memory_id}' already exists."
            ) from exc

    async def get(self, memory_id: UUID) -> MemoryRecord | None:
        """Retrieve a memory record by ID.

        Args:
            memory_id: The UUID of the record.

        Returns:
            The matching MemoryRecord, or None if not found.
        """
        pool = await self._get_pool()
        query = "SELECT * FROM memory_records WHERE memory_id = $1"

        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, memory_id)

        if row is None:
            return None
        return _deserialize_row(row)

    async def update(self, memory: MemoryRecord) -> MemoryRecord:
        """Update an existing memory record in PostgreSQL.

        Preserves original memory_id and created_at timestamps. Updates updated_at.

        Args:
            memory: The modified MemoryRecord with an existing memory_id.

        Returns:
            A deep copy of the updated MemoryRecord.

        Raises:
            MemoryNotFoundError: If no record exists with memory.memory_id.
            ImmutableMemoryError: If the existing record is immutable.
        """
        pool = await self._get_pool()

        select_query = "SELECT * FROM memory_records WHERE memory_id = $1 FOR UPDATE"
        update_query = """
            UPDATE memory_records SET
                memory_type = $2,
                content = $3,
                structured_payload = $4::jsonb,
                source = $5,
                confidence = $6,
                importance = $7,
                relevance = $8,
                project_id = $9,
                session_id = $10,
                provenance = $11::jsonb,
                embedding_ref = $12::jsonb,
                access_metadata = $13::jsonb,
                retention_metadata = $14::jsonb,
                updated_at = $15
            WHERE memory_id = $1
        """

        now = _utcnow()

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(select_query, memory.memory_id)
                if row is None:
                    raise MemoryNotFoundError(
                        f"Memory record with ID '{memory.memory_id}' not found."
                    )

                existing = _deserialize_row(row)
                if existing.retention_metadata.is_immutable:
                    raise ImmutableMemoryError(
                        f"Memory record '{memory.memory_id}' is immutable and cannot be updated."
                    )

                updated = memory.model_copy(deep=True)
                updated.created_at = existing.created_at
                updated.access_metadata.created_at = existing.access_metadata.created_at
                updated.updated_at = now
                updated.access_metadata.updated_at = now

                source_val = (updated.source or updated.provenance.source_type).value
                emb_json = (
                    json.dumps(updated.embedding_ref.model_dump(mode="json"))
                    if updated.embedding_ref
                    else None
                )

                params = (
                    updated.memory_id,
                    updated.memory_type.value,
                    updated.content,
                    json.dumps(updated.structured_payload),
                    source_val,
                    updated.confidence,
                    updated.importance,
                    updated.relevance,
                    updated.project_id,
                    updated.session_id,
                    json.dumps(updated.provenance.model_dump(mode="json")),
                    emb_json,
                    json.dumps(updated.access_metadata.model_dump(mode="json")),
                    json.dumps(updated.retention_metadata.model_dump(mode="json")),
                    now,
                )

                await conn.execute(update_query, *params)
                return updated

    async def delete(self, memory_id: UUID, *, raise_if_missing: bool = False) -> bool:
        """Delete a memory record by ID.

        Args:
            memory_id: The UUID of the record to delete.
            raise_if_missing: If True, raises MemoryNotFoundError when missing.

        Returns:
            True if the record was deleted, False if not found.

        Raises:
            MemoryNotFoundError: If raise_if_missing is True and the record was not found.
        """
        pool = await self._get_pool()
        query = "DELETE FROM memory_records WHERE memory_id = $1"

        async with pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(query, memory_id)

        # Result format is usually "DELETE <n>"
        count = 0
        if result and isinstance(result, str):
            parts = result.strip().split()
            if parts and parts[-1].isdigit():
                count = int(parts[-1])

        if count > 0:
            return True

        if raise_if_missing:
            raise MemoryNotFoundError(
                f"Memory record with ID '{memory_id}' not found."
            )
        return False

    async def query(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Query memory records using parameterized SQL with AND semantics.

        Args:
            query: The MemoryQuery containing filter criteria.

        Returns:
            A list of matching MemoryRecords ordered deterministically.
        """
        pool = await self._get_pool()
        clauses: list[str] = []
        params: list[Any] = []

        if query.memory_types:
            params.append([m.value for m in query.memory_types])
            clauses.append(f"memory_type = ANY(${len(params)}::varchar[])")

        if query.project_id is not None:
            params.append(query.project_id)
            if query.include_global:
                clauses.append(f"(project_id = ${len(params)} OR project_id IS NULL)")
            else:
                clauses.append(f"project_id = ${len(params)}")

        if query.session_id is not None:
            params.append(query.session_id)
            clauses.append(f"session_id = ${len(params)}")

        if query.sources:
            params.append([s.value for s in query.sources])
            clauses.append(f"source = ANY(${len(params)}::varchar[])")

        if query.min_confidence is not None:
            params.append(query.min_confidence)
            clauses.append(f"confidence >= ${len(params)}")

        if query.min_importance is not None:
            params.append(query.min_importance)
            clauses.append(f"importance >= ${len(params)}")

        if query.created_after is not None:
            params.append(query.created_after)
            clauses.append(f"created_at >= ${len(params)}")

        if query.created_before is not None:
            params.append(query.created_before)
            clauses.append(f"created_at <= ${len(params)}")

        if query.updated_after is not None:
            params.append(query.updated_after)
            clauses.append(f"updated_at >= ${len(params)}")

        if query.updated_before is not None:
            params.append(query.updated_before)
            clauses.append(f"updated_at <= ${len(params)}")

        if query.is_archived is not None:
            params.append(query.is_archived)
            clauses.append(f"((retention_metadata->>'is_archived')::boolean = ${len(params)})")

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        limit_sql = ""
        if query.limit is not None:
            params.append(query.limit)
            limit_sql = f"LIMIT ${len(params)}"

        sql = (
            f"SELECT * FROM memory_records {where_sql} "
            f"ORDER BY created_at DESC, memory_id ASC {limit_sql}"
        )

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [_deserialize_row(row) for row in rows]

    async def list_all(self) -> list[MemoryRecord]:
        """List all records in deterministic order."""
        pool = await self._get_pool()
        sql = "SELECT * FROM memory_records ORDER BY created_at DESC, memory_id ASC"

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql)

        return [_deserialize_row(row) for row in rows]

    async def count(self) -> int:
        """Return the total number of records in PostgreSQL."""
        pool = await self._get_pool()
        sql = "SELECT COUNT(*) FROM memory_records"

        async with pool.acquire() as conn:
            val = await conn.fetchval(sql)

        return int(val) if val is not None else 0
