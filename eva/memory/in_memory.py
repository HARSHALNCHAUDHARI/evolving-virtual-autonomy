"""In-memory repository implementation for the EVA memory subsystem.

Designed for development, offline testing, and contract verification prior to
production storage engines (PostgreSQL/pgvector/Redis). This repository does
not provide persistent storage across application restarts.
"""

from __future__ import annotations

import asyncio
from typing import Sequence
from uuid import UUID

from eva.memory.models import (
    DuplicateMemoryError,
    ImmutableMemoryError,
    MemoryNotFoundError,
    MemoryQuery,
    MemoryRecord,
    _utcnow,
)
from eva.memory.repository import MemoryRepository


class InMemoryMemoryRepository(MemoryRepository):
    """In-memory implementation of the MemoryRepository contract.

    Stores MemoryRecord objects in an internal dictionary. Provides isolated deep
    copies on storage and retrieval to ensure external mutations do not affect the
    repository state.
    """

    def __init__(self, initial_records: Sequence[MemoryRecord] | None = None) -> None:
        self._records: dict[UUID, MemoryRecord] = {}
        self._lock = asyncio.Lock()

        if initial_records:
            for record in initial_records:
                self._records[record.memory_id] = record.model_copy(deep=True)

    async def save(self, memory: MemoryRecord) -> MemoryRecord:
        """Persist a new memory record in memory.

        Args:
            memory: The MemoryRecord to store.

        Returns:
            A deep copy of the stored MemoryRecord.

        Raises:
            DuplicateMemoryError: If a record with memory.memory_id already exists.
        """
        async with self._lock:
            if memory.memory_id in self._records:
                raise DuplicateMemoryError(
                    f"Memory record with ID '{memory.memory_id}' already exists."
                )

            stored = memory.model_copy(deep=True)
            self._records[memory.memory_id] = stored
            return stored.model_copy(deep=True)

    async def get(self, memory_id: UUID) -> MemoryRecord | None:
        """Retrieve a memory record by ID.

        Args:
            memory_id: The UUID of the record.

        Returns:
            A deep copy of the matching MemoryRecord, or None if not found.
        """
        async with self._lock:
            record = self._records.get(memory_id)
            if record is None:
                return None
            return record.model_copy(deep=True)

    async def update(self, memory: MemoryRecord) -> MemoryRecord:
        """Update an existing memory record.

        Preserves original memory_id and created_at timestamps. Updates updated_at.

        Args:
            memory: The modified MemoryRecord with an existing memory_id.

        Returns:
            A deep copy of the updated MemoryRecord.

        Raises:
            MemoryNotFoundError: If no record exists with memory.memory_id.
            ImmutableMemoryError: If the existing record is immutable.
        """
        async with self._lock:
            existing = self._records.get(memory.memory_id)
            if existing is None:
                raise MemoryNotFoundError(
                    f"Memory record with ID '{memory.memory_id}' not found."
                )

            if existing.retention_metadata.is_immutable:
                raise ImmutableMemoryError(
                    f"Memory record '{memory.memory_id}' is immutable and cannot be updated."
                )

            now = _utcnow()
            updated = memory.model_copy(deep=True)

            # Preserve immutable historical timestamps
            updated.created_at = existing.created_at
            updated.access_metadata.created_at = existing.access_metadata.created_at

            # Refresh updated timestamps
            updated.updated_at = now
            updated.access_metadata.updated_at = now

            self._records[memory.memory_id] = updated
            return updated.model_copy(deep=True)

    async def delete(self, memory_id: UUID, *, raise_if_missing: bool = False) -> bool:
        """Delete a memory record by ID.

        Args:
            memory_id: The UUID of the record to delete.
            raise_if_missing: If True, raises MemoryNotFoundError when missing.

        Returns:
            True if the record was found and deleted, False if not found.

        Raises:
            MemoryNotFoundError: If raise_if_missing is True and the record was not found.
        """
        async with self._lock:
            if memory_id in self._records:
                del self._records[memory_id]
                return True

            if raise_if_missing:
                raise MemoryNotFoundError(
                    f"Memory record with ID '{memory_id}' not found."
                )
            return False

    async def query(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Query memory records using deterministic criteria with AND semantics.

        Args:
            query: The MemoryQuery filter criteria.

        Returns:
            A list of matching deep-copied MemoryRecords in deterministic order.
        """
        async with self._lock:
            matches: list[MemoryRecord] = []

            for record in self._records.values():
                # Filter by memory type
                if query.memory_types and record.memory_type not in query.memory_types:
                    continue

                # Filter by project ID (with optional global inclusion)
                if query.project_id is not None:
                    if query.include_global:
                        if record.project_id is not None and record.project_id != query.project_id:
                            continue
                    else:
                        if record.project_id != query.project_id:
                            continue

                # Filter by session ID
                if query.session_id is not None and record.session_id != query.session_id:
                    continue

                # Filter by source / provenance
                if query.sources:
                    rec_source = record.source or record.provenance.source_type
                    if rec_source not in query.sources:
                        continue

                # Filter by minimum confidence
                if query.min_confidence is not None and record.confidence < query.min_confidence:
                    continue

                # Filter by minimum importance
                if query.min_importance is not None and record.importance < query.min_importance:
                    continue

                # Filter by creation time
                if query.created_after is not None and record.created_at < query.created_after:
                    continue
                if query.created_before is not None and record.created_at > query.created_before:
                    continue

                # Filter by update time
                if query.updated_after is not None and record.updated_at < query.updated_after:
                    continue
                if query.updated_before is not None and record.updated_at > query.updated_before:
                    continue

                # Filter by archived status
                if query.is_archived is not None:
                    if record.retention_metadata.is_archived != query.is_archived:
                        continue

                matches.append(record)

            # Deterministic ordering: newest created_at first, tie-break by memory_id
            matches.sort(
                key=lambda r: (-r.created_at.timestamp(), str(r.memory_id))
            )

            # Apply limit if specified
            if query.limit is not None:
                matches = matches[:query.limit]

            return [r.model_copy(deep=True) for r in matches]

    async def list_all(self) -> list[MemoryRecord]:
        """List all records in deterministic order."""
        async with self._lock:
            records = list(self._records.values())
            records.sort(
                key=lambda r: (-r.created_at.timestamp(), str(r.memory_id))
            )
            return [r.model_copy(deep=True) for r in records]

    async def count(self) -> int:
        """Return the total number of records."""
        async with self._lock:
            return len(self._records)

    async def clear(self) -> None:
        """Clear all records from the repository (testing helper)."""
        async with self._lock:
            self._records.clear()
