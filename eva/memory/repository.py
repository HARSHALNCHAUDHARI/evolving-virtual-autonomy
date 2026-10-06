"""Storage-agnostic repository interface for the EVA memory subsystem.

Defines the abstract persistence contract implemented by memory repositories
(e.g., InMemoryMemoryRepository for development and testing, or future database
repositories).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from eva.memory.models import MemoryQuery, MemoryRecord


@runtime_checkable
class MemoryRepository(Protocol):
    """Storage-agnostic persistence contract for EVA memory records.

    Provides asynchronous CRUD and deterministic query operations. Implementations
    must remain agnostic to whether backing storage is in-memory, relational,
    document, or vector-based.
    """

    async def save(self, memory: MemoryRecord) -> MemoryRecord:
        """Persist a new memory record.

        Args:
            memory: The MemoryRecord to store.

        Returns:
            The persisted MemoryRecord.

        Raises:
            DuplicateMemoryError: If a record with memory.memory_id already exists.
        """
        ...

    async def get(self, memory_id: UUID) -> MemoryRecord | None:
        """Retrieve a memory record by unique identifier.

        Args:
            memory_id: The UUID of the record to retrieve.

        Returns:
            The matching MemoryRecord, or None if not found.
        """
        ...

    async def update(self, memory: MemoryRecord) -> MemoryRecord:
        """Update an existing memory record.

        Args:
            memory: The modified MemoryRecord with an existing memory_id.

        Returns:
            The updated MemoryRecord with refreshed updated_at timestamp.

        Raises:
            MemoryNotFoundError: If no record exists with memory.memory_id.
            ImmutableMemoryError: If the existing record is immutable.
        """
        ...

    async def delete(self, memory_id: UUID, *, raise_if_missing: bool = False) -> bool:
        """Delete a memory record by unique identifier.

        Args:
            memory_id: The UUID of the record to delete.
            raise_if_missing: If True, raises MemoryNotFoundError when missing.

        Returns:
            True if the record was found and deleted, False if not found.

        Raises:
            MemoryNotFoundError: If raise_if_missing is True and the record was not found.
        """
        ...

    async def query(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Query memory records using deterministic criteria.

        Args:
            query: The MemoryQuery containing filter parameters.

        Returns:
            A list of matching MemoryRecords, sorted deterministically.
        """
        ...

    async def list_all(self) -> list[MemoryRecord]:
        """List all memory records.

        Returns:
            A list of all records in the repository.
        """
        ...

    async def count(self) -> int:
        """Return the total number of records in the repository."""
        ...
