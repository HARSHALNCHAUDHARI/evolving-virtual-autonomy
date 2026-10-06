"""Storage-agnostic contract for EVA's Working Memory repository.

Defines the interface for ephemeral, session-scoped working memory operations.
Implemented by InMemoryWorkingMemoryRepository and RedisWorkingMemoryRepository.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from eva.memory.working.models import WorkingMemoryEntry


@runtime_checkable
class WorkingMemoryRepository(Protocol):
    """Protocol for fast, temporary, session-scoped working memory access."""

    async def set(self, entry: WorkingMemoryEntry) -> WorkingMemoryEntry:
        """Store or replace a working memory entry.

        Args:
            entry: The WorkingMemoryEntry to store.

        Returns:
            The stored WorkingMemoryEntry.
        """
        ...

    async def get(
        self,
        session_id: str,
        key: str,
        *,
        namespace: str = "default",
    ) -> WorkingMemoryEntry | None:
        """Retrieve a working memory entry by session, key, and namespace.

        Returns:
            The WorkingMemoryEntry, or None if missing or expired.
        """
        ...

    async def update(self, entry: WorkingMemoryEntry) -> WorkingMemoryEntry:
        """Update an existing working memory entry.

        Args:
            entry: The modified WorkingMemoryEntry.

        Returns:
            The updated WorkingMemoryEntry.

        Raises:
            WorkingMemoryNotFoundError: If the entry does not exist or has expired.
        """
        ...

    async def delete(
        self,
        session_id: str,
        key: str,
        *,
        namespace: str = "default",
        raise_if_missing: bool = False,
    ) -> bool:
        """Delete a working memory entry.

        Args:
            session_id: Session identifier.
            key: Entry key.
            namespace: Context namespace.
            raise_if_missing: If True, raises WorkingMemoryNotFoundError when absent.

        Returns:
            True if deleted, False if not found.

        Raises:
            WorkingMemoryNotFoundError: If raise_if_missing is True and the entry was missing.
        """
        ...

    async def exists(
        self,
        session_id: str,
        key: str,
        *,
        namespace: str = "default",
    ) -> bool:
        """Check whether a non-expired entry exists."""
        ...

    async def clear_session(
        self,
        session_id: str,
        *,
        namespace: str | None = None,
    ) -> int:
        """Purge working memory entries for a specific session.

        Args:
            session_id: Target session to purge.
            namespace: Optional namespace restriction. If None, purges all namespaces
                for this session.

        Returns:
            Number of entries removed.
        """
        ...

    async def list_session(
        self,
        session_id: str,
        *,
        namespace: str | None = None,
        limit: int | None = None,
    ) -> list[WorkingMemoryEntry]:
        """List active, non-expired working memory entries for a session.

        Args:
            session_id: Session identifier.
            namespace: Optional namespace filter.
            limit: Maximum entries to return.

        Returns:
            Deterministic list of matching WorkingMemoryEntries.
        """
        ...

    async def set_ttl(
        self,
        session_id: str,
        key: str,
        ttl_seconds: int | None,
        *,
        namespace: str = "default",
    ) -> bool:
        """Update or clear the expiration TTL for an existing entry.

        Args:
            session_id: Session identifier.
            key: Entry key.
            ttl_seconds: Seconds until expiration, or None to make persistent.
            namespace: Context namespace.

        Returns:
            True if TTL was updated, False if the entry does not exist or has expired.
        """
        ...
