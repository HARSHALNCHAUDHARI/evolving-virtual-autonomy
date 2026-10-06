"""In-memory Working Memory repository implementation.

Designed for unit testing, offline development, and validating the
WorkingMemoryRepository contract without requiring a running Redis instance.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Sequence

from eva.memory.working.models import (
    WorkingMemoryEntry,
    WorkingMemoryNotFoundError,
    _ensure_timezone,
    _utcnow,
)
from eva.memory.working.repository import WorkingMemoryRepository
from eva.memory.working.serialization import validate_json_compatible


class InMemoryWorkingMemoryRepository(WorkingMemoryRepository):
    """In-memory working memory repository with session isolation and TTL simulation."""

    def __init__(self, initial_entries: Sequence[WorkingMemoryEntry] | None = None) -> None:
        # Keyed by (session_id, namespace, key)
        self._entries: dict[tuple[str, str, str], WorkingMemoryEntry] = {}
        self._lock = asyncio.Lock()

        if initial_entries:
            for entry in initial_entries:
                self._entries[(entry.session_id, entry.namespace, entry.key)] = entry.model_copy(
                    deep=True
                )

    def _is_expired(self, entry: WorkingMemoryEntry, now: Any = None) -> bool:
        """Internal helper to evaluate expiration."""
        return entry.is_expired(now)

    async def set(self, entry: WorkingMemoryEntry) -> WorkingMemoryEntry:
        """Store or replace an entry."""
        # Ensure value is JSON-compatible
        validate_json_compatible(entry.value)

        async with self._lock:
            stored = entry.model_copy(deep=True)
            stored_key = (stored.session_id, stored.namespace, stored.key)

            # If entry has TTL <= 0, it expires immediately
            if stored.ttl_seconds is not None and stored.ttl_seconds <= 0:
                self._entries.pop(stored_key, None)
                return stored

            self._entries[stored_key] = stored
            return stored.model_copy(deep=True)

    async def get(
        self,
        session_id: str,
        key: str,
        *,
        namespace: str = "default",
    ) -> WorkingMemoryEntry | None:
        """Retrieve an entry, automatically cleaning up if expired."""
        map_key = (session_id, namespace, key)
        async with self._lock:
            entry = self._entries.get(map_key)
            if entry is None:
                return None

            if self._is_expired(entry):
                del self._entries[map_key]
                return None

            return entry.model_copy(deep=True)

    async def update(self, entry: WorkingMemoryEntry) -> WorkingMemoryEntry:
        """Update an existing entry, refreshing updated_at."""
        validate_json_compatible(entry.value)
        map_key = (entry.session_id, entry.namespace, entry.key)

        async with self._lock:
            existing = self._entries.get(map_key)
            if existing is None or self._is_expired(existing):
                self._entries.pop(map_key, None)
                raise WorkingMemoryNotFoundError(
                    f"Working memory entry '{entry.key}' not found in session '{entry.session_id}'."
                )

            now = _utcnow()
            updated = entry.model_copy(deep=True)
            updated.created_at = existing.created_at
            updated.updated_at = now

            # If new entry did not explicitly specify TTL, preserve existing TTL if any
            if updated.ttl_seconds is None and existing.expires_at is not None:
                updated.expires_at = existing.expires_at
                remaining = max(0, int((existing.expires_at - now).total_seconds()))
                updated.ttl_seconds = remaining

            if updated.ttl_seconds is not None and updated.ttl_seconds <= 0:
                self._entries.pop(map_key, None)
                return updated

            self._entries[map_key] = updated
            return updated.model_copy(deep=True)

    async def delete(
        self,
        session_id: str,
        key: str,
        *,
        namespace: str = "default",
        raise_if_missing: bool = False,
    ) -> bool:
        """Delete an entry."""
        map_key = (session_id, namespace, key)
        async with self._lock:
            entry = self._entries.pop(map_key, None)
            if entry is not None and not self._is_expired(entry):
                return True

            if raise_if_missing:
                raise WorkingMemoryNotFoundError(
                    f"Working memory entry '{key}' not found in session '{session_id}'."
                )
            return False

    async def exists(
        self,
        session_id: str,
        key: str,
        *,
        namespace: str = "default",
    ) -> bool:
        """Check whether a non-expired entry exists."""
        map_key = (session_id, namespace, key)
        async with self._lock:
            entry = self._entries.get(map_key)
            if entry is None:
                return False
            if self._is_expired(entry):
                del self._entries[map_key]
                return False
            return True

    async def clear_session(
        self,
        session_id: str,
        *,
        namespace: str | None = None,
    ) -> int:
        """Remove all working memory entries for a given session."""
        async with self._lock:
            to_remove = [
                k
                for k in self._entries.keys()
                if k[0] == session_id and (namespace is None or k[1] == namespace)
            ]
            for k in to_remove:
                del self._entries[k]
            return len(to_remove)

    async def list_session(
        self,
        session_id: str,
        *,
        namespace: str | None = None,
        limit: int | None = None,
    ) -> list[WorkingMemoryEntry]:
        """List active non-expired entries for a session."""
        now = _utcnow()
        async with self._lock:
            expired_keys: list[tuple[str, str, str]] = []
            valid_entries: list[WorkingMemoryEntry] = []

            for k, entry in self._entries.items():
                if k[0] != session_id:
                    continue
                if namespace is not None and k[1] != namespace:
                    continue

                if self._is_expired(entry, now):
                    expired_keys.append(k)
                else:
                    valid_entries.append(entry)

            # Purge expired items found during scan
            for k in expired_keys:
                del self._entries[k]

            # Deterministic ordering: sorted by key ascending
            valid_entries.sort(key=lambda e: (e.namespace, e.key))

            if limit is not None:
                valid_entries = valid_entries[:limit]

            return [e.model_copy(deep=True) for e in valid_entries]

    async def set_ttl(
        self,
        session_id: str,
        key: str,
        ttl_seconds: int | None,
        *,
        namespace: str = "default",
    ) -> bool:
        """Update or clear the expiration TTL for an existing entry."""
        map_key = (session_id, namespace, key)
        async with self._lock:
            entry = self._entries.get(map_key)
            if entry is None or self._is_expired(entry):
                self._entries.pop(map_key, None)
                return False

            now = _utcnow()
            if ttl_seconds is None or ttl_seconds < 0:
                # Make persistent (remove expiration)
                entry.expires_at = None
                entry.ttl_seconds = None
            elif ttl_seconds == 0:
                # Expire immediately
                del self._entries[map_key]
                return True
            else:
                entry.ttl_seconds = ttl_seconds
                entry.expires_at = now + timedelta(seconds=ttl_seconds)

            entry.updated_at = now
            return True

    async def count(self) -> int:
        """Return total active entries across all sessions."""
        async with self._lock:
            now = _utcnow()
            return sum(1 for e in self._entries.values() if not self._is_expired(e, now))

    async def clear_all(self) -> None:
        """Helper to clear all state (for tests)."""
        async with self._lock:
            self._entries.clear()
