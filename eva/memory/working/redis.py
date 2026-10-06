"""Redis implementation of EVA's WorkingMemoryRepository contract.

Uses redis.asyncio with connection pooling, session-scoped namespaced keys,
safe JSON serialization, and native Redis TTL expiration semantics.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import redis.asyncio as aioredis

from eva.memory.working.config import RedisConfig
from eva.memory.working.connection import RedisConnectionManager
from eva.memory.working.models import (
    WorkingMemoryConnectionError,
    WorkingMemoryEntry,
    WorkingMemoryError,
    WorkingMemoryNotFoundError,
    _utcnow,
)
from eva.memory.working.repository import WorkingMemoryRepository
from eva.memory.working.serialization import (
    deserialize_entry,
    serialize_entry,
    validate_json_compatible,
)

logger = logging.getLogger(__name__)


class RedisWorkingMemoryRepository(WorkingMemoryRepository):
    """Redis-backed repository for ephemeral, session-scoped working memory."""

    def __init__(
        self,
        client: aioredis.Redis | RedisConnectionManager | None = None,
        config: RedisConfig | None = None,
    ) -> None:
        self.config = config or RedisConfig()
        self._client: aioredis.Redis | None = None
        self._manager: RedisConnectionManager | None = None

        if isinstance(client, RedisConnectionManager):
            self._manager = client
            self.config = client.config
        elif client is not None:
            self._client = client
        else:
            self._manager = RedisConnectionManager(config=self.config)

    async def _get_client(self) -> aioredis.Redis:
        """Obtain or initialize the active Redis client."""
        if self._client is not None:
            return self._client
        if self._manager is not None:
            self._client = await self._manager.start()
            return self._client
        raise WorkingMemoryConnectionError("Redis client is not configured or started.")

    async def start(self) -> None:
        """Start the backing Redis connection pool."""
        await self._get_client()

    async def stop(self) -> None:
        """Stop the backing connection pool."""
        if self._manager is not None:
            await self._manager.stop()
            self._client = None

    async def is_available(self) -> bool:
        """Check whether Redis is reachable and responsive."""
        try:
            client = await self._get_client()
            return bool(await client.ping())
        except Exception:
            return False

    async def set(self, entry: WorkingMemoryEntry) -> WorkingMemoryEntry:
        """Store or replace an entry with TTL support."""
        validate_json_compatible(entry.value)
        client = await self._get_client()
        redis_key = self.config.build_redis_key(entry.session_id, entry.key, entry.namespace)

        # If TTL <= 0, delete and return immediately
        if entry.ttl_seconds is not None and entry.ttl_seconds <= 0:
            await client.delete(redis_key)
            return entry.model_copy(deep=True)

        payload = serialize_entry(entry)

        try:
            if entry.ttl_seconds is not None and entry.ttl_seconds > 0:
                await client.set(redis_key, payload, ex=entry.ttl_seconds)
            else:
                await client.set(redis_key, payload)
            return entry.model_copy(deep=True)
        except Exception as exc:
            raise WorkingMemoryError(f"Failed to set working memory in Redis: {exc}") from exc

    async def get(
        self,
        session_id: str,
        key: str,
        *,
        namespace: str = "default",
    ) -> WorkingMemoryEntry | None:
        """Retrieve a working memory entry by session, key, and namespace."""
        client = await self._get_client()
        redis_key = self.config.build_redis_key(session_id, key, namespace)

        try:
            data = await client.get(redis_key)
            if data is None:
                return None
            return deserialize_entry(data)
        except Exception as exc:
            raise WorkingMemoryError(f"Failed to get working memory from Redis: {exc}") from exc

    async def update(self, entry: WorkingMemoryEntry) -> WorkingMemoryEntry:
        """Update an existing working memory entry, preserving creation timestamp."""
        validate_json_compatible(entry.value)
        client = await self._get_client()
        redis_key = self.config.build_redis_key(entry.session_id, entry.key, entry.namespace)

        try:
            existing_data = await client.get(redis_key)
            if existing_data is None:
                raise WorkingMemoryNotFoundError(
                    f"Working memory entry '{entry.key}' not found in session '{entry.session_id}'."
                )

            existing = deserialize_entry(existing_data)
            now = _utcnow()
            updated = entry.model_copy(deep=True)
            updated.created_at = existing.created_at
            updated.updated_at = now

            # If TTL was not explicitly specified on update, preserve remaining TTL from Redis
            if updated.ttl_seconds is None:
                remaining_ttl = await client.ttl(redis_key)
                if remaining_ttl > 0:
                    updated.ttl_seconds = remaining_ttl
                    updated.expires_at = now + timedelta(seconds=remaining_ttl)

            payload = serialize_entry(updated)

            if updated.ttl_seconds is not None and updated.ttl_seconds > 0:
                await client.set(redis_key, payload, ex=updated.ttl_seconds)
            elif updated.ttl_seconds is not None and updated.ttl_seconds <= 0:
                await client.delete(redis_key)
            else:
                await client.set(redis_key, payload)

            return updated
        except (WorkingMemoryNotFoundError, WorkingMemoryError):
            raise
        except Exception as exc:
            raise WorkingMemoryError(f"Failed to update working memory in Redis: {exc}") from exc

    async def delete(
        self,
        session_id: str,
        key: str,
        *,
        namespace: str = "default",
        raise_if_missing: bool = False,
    ) -> bool:
        """Delete an entry from Redis."""
        client = await self._get_client()
        redis_key = self.config.build_redis_key(session_id, key, namespace)

        try:
            del_count = await client.delete(redis_key)
            if del_count > 0:
                return True
            if raise_if_missing:
                raise WorkingMemoryNotFoundError(
                    f"Working memory entry '{key}' not found in session '{session_id}'."
                )
            return False
        except WorkingMemoryNotFoundError:
            raise
        except Exception as exc:
            raise WorkingMemoryError(f"Failed to delete working memory in Redis: {exc}") from exc

    async def exists(
        self,
        session_id: str,
        key: str,
        *,
        namespace: str = "default",
    ) -> bool:
        """Check whether a key exists in Redis."""
        client = await self._get_client()
        redis_key = self.config.build_redis_key(session_id, key, namespace)
        try:
            return bool(await client.exists(redis_key))
        except Exception as exc:
            raise WorkingMemoryError(f"Failed to check existence in Redis: {exc}") from exc

    async def clear_session(
        self,
        session_id: str,
        *,
        namespace: str | None = None,
    ) -> int:
        """Purge keys matching the session (and optional namespace) pattern."""
        client = await self._get_client()
        pattern = self.config.build_scan_pattern(session_id, namespace)

        deleted_count = 0
        try:
            keys_to_delete: list[str] = []
            async for k in client.scan_iter(match=pattern, count=100):
                keys_to_delete.append(k)

            if keys_to_delete:
                # Delete in chunks using pipeline
                pipe = client.pipeline()
                chunk_size = 100
                for i in range(0, len(keys_to_delete), chunk_size):
                    chunk = keys_to_delete[i : i + chunk_size]
                    for key in chunk:
                        pipe.delete(key)
                    results = await pipe.execute()
                    deleted_count += sum(1 for r in results if r > 0)

            return deleted_count
        except Exception as exc:
            raise WorkingMemoryError(f"Failed to clear session in Redis: {exc}") from exc

    async def list_session(
        self,
        session_id: str,
        *,
        namespace: str | None = None,
        limit: int | None = None,
    ) -> list[WorkingMemoryEntry]:
        """List active entries for a session deterministically."""
        client = await self._get_client()
        pattern = self.config.build_scan_pattern(session_id, namespace)

        try:
            keys: list[str] = []
            async for k in client.scan_iter(match=pattern, count=100):
                keys.append(k)

            if not keys:
                return []

            # Retrieve payloads via mget
            entries: list[WorkingMemoryEntry] = []
            chunk_size = 100
            for i in range(0, len(keys), chunk_size):
                chunk = keys[i : i + chunk_size]
                values = await client.mget(chunk)
                for val in values:
                    if val is not None:
                        try:
                            entries.append(deserialize_entry(val))
                        except Exception:
                            logger.warning("Skipping corrupted working memory entry in Redis")

            # Deterministic ordering by namespace then key ascending
            entries.sort(key=lambda e: (e.namespace, e.key))

            if limit is not None:
                entries = entries[:limit]

            return entries
        except Exception as exc:
            raise WorkingMemoryError(f"Failed to list session in Redis: {exc}") from exc

    async def set_ttl(
        self,
        session_id: str,
        key: str,
        ttl_seconds: int | None,
        *,
        namespace: str = "default",
    ) -> bool:
        """Update or clear the expiration TTL for an existing entry."""
        client = await self._get_client()
        redis_key = self.config.build_redis_key(session_id, key, namespace)

        try:
            # Check if key exists
            existing_data = await client.get(redis_key)
            if existing_data is None:
                return False

            now = _utcnow()
            entry = deserialize_entry(existing_data)
            entry.updated_at = now

            if ttl_seconds is None or ttl_seconds < 0:
                # Remove expiration
                entry.expires_at = None
                entry.ttl_seconds = None
                payload = serialize_entry(entry)
                await client.set(redis_key, payload)
                await client.persist(redis_key)
                return True

            if ttl_seconds == 0:
                await client.delete(redis_key)
                return True

            entry.ttl_seconds = ttl_seconds
            entry.expires_at = now + timedelta(seconds=ttl_seconds)
            payload = serialize_entry(entry)
            await client.set(redis_key, payload, ex=ttl_seconds)
            return True
        except Exception as exc:
            raise WorkingMemoryError(f"Failed to set TTL in Redis: {exc}") from exc
