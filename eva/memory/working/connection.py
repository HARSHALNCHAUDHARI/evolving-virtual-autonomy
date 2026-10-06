"""Connection pool management for Redis Working Memory repository.

Encapsulates redis.asyncio ConnectionPool lifecycle and error translation
into WorkingMemoryConnectionError without leaking credentials.
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

from eva.memory.working.config import RedisConfig
from eva.memory.working.models import WorkingMemoryConnectionError

logger = logging.getLogger(__name__)


class RedisConnectionManager:
    """Manages the connection pool and client lifecycle for Redis."""

    def __init__(self, config: RedisConfig | None = None) -> None:
        self.config = config or RedisConfig()
        self._pool: aioredis.ConnectionPool | None = None
        self._client: aioredis.Redis | None = None

    @property
    def client(self) -> aioredis.Redis:
        """Return the active Redis client or raise an error if not connected."""
        if self._client is None:
            raise WorkingMemoryConnectionError(
                "Redis client is not started. Call start() before executing operations."
            )
        return self._client

    async def is_connected(self) -> bool:
        """Check whether Redis is reachable by executing a PING."""
        if self._client is None:
            return False
        try:
            return bool(await self._client.ping())
        except Exception:
            return False

    async def start(self) -> aioredis.Redis:
        """Initialize the connection pool and client.

        Returns:
            The initialized aioredis.Redis client instance.

        Raises:
            WorkingMemoryConnectionError: If connection or ping fails.
        """
        if self._client is not None:
            return self._client

        try:
            logger.info("Initializing Redis connection pool at %s", self.config.masked_url)
            self._pool = aioredis.ConnectionPool.from_url(
                self.config.effective_url,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=True,
            )
            client = aioredis.Redis(connection_pool=self._pool)
            # Verify connectivity
            await client.ping()
            self._client = client
            return self._client
        except Exception as exc:
            logger.error(
                "Failed to connect to Redis at %s: %s",
                self.config.masked_url,
                exc.__class__.__name__,
            )
            # Clean up pool if created
            if self._pool is not None:
                try:
                    await self._pool.disconnect()
                except Exception:
                    pass
                self._pool = None
            raise WorkingMemoryConnectionError(
                f"Could not connect to Redis at {self.config.masked_url}: {exc}"
            ) from exc

    async def stop(self) -> None:
        """Gracefully close client connections and pool."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

        if self._pool is not None:
            try:
                await self._pool.disconnect()
            except Exception:
                pass
            self._pool = None
