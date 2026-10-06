"""Connection pool management for PostgreSQL memory repository.

Encapsulates asyncpg.Pool lifecycle, connection acquisition, and safe error
translation into EVA's DatabaseConnectionError hierarchy without leaking credentials.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg

from eva.memory.models import DatabaseConnectionError
from eva.memory.postgres.config import PostgresConfig

logger = logging.getLogger(__name__)


class PostgresPoolManager:
    """Manages the lifecycle of an asyncpg connection pool."""

    def __init__(self, config: PostgresConfig | None = None) -> None:
        self.config = config or PostgresConfig()
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        """Return the active connection pool or raise an error if not connected."""
        if self._pool is None:
            raise DatabaseConnectionError(
                "PostgreSQL connection pool is not started. Call start() before querying."
            )
        return self._pool

    async def is_connected(self) -> bool:
        """Check whether the pool is active and capable of acquiring a connection."""
        if self._pool is None:
            return False
        try:
            async with self._pool.acquire() as conn:
                res = await conn.fetchval("SELECT 1")
                return res == 1
        except Exception:
            return False

    async def start(self) -> asyncpg.Pool:
        """Establish and initialize the connection pool.

        Returns:
            The initialized asyncpg.Pool instance.

        Raises:
            DatabaseConnectionError: If connection fails, logging masked credentials only.
        """
        if self._pool is not None:
            return self._pool

        try:
            logger.info("Initializing PostgreSQL pool at %s", self.config.masked_dsn)
            self._pool = await asyncpg.create_pool(
                dsn=self.config.effective_dsn,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                timeout=self.config.connect_timeout,
                command_timeout=self.config.command_timeout,
            )
            return self._pool
        except Exception as exc:
            logger.error(
                "Failed to connect to PostgreSQL at %s: %s",
                self.config.masked_dsn,
                exc.__class__.__name__,
            )
            raise DatabaseConnectionError(
                f"Could not connect to PostgreSQL at {self.config.masked_dsn}: {exc}"
            ) from exc

    async def stop(self) -> None:
        """Gracefully terminate all connections in the pool."""
        if self._pool is not None:
            logger.info("Closing PostgreSQL pool for %s", self.config.masked_dsn)
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """Context manager to acquire and automatically return a connection."""
        pool = self.pool
        async with pool.acquire() as conn:
            yield conn
