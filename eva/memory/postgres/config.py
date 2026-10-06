"""Typed PostgreSQL configuration for EVA Memory Subsystem.

Provides secure connection settings, environment loading, credential masking,
and connection pooling parameters. Never exposes raw passwords in logs or
string representations.
"""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from eva.config import EVASettings, get_settings


class PostgresConfig(BaseModel):
    """Configuration settings for PostgreSQL memory persistence."""

    dsn: str | None = None
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str | None = None
    database: str = "eva"
    min_connections: int = Field(default=2, ge=1)
    max_connections: int = Field(default=10, ge=1)
    connect_timeout: float = Field(default=10.0, gt=0.0)
    command_timeout: float = Field(default=30.0, gt=0.0)

    @property
    def effective_dsn(self) -> str:
        """Return the fully qualified DSN, either explicit or constructed."""
        if self.dsn:
            return self.dsn

        auth = self.user
        if self.password:
            auth = f"{auth}:{self.password}"
        return f"postgresql://{auth}@{self.host}:{self.port}/{self.database}"

    @property
    def masked_dsn(self) -> str:
        """Return the DSN with passwords safely masked for logging and errors."""
        raw = self.effective_dsn
        # Mask password in URI pattern: :password@ -> :***@
        return re.sub(r":([^/@]+)@", r":***@", raw)

    def __repr__(self) -> str:
        return (
            f"PostgresConfig(host={self.host!r}, port={self.port}, "
            f"database={self.database!r}, user={self.user!r}, "
            f"min_connections={self.min_connections}, "
            f"max_connections={self.max_connections}, dsn={self.masked_dsn!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def from_settings(cls, settings: EVASettings | None = None) -> PostgresConfig:
        """Construct PostgresConfig from EVASettings or central configuration."""
        cfg = settings or get_settings()

        dsn = cfg.postgres_dsn or cfg.database_url
        if dsn:
            parsed = urlparse(dsn)
            return cls(
                dsn=dsn,
                host=parsed.hostname or "localhost",
                port=parsed.port or 5432,
                user=parsed.username or "postgres",
                password=parsed.password,
                database=(parsed.path or "/eva").lstrip("/") or "eva",
                min_connections=cfg.postgres_min_connections,
                max_connections=cfg.postgres_max_connections,
                connect_timeout=cfg.postgres_connect_timeout,
            )

        return cls(
            min_connections=cfg.postgres_min_connections,
            max_connections=cfg.postgres_max_connections,
            connect_timeout=cfg.postgres_connect_timeout,
        )

    @classmethod
    def from_env(cls) -> PostgresConfig:
        """Construct PostgresConfig from system environment variables."""
        dsn = os.getenv("EVA_POSTGRES_DSN") or os.getenv("DATABASE_URL")
        host = os.getenv("EVA_POSTGRES_HOST", "localhost")
        port = int(os.getenv("EVA_POSTGRES_PORT", "5432"))
        user = os.getenv("EVA_POSTGRES_USER", "postgres")
        password = os.getenv("EVA_POSTGRES_PASSWORD")
        database = os.getenv("EVA_POSTGRES_DB", "eva")

        return cls(
            dsn=dsn,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
