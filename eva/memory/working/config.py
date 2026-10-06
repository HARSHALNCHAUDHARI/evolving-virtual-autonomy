"""Configuration and key-namespacing definitions for Redis Working Memory."""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from eva.config import EVASettings, get_settings


class RedisConfig(BaseModel):
    """Configuration settings for Redis working-memory persistence."""

    url: str | None = None
    host: str = "localhost"
    port: int = 6379
    db: int = Field(default=0, ge=0)
    password: str | None = None
    socket_timeout: float = Field(default=5.0, gt=0.0)
    socket_connect_timeout: float = Field(default=5.0, gt=0.0)
    max_connections: int = Field(default=10, ge=1)
    key_prefix: str = "eva:working"
    default_ttl_seconds: int | None = None

    @property
    def effective_url(self) -> str:
        """Return the fully qualified connection URL, explicit or synthesized."""
        if self.url:
            return self.url

        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

    @property
    def masked_url(self) -> str:
        """Return the connection URL with password redacted for logging and errors."""
        raw = self.effective_url
        return re.sub(r":([^/@]+)@", r":***@", raw)

    def build_redis_key(self, session_id: str, key: str, namespace: str = "default") -> str:
        """Construct an isolated, namespaced key string for Redis storage."""
        clean_prefix = self.key_prefix.strip(":")
        clean_sess = session_id.strip()
        clean_ns = namespace.strip()
        clean_key = key.strip()
        return f"{clean_prefix}:{clean_sess}:{clean_ns}:{clean_key}"

    def build_scan_pattern(self, session_id: str, namespace: str | None = None) -> str:
        """Construct a scoped wildcard scan pattern for session / namespace retrieval."""
        clean_prefix = self.key_prefix.strip(":")
        clean_sess = session_id.strip()
        if namespace is not None:
            clean_ns = namespace.strip()
            return f"{clean_prefix}:{clean_sess}:{clean_ns}:*"
        return f"{clean_prefix}:{clean_sess}:*"

    def __repr__(self) -> str:
        return (
            f"RedisConfig(host={self.host!r}, port={self.port}, db={self.db}, "
            f"prefix={self.key_prefix!r}, url={self.masked_url!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def from_settings(cls, settings: EVASettings | None = None) -> RedisConfig:
        """Construct RedisConfig from EVASettings or global configuration."""
        cfg = settings or get_settings()
        url = cfg.redis_url

        if url:
            parsed = urlparse(url)
            db_num = 0
            if parsed.path and parsed.path.lstrip("/").isdigit():
                db_num = int(parsed.path.lstrip("/"))
            return cls(
                url=url,
                host=parsed.hostname or "localhost",
                port=parsed.port or 6379,
                password=parsed.password,
                db=db_num,
                max_connections=cfg.redis_max_connections,
                socket_timeout=cfg.redis_socket_timeout,
                key_prefix=cfg.redis_key_prefix,
                default_ttl_seconds=cfg.redis_default_ttl_seconds,
            )

        return cls(
            max_connections=cfg.redis_max_connections,
            socket_timeout=cfg.redis_socket_timeout,
            key_prefix=cfg.redis_key_prefix,
            default_ttl_seconds=cfg.redis_default_ttl_seconds,
        )

    @classmethod
    def from_env(cls) -> RedisConfig:
        """Construct RedisConfig from system environment variables."""
        url = os.getenv("EVA_REDIS_URL") or os.getenv("REDIS_URL")
        host = os.getenv("EVA_REDIS_HOST", "localhost")
        port = int(os.getenv("EVA_REDIS_PORT", "6379"))
        db = int(os.getenv("EVA_REDIS_DB", "0"))
        password = os.getenv("EVA_REDIS_PASSWORD")
        prefix = os.getenv("EVA_REDIS_KEY_PREFIX", "eva:working")

        return cls(
            url=url,
            host=host,
            port=port,
            db=db,
            password=password,
            key_prefix=prefix,
        )
