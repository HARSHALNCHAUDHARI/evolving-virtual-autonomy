"""Unit tests for RedisConfig and key generation."""

from __future__ import annotations

import os

import pytest

from eva.config import EVASettings
from eva.memory.working.config import RedisConfig


def test_redis_config_defaults() -> None:
    cfg = RedisConfig()
    assert cfg.url is None
    assert cfg.host == "localhost"
    assert cfg.port == 6379
    assert cfg.db == 0
    assert cfg.password is None
    assert cfg.key_prefix == "eva:working"
    assert cfg.effective_url == "redis://localhost:6379/0"
    assert cfg.masked_url == "redis://localhost:6379/0"


def test_redis_config_with_password_masked() -> None:
    cfg = RedisConfig(password="supersecret123")
    assert "supersecret123" in cfg.effective_url
    assert "supersecret123" not in cfg.masked_url
    assert "***" in cfg.masked_url
    assert "supersecret123" not in repr(cfg)
    assert "supersecret123" not in str(cfg)


def test_redis_config_explicit_url() -> None:
    cfg = RedisConfig(url="rediss://user:topsecret@redis.internal.org:6380/2")
    assert cfg.effective_url == "rediss://user:topsecret@redis.internal.org:6380/2"
    assert "topsecret" not in cfg.masked_url
    assert "***" in cfg.masked_url


def test_build_redis_key() -> None:
    cfg = RedisConfig(key_prefix="eva:working")
    key = cfg.build_redis_key("sess-123", "active_goal", namespace="tasks")
    assert key == "eva:working:sess-123:tasks:active_goal"

    # Default namespace
    key_default = cfg.build_redis_key("sess-123", "user_prompt")
    assert key_default == "eva:working:sess-123:default:user_prompt"


def test_build_scan_pattern() -> None:
    cfg = RedisConfig(key_prefix="eva:working")
    # All namespaces for session
    pattern_all = cfg.build_scan_pattern("sess-123")
    assert pattern_all == "eva:working:sess-123:*"

    # Specific namespace for session
    pattern_ns = cfg.build_scan_pattern("sess-123", namespace="tasks")
    assert pattern_ns == "eva:working:sess-123:tasks:*"


def test_from_settings() -> None:
    settings = EVASettings(
        redis_url="redis://:my_pwd@cache.example.com:6388/3",
        redis_max_connections=25,
        redis_socket_timeout=2.5,
        redis_key_prefix="custom:eva",
        redis_default_ttl_seconds=3600,
    )
    cfg = RedisConfig.from_settings(settings)
    assert cfg.url == "redis://:my_pwd@cache.example.com:6388/3"
    assert cfg.host == "cache.example.com"
    assert cfg.port == 6388
    assert cfg.db == 3
    assert cfg.password == "my_pwd"
    assert cfg.max_connections == 25
    assert cfg.socket_timeout == 2.5
    assert cfg.key_prefix == "custom:eva"
    assert cfg.default_ttl_seconds == 3600


def test_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVA_REDIS_URL", "redis://redis-host:6379/1")
    monkeypatch.setenv("EVA_REDIS_KEY_PREFIX", "env:eva")

    cfg = RedisConfig.from_env()
    assert cfg.url == "redis://redis-host:6379/1"
    assert cfg.key_prefix == "env:eva"
