"""Unit tests for PostgreSQL configuration, DSN masking, and pool manager."""

import pytest

from eva.config import EVASettings
from eva.memory.models import DatabaseConnectionError
from eva.memory.postgres.config import PostgresConfig
from eva.memory.postgres.connection import PostgresPoolManager


def test_postgres_config_defaults() -> None:
    """Verify default PostgreSQL configuration values."""
    config = PostgresConfig()
    assert config.host == "localhost"
    assert config.port == 5432
    assert config.user == "postgres"
    assert config.password is None
    assert config.database == "eva"
    assert config.min_connections == 2
    assert config.max_connections == 10
    assert config.connect_timeout == 10.0
    assert "postgresql://postgres@localhost:5432/eva" in config.effective_dsn


def test_postgres_config_credential_masking() -> None:
    """Verify that sensitive credentials are never leaked in logs or repr."""
    config = PostgresConfig(
        user="eva_user",
        password="super_secret_password_123",
        host="db.internal.net",
        database="production_eva",
    )

    # Effective DSN has real password
    assert "super_secret_password_123" in config.effective_dsn

    # Masked DSN must redact password
    assert "super_secret_password_123" not in config.masked_dsn
    assert ":***@" in config.masked_dsn
    assert "postgresql://eva_user:***@db.internal.net:5432/production_eva" in config.masked_dsn

    # __repr__ and __str__ must use masked DSN and never expose the password
    repr_str = repr(config)
    assert "super_secret_password_123" not in repr_str
    assert ":***@" in repr_str


def test_postgres_config_from_settings() -> None:
    """Verify constructing PostgresConfig from EVASettings."""
    settings = EVASettings(
        database_url="postgresql://custom_user:secret_pass@127.0.0.1:5433/custom_db",
        postgres_min_connections=4,
        postgres_max_connections=20,
    )

    config = PostgresConfig.from_settings(settings)
    assert config.user == "custom_user"
    assert config.password == "secret_pass"
    assert config.host == "127.0.0.1"
    assert config.port == 5433
    assert config.database == "custom_db"
    assert config.min_connections == 4
    assert config.max_connections == 20
    assert "secret_pass" not in config.masked_dsn


def test_pool_manager_unstarted_error() -> None:
    """Accessing pool on unstarted PostgresPoolManager raises DatabaseConnectionError."""
    manager = PostgresPoolManager()
    with pytest.raises(DatabaseConnectionError) as exc_info:
        _ = manager.pool
    assert "not started" in str(exc_info.value).lower()
