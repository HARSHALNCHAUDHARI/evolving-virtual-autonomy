"""Unit tests for EVA WorkingMemoryEntry and exception hierarchy."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from eva.memory.models import MemoryError
from eva.memory.working.models import (
    WorkingMemoryConfigurationError,
    WorkingMemoryConnectionError,
    WorkingMemoryEntry,
    WorkingMemoryError,
    WorkingMemoryNotFoundError,
    WorkingMemorySerializationError,
)


def test_working_memory_exception_hierarchy() -> None:
    """Verify that all working memory exceptions inherit from WorkingMemoryError and MemoryError."""
    exceptions = [
        WorkingMemoryNotFoundError,
        WorkingMemorySerializationError,
        WorkingMemoryConnectionError,
        WorkingMemoryConfigurationError,
    ]
    for exc in exceptions:
        assert issubclass(exc, WorkingMemoryError)
        assert issubclass(exc, MemoryError)

    err = WorkingMemoryNotFoundError("Key abc not found in session s1")
    assert "abc" in str(err)
    assert isinstance(err, Exception)


def test_working_memory_entry_validation() -> None:
    """Test valid instantiation, string stripping, and validation bounds."""
    entry = WorkingMemoryEntry(
        session_id="  session-alpha  ",
        key="  active_project  ",
        value={"project_id": "eva", "version": "0.1.0"},
        namespace="  context  ",
    )

    assert entry.session_id == "session-alpha"
    assert entry.key == "active_project"
    assert entry.namespace == "context"
    assert entry.value == {"project_id": "eva", "version": "0.1.0"}
    assert entry.created_at.tzinfo is not None
    assert entry.updated_at.tzinfo is not None
    assert entry.expires_at is None
    assert entry.ttl_seconds is None

    # Empty string validation
    with pytest.raises(ValidationError):
        WorkingMemoryEntry(session_id="", key="k", value="v")

    with pytest.raises(ValidationError):
        WorkingMemoryEntry(session_id="s", key="   ", value="v")

    with pytest.raises(ValidationError):
        WorkingMemoryEntry(session_id="s", key="k", value="v", namespace="  ")


def test_ttl_and_expiration_calculation() -> None:
    """Verify synchronization between ttl_seconds and expires_at."""
    # When ttl_seconds is provided, expires_at is calculated
    entry_ttl = WorkingMemoryEntry(
        session_id="s1",
        key="turn_scratch",
        value="partial reasoning",
        ttl_seconds=300,
    )
    assert entry_ttl.ttl_seconds == 300
    assert entry_ttl.expires_at is not None
    delta = (entry_ttl.expires_at - entry_ttl.created_at).total_seconds()
    assert abs(delta - 300) < 1.0

    # When expires_at is provided, ttl_seconds is calculated
    future_time = datetime.now(timezone.utc) + timedelta(seconds=600)
    entry_exp = WorkingMemoryEntry(
        session_id="s1",
        key="turn_scratch",
        value="partial reasoning",
        expires_at=future_time,
    )
    assert entry_exp.expires_at == future_time
    assert entry_exp.ttl_seconds is not None
    assert abs(entry_exp.ttl_seconds - 600) <= 2


def test_is_expired_method() -> None:
    """Test is_expired checking against current and simulated timestamps."""
    now = datetime.now(timezone.utc)
    entry_active = WorkingMemoryEntry(
        session_id="s1",
        key="active",
        value=123,
        expires_at=now + timedelta(seconds=100),
    )
    assert entry_active.is_expired(now) is False
    assert entry_active.is_expired(now + timedelta(seconds=50)) is False
    assert entry_active.is_expired(now + timedelta(seconds=150)) is True

    entry_no_ttl = WorkingMemoryEntry(
        session_id="s1",
        key="persistent",
        value=True,
    )
    assert entry_no_ttl.is_expired(now + timedelta(days=365)) is False
