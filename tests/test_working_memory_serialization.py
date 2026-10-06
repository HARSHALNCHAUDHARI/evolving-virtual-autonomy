"""Unit tests for safe working memory serialization and deserialization."""

import pytest

from eva.memory.working.models import (
    WorkingMemoryEntry,
    WorkingMemorySerializationError,
)
from eva.memory.working.serialization import (
    deserialize_entry,
    serialize_entry,
    validate_json_compatible,
)


def test_validate_json_compatible_types() -> None:
    """Verify JSON compatible validation for allowed and disallowed types."""
    valid_values = [
        "a string",
        42,
        3.14159,
        True,
        False,
        None,
        ["list", "of", "primitives", 1, 2],
        {"nested": {"dict": True, "count": 10}},
    ]
    for val in valid_values:
        validate_json_compatible(val)

    # Disallowed non-JSON-compatible types
    class CustomObject:
        pass

    with pytest.raises(WorkingMemorySerializationError):
        validate_json_compatible(CustomObject())

    with pytest.raises(WorkingMemorySerializationError):
        validate_json_compatible(lambda x: x)


def test_round_trip_serialization() -> None:
    """Test full serialize and deserialize round-trip for WorkingMemoryEntry."""
    entry = WorkingMemoryEntry(
        session_id="session_42",
        key="recent_observations",
        value=[
            {"tool": "read_file", "path": "/etc/hosts", "success": True},
            {"status": "ok", "retry_count": 0},
        ],
        namespace="telemetry",
        ttl_seconds=120,
        metadata={"source": "cli_runner"},
    )

    serialized = serialize_entry(entry)
    assert isinstance(serialized, str)
    assert "recent_observations" in serialized
    assert "session_42" in serialized

    # Deserialize back
    restored = deserialize_entry(serialized)
    assert restored.session_id == entry.session_id
    assert restored.key == entry.key
    assert restored.namespace == entry.namespace
    assert restored.value == entry.value
    assert restored.ttl_seconds == entry.ttl_seconds
    assert restored.metadata == entry.metadata
    assert restored.created_at == entry.created_at


def test_deserialize_bytes_input() -> None:
    """Verify deserialize_entry supports UTF-8 encoded bytes from Redis."""
    entry = WorkingMemoryEntry(
        session_id="sess_bytes",
        key="byte_key",
        value={"answer": 42},
    )
    raw_bytes = serialize_entry(entry).encode("utf-8")

    restored = deserialize_entry(raw_bytes)
    assert restored.session_id == "sess_bytes"
    assert restored.value == {"answer": 42}


def test_corrupted_json_deserialization_error() -> None:
    """Corrupted JSON strings must raise WorkingMemorySerializationError."""
    with pytest.raises(WorkingMemorySerializationError):
        deserialize_entry("not a valid json { broken")
