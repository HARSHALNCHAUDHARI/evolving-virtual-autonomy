"""Safe JSON serialization for EVA working memory entries.

Restricts values to JSON-compatible data structures (strings, numbers, booleans,
null, lists, dictionaries) and rejects unsafe serialization (pickle, raw callables,
or arbitrary custom objects).
"""

from __future__ import annotations

import json
from typing import Any

from eva.memory.working.models import (
    WorkingMemoryEntry,
    WorkingMemorySerializationError,
)


def validate_json_compatible(value: Any) -> None:
    """Ensure a value consists solely of standard JSON-serializable types.

    Raises:
        WorkingMemorySerializationError: If non-serializable objects are detected.
    """
    try:
        json.dumps(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WorkingMemorySerializationError(
            f"Value contains non-JSON-compatible type: {exc}"
        ) from exc


def serialize_entry(entry: WorkingMemoryEntry) -> str:
    """Serialize a WorkingMemoryEntry to a JSON string.

    Args:
        entry: The WorkingMemoryEntry to serialize.

    Returns:
        JSON string representation.

    Raises:
        WorkingMemorySerializationError: If serialization fails.
    """
    try:
        # Validate that the entry value is safely JSON-compatible
        validate_json_compatible(entry.value)
        return entry.model_dump_json()
    except WorkingMemorySerializationError:
        raise
    except Exception as exc:
        raise WorkingMemorySerializationError(
            f"Failed to serialize working memory entry: {exc}"
        ) from exc


def deserialize_entry(data: str | bytes) -> WorkingMemoryEntry:
    """Deserialize a JSON string or bytes into a WorkingMemoryEntry.

    Args:
        data: Serialized JSON payload from Redis or in-memory store.

    Returns:
        The validated WorkingMemoryEntry instance.

    Raises:
        WorkingMemorySerializationError: If deserialization or model validation fails.
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")

    try:
        raw_dict = json.loads(data)
        return WorkingMemoryEntry.model_validate(raw_dict)
    except Exception as exc:
        raise WorkingMemorySerializationError(
            f"Failed to deserialize working memory entry: {exc}"
        ) from exc
