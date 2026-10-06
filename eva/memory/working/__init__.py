"""EVA Working Memory subsystem.

Provides temporary, session-scoped working state management backed by Redis
and in-memory test implementations.
"""

from eva.memory.working.config import RedisConfig
from eva.memory.working.connection import RedisConnectionManager
from eva.memory.working.in_memory import InMemoryWorkingMemoryRepository
from eva.memory.working.models import (
    WorkingMemoryConfigurationError,
    WorkingMemoryConnectionError,
    WorkingMemoryEntry,
    WorkingMemoryError,
    WorkingMemoryNotFoundError,
    WorkingMemorySerializationError,
)
from eva.memory.working.redis import RedisWorkingMemoryRepository
from eva.memory.working.repository import WorkingMemoryRepository
from eva.memory.working.serialization import deserialize_entry, serialize_entry

__all__ = [
    "InMemoryWorkingMemoryRepository",
    "RedisConfig",
    "RedisConnectionManager",
    "RedisWorkingMemoryRepository",
    "WorkingMemoryConfigurationError",
    "WorkingMemoryConnectionError",
    "WorkingMemoryEntry",
    "WorkingMemoryError",
    "WorkingMemoryNotFoundError",
    "WorkingMemoryRepository",
    "WorkingMemorySerializationError",
    "deserialize_entry",
    "serialize_entry",
]
