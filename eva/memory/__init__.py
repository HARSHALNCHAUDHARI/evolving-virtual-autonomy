"""EVA Memory Subsystem.

Provides typed data models, lifecycle contracts, and repository abstractions
for EVA's persistent and working memory hierarchy.
"""

from eva.memory.in_memory import InMemoryMemoryRepository
from eva.memory.models import (
    AccessMetadata,
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseMigrationError,
    DuplicateMemoryError,
    EmbeddingReference,
    ImmutableMemoryError,
    MemoryError,
    MemoryNotFoundError,
    MemoryQuery,
    MemoryRecord,
    MemorySource,
    MemoryType,
    MemoryUpdateError,
    ProvenanceRecord,
    RetentionMetadata,
)
from eva.memory.postgres import PostgresConfig, PostgreSQLMemoryRepository
from eva.memory.repository import MemoryRepository
from eva.memory.working import (
    InMemoryWorkingMemoryRepository,
    RedisConfig,
    RedisWorkingMemoryRepository,
    WorkingMemoryConfigurationError,
    WorkingMemoryConnectionError,
    WorkingMemoryEntry,
    WorkingMemoryError,
    WorkingMemoryNotFoundError,
    WorkingMemoryRepository,
    WorkingMemorySerializationError,
)

__all__ = [
    "AccessMetadata",
    "DatabaseConfigurationError",
    "DatabaseConnectionError",
    "DatabaseError",
    "DatabaseMigrationError",
    "DuplicateMemoryError",
    "EmbeddingReference",
    "ImmutableMemoryError",
    "InMemoryMemoryRepository",
    "InMemoryWorkingMemoryRepository",
    "MemoryError",
    "MemoryNotFoundError",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryRepository",
    "MemorySource",
    "MemoryType",
    "MemoryUpdateError",
    "PostgresConfig",
    "PostgreSQLMemoryRepository",
    "ProvenanceRecord",
    "RedisConfig",
    "RedisWorkingMemoryRepository",
    "RetentionMetadata",
    "WorkingMemoryConfigurationError",
    "WorkingMemoryConnectionError",
    "WorkingMemoryEntry",
    "WorkingMemoryError",
    "WorkingMemoryNotFoundError",
    "WorkingMemoryRepository",
    "WorkingMemorySerializationError",
]
