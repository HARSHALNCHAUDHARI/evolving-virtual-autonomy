"""Unit tests for the memory repository interface and exception hierarchy."""

import inspect

from eva.memory.in_memory import InMemoryMemoryRepository
from eva.memory.models import (
    DuplicateMemoryError,
    ImmutableMemoryError,
    MemoryError,
    MemoryNotFoundError,
    MemoryUpdateError,
)
from eva.memory.repository import MemoryRepository


def test_exception_hierarchy() -> None:
    """Verify that memory exception classes inherit properly from MemoryError."""
    assert issubclass(MemoryNotFoundError, MemoryError)
    assert issubclass(DuplicateMemoryError, MemoryError)
    assert issubclass(MemoryUpdateError, MemoryError)
    assert issubclass(ImmutableMemoryError, MemoryUpdateError)
    assert issubclass(ImmutableMemoryError, MemoryError)

    # Test instantiation with custom messages
    err = MemoryNotFoundError("Record abc not found")
    assert str(err) == "Record abc not found"
    assert isinstance(err, Exception)


def test_repository_protocol_conformance() -> None:
    """Verify that InMemoryMemoryRepository conforms to the MemoryRepository protocol."""
    repo = InMemoryMemoryRepository()
    assert isinstance(repo, MemoryRepository)

    # Verify expected async methods exist on the repository
    expected_methods = ["save", "get", "update", "delete", "query", "list_all", "count"]
    for method_name in expected_methods:
        method = getattr(repo, method_name, None)
        assert method is not None, f"Method {method_name} missing from repository"
        assert callable(method)
        assert inspect.iscoroutinefunction(method), f"Method {method_name} must be async"
