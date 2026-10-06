"""Unit tests for InMemoryWorkingMemoryRepository."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from eva.memory.working.in_memory import InMemoryWorkingMemoryRepository
from eva.memory.working.models import WorkingMemoryEntry, WorkingMemoryNotFoundError


@pytest.fixture
def repo() -> InMemoryWorkingMemoryRepository:
    return InMemoryWorkingMemoryRepository()


@pytest.mark.asyncio
async def test_set_and_get(repo: InMemoryWorkingMemoryRepository) -> None:
    entry = WorkingMemoryEntry(
        session_id="session-1",
        key="goal",
        value={"text": "Write code", "priority": 1},
        namespace="active_task",
    )
    saved = await repo.set(entry)
    assert saved.key == "goal"
    assert saved.value == {"text": "Write code", "priority": 1}

    retrieved = await repo.get("session-1", "goal", namespace="active_task")
    assert retrieved is not None
    assert retrieved.key == "goal"
    assert retrieved.value == {"text": "Write code", "priority": 1}
    assert retrieved.session_id == "session-1"
    assert retrieved.namespace == "active_task"


@pytest.mark.asyncio
async def test_get_nonexistent(repo: InMemoryWorkingMemoryRepository) -> None:
    assert await repo.get("session-unknown", "missing") is None


@pytest.mark.asyncio
async def test_session_isolation(repo: InMemoryWorkingMemoryRepository) -> None:
    e1 = WorkingMemoryEntry(session_id="s1", key="topic", value="Machine Learning")
    e2 = WorkingMemoryEntry(session_id="s2", key="topic", value="Web Development")

    await repo.set(e1)
    await repo.set(e2)

    s1_val = await repo.get("s1", "topic")
    s2_val = await repo.get("s2", "topic")

    assert s1_val is not None and s1_val.value == "Machine Learning"
    assert s2_val is not None and s2_val.value == "Web Development"


@pytest.mark.asyncio
async def test_namespace_isolation(repo: InMemoryWorkingMemoryRepository) -> None:
    e1 = WorkingMemoryEntry(session_id="s1", key="target", value="backend", namespace="arch")
    e2 = WorkingMemoryEntry(session_id="s1", key="target", value="react", namespace="frontend")

    await repo.set(e1)
    await repo.set(e2)

    arch_entry = await repo.get("s1", "target", namespace="arch")
    frontend_entry = await repo.get("s1", "target", namespace="frontend")

    assert arch_entry is not None and arch_entry.value == "backend"
    assert frontend_entry is not None and frontend_entry.value == "react"
    assert await repo.get("s1", "target", namespace="default") is None


@pytest.mark.asyncio
async def test_update(repo: InMemoryWorkingMemoryRepository) -> None:
    entry = WorkingMemoryEntry(session_id="s1", key="counter", value=1)
    await repo.set(entry)

    entry.value = 2
    updated = await repo.update(entry)
    assert updated.value == 2
    assert updated.updated_at >= entry.created_at

    retrieved = await repo.get("s1", "counter")
    assert retrieved is not None and retrieved.value == 2


@pytest.mark.asyncio
async def test_update_nonexistent_raises(repo: InMemoryWorkingMemoryRepository) -> None:
    entry = WorkingMemoryEntry(session_id="s1", key="missing", value="test")
    with pytest.raises(WorkingMemoryNotFoundError):
        await repo.update(entry)


@pytest.mark.asyncio
async def test_delete(repo: InMemoryWorkingMemoryRepository) -> None:
    entry = WorkingMemoryEntry(session_id="s1", key="temp", value="val")
    await repo.set(entry)

    assert await repo.exists("s1", "temp") is True
    assert await repo.delete("s1", "temp") is True
    assert await repo.exists("s1", "temp") is False
    assert await repo.delete("s1", "temp") is False

    with pytest.raises(WorkingMemoryNotFoundError):
        await repo.delete("s1", "temp", raise_if_missing=True)


@pytest.mark.asyncio
async def test_clear_session_all_namespaces(repo: InMemoryWorkingMemoryRepository) -> None:
    await repo.set(WorkingMemoryEntry(session_id="s1", key="k1", value="v1", namespace="ns1"))
    await repo.set(WorkingMemoryEntry(session_id="s1", key="k2", value="v2", namespace="ns2"))
    await repo.set(WorkingMemoryEntry(session_id="s2", key="k1", value="other"))

    deleted = await repo.clear_session("s1")
    assert deleted == 2

    assert await repo.get("s1", "k1", namespace="ns1") is None
    assert await repo.get("s1", "k2", namespace="ns2") is None
    # s2 remains untouched
    assert await repo.get("s2", "k1") is not None


@pytest.mark.asyncio
async def test_clear_session_specific_namespace(repo: InMemoryWorkingMemoryRepository) -> None:
    await repo.set(WorkingMemoryEntry(session_id="s1", key="k1", value="v1", namespace="ns1"))
    await repo.set(WorkingMemoryEntry(session_id="s1", key="k2", value="v2", namespace="ns2"))

    deleted = await repo.clear_session("s1", namespace="ns1")
    assert deleted == 1

    assert await repo.get("s1", "k1", namespace="ns1") is None
    assert await repo.get("s1", "k2", namespace="ns2") is not None


@pytest.mark.asyncio
async def test_list_session(repo: InMemoryWorkingMemoryRepository) -> None:
    await repo.set(WorkingMemoryEntry(session_id="s1", key="b", value="val_b"))
    await repo.set(WorkingMemoryEntry(session_id="s1", key="a", value="val_a"))
    await repo.set(WorkingMemoryEntry(session_id="s1", key="c", value="val_c", namespace="other"))
    await repo.set(WorkingMemoryEntry(session_id="s2", key="a", value="s2_a"))

    all_s1 = await repo.list_session("s1")
    assert len(all_s1) == 3
    # Check deterministic ordering: namespace then key
    assert [(e.namespace, e.key) for e in all_s1] == [
        ("default", "a"),
        ("default", "b"),
        ("other", "c"),
    ]

    limited = await repo.list_session("s1", limit=2)
    assert len(limited) == 2
    assert [e.key for e in limited] == ["a", "b"]

    ns_filtered = await repo.list_session("s1", namespace="other")
    assert len(ns_filtered) == 1
    assert ns_filtered[0].key == "c"


@pytest.mark.asyncio
async def test_ttl_expiration(repo: InMemoryWorkingMemoryRepository) -> None:
    # Explicit past expires_at
    expired_entry = WorkingMemoryEntry(
        session_id="s1",
        key="expired_k",
        value="should_vanish",
        expires_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    await repo.set(expired_entry)

    assert await repo.get("s1", "expired_k") is None
    assert await repo.exists("s1", "expired_k") is False

    listed = await repo.list_session("s1")
    assert len(listed) == 0


@pytest.mark.asyncio
async def test_ttl_immediate_expiry(repo: InMemoryWorkingMemoryRepository) -> None:
    entry = WorkingMemoryEntry(
        session_id="s1",
        key="instant_expire",
        value="gone",
        ttl_seconds=0,
    )
    await repo.set(entry)
    assert await repo.get("s1", "instant_expire") is None


@pytest.mark.asyncio
async def test_set_ttl_operation(repo: InMemoryWorkingMemoryRepository) -> None:
    entry = WorkingMemoryEntry(session_id="s1", key="k", value="val")
    await repo.set(entry)

    # Entry is persistent initially
    saved = await repo.get("s1", "k")
    assert saved is not None and saved.expires_at is None

    # Set positive TTL
    success = await repo.set_ttl("s1", "k", 300)
    assert success is True
    updated = await repo.get("s1", "k")
    assert updated is not None
    assert updated.ttl_seconds == 300
    assert updated.expires_at is not None

    # Clear TTL (make persistent again)
    success = await repo.set_ttl("s1", "k", None)
    assert success is True
    persistent = await repo.get("s1", "k")
    assert persistent is not None
    assert persistent.expires_at is None
    assert persistent.ttl_seconds is None

    # Set TTL = 0 expires immediately
    success = await repo.set_ttl("s1", "k", 0)
    assert success is True
    assert await repo.get("s1", "k") is None


@pytest.mark.asyncio
async def test_mutation_safety(repo: InMemoryWorkingMemoryRepository) -> None:
    original = {"items": [1, 2, 3]}
    entry = WorkingMemoryEntry(session_id="s1", key="list", value=original)
    await repo.set(entry)

    original["items"].append(4)  # type: ignore[attr-defined]

    stored = await repo.get("s1", "list")
    assert stored is not None
    assert stored.value == {"items": [1, 2, 3]}
