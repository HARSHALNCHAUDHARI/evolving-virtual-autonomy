"""Unit tests for RedisWorkingMemoryRepository using mocked Redis client."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as aioredis

from eva.memory.working.config import RedisConfig
from eva.memory.working.models import (
    WorkingMemoryConnectionError,
    WorkingMemoryEntry,
    WorkingMemoryError,
    WorkingMemoryNotFoundError,
)
from eva.memory.working.redis import RedisWorkingMemoryRepository
from eva.memory.working.serialization import serialize_entry


class FakeRedisClient:
    """In-memory async fake for redis.asyncio.Redis to test repository operations offline."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    async def ping(self) -> bool:
        return True

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.store[key] = value
        if ex is not None:
            self.ttls[key] = ex
        elif key in self.ttls:
            del self.ttls[key]
        return True

    async def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                self.ttls.pop(k, None)
                count += 1
        return count

    async def exists(self, key: str) -> int:
        return 1 if key in self.store else 0

    async def ttl(self, key: str) -> int:
        return self.ttls.get(key, -1)

    async def persist(self, key: str) -> bool:
        if key in self.ttls:
            del self.ttls[key]
            return True
        return False

    async def scan_iter(self, match: str = "*", count: int = 100):
        # Translate glob match pattern
        prefix = match.rstrip("*")
        for k in list(self.store.keys()):
            if k.startswith(prefix):
                yield k

    async def mget(self, keys: list[str]) -> list[str | None]:
        return [self.store.get(k) for k in keys]

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, client: FakeRedisClient) -> None:
        self.client = client
        self.commands: list[tuple[str, tuple]] = []

    def delete(self, key: str) -> FakePipeline:
        self.commands.append(("delete", (key,)))
        return self

    async def execute(self) -> list[int]:
        results: list[int] = []
        for cmd, args in self.commands:
            if cmd == "delete":
                res = await self.client.delete(*args)
                results.append(res)
        self.commands.clear()
        return results


@pytest.fixture
def fake_client() -> FakeRedisClient:
    return FakeRedisClient()


@pytest.fixture
def repo(fake_client: FakeRedisClient) -> RedisWorkingMemoryRepository:
    config = RedisConfig(key_prefix="test:working")
    return RedisWorkingMemoryRepository(client=fake_client, config=config)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_redis_set_and_get(repo: RedisWorkingMemoryRepository) -> None:
    entry = WorkingMemoryEntry(
        session_id="s1",
        key="active_task",
        value={"id": 42, "status": "in_progress"},
        namespace="tasks",
        ttl_seconds=300,
    )
    saved = await repo.set(entry)
    assert saved.key == "active_task"

    retrieved = await repo.get("s1", "active_task", namespace="tasks")
    assert retrieved is not None
    assert retrieved.session_id == "s1"
    assert retrieved.key == "active_task"
    assert retrieved.value == {"id": 42, "status": "in_progress"}
    assert retrieved.namespace == "tasks"


@pytest.mark.asyncio
async def test_redis_get_missing(repo: RedisWorkingMemoryRepository) -> None:
    assert await repo.get("s1", "nonexistent") is None


@pytest.mark.asyncio
async def test_redis_exists(repo: RedisWorkingMemoryRepository) -> None:
    entry = WorkingMemoryEntry(session_id="s1", key="flag", value=True)
    await repo.set(entry)

    assert await repo.exists("s1", "flag") is True
    assert await repo.exists("s1", "not_there") is False


@pytest.mark.asyncio
async def test_redis_update(repo: RedisWorkingMemoryRepository) -> None:
    entry = WorkingMemoryEntry(session_id="s1", key="state", value="init")
    await repo.set(entry)

    entry.value = "ready"
    updated = await repo.update(entry)
    assert updated.value == "ready"

    retrieved = await repo.get("s1", "state")
    assert retrieved is not None and retrieved.value == "ready"


@pytest.mark.asyncio
async def test_redis_update_missing_raises(repo: RedisWorkingMemoryRepository) -> None:
    entry = WorkingMemoryEntry(session_id="s1", key="missing", value="val")
    with pytest.raises(WorkingMemoryNotFoundError):
        await repo.update(entry)


@pytest.mark.asyncio
async def test_redis_delete(repo: RedisWorkingMemoryRepository) -> None:
    entry = WorkingMemoryEntry(session_id="s1", key="temp", value=123)
    await repo.set(entry)

    assert await repo.delete("s1", "temp") is True
    assert await repo.delete("s1", "temp") is False

    with pytest.raises(WorkingMemoryNotFoundError):
        await repo.delete("s1", "temp", raise_if_missing=True)


@pytest.mark.asyncio
async def test_redis_clear_session(repo: RedisWorkingMemoryRepository) -> None:
    await repo.set(WorkingMemoryEntry(session_id="s1", key="k1", value="v1", namespace="ns1"))
    await repo.set(WorkingMemoryEntry(session_id="s1", key="k2", value="v2", namespace="ns2"))
    await repo.set(WorkingMemoryEntry(session_id="s2", key="k1", value="keep_me"))

    deleted = await repo.clear_session("s1")
    assert deleted == 2

    assert await repo.get("s1", "k1", namespace="ns1") is None
    assert await repo.get("s1", "k2", namespace="ns2") is None
    assert await repo.get("s2", "k1") is not None


@pytest.mark.asyncio
async def test_redis_list_session(repo: RedisWorkingMemoryRepository) -> None:
    await repo.set(WorkingMemoryEntry(session_id="s1", key="b", value="val_b"))
    await repo.set(WorkingMemoryEntry(session_id="s1", key="a", value="val_a"))
    await repo.set(WorkingMemoryEntry(session_id="s1", key="z", value="val_z", namespace="ns_extra"))

    entries = await repo.list_session("s1")
    assert len(entries) == 3
    # Check deterministic ordering: namespace then key
    assert [(e.namespace, e.key) for e in entries] == [
        ("default", "a"),
        ("default", "b"),
        ("ns_extra", "z"),
    ]

    limited = await repo.list_session("s1", limit=1)
    assert len(limited) == 1
    assert limited[0].key == "a"


@pytest.mark.asyncio
async def test_redis_set_ttl(repo: RedisWorkingMemoryRepository, fake_client: FakeRedisClient) -> None:
    entry = WorkingMemoryEntry(session_id="s1", key="k", value="v")
    await repo.set(entry)

    # Set TTL
    ok = await repo.set_ttl("s1", "k", 600)
    assert ok is True
    redis_key = repo.config.build_redis_key("s1", "k")
    assert fake_client.ttls.get(redis_key) == 600

    # Clear TTL (persist)
    ok = await repo.set_ttl("s1", "k", None)
    assert ok is True
    assert redis_key not in fake_client.ttls

    # TTL = 0 expires immediately
    ok = await repo.set_ttl("s1", "k", 0)
    assert ok is True
    assert await repo.get("s1", "k") is None


@pytest.mark.asyncio
async def test_redis_availability_check(repo: RedisWorkingMemoryRepository) -> None:
    assert await repo.is_available() is True


@pytest.mark.asyncio
async def test_redis_unreachable_availability_check() -> None:
    mock_client = AsyncMock()
    mock_client.ping.side_effect = ConnectionError("Connection refused")
    repo = RedisWorkingMemoryRepository(client=mock_client)  # type: ignore[arg-type]
    assert await repo.is_available() is False


@pytest.mark.asyncio
async def test_redis_operation_error_wrapping() -> None:
    mock_client = AsyncMock()
    mock_client.get.side_effect = aioredis.RedisError("Cluster down")
    repo = RedisWorkingMemoryRepository(client=mock_client)  # type: ignore[arg-type]

    with pytest.raises(WorkingMemoryError, match="Failed to get working memory from Redis"):
        await repo.get("s1", "key")
