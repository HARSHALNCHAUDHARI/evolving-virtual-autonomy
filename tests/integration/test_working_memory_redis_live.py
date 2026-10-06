"""Optional live integration tests for RedisWorkingMemoryRepository.

These tests run only when a live Redis instance is available via EVA_REDIS_URL
or default localhost:6379. If Redis is unreachable, the tests skip gracefully.
Never uses FLUSHDB or FLUSHALL; operates strictly within a dedicated UUID-isolated prefix.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
import redis.asyncio as aioredis

from eva.memory.working.config import RedisConfig
from eva.memory.working.models import WorkingMemoryEntry
from eva.memory.working.redis import RedisWorkingMemoryRepository

pytestmark = pytest.mark.integration


async def _get_live_client(config: RedisConfig) -> aioredis.Redis | None:
    """Attempt to establish a live Redis connection with a short timeout."""
    try:
        client = aioredis.from_url(
            config.effective_url,
            socket_connect_timeout=1.0,
            socket_timeout=2.0,
            decode_responses=True,
        )
        await client.ping()
        return client
    except Exception:
        return None


@pytest.mark.asyncio
async def test_live_redis_working_memory_lifecycle() -> None:
    """Execute live working memory operations against a real Redis server if available."""
    # Isolated test prefix using UUID to prevent any test collisions
    test_run_id = uuid4().hex[:8]
    config = RedisConfig.from_env()
    config.key_prefix = f"eva:test:live:{test_run_id}"

    client = await _get_live_client(config)
    if client is None:
        pytest.skip("Live Redis server is not reachable. Skipping integration test.")

    repo = RedisWorkingMemoryRepository(client=client, config=config)

    session_id = f"sess-{test_run_id}"

    try:
        # 1. Set entry
        entry = WorkingMemoryEntry(
            session_id=session_id,
            key="current_focus",
            value={"project": "EVA", "step": "Milestone 1E.4"},
            namespace="planning",
            ttl_seconds=60,
        )
        await repo.set(entry)

        # 2. Get entry
        fetched = await repo.get(session_id, "current_focus", namespace="planning")
        assert fetched is not None
        assert fetched.value == {"project": "EVA", "step": "Milestone 1E.4"}

        # 3. Exists
        assert await repo.exists(session_id, "current_focus", namespace="planning") is True
        assert await repo.exists(session_id, "missing_key", namespace="planning") is False

        # 4. Update entry
        entry.value = {"project": "EVA", "step": "Milestone 1E.4", "status": "active"}
        updated = await repo.update(entry)
        assert updated.value["status"] == "active"

        fetched_after_update = await repo.get(session_id, "current_focus", namespace="planning")
        assert fetched_after_update is not None
        assert fetched_after_update.value["status"] == "active"

        # 5. List session entries
        second_entry = WorkingMemoryEntry(
            session_id=session_id,
            key="recent_tool",
            value="pytest",
            namespace="planning",
        )
        await repo.set(second_entry)

        entries = await repo.list_session(session_id, namespace="planning")
        assert len(entries) == 2
        assert [e.key for e in entries] == ["current_focus", "recent_tool"]

        # 6. TTL manipulation
        ttl_ok = await repo.set_ttl(session_id, "recent_tool", 120, namespace="planning")
        assert ttl_ok is True

        # 7. Delete individual entry
        deleted = await repo.delete(session_id, "recent_tool", namespace="planning")
        assert deleted is True
        assert await repo.exists(session_id, "recent_tool", namespace="planning") is False

        # 8. Clear entire session
        cleared_count = await repo.clear_session(session_id)
        assert cleared_count == 1
        assert await repo.get(session_id, "current_focus", namespace="planning") is None

    finally:
        # Clean up any leftover keys for this test session
        await repo.clear_session(session_id)
        await client.aclose()
