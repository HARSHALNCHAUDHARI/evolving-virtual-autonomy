"""Unit tests for the InMemoryMemoryRepository implementation."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from eva.memory.in_memory import InMemoryMemoryRepository
from eva.memory.models import (
    DuplicateMemoryError,
    ImmutableMemoryError,
    MemoryNotFoundError,
    MemoryQuery,
    MemoryRecord,
    MemorySource,
    MemoryType,
    ProvenanceRecord,
    RetentionMetadata,
)


def _make_record(
    content: str = "Test memory",
    memory_type: MemoryType = MemoryType.WORKING,
    source: MemorySource = MemorySource.USER_STATEMENT,
    confidence: float = 1.0,
    importance: float = 0.5,
    project_id: str | None = None,
    session_id: str | None = None,
    is_immutable: bool = False,
    is_archived: bool = False,
    created_at: datetime | None = None,
) -> MemoryRecord:
    prov = ProvenanceRecord(source_type=source)
    kwargs = {
        "memory_type": memory_type,
        "content": content,
        "provenance": prov,
        "confidence": confidence,
        "importance": importance,
        "project_id": project_id,
        "session_id": session_id,
        "retention_metadata": RetentionMetadata(
            is_immutable=is_immutable,
            is_archived=is_archived,
        ),
    }
    if created_at is not None:
        kwargs["created_at"] = created_at
        kwargs["updated_at"] = created_at
    return MemoryRecord(**kwargs)


@pytest.mark.asyncio
async def test_save_and_get() -> None:
    """Test saving a record and retrieving it by ID with mutation isolation."""
    repo = InMemoryMemoryRepository()
    record = _make_record(content="Learn Python async")

    saved = await repo.save(record)
    assert saved.memory_id == record.memory_id
    assert saved.content == "Learn Python async"

    # Verify retrieved record
    retrieved = await repo.get(record.memory_id)
    assert retrieved is not None
    assert retrieved.memory_id == record.memory_id
    assert retrieved.content == "Learn Python async"

    # Verify safe handling of mutable state: mutating retrieved doesn't mutate repo state
    retrieved.content = "Mutated content externally"
    second_get = await repo.get(record.memory_id)
    assert second_get is not None
    assert second_get.content == "Learn Python async"


@pytest.mark.asyncio
async def test_save_duplicate_id_raises() -> None:
    """Saving a record with an existing ID must raise DuplicateMemoryError."""
    repo = InMemoryMemoryRepository()
    record = _make_record(content="Initial entry")
    await repo.save(record)

    duplicate = record.model_copy()
    duplicate.content = "Conflicting duplicate"

    with pytest.raises(DuplicateMemoryError) as exc_info:
        await repo.save(duplicate)
    assert str(record.memory_id) in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_nonexistent_returns_none() -> None:
    """Retrieving an unknown memory ID returns None."""
    repo = InMemoryMemoryRepository()
    assert await repo.get(uuid4()) is None


@pytest.mark.asyncio
async def test_update_existing_record() -> None:
    """Updating a record must preserve created_at and memory_id while refreshing updated_at."""
    repo = InMemoryMemoryRepository()
    initial_time = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    record = _make_record(content="Before update", created_at=initial_time)
    await repo.save(record)

    # Attempt to tamper with created_at on update
    record_to_update = record.model_copy(deep=True)
    record_to_update.content = "After update"
    record_to_update.importance = 0.95
    record_to_update.created_at = datetime(2030, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    updated = await repo.update(record_to_update)
    assert updated.memory_id == record.memory_id
    assert updated.content == "After update"
    assert updated.importance == 0.95

    # created_at must be preserved from the original record
    assert updated.created_at == initial_time
    assert updated.access_metadata.created_at == initial_time

    # updated_at must be refreshed
    assert updated.updated_at > initial_time
    assert updated.access_metadata.updated_at == updated.updated_at

    # Verify state in repository
    persisted = await repo.get(record.memory_id)
    assert persisted is not None
    assert persisted.content == "After update"
    assert persisted.created_at == initial_time


@pytest.mark.asyncio
async def test_update_nonexistent_raises() -> None:
    """Updating a nonexistent record must raise MemoryNotFoundError."""
    repo = InMemoryMemoryRepository()
    unknown = _make_record(content="Missing record")

    with pytest.raises(MemoryNotFoundError):
        await repo.update(unknown)


@pytest.mark.asyncio
async def test_update_immutable_raises() -> None:
    """Attempting to update an immutable record must raise ImmutableMemoryError."""
    repo = InMemoryMemoryRepository()
    immutable_record = _make_record(
        content="System Safety Directive: Never rm -rf /",
        memory_type=MemoryType.PREFERENCE,
        is_immutable=True,
    )
    await repo.save(immutable_record)

    tampered = immutable_record.model_copy(deep=True)
    tampered.content = "Modified safety policy"

    with pytest.raises(ImmutableMemoryError):
        await repo.update(tampered)


@pytest.mark.asyncio
async def test_delete_existing_and_missing() -> None:
    """Test deletion of existing and missing records."""
    repo = InMemoryMemoryRepository()
    record = _make_record(content="Temporary note")
    await repo.save(record)
    assert await repo.count() == 1

    # Delete existing
    assert await repo.delete(record.memory_id) is True
    assert await repo.count() == 0
    assert await repo.get(record.memory_id) is None

    # Delete missing without raise
    unknown_id = uuid4()
    assert await repo.delete(unknown_id, raise_if_missing=False) is False

    # Delete missing with raise_if_missing=True
    with pytest.raises(MemoryNotFoundError):
        await repo.delete(unknown_id, raise_if_missing=True)


@pytest.mark.asyncio
async def test_count_and_list_all() -> None:
    """Test count and list_all operations."""
    repo = InMemoryMemoryRepository()
    assert await repo.count() == 0
    assert await repo.list_all() == []

    r1 = _make_record(content="Item 1")
    r2 = _make_record(content="Item 2")
    await repo.save(r1)
    await repo.save(r2)

    assert await repo.count() == 2
    all_items = await repo.list_all()
    assert len(all_items) == 2
    ids = {item.memory_id for item in all_items}
    assert ids == {r1.memory_id, r2.memory_id}


@pytest.mark.asyncio
async def test_query_filter_by_memory_types() -> None:
    """Test querying by single and multiple memory types."""
    repo = InMemoryMemoryRepository()
    r_work = _make_record(content="W1", memory_type=MemoryType.WORKING)
    r_pref = _make_record(content="P1", memory_type=MemoryType.PREFERENCE)
    r_ep = _make_record(content="E1", memory_type=MemoryType.EPISODIC)

    await repo.save(r_work)
    await repo.save(r_pref)
    await repo.save(r_ep)

    # Filter single type
    q1 = MemoryQuery(memory_types=[MemoryType.PREFERENCE])
    results1 = await repo.query(q1)
    assert len(results1) == 1
    assert results1[0].memory_id == r_pref.memory_id

    # Filter multiple types
    q2 = MemoryQuery(memory_types=[MemoryType.WORKING, MemoryType.EPISODIC])
    results2 = await repo.query(q2)
    assert len(results2) == 2
    res_ids = {r.memory_id for r in results2}
    assert res_ids == {r_work.memory_id, r_ep.memory_id}


@pytest.mark.asyncio
async def test_query_filter_project_isolation() -> None:
    """Test workspace/project isolation with and without global inheritance."""
    repo = InMemoryMemoryRepository()
    r_global = _make_record(content="Global preference", project_id=None)
    r_eva = _make_record(content="EVA architecture", project_id="eva")
    r_portal = _make_record(content="Portal dashboard", project_id="portal")

    await repo.save(r_global)
    await repo.save(r_eva)
    await repo.save(r_portal)

    # Query project 'eva' with global inheritance (default)
    q_eva = MemoryQuery(project_id="eva", include_global=True)
    res_eva = await repo.query(q_eva)
    eva_ids = {r.memory_id for r in res_eva}
    assert eva_ids == {r_global.memory_id, r_eva.memory_id}
    assert r_portal.memory_id not in eva_ids

    # Query project 'eva' strictly (without global)
    q_eva_strict = MemoryQuery(project_id="eva", include_global=False)
    res_strict = await repo.query(q_eva_strict)
    assert len(res_strict) == 1
    assert res_strict[0].memory_id == r_eva.memory_id


@pytest.mark.asyncio
async def test_query_filter_session_isolation() -> None:
    """Test session boundary filtering."""
    repo = InMemoryMemoryRepository()
    r_s1 = _make_record(content="Session 1 turn", session_id="sess-001")
    r_s2 = _make_record(content="Session 2 turn", session_id="sess-002")
    await repo.save(r_s1)
    await repo.save(r_s2)

    q = MemoryQuery(session_id="sess-001")
    results = await repo.query(q)
    assert len(results) == 1
    assert results[0].memory_id == r_s1.memory_id


@pytest.mark.asyncio
async def test_query_filter_source_provenance() -> None:
    """Test filtering by memory source."""
    repo = InMemoryMemoryRepository()
    r_user = _make_record(content="User says hello", source=MemorySource.USER_STATEMENT)
    r_tool = _make_record(content="File created", source=MemorySource.TOOL_RESULT)
    r_infer = _make_record(content="User may be developer", source=MemorySource.MODEL_INFERENCE)

    await repo.save(r_user)
    await repo.save(r_tool)
    await repo.save(r_infer)

    q = MemoryQuery(sources=[MemorySource.USER_STATEMENT, MemorySource.TOOL_RESULT])
    results = await repo.query(q)
    ids = {r.memory_id for r in results}
    assert ids == {r_user.memory_id, r_tool.memory_id}
    assert r_infer.memory_id not in ids


@pytest.mark.asyncio
async def test_query_filter_confidence_and_importance() -> None:
    """Test filtering by minimum confidence and importance thresholds."""
    repo = InMemoryMemoryRepository()
    r_low_conf = _make_record(content="Speculation", confidence=0.3, importance=0.9)
    r_high_conf = _make_record(content="Fact", confidence=0.95, importance=0.4)
    r_high_both = _make_record(content="Critical fact", confidence=0.99, importance=0.95)

    await repo.save(r_low_conf)
    await repo.save(r_high_conf)
    await repo.save(r_high_both)

    # Minimum confidence >= 0.9
    q_conf = MemoryQuery(min_confidence=0.9)
    res_conf = await repo.query(q_conf)
    assert {r.memory_id for r in res_conf} == {r_high_conf.memory_id, r_high_both.memory_id}

    # Minimum importance >= 0.8
    q_imp = MemoryQuery(min_importance=0.8)
    res_imp = await repo.query(q_imp)
    assert {r.memory_id for r in res_imp} == {r_low_conf.memory_id, r_high_both.memory_id}

    # Combined confidence >= 0.9 AND importance >= 0.8
    q_both = MemoryQuery(min_confidence=0.9, min_importance=0.8)
    res_both = await repo.query(q_both)
    assert len(res_both) == 1
    assert res_both[0].memory_id == r_high_both.memory_id


@pytest.mark.asyncio
async def test_query_filter_time_range() -> None:
    """Test filtering by creation and update time ranges."""
    repo = InMemoryMemoryRepository()
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=5)
    t2 = t0 + timedelta(days=10)

    r0 = _make_record(content="Early", created_at=t0)
    r1 = _make_record(content="Mid", created_at=t1)
    r2 = _make_record(content="Late", created_at=t2)

    await repo.save(r0)
    await repo.save(r1)
    await repo.save(r2)

    # Window between day 2 and day 7
    q = MemoryQuery(
        created_after=t0 + timedelta(days=2),
        created_before=t0 + timedelta(days=7),
    )
    results = await repo.query(q)
    assert len(results) == 1
    assert results[0].memory_id == r1.memory_id


@pytest.mark.asyncio
async def test_query_filter_is_archived() -> None:
    """Test default active-only filtering and archived overrides."""
    repo = InMemoryMemoryRepository()
    r_active = _make_record(content="Active memory", is_archived=False)
    r_archived = _make_record(content="Archived memory", is_archived=True)

    await repo.save(r_active)
    await repo.save(r_archived)

    # Default is_archived=False (active only)
    q_active = MemoryQuery()
    res_active = await repo.query(q_active)
    assert len(res_active) == 1
    assert res_active[0].memory_id == r_active.memory_id

    # Query archived only
    q_archived = MemoryQuery(is_archived=True)
    res_archived = await repo.query(q_archived)
    assert len(res_archived) == 1
    assert res_archived[0].memory_id == r_archived.memory_id

    # Query both active and archived
    q_all = MemoryQuery(is_archived=None)
    res_all = await repo.query(q_all)
    assert len(res_all) == 2


@pytest.mark.asyncio
async def test_query_limit_and_deterministic_order() -> None:
    """Test query limit and deterministic reverse-chronological ordering."""
    repo = InMemoryMemoryRepository()
    base_time = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    records = [
        _make_record(content=f"Item {i}", created_at=base_time + timedelta(minutes=i))
        for i in range(5)
    ]
    for r in records:
        await repo.save(r)

    # Query with limit 3: should return items 4, 3, 2 (newest first)
    q = MemoryQuery(limit=3)
    results = await repo.query(q)
    assert len(results) == 3
    assert results[0].content == "Item 4"
    assert results[1].content == "Item 3"
    assert results[2].content == "Item 2"


@pytest.mark.asyncio
async def test_in_memory_repo_clear() -> None:
    """Test repository clear helper."""
    repo = InMemoryMemoryRepository()
    await repo.save(_make_record(content="To clear"))
    assert await repo.count() == 1
    await repo.clear()
    assert await repo.count() == 0
