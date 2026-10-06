"""Tests for SemanticMemoryRetriever with cosine similarity and metadata isolation."""

from __future__ import annotations

import pytest
from uuid import uuid4

from eva.memory.embeddings.mock import MockEmbeddingProvider
from eva.memory.embeddings.service import EmbeddingService
from eva.memory.in_memory import InMemoryMemoryRepository
from eva.memory.models import (
    MemoryQuery,
    MemoryRecord,
    MemorySource,
    MemoryType,
    ProvenanceRecord,
)
from eva.memory.retrieval.semantic import (
    SemanticMemoryRetriever,
    compute_cosine_similarity,
)


def test_compute_cosine_similarity() -> None:
    """Cosine similarity calculation is mathematically accurate and clamped to [0, 1]."""
    # Identical vectors
    v1 = [1.0, 0.0, 0.0]
    assert compute_cosine_similarity(v1, v1) == 1.0

    # Orthogonal vectors
    v2 = [0.0, 1.0, 0.0]
    assert compute_cosine_similarity(v1, v2) == 0.0

    # Opposite vectors clamped to 0.0
    v3 = [-1.0, 0.0, 0.0]
    assert compute_cosine_similarity(v1, v3) == 0.0

    # Empty or mismatched vectors
    assert compute_cosine_similarity([], [1.0]) == 0.0
    assert compute_cosine_similarity([1.0, 2.0], [1.0]) == 0.0


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    provider = MockEmbeddingProvider(dimension=64)
    return EmbeddingService(provider=provider, expected_dimension=64)


@pytest.fixture
def memory_repo() -> InMemoryMemoryRepository:
    return InMemoryMemoryRepository()


@pytest.mark.asyncio
async def test_semantic_retrieval_ranking_and_scores(
    memory_repo: InMemoryMemoryRepository,
    mock_embedding_service: EmbeddingService,
) -> None:
    """Semantic retrieval orders records by similarity score descending."""
    retriever = SemanticMemoryRetriever(
        repository=memory_repo,
        embedding_service=mock_embedding_service,
    )

    # Create memory 1: exact match with query concept
    m1 = MemoryRecord(
        memory_type=MemoryType.PREFERENCE,
        content="User prefers dark theme with high contrast fonts.",
        provenance=ProvenanceRecord(source_type=MemorySource.USER_STATEMENT),
        importance=0.8,
    )
    # Create memory 2: completely unrelated concept
    m2 = MemoryRecord(
        memory_type=MemoryType.EPISODIC,
        content="PostgreSQL database replication topology configuration.",
        provenance=ProvenanceRecord(source_type=MemorySource.EVA_OBSERVATION),
        importance=0.5,
    )

    m1_emb = await mock_embedding_service.attach_embedding(m1)
    m2_emb = await mock_embedding_service.attach_embedding(m2)

    await memory_repo.save(m1_emb)
    await memory_repo.save(m2_emb)

    results = await retriever.search(MemoryQuery(query_text="dark theme UI preference"))

    assert len(results) >= 1
    # First result should be m1
    assert results[0].memory.memory_id == m1.memory_id
    # Scores must be in descending order
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score
        assert results[i].explanation.semantic_similarity == pytest.approx(results[i].score, rel=1e-3)


@pytest.mark.asyncio
async def test_semantic_retrieval_skips_records_without_embeddings(
    memory_repo: InMemoryMemoryRepository,
    mock_embedding_service: EmbeddingService,
) -> None:
    """Records lacking an embedding vector are safely skipped rather than crashing."""
    retriever = SemanticMemoryRetriever(
        repository=memory_repo,
        embedding_service=mock_embedding_service,
    )

    # Record without embedding
    m_no_emb = MemoryRecord(
        content="Record without any vector representation.",
        memory_type=MemoryType.EPISODIC,
        provenance=ProvenanceRecord(source_type=MemorySource.SYSTEM_EVENT),
    )
    # Record with embedding
    m_with_emb = MemoryRecord(
        content="Record with dense embedding attached.",
        memory_type=MemoryType.EPISODIC,
        provenance=ProvenanceRecord(source_type=MemorySource.SYSTEM_EVENT),
    )
    m_with_emb = await mock_embedding_service.attach_embedding(m_with_emb)

    await memory_repo.save(m_no_emb)
    await memory_repo.save(m_with_emb)

    results = await retriever.search(MemoryQuery(query_text="vector representation"))
    assert len(results) == 1
    assert results[0].memory.memory_id == m_with_emb.memory_id


@pytest.mark.asyncio
async def test_semantic_retrieval_empty_query_text(
    memory_repo: InMemoryMemoryRepository,
    mock_embedding_service: EmbeddingService,
) -> None:
    """Empty or whitespace-only query text returns empty results without calling provider."""
    retriever = SemanticMemoryRetriever(
        repository=memory_repo,
        embedding_service=mock_embedding_service,
    )

    m = await mock_embedding_service.attach_embedding(
        MemoryRecord(
            content="Sample memory.",
            memory_type=MemoryType.EPISODIC,
            provenance=ProvenanceRecord(source_type=MemorySource.SYSTEM_EVENT),
        )
    )
    await memory_repo.save(m)

    assert await retriever.search(MemoryQuery(query_text="")) == []
    assert await retriever.search(MemoryQuery(query_text="   ")) == []


@pytest.mark.asyncio
async def test_semantic_retrieval_similarity_threshold(
    memory_repo: InMemoryMemoryRepository,
    mock_embedding_service: EmbeddingService,
) -> None:
    """Candidates below similarity_threshold are excluded."""
    retriever = SemanticMemoryRetriever(
        repository=memory_repo,
        embedding_service=mock_embedding_service,
        similarity_threshold=0.9999,  # Very strict threshold
    )

    m = await mock_embedding_service.attach_embedding(
        MemoryRecord(
            content="General unrelated information.",
            memory_type=MemoryType.EPISODIC,
            provenance=ProvenanceRecord(source_type=MemorySource.SYSTEM_EVENT),
        )
    )
    await memory_repo.save(m)

    # Completely different text will not meet 0.9999 threshold
    results = await retriever.search(MemoryQuery(query_text="Specialized query keywords"))
    assert len(results) == 0


@pytest.mark.asyncio
async def test_semantic_retrieval_project_and_session_isolation(
    memory_repo: InMemoryMemoryRepository,
    mock_embedding_service: EmbeddingService,
) -> None:
    """Semantic retrieval enforces strict project and session isolation."""
    retriever = SemanticMemoryRetriever(
        repository=memory_repo,
        embedding_service=mock_embedding_service,
    )

    proj_a = "project-alpha"
    proj_b = "project-beta"
    sess_1 = "session-001"
    sess_2 = "session-002"

    m_alpha = await mock_embedding_service.attach_embedding(
        MemoryRecord(
            content="Alpha project authentication secret key.",
            memory_type=MemoryType.PROJECT,
            project_id=proj_a,
            session_id=sess_1,
            provenance=ProvenanceRecord(source_type=MemorySource.USER_STATEMENT),
        )
    )
    m_beta = await mock_embedding_service.attach_embedding(
        MemoryRecord(
            content="Beta project authentication secret key.",
            memory_type=MemoryType.PROJECT,
            project_id=proj_b,
            session_id=sess_2,
            provenance=ProvenanceRecord(source_type=MemorySource.USER_STATEMENT),
        )
    )

    await memory_repo.save(m_alpha)
    await memory_repo.save(m_beta)

    # Query for Alpha project: must NOT return Beta project memory
    results_alpha = await retriever.search(
        MemoryQuery(query_text="authentication secret key", project_id=proj_a)
    )
    assert len(results_alpha) == 1
    assert results_alpha[0].memory.memory_id == m_alpha.memory_id
    assert results_alpha[0].memory.project_id == proj_a

    # Query with session isolation
    results_sess = await retriever.search(
        MemoryQuery(query_text="authentication secret key", session_id=sess_2)
    )
    assert len(results_sess) == 1
    assert results_sess[0].memory.memory_id == m_beta.memory_id


@pytest.mark.asyncio
async def test_semantic_retrieval_pagination_and_count(
    memory_repo: InMemoryMemoryRepository,
    mock_embedding_service: EmbeddingService,
) -> None:
    """Pagination (limit, offset) and count work accurately in semantic search."""
    retriever = SemanticMemoryRetriever(
        repository=memory_repo,
        embedding_service=mock_embedding_service,
    )

    for i in range(5):
        m = await mock_embedding_service.attach_embedding(
            MemoryRecord(
                content=f"Memory record batch entry number {i}.",
                memory_type=MemoryType.EPISODIC,
                provenance=ProvenanceRecord(source_type=MemorySource.SYSTEM_EVENT),
            )
        )
        await memory_repo.save(m)

    query = MemoryQuery(query_text="batch entry", limit=2, offset=1)
    results = await retriever.search(query)
    assert len(results) == 2

    total = await retriever.count(query)
    assert total == 5

    response = await retriever.search_with_response(query)
    assert len(response.results) == 2
    assert response.total_candidates == 5
