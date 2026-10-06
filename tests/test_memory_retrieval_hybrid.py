"""Tests for HybridMemoryRetriever combining deterministic keyword and vector search."""

from __future__ import annotations

import pytest
from uuid import uuid4

from eva.memory.embeddings.mock import MockEmbeddingProvider
from eva.memory.embeddings.service import EmbeddingService
from eva.memory.in_memory import InMemoryMemoryRepository
from eva.memory.models import (
    MemoryQuery,
    MemoryRecord,
    MemoryRetrievalError,
    MemorySource,
    MemoryType,
    ProvenanceRecord,
)
from eva.memory.retrieval.hybrid import HybridMemoryRetriever, HybridWeights
from eva.memory.retrieval.retriever import DeterministicMemoryRetriever
from eva.memory.retrieval.semantic import SemanticMemoryRetriever


def test_hybrid_weights_validation() -> None:
    """HybridWeights ensures configured weights are normalized and sum to 1.0."""
    # Valid default weights (0.45 + 0.25 + 0.15 + 0.10 + 0.05 = 1.0)
    w = HybridWeights()
    assert w.semantic_weight == 0.45

    # Valid custom weights
    custom = HybridWeights(
        semantic_weight=0.50,
        keyword_weight=0.30,
        importance_weight=0.10,
        recency_weight=0.05,
        confidence_weight=0.05,
    )
    assert custom.semantic_weight == 0.50

    # Invalid weights not summing to 1.0
    with pytest.raises(ValueError, match="must sum to 1.0"):
        HybridWeights(
            semantic_weight=0.80,
            keyword_weight=0.80,
            importance_weight=0.10,
            recency_weight=0.10,
            confidence_weight=0.10,
        )


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    provider = MockEmbeddingProvider(dimension=64)
    return EmbeddingService(provider=provider, expected_dimension=64)


@pytest.fixture
def memory_repo() -> InMemoryMemoryRepository:
    return InMemoryMemoryRepository()


@pytest.mark.asyncio
async def test_hybrid_combined_ranking_and_formula(
    memory_repo: InMemoryMemoryRepository,
    mock_embedding_service: EmbeddingService,
) -> None:
    """Hybrid score accurately computes composite weighted score."""
    provider = MockEmbeddingProvider(
        dimension=64,
        preset_vectors={
            "deploy eva orchestrator to production kubernetes cluster.": [1.0] + [0.0] * 63,
            "deploy eva production cluster": [0.8, 0.6] + [0.0] * 62,
        },
    )
    custom_emb_service = EmbeddingService(provider=provider, expected_dimension=64)

    det_retriever = DeterministicMemoryRetriever(repository=memory_repo)
    sem_retriever = SemanticMemoryRetriever(
        repository=memory_repo,
        embedding_service=custom_emb_service,
    )
    weights = HybridWeights(
        semantic_weight=0.40,
        keyword_weight=0.30,
        importance_weight=0.15,
        recency_weight=0.10,
        confidence_weight=0.05,
    )
    hybrid_retriever = HybridMemoryRetriever(
        deterministic_retriever=det_retriever,
        semantic_retriever=sem_retriever,
        weights=weights,
    )

    m = await custom_emb_service.attach_embedding(
        MemoryRecord(
            content="Deploy EVA orchestrator to production kubernetes cluster.",
            memory_type=MemoryType.EPISODIC,
            importance=0.9,
            confidence=0.8,
            provenance=ProvenanceRecord(source_type=MemorySource.USER_STATEMENT),
        )
    )
    await memory_repo.save(m)

    query = MemoryQuery(query_text="deploy EVA production cluster")
    results = await hybrid_retriever.search(query)

    assert len(results) == 1
    res = results[0]
    exp = res.explanation

    # Verify score breakdown is populated
    assert exp.semantic_similarity > 0.0
    assert exp.text_score > 0.0
    assert exp.importance_score == pytest.approx(0.9, abs=1e-3)
    assert exp.confidence_score == pytest.approx(0.8, abs=1e-3)
    assert exp.recency_score > 0.0

    # Verify composite score calculation
    expected_score = (
        (0.40 * exp.semantic_similarity)
        + (0.30 * exp.text_score)
        + (0.15 * m.importance)
        + (0.10 * exp.recency_score)
        + (0.05 * m.confidence)
    )
    assert res.score == pytest.approx(expected_score, abs=1e-5)
    assert any("Hybrid composite score:" in r for r in exp.reasons)


@pytest.mark.asyncio
async def test_hybrid_fallback_on_semantic_failure(
    memory_repo: InMemoryMemoryRepository,
) -> None:
    """Hybrid retriever falls back to deterministic retrieval if semantic retrieval fails."""
    det_retriever = DeterministicMemoryRetriever(repository=memory_repo)

    class BrokenSemanticRetriever(SemanticMemoryRetriever):
        async def _get_scored_candidates(self, query: MemoryQuery):
            raise MemoryRetrievalError("Vector database cluster unavailable")

    broken_sem = BrokenSemanticRetriever(
        repository=memory_repo,
        embedding_service=EmbeddingService(MockEmbeddingProvider()),
    )

    m = MemoryRecord(
        content="Important database password rotation guideline.",
        memory_type=MemoryType.PROCEDURAL,
        importance=0.9,
        provenance=ProvenanceRecord(source_type=MemorySource.SYSTEM_EVENT),
    )
    await memory_repo.save(m)

    # 1. With fallback_to_deterministic=True (default)
    hybrid_with_fallback = HybridMemoryRetriever(
        deterministic_retriever=det_retriever,
        semantic_retriever=broken_sem,
        fallback_to_deterministic=True,
    )
    results = await hybrid_with_fallback.search(MemoryQuery(query_text="database password rotation"))
    assert len(results) == 1
    assert any("Semantic retrieval unavailable" in r for r in results[0].explanation.reasons)

    # 2. With fallback_to_deterministic=False: raises MemoryRetrievalError
    hybrid_no_fallback = HybridMemoryRetriever(
        deterministic_retriever=det_retriever,
        semantic_retriever=broken_sem,
        fallback_to_deterministic=False,
    )
    with pytest.raises(MemoryRetrievalError, match="Semantic retrieval failed"):
        await hybrid_no_fallback.search(MemoryQuery(query_text="database password"))


@pytest.mark.asyncio
async def test_hybrid_without_semantic_retriever(
    memory_repo: InMemoryMemoryRepository,
) -> None:
    """When semantic_retriever is None, HybridMemoryRetriever functions as deterministic retriever."""
    det_retriever = DeterministicMemoryRetriever(repository=memory_repo)
    hybrid_retriever = HybridMemoryRetriever(
        deterministic_retriever=det_retriever,
        semantic_retriever=None,
    )

    m = MemoryRecord(
        content="Quick reference note on python typing.",
        memory_type=MemoryType.EPISODIC,
        provenance=ProvenanceRecord(source_type=MemorySource.USER_STATEMENT),
    )
    await memory_repo.save(m)

    results = await hybrid_retriever.search(MemoryQuery(query_text="python typing"))
    assert len(results) == 1
    assert results[0].memory.memory_id == m.memory_id


@pytest.mark.asyncio
async def test_hybrid_search_with_response_and_pagination(
    memory_repo: InMemoryMemoryRepository,
    mock_embedding_service: EmbeddingService,
) -> None:
    """search_with_response container returns correct total candidates and pagination."""
    det_retriever = DeterministicMemoryRetriever(repository=memory_repo)
    sem_retriever = SemanticMemoryRetriever(
        repository=memory_repo,
        embedding_service=mock_embedding_service,
    )
    hybrid_retriever = HybridMemoryRetriever(
        deterministic_retriever=det_retriever,
        semantic_retriever=sem_retriever,
    )

    for i in range(4):
        m = await mock_embedding_service.attach_embedding(
            MemoryRecord(
                content=f"Common project note {i} about engineering standards.",
                memory_type=MemoryType.EPISODIC,
                provenance=ProvenanceRecord(source_type=MemorySource.SYSTEM_EVENT),
            )
        )
        await memory_repo.save(m)

    resp = await hybrid_retriever.search_with_response(
        MemoryQuery(query_text="engineering standards", limit=2, offset=1)
    )
    assert len(resp.results) == 2
    assert resp.total_candidates == 4
    assert await hybrid_retriever.count(MemoryQuery(query_text="engineering standards")) == 4
