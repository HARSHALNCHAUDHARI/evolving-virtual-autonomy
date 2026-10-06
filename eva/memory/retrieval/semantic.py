"""Semantic memory retriever using dense vector similarity."""

from __future__ import annotations

import logging
import math
from typing import Any

from eva.memory.models import InvalidMemoryQueryError, MemoryQuery, MemoryRecord, MemoryRetrievalError, _utcnow
from eva.memory.repository import MemoryRepository
from eva.memory.embeddings.service import EmbeddingService
from eva.memory.retrieval.models import MatchExplanation, MemorySearchResponse, MemorySearchResult
from eva.memory.retrieval.ranking import ranking_sort_key
from eva.memory.retrieval.retriever import MemoryRetriever

logger = logging.getLogger(__name__)


def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two dense float vectors clamped to [0.0, 1.0].

    Returns 1.0 for identically oriented vectors, 0.0 for orthogonal or opposing vectors.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = dot / (norm_a * norm_b)
    # Cosine similarity for unit vectors in semantic space: clamp to [0.0, 1.0]
    return max(0.0, min(1.0, float(sim)))


class SemanticMemoryRetriever(MemoryRetriever):
    """Semantic vector-based retriever powered by embeddings and pgvector/in-memory cosine search.

    Follows the semantic retrieval lifecycle:
        query text -> embedding service -> query vector -> vector search -> candidates -> similarity ranking.
    """

    def __init__(
        self,
        repository: MemoryRepository,
        embedding_service: EmbeddingService,
        *,
        similarity_threshold: float = 0.0,
    ) -> None:
        """Initialize the semantic retriever.

        Args:
            repository: Storage repository (PostgreSQLMemoryRepository or InMemoryMemoryRepository).
            embedding_service: Service generating and normalizing dense embeddings.
            similarity_threshold: Minimum cosine similarity required to include a candidate (default 0.0).
        """
        self.repository = repository
        self.embedding_service = embedding_service
        self.similarity_threshold = max(0.0, min(1.0, similarity_threshold))

    def _validate_query(self, query: MemoryQuery) -> None:
        """Validate query parameters and boundaries."""
        if query.offset < 0:
            raise InvalidMemoryQueryError(f"Offset cannot be negative, got {query.offset}")
        if query.limit is not None and query.limit <= 0:
            raise InvalidMemoryQueryError(f"Limit must be positive, got {query.limit}")
        if (
            query.created_after is not None
            and query.created_before is not None
            and query.created_after > query.created_before
        ):
            raise InvalidMemoryQueryError(
                f"created_after ({query.created_after}) cannot be after created_before ({query.created_before})"
            )
        if (
            query.updated_after is not None
            and query.updated_before is not None
            and query.updated_after > query.updated_before
        ):
            raise InvalidMemoryQueryError(
                f"updated_after ({query.updated_after}) cannot be after updated_before ({query.updated_before})"
            )

    async def _get_scored_candidates(
        self, query: MemoryQuery
    ) -> list[MemorySearchResult]:
        """Generate query vector and retrieve semantically scored candidates."""
        self._validate_query(query)

        # If no query text is supplied, semantic retrieval cannot compute similarity
        query_text = (query.query_text or "").strip()
        if not query_text:
            return []

        # Generate query embedding
        try:
            query_vector = await self.embedding_service.embed_text(query_text)
        except Exception as exc:
            logger.error("Failed to generate embedding for semantic query: %s", exc)
            raise MemoryRetrievalError(
                f"Semantic embedding generation failed: {exc}"
            ) from exc

        candidates_with_scores: list[tuple[MemoryRecord, float]] = []

        # Check if underlying repository supports native vector search (e.g. pgvector)
        if hasattr(self.repository, "search_vector"):
            try:
                # Retrieve up to maximum reasonable candidates (or query.limit if unbounded fetch not desired)
                # We request full candidates to allow global ranking
                max_fetch = 1000 if query.limit is None else max(100, query.limit * 10)
                candidates_with_scores = await self.repository.search_vector(
                    query_vector=query_vector,
                    query=query,
                    limit=max_fetch,
                    similarity_threshold=self.similarity_threshold,
                )
            except Exception as exc:
                logger.error("pgvector search failure in repository: %s", exc)
                raise MemoryRetrievalError(f"Vector search failed: {exc}") from exc
        else:
            # Fallback to in-memory cosine similarity calculation across repository records
            try:
                repo_query = query.model_copy(update={"limit": None, "offset": 0})
                candidates = await self.repository.query(repo_query)
            except Exception as exc:
                logger.error("Repository query failure during semantic retrieval: %s", exc)
                raise MemoryRetrievalError(
                    f"Failed to query candidate memories for semantic search: {exc}"
                ) from exc

            # Metadata filters
            if query.metadata_filters:
                filtered = []
                for cand in candidates:
                    match = True
                    for k, v in query.metadata_filters.items():
                        if cand.structured_payload.get(k) != v:
                            match = False
                            break
                    if match:
                        filtered.append(cand)
                candidates = filtered

            # Strict session isolation check
            if query.session_id is not None:
                candidates = [c for c in candidates if c.session_id == query.session_id]

            # Strict project isolation check
            if query.project_id is not None:
                if query.include_global:
                    candidates = [
                        c for c in candidates
                        if c.project_id == query.project_id or c.project_id is None
                    ]
                else:
                    candidates = [c for c in candidates if c.project_id == query.project_id]

            for cand in candidates:
                if cand.embedding is None:
                    # Skip memories without embeddings
                    continue
                sim = compute_cosine_similarity(query_vector, cand.embedding)
                if sim >= self.similarity_threshold:
                    candidates_with_scores.append((cand, sim))

        # Convert to MemorySearchResult with explanation
        results: list[MemorySearchResult] = []
        for rec, sim in candidates_with_scores:
            clamped_sim = max(0.0, min(1.0, float(sim)))
            explanation = MatchExplanation(
                exact_phrase_matched=False,
                matched_tokens=[],
                token_overlap_ratio=0.0,
                text_score=0.0,
                semantic_similarity=round(clamped_sim, 4),
                semantic_score=round(clamped_sim, 4),
                importance_score=round(rec.importance, 4),
                confidence_score=round(rec.confidence, 4),
                recency_score=1.0,
                reasons=[f"Semantic similarity: {clamped_sim:.4f}"],
            )
            results.append(
                MemorySearchResult(
                    memory=rec,
                    score=round(clamped_sim, 6),
                    explanation=explanation,
                )
            )

        # Deterministic sorting: score DESC, importance DESC, updated_at DESC, created_at DESC, memory_id ASC
        results.sort(key=ranking_sort_key)
        return results

    async def search(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """Perform semantic search and return ranked results respecting pagination."""
        scored_results = await self._get_scored_candidates(query)
        start = query.offset
        if query.limit is not None:
            return scored_results[start : start + query.limit]
        return scored_results[start:]

    async def count(self, query: MemoryQuery) -> int:
        """Count total candidate memories meeting semantic similarity criteria."""
        scored_results = await self._get_scored_candidates(query)
        return len(scored_results)

    async def search_with_response(self, query: MemoryQuery) -> MemorySearchResponse:
        """Perform semantic search and return wrapped response."""
        scored_results = await self._get_scored_candidates(query)
        total_count = len(scored_results)
        start = query.offset
        if query.limit is not None:
            paginated = scored_results[start : start + query.limit]
        else:
            paginated = scored_results[start:]

        return MemorySearchResponse(
            results=paginated,
            total_candidates=total_count,
            query=query,
        )
