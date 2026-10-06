"""Hybrid Memory Retriever combining deterministic keyword and semantic vector search."""

from __future__ import annotations

import logging
import math
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from eva.memory.models import InvalidMemoryQueryError, MemoryQuery, MemoryRecord, MemoryRetrievalError, _utcnow
from eva.memory.retrieval.models import MatchExplanation, MemorySearchResponse, MemorySearchResult
from eva.memory.retrieval.ranking import (
    DEFAULT_DECAY_RATE,
    compute_recency_score,
    ranking_sort_key,
)
from eva.memory.retrieval.retriever import DeterministicMemoryRetriever, MemoryRetriever
from eva.memory.retrieval.semantic import SemanticMemoryRetriever

logger = logging.getLogger(__name__)


class HybridWeights(BaseModel):
    """Configurable weights for hybrid memory scoring.

    Formula:
        Score = (semantic_weight * S_semantic)
              + (keyword_weight * S_keyword)
              + (importance_weight * S_importance)
              + (recency_weight * S_recency)
              + (confidence_weight * S_confidence)

    All weights must be non-negative and sum to 1.0.
    """

    semantic_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    importance_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    confidence_weight: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum(self) -> HybridWeights:
        total = (
            self.semantic_weight
            + self.keyword_weight
            + self.importance_weight
            + self.recency_weight
            + self.confidence_weight
        )
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(
                f"Hybrid weights must sum to 1.0, got sum = {total:.4f}. "
                "Verify weight configuration."
            )
        return self


class HybridMemoryRetriever(MemoryRetriever):
    """Hybrid memory retriever merging deterministic keyword matching with semantic vector similarity.

    Provides explainable, deterministic composite ranking and degrades gracefully
    to deterministic retrieval if vector infrastructure or embeddings are unavailable.
    """

    def __init__(
        self,
        deterministic_retriever: DeterministicMemoryRetriever,
        semantic_retriever: SemanticMemoryRetriever | None = None,
        *,
        weights: HybridWeights | None = None,
        fallback_to_deterministic: bool = True,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            deterministic_retriever: Primary deterministic retrieval engine.
            semantic_retriever: Optional semantic vector retriever.
            weights: Normalized scoring weights. Defaults to 45% semantic, 25% keyword, 15% importance, 10% recency, 5% confidence.
            fallback_to_deterministic: Whether to gracefully fall back if semantic search fails.
        """
        self.deterministic_retriever = deterministic_retriever
        self.semantic_retriever = semantic_retriever
        self.weights = weights or HybridWeights()
        self.fallback_to_deterministic = fallback_to_deterministic

    async def _get_scored_candidates(
        self, query: MemoryQuery
    ) -> list[MemorySearchResult]:
        """Retrieve, fuse, re-rank, and explain hybrid memory candidates."""
        # Query without pagination for global re-ranking across both channels
        unpaginated_query = query.model_copy(update={"limit": None, "offset": 0})

        # 1. Deterministic candidates
        deterministic_results: list[MemorySearchResult] = []
        try:
            deterministic_results = await self.deterministic_retriever._get_scored_candidates(
                unpaginated_query
            )
        except Exception as exc:
            logger.error("Deterministic retrieval failed during hybrid retrieval: %s", exc)
            raise

        # If semantic retriever is omitted or query text is absent, use deterministic results directly
        query_text = (query.query_text or "").strip()
        if self.semantic_retriever is None or not query_text:
            return deterministic_results

        # 2. Semantic candidates
        semantic_results: list[MemorySearchResult] = []
        try:
            semantic_results = await self.semantic_retriever._get_scored_candidates(
                unpaginated_query
            )
        except Exception as exc:
            if self.fallback_to_deterministic:
                logger.warning(
                    "Semantic search failed during hybrid retrieval (%s); "
                    "falling back to deterministic ranking.",
                    exc,
                )
                for res in deterministic_results:
                    res.explanation.reasons.append(
                        "Note: Semantic retrieval unavailable; fell back to deterministic ranking."
                    )
                return deterministic_results
            raise MemoryRetrievalError(
                f"Semantic retrieval failed during hybrid search: {exc}"
            ) from exc

        # 3. Fuse candidates across both channels
        records_map: dict[UUID, MemoryRecord] = {}
        det_map: dict[UUID, MemorySearchResult] = {}
        sem_map: dict[UUID, MemorySearchResult] = {}

        for res in deterministic_results:
            records_map[res.memory.memory_id] = res.memory
            det_map[res.memory.memory_id] = res

        for res in semantic_results:
            records_map[res.memory.memory_id] = res.memory
            sem_map[res.memory.memory_id] = res

        now = _utcnow()
        fused_results: list[MemorySearchResult] = []

        for mem_id, record in records_map.items():
            det_res = det_map.get(mem_id)
            sem_res = sem_map.get(mem_id)

            keyword_score = det_res.explanation.text_score if det_res else 0.0
            semantic_sim = sem_res.explanation.semantic_similarity if sem_res else 0.0

            recency_score = compute_recency_score(
                record.updated_at,
                is_immutable=record.retention_metadata.is_immutable,
                decay_rate=record.access_metadata.decay_rate or DEFAULT_DECAY_RATE,
                now=now,
            )

            # Hybrid composite score calculation
            composite_score = (
                (self.weights.semantic_weight * semantic_sim)
                + (self.weights.keyword_weight * keyword_score)
                + (self.weights.importance_weight * record.importance)
                + (self.weights.recency_weight * recency_score)
                + (self.weights.confidence_weight * record.confidence)
            )
            clamped_score = max(0.0, min(1.0, float(composite_score)))

            reasons: list[str] = [
                f"Hybrid composite score: {clamped_score:.4f}",
                f"Semantic similarity: {semantic_sim:.4f} (weight {self.weights.semantic_weight:.2f})",
                f"Keyword score: {keyword_score:.4f} (weight {self.weights.keyword_weight:.2f})",
                f"Importance: {record.importance:.2f} (weight {self.weights.importance_weight:.2f})",
                f"Recency: {recency_score:.3f} (weight {self.weights.recency_weight:.2f})",
                f"Confidence: {record.confidence:.2f} (weight {self.weights.confidence_weight:.2f})",
            ]

            explanation = MatchExplanation(
                exact_phrase_matched=det_res.explanation.exact_phrase_matched if det_res else False,
                matched_tokens=det_res.explanation.matched_tokens if det_res else [],
                token_overlap_ratio=det_res.explanation.token_overlap_ratio if det_res else 0.0,
                text_score=round(keyword_score, 4),
                semantic_similarity=round(semantic_sim, 4),
                semantic_score=round(self.weights.semantic_weight * semantic_sim, 4),
                importance_score=round(record.importance, 4),
                confidence_score=round(record.confidence, 4),
                recency_score=round(recency_score, 4),
                matched_filters=det_res.explanation.matched_filters if det_res else [],
                reasons=reasons,
            )

            fused_results.append(
                MemorySearchResult(
                    memory=record,
                    score=round(clamped_score, 6),
                    explanation=explanation,
                )
            )

        # Deterministic ranking with tie-breaking
        fused_results.sort(key=ranking_sort_key)
        return fused_results

    async def search(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """Perform hybrid search and return paginated ranked results."""
        scored_results = await self._get_scored_candidates(query)
        start = query.offset
        if query.limit is not None:
            return scored_results[start : start + query.limit]
        return scored_results[start:]

    async def count(self, query: MemoryQuery) -> int:
        """Count total candidate memories matching hybrid search criteria."""
        scored_results = await self._get_scored_candidates(query)
        return len(scored_results)

    async def search_with_response(self, query: MemoryQuery) -> MemorySearchResponse:
        """Perform hybrid search and return wrapped response."""
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
