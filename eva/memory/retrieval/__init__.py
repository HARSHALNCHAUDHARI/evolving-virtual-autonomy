"""EVA Memory Retrieval Foundation package.

Provides deterministic keyword/metadata memory search, explainable ranking,
session/context isolation, and pagination.
"""

from __future__ import annotations

from eva.memory.retrieval.exceptions import (
    InvalidMemoryQueryError,
    MemoryRetrievalError,
)
from eva.memory.retrieval.models import (
    MatchExplanation,
    MemorySearchResponse,
    MemorySearchResult,
)
from eva.memory.retrieval.ranking import (
    compute_recency_score,
    normalize_text,
    ranking_sort_key,
    score_memory_record,
    tokenize,
)
from eva.memory.retrieval.hybrid import (
    HybridMemoryRetriever,
    HybridWeights,
)
from eva.memory.retrieval.retriever import (
    DeterministicMemoryRetriever,
    MemoryRetriever,
)
from eva.memory.retrieval.semantic import (
    SemanticMemoryRetriever,
    compute_cosine_similarity,
)

__all__ = [
    "MemoryRetriever",
    "DeterministicMemoryRetriever",
    "SemanticMemoryRetriever",
    "HybridMemoryRetriever",
    "HybridWeights",
    "MemorySearchResult",
    "MatchExplanation",
    "MemorySearchResponse",
    "MemoryRetrievalError",
    "InvalidMemoryQueryError",
    "normalize_text",
    "tokenize",
    "compute_recency_score",
    "compute_cosine_similarity",
    "score_memory_record",
    "ranking_sort_key",
]
