"""Typed retrieval models and result structures for EVA memory retrieval."""

from __future__ import annotations

from typing import Any, Iterator

from pydantic import BaseModel, Field

from eva.memory.models import MemoryQuery, MemoryRecord


class MatchExplanation(BaseModel):
    """Structured explanation detailing why a memory record matched and its score components."""

    exact_phrase_matched: bool = False
    matched_tokens: list[str] = Field(default_factory=list)
    token_overlap_ratio: float = 0.0
    text_score: float = 0.0
    importance_score: float = 0.0
    confidence_score: float = 0.0
    recency_score: float = 0.0
    matched_filters: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class MemorySearchResult(BaseModel):
    """Scored search result wrapping a matched MemoryRecord and its ranking rationale."""

    memory: MemoryRecord
    score: float = Field(ge=0.0, le=1.0)
    explanation: MatchExplanation = Field(default_factory=MatchExplanation)

    @property
    def match_reasons(self) -> list[str]:
        """Convenience property for human-readable match explanations."""
        return self.explanation.reasons


class MemorySearchResponse(BaseModel):
    """Collection model for retrieval responses with total counts and original query."""

    results: list[MemorySearchResult] = Field(default_factory=list)
    total_candidates: int = Field(default=0, ge=0)
    query: MemoryQuery

    def __iter__(self) -> Iterator[MemorySearchResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> MemorySearchResult:
        return self.results[index]
