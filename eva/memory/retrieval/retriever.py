"""Deterministic Memory Retriever implementation and protocol."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from eva.memory.models import InvalidMemoryQueryError, MemoryQuery, MemoryRetrievalError, _utcnow
from eva.memory.repository import MemoryRepository
from eva.memory.retrieval.models import MemorySearchResponse, MemorySearchResult
from eva.memory.retrieval.ranking import ranking_sort_key, score_memory_record

logger = logging.getLogger(__name__)


@runtime_checkable
class MemoryRetriever(Protocol):
    """Protocol for deterministic memory retrieval operations."""

    async def search(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """Retrieve and rank matching memories deterministically.

        Args:
            query: The MemoryQuery specifying search and filter criteria.

        Returns:
            Ranked list of MemorySearchResults matching the query.
        """
        ...

    async def count(self, query: MemoryQuery) -> int:
        """Count the total number of candidate memories matching query criteria.

        Args:
            query: The MemoryQuery specifying search and filter criteria.

        Returns:
            Integer count of matching candidate memories before pagination.
        """
        ...


class DeterministicMemoryRetriever(MemoryRetriever):
    """Reference implementation of deterministic memory retrieval.

    Operates across any storage repository implementing the MemoryRepository protocol.
    Performs query normalization, candidate retrieval, metadata filtering,
    deterministic scoring, tie-breaking, and pagination.
    """

    def __init__(self, repository: MemoryRepository) -> None:
        self.repository = repository

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
        """Internal helper to retrieve, filter, score, and rank candidates."""
        self._validate_query(query)

        try:
            # Query candidate records from the underlying repository without pagination
            # to ensure ranking is computed globally across all matching records
            repo_query = query.model_copy(update={"limit": None, "offset": 0})
            candidates = await self.repository.query(repo_query)
        except (InvalidMemoryQueryError, ValidationError):
            raise
        except Exception as exc:
            logger.error("Repository query failure during memory retrieval: %s", exc)
            raise MemoryRetrievalError(f"Failed to retrieve memory candidates: {exc}") from exc

        # Filter by structured payload metadata if requested
        if query.metadata_filters:
            filtered_candidates = []
            for cand in candidates:
                match = True
                for k, v in query.metadata_filters.items():
                    if cand.structured_payload.get(k) != v:
                        match = False
                        break
                if match:
                    filtered_candidates.append(cand)
            candidates = filtered_candidates

        # Strict session isolation: if session_id is provided, reject mismatched records
        if query.session_id is not None:
            candidates = [c for c in candidates if c.session_id == query.session_id]

        # Score and filter candidates against text / keyword / metadata signals
        now = _utcnow()
        scored_results: list[MemorySearchResult] = []
        for cand in candidates:
            scored = score_memory_record(cand, query, now=now)
            if scored is not None:
                scored_results.append(scored)

        # Deterministic ranking with tie-breaking
        scored_results.sort(key=ranking_sort_key)
        return scored_results

    async def search(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """Retrieve and rank matching memories deterministically."""
        scored_results = await self._get_scored_candidates(query)

        # Apply pagination (offset & limit)
        start = query.offset
        if query.limit is not None:
            return scored_results[start : start + query.limit]
        return scored_results[start:]

    async def count(self, query: MemoryQuery) -> int:
        """Count the total number of candidate memories matching query criteria."""
        scored_results = await self._get_scored_candidates(query)
        return len(scored_results)

    async def search_with_response(self, query: MemoryQuery) -> MemorySearchResponse:
        """Retrieve results and return wrapped in a MemorySearchResponse container."""
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
