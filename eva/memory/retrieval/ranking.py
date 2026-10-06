"""Deterministic scoring, text normalization, and ranking utilities for memory retrieval."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

from eva.memory.models import MemoryQuery, MemoryRecord, _ensure_timezone, _utcnow
from eva.memory.retrieval.models import MatchExplanation, MemorySearchResult

# Keyword / Text search weights (sum = 1.00)
TEXT_WEIGHT = 0.60
IMPORTANCE_WEIGHT = 0.20
RECENCY_WEIGHT = 0.10
CONFIDENCE_WEIGHT = 0.10

# Metadata-only search weights (sum = 1.00)
METADATA_IMPORTANCE_WEIGHT = 0.50
METADATA_RECENCY_WEIGHT = 0.30
METADATA_CONFIDENCE_WEIGHT = 0.20

# Recency decay rate per day
DEFAULT_DECAY_RATE = 0.01


def normalize_text(text: str) -> str:
    """Normalize text by trimming, lowercasing, and collapsing internal whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> list[str]:
    """Extract ordered alphanumeric word tokens from normalized text."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


def compute_recency_score(
    updated_at: datetime,
    *,
    is_immutable: bool = False,
    decay_rate: float = DEFAULT_DECAY_RATE,
    now: datetime | None = None,
) -> float:
    """Calculate exponential recency decay score in (0, 1].

    Immutable memories do not suffer temporal decay (recency score = 1.0).
    """
    if is_immutable or decay_rate <= 0.0:
        return 1.0

    current_time = _ensure_timezone(now) or _utcnow()
    ref_time = _ensure_timezone(updated_at) or current_time

    delta_seconds = max(0.0, (current_time - ref_time).total_seconds())
    delta_days = delta_seconds / 86400.0

    return float(math.exp(-decay_rate * delta_days))


def score_memory_record(
    record: MemoryRecord,
    query: MemoryQuery,
    *,
    now: datetime | None = None,
) -> MemorySearchResult | None:
    """Score a candidate memory record against query criteria.

    Returns:
        MemorySearchResult if the candidate matches text/filter criteria,
        or None if the candidate fails text matching.
    """
    ref_now = _ensure_timezone(now) or _utcnow()
    query_text = normalize_text(query.query_text or "")
    reasons: list[str] = []
    matched_filters: list[str] = []

    # Recency score
    recency_score = compute_recency_score(
        record.updated_at,
        is_immutable=record.retention_metadata.is_immutable,
        decay_rate=record.access_metadata.decay_rate or DEFAULT_DECAY_RATE,
        now=ref_now,
    )

    if query_text:
        # 1. Text / Keyword Matching
        content_norm = normalize_text(record.content)
        query_tokens = list(dict.fromkeys(tokenize(query_text)))
        content_tokens = set(tokenize(content_norm))

        # Include string values from structured payload in content tokens
        if record.structured_payload:
            for val in record.structured_payload.values():
                if isinstance(val, str):
                    content_tokens.update(tokenize(val))

        matched_tokens = [tok for tok in query_tokens if tok in content_tokens]
        exact_phrase_matched = query_text in content_norm

        # If query has tokens and none match, and no phrase match, exclude record
        if query_tokens and not matched_tokens and not exact_phrase_matched:
            return None

        token_overlap_ratio = (
            len(matched_tokens) / len(query_tokens) if query_tokens else 0.0
        )
        text_score = (0.60 * token_overlap_ratio) + (0.40 if exact_phrase_matched else 0.0)

        # Composite score calculation
        score = (
            (TEXT_WEIGHT * text_score)
            + (IMPORTANCE_WEIGHT * record.importance)
            + (RECENCY_WEIGHT * recency_score)
            + (CONFIDENCE_WEIGHT * record.confidence)
        )

        if exact_phrase_matched:
            reasons.append(f"Exact phrase matched: '{query_text}'")
        if matched_tokens:
            reasons.append(
                f"Token overlap: {len(matched_tokens)}/{len(query_tokens)} "
                f"({token_overlap_ratio:.0%}) [{', '.join(matched_tokens)}]"
            )
        reasons.append(f"Importance: {record.importance:.2f}")
        reasons.append(f"Confidence: {record.confidence:.2f}")
        reasons.append(f"Recency factor: {recency_score:.3f}")

        explanation = MatchExplanation(
            exact_phrase_matched=exact_phrase_matched,
            matched_tokens=matched_tokens,
            token_overlap_ratio=token_overlap_ratio,
            text_score=round(text_score, 4),
            importance_score=round(record.importance, 4),
            confidence_score=round(record.confidence, 4),
            recency_score=round(recency_score, 4),
            matched_filters=matched_filters,
            reasons=reasons,
        )

    else:
        # 2. Metadata-only matching
        score = (
            (METADATA_IMPORTANCE_WEIGHT * record.importance)
            + (METADATA_RECENCY_WEIGHT * recency_score)
            + (METADATA_CONFIDENCE_WEIGHT * record.confidence)
        )

        reasons.append("Matched metadata filters")
        reasons.append(f"Importance: {record.importance:.2f}")
        reasons.append(f"Confidence: {record.confidence:.2f}")
        reasons.append(f"Recency factor: {recency_score:.3f}")

        explanation = MatchExplanation(
            exact_phrase_matched=False,
            matched_tokens=[],
            token_overlap_ratio=0.0,
            text_score=0.0,
            importance_score=round(record.importance, 4),
            confidence_score=round(record.confidence, 4),
            recency_score=round(recency_score, 4),
            matched_filters=matched_filters,
            reasons=reasons,
        )

    # Clamped to [0.0, 1.0]
    final_score = max(0.0, min(1.0, float(score)))

    return MemorySearchResult(
        memory=record,
        score=round(final_score, 6),
        explanation=explanation,
    )


def ranking_sort_key(result: MemorySearchResult) -> tuple[float, float, float, float, str]:
    """Deterministic tie-breaking sort key for ranking MemorySearchResults.

    Order:
        1. Score descending
        2. Importance descending
        3. Updated_at descending
        4. Created_at descending
        5. Memory_id ascending
    """
    rec = result.memory
    score_rounded = round(result.score, 6)
    importance_rounded = round(rec.importance, 6)
    updated_ts = rec.updated_at.timestamp()
    created_ts = rec.created_at.timestamp()
    memory_id_str = str(rec.memory_id)

    return (
        -score_rounded,
        -importance_rounded,
        -updated_ts,
        -created_ts,
        memory_id_str,
    )
