"""Embedding subsystem for EVA Memory.

Provides provider-neutral vector generation, mock embedding implementation,
and embedding lifecycle utilities.
"""

from __future__ import annotations

from eva.memory.embeddings.config import EmbeddingConfig
from eva.memory.embeddings.mock import MockEmbeddingProvider
from eva.memory.embeddings.models import (
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResult,
)
from eva.memory.embeddings.provider import EmbeddingProvider
from eva.memory.embeddings.service import EmbeddingService

__all__ = [
    "EmbeddingConfig",
    "EmbeddingDimensionMismatchError",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingResult",
    "EmbeddingService",
    "MockEmbeddingProvider",
]
