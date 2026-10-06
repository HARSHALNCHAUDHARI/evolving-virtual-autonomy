"""Embedding service responsible for vector generation, normalization, and lifecycle management."""

from __future__ import annotations

import logging
import re
from typing import Sequence

from eva.memory.embeddings.config import EmbeddingConfig
from eva.memory.embeddings.mock import MockEmbeddingProvider
from eva.memory.embeddings.models import (
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingRequest,
)
from eva.memory.embeddings.provider import EmbeddingProvider
from eva.memory.models import EmbeddingReference, MemoryRecord

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service coordinating vector generation, dimension validation, and memory record embedding attachment."""

    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
        config: EmbeddingConfig | None = None,
        expected_dimension: int | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig()
        if expected_dimension is not None:
            self.config.dimension = expected_dimension
        self.provider = provider or MockEmbeddingProvider(
            dimension=self.config.dimension,
            model_name=self.config.model,
        )

    @property
    def dimension(self) -> int:
        """Return configured target vector dimensionality."""
        return self.config.dimension

    @property
    def model(self) -> str:
        """Return configured model name."""
        return self.config.model

    async def is_available(self) -> bool:
        """Check whether the backing embedding provider is functional."""
        try:
            return await self.provider.is_available()
        except Exception:
            return False

    async def embed_text(self, text: str) -> list[float]:
        """Generate and validate an embedding vector for a single text input.

        Args:
            text: Input text to vectorize.

        Returns:
            Unit-norm float vector matching configured dimensions.

        Raises:
            EmbeddingProviderError: If the provider is unavailable or fails.
            EmbeddingDimensionMismatchError: If the returned vector has incorrect dimensions.
        """
        clean_text = re.sub(r"\s+", " ", text.strip().lower())
        if not clean_text:
            raise EmbeddingError("Cannot generate embedding for empty text string.")

        req = EmbeddingRequest(
            texts=[clean_text],
            model=self.config.model,
            dimension=self.config.dimension,
        )

        try:
            resp = await self.provider.embed(req)
        except (EmbeddingProviderError, EmbeddingDimensionMismatchError):
            raise
        except Exception as exc:
            logger.error("Embedding generation failed unexpectedly: %s", exc)
            raise EmbeddingProviderError(f"Embedding generation failed: {exc}") from exc

        if not resp.embeddings:
            raise EmbeddingProviderError("Embedding provider returned an empty response.")

        vector = resp.first_vector
        if len(vector) != self.config.dimension:
            raise EmbeddingDimensionMismatchError(
                f"Generated embedding dimension {len(vector)} does not match expected {self.config.dimension}."
            )

        return vector

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings in batch for multiple texts."""
        clean_texts = [t.strip() for t in texts if t.strip()]
        if not clean_texts:
            return []

        req = EmbeddingRequest(
            texts=clean_texts,
            model=self.config.model,
            dimension=self.config.dimension,
        )
        resp = await self.provider.embed(req)
        return resp.vectors

    async def attach_embedding(
        self,
        record: MemoryRecord,
        *,
        force: bool = False,
    ) -> MemoryRecord:
        """Generate and attach dense vector representation to a MemoryRecord.

        Args:
            record: The MemoryRecord to augment.
            force: If True, regenerates the embedding even if one already exists.

        Returns:
            The augmented MemoryRecord with populated embedding and embedding_ref.
        """
        # If record already has a valid embedding matching current dimension and not forced, reuse
        if not force and record.embedding is not None and len(record.embedding) == self.dimension:
            return record

        vector = await self.embed_text(record.content)
        record.embedding = vector
        record.embedding_ref = EmbeddingReference(
            model=self.config.model,
            dimensions=self.dimension,
            version=1,
        )
        return record
