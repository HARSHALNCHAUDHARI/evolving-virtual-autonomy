"""Storage- and vendor-neutral protocol for embedding providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from eva.memory.embeddings.models import EmbeddingRequest, EmbeddingResponse


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for dense vector embedding generation providers."""

    @property
    def name(self) -> str:
        """Return the provider identifier name."""
        ...

    @property
    def default_dimension(self) -> int:
        """Return the default vector dimensionality."""
        ...

    @property
    def default_model(self) -> str:
        """Return the default embedding model identifier."""
        ...

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate dense vector embeddings for the requested texts.

        Args:
            request: The EmbeddingRequest containing input texts, model, and dimension.

        Returns:
            The EmbeddingResponse containing generated vectors and metadata.

        Raises:
            EmbeddingProviderError: If the underlying model or service fails.
            EmbeddingDimensionMismatchError: If generated vectors do not match dimension.
        """
        ...

    async def is_available(self) -> bool:
        """Check whether the provider is currently available and healthy."""
        ...
