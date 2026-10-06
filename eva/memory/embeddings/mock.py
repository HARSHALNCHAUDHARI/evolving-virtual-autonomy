"""Deterministic MockEmbeddingProvider for offline testing and development."""

from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Sequence

from eva.memory.embeddings.models import (
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResult,
)
from eva.memory.embeddings.provider import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic, zero-network mock embedding provider.

    Generates reproducible unit-norm vectors by seeding a pseudo-random generator
    with the SHA-256 hash of the normalized input text.
    """

    def __init__(
        self,
        dimension: int = 768,
        model_name: str = "mock-embedding-v1",
        preset_vectors: dict[str, list[float]] | None = None,
        available: bool = True,
    ) -> None:
        self._dimension = dimension
        self._model_name = model_name
        self._preset_vectors = preset_vectors or {}
        self._available = available

    @property
    def name(self) -> str:
        return "mock"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def default_dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def default_model(self) -> str:
        return self._model_name

    def set_availability(self, available: bool) -> None:
        """Dynamically toggle mock provider availability for testing."""
        self._available = available

    def register_vector(self, text: str, vector: Sequence[float]) -> None:
        """Register a preset vector for a specific input text."""
        norm_text = re.sub(r"\s+", " ", text.strip().lower())
        if len(vector) != self._dimension:
            raise EmbeddingDimensionMismatchError(
                f"Preset vector length {len(vector)} does not match dimension {self._dimension}."
            )
        self._preset_vectors[norm_text] = list(vector)

    def _generate_vector(self, text: str, target_dim: int) -> list[float]:
        """Deterministically generate a unit-norm vector from text content."""
        # Normalize internal and surrounding whitespace
        norm_text = re.sub(r"\s+", " ", text.strip().lower())

        # Check preset vectors
        if norm_text in self._preset_vectors:
            vec = self._preset_vectors[norm_text]
            if len(vec) != target_dim:
                raise EmbeddingDimensionMismatchError(
                    f"Preset vector length {len(vec)} does not match requested dimension {target_dim}."
                )
            return vec

        # Use SHA-256 hash as deterministic seed
        seed_int = int(hashlib.sha256(norm_text.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed_int)

        # Generate Gaussian samples and normalize to L2 unit length
        raw = [rng.gauss(0.0, 1.0) for _ in range(target_dim)]
        norm = math.sqrt(sum(x * x for x in raw))
        if norm == 0.0:
            norm = 1.0

        return [round(x / norm, 6) for x in raw]

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate deterministic embeddings for all texts in request."""
        if not self._available:
            raise EmbeddingProviderError("MockEmbeddingProvider is currently marked as unavailable.")

        target_dim = request.dimension or self._dimension
        model = request.model or self._model_name

        results: list[EmbeddingResult] = []
        for text in request.texts:
            vec = self._generate_vector(text, target_dim)
            results.append(
                EmbeddingResult(
                    text=text,
                    vector=vec,
                    dimensions=target_dim,
                )
            )

        return EmbeddingResponse(
            embeddings=results,
            model=model,
            dimensions=target_dim,
            usage={"prompt_tokens": sum(len(t.split()) for t in request.texts)},
        )

    async def is_available(self) -> bool:
        return self._available
