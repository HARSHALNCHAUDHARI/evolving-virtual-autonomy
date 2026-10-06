"""Typed request and response models for embedding providers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from eva.memory.models import MemoryError


class EmbeddingError(MemoryError):
    """Base exception for embedding operations."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when an embedding provider fails to generate vectors."""


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when an embedding vector dimension does not match expected dimensions."""


class EmbeddingRequest(BaseModel):
    """Request payload for generating dense vector embeddings."""

    texts: list[str]
    model: str | None = None
    dimension: int | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _validate_texts(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Support inputs and single text parameter as convenience
            if "inputs" in data and "texts" not in data:
                data["texts"] = data.pop("inputs")
            elif "text" in data and "texts" not in data:
                data["texts"] = [data.pop("text")]
            elif "texts" in data and isinstance(data["texts"], str):
                data["texts"] = [data["texts"]]
        return data

    @model_validator(mode="after")
    def _validate_non_empty(self) -> EmbeddingRequest:
        if not self.texts:
            raise ValueError("inputs list cannot be empty")
        return self


class EmbeddingResult(BaseModel):
    """Result vector for an individual text input."""

    text: str
    vector: list[float]
    dimensions: int = Field(gt=0)
    model: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "embedding" in data and "vector" not in data:
                data["vector"] = data.pop("embedding")
            if "dimension" in data and "dimensions" not in data:
                data["dimensions"] = data.pop("dimension")
            if "input_text" in data and "text" not in data:
                data["text"] = data.pop("input_text")
        return data

    @model_validator(mode="after")
    def _verify_dimensions(self) -> EmbeddingResult:
        if not self.vector:
            raise ValueError("Embedding vector cannot be empty.")
        if len(self.vector) != self.dimensions:
            raise ValueError(
                f"Dimension mismatch: Vector length {len(self.vector)} does not match specified dimensions {self.dimensions}."
            )
        return self

    @property
    def embedding(self) -> list[float]:
        return self.vector

    @property
    def dimension(self) -> int:
        return self.dimensions


class EmbeddingResponse(BaseModel):
    """Standardized response from an EmbeddingProvider."""

    embeddings: list[EmbeddingResult]
    model: str
    dimensions: int = Field(gt=0)
    usage: dict[str, Any] = Field(default_factory=dict)

    @property
    def results(self) -> list[EmbeddingResult]:
        """Convenience alias for embeddings."""
        return self.embeddings

    @property
    def vectors(self) -> list[list[float]]:
        """Convenience property returning raw vector lists."""
        return [item.vector for item in self.embeddings]

    @property
    def first_vector(self) -> list[float]:
        """Convenience property returning the first vector."""
        if not self.embeddings:
            raise IndexError("EmbeddingResponse contains no vectors.")
        return self.embeddings[0].vector
