"""Configuration for the embedding subsystem."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from eva.config import EVASettings, get_settings


class EmbeddingConfig(BaseModel):
    """Configuration settings for vector embeddings."""

    provider: str = "mock"
    model: str = "mock-embedding-v1"
    dimension: int = Field(default=768, gt=0)
    batch_size: int = Field(default=32, gt=0)
    timeout: float = Field(default=10.0, gt=0.0)

    @classmethod
    def from_settings(cls, settings: EVASettings | None = None) -> EmbeddingConfig:
        """Construct EmbeddingConfig from EVASettings or global configuration."""
        cfg = settings or get_settings()
        return cls(
            provider=cfg.embedding_provider,
            model=cfg.embedding_model,
            dimension=cfg.embedding_dimension,
        )

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        """Construct EmbeddingConfig from system environment variables."""
        provider = os.getenv("EVA_EMBEDDING_PROVIDER", "mock")
        model = os.getenv("EVA_EMBEDDING_MODEL", "mock-embedding-v1")
        dimension = int(os.getenv("EVA_EMBEDDING_DIMENSION", "768"))

        return cls(
            provider=provider,
            model=model,
            dimension=dimension,
        )
