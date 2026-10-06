"""Tests for EVA embedding abstraction, mock provider, and embedding service."""

from __future__ import annotations

import math
import pytest
from uuid import uuid4

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
from eva.memory.models import MemoryRecord, MemorySource, MemoryType, ProvenanceRecord


def test_embedding_request_validation() -> None:
    """EmbeddingRequest validates inputs correctly."""
    req = EmbeddingRequest(inputs=["hello world", "test memory"])
    assert len(req.texts) == 2
    assert req.texts[0] == "hello world"

    # Empty inputs list raises ValueError
    with pytest.raises(ValueError, match="inputs list cannot be empty"):
        EmbeddingRequest(inputs=[])


def test_embedding_result_validation() -> None:
    """EmbeddingResult validates dimensions and empty vectors."""
    res = EmbeddingResult(
        embedding=[0.1, 0.2, 0.3],
        dimension=3,
        model="mock-v1",
        input_text="test",
    )
    assert res.dimension == 3
    assert len(res.embedding) == 3

    # Empty vector rejected
    with pytest.raises(ValueError, match="Embedding vector cannot be empty"):
        EmbeddingResult(
            embedding=[],
            dimension=3,
            model="mock-v1",
            input_text="test",
        )

    # Dimension mismatch rejected
    with pytest.raises(ValueError, match="Dimension mismatch"):
        EmbeddingResult(
            embedding=[0.1, 0.2],
            dimension=3,
            model="mock-v1",
            input_text="test",
        )


@pytest.mark.asyncio
async def test_mock_embedding_provider_deterministic() -> None:
    """MockEmbeddingProvider produces identical vectors for identical normalized inputs."""
    provider = MockEmbeddingProvider(dimension=64)
    assert provider.dimension == 64
    assert provider.model_name == "mock-embedding-v1"
    assert await provider.is_available()

    req = EmbeddingRequest(inputs=["user prefers dark mode", "  user prefers   dark mode  "])
    resp = await provider.embed(req)

    assert len(resp.results) == 2
    # Normalization ensures identical vectors
    assert resp.results[0].embedding == resp.results[1].embedding


@pytest.mark.asyncio
async def test_mock_embedding_provider_different_inputs() -> None:
    """MockEmbeddingProvider produces distinct vectors for different inputs."""
    provider = MockEmbeddingProvider(dimension=64)
    resp = await provider.embed(
        EmbeddingRequest(inputs=["cats are friendly", "quantum electrodynamics formulation"])
    )
    vec1 = resp.results[0].embedding
    vec2 = resp.results[1].embedding

    assert vec1 != vec2
    # Verify unit normalization: L2 norm is ~1.0
    norm1 = math.sqrt(sum(x * x for x in vec1))
    norm2 = math.sqrt(sum(x * x for x in vec2))
    assert math.isclose(norm1, 1.0, rel_tol=1e-5)
    assert math.isclose(norm2, 1.0, rel_tol=1e-5)


@pytest.mark.asyncio
async def test_mock_embedding_provider_preset_vectors() -> None:
    """MockEmbeddingProvider respects preset vectors."""
    target_vec = [1.0] + [0.0] * 63
    provider = MockEmbeddingProvider(
        dimension=64,
        preset_vectors={"special query": target_vec},
    )

    resp = await provider.embed(EmbeddingRequest(inputs=["special query"]))
    assert resp.results[0].embedding == target_vec


@pytest.mark.asyncio
async def test_embedding_service_text_normalization() -> None:
    """EmbeddingService normalizes text whitespace before calling provider."""
    provider = MockEmbeddingProvider(dimension=32)
    service = EmbeddingService(provider=provider, expected_dimension=32)

    res1 = await service.embed_text("  hello    world  ")
    res2 = await service.embed_text("hello world")

    assert res1 == res2
    assert len(res1) == 32


class MismatchedProvider:
    """Provider returning different dimension than expected."""

    model_name = "mismatched"
    default_dimension = 64
    default_model = "mismatched"

    async def is_available(self) -> bool:
        return True

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(
            embeddings=[
                EmbeddingResult(
                    text=request.texts[0],
                    vector=[0.1] * 64,
                    dimensions=64,
                )
            ],
            model="mismatched",
            dimensions=64,
        )


@pytest.mark.asyncio
async def test_embedding_service_dimension_mismatch() -> None:
    """EmbeddingService detects dimension mismatch between config and provider."""
    service = EmbeddingService(
        provider=MismatchedProvider(),
        config=EmbeddingConfig(dimension=128),
    )

    with pytest.raises(EmbeddingDimensionMismatchError, match="does not match expected 128"):
        await service.embed_text("sample input")


class FailingProvider:
    """Failing provider mock for failure tests."""

    model_name = "failing-model"
    default_dimension = 32
    default_model = "failing-model"

    async def is_available(self) -> bool:
        return False

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise RuntimeError("Network timeout connecting to embedding provider")


@pytest.mark.asyncio
async def test_embedding_service_provider_failure() -> None:
    """EmbeddingService wraps provider exceptions into EmbeddingProviderError."""
    service = EmbeddingService(provider=FailingProvider(), expected_dimension=32)

    with pytest.raises(EmbeddingProviderError, match="Network timeout"):
        await service.embed_text("test")


@pytest.mark.asyncio
async def test_embedding_service_attach_embedding() -> None:
    """EmbeddingService attaches embedding to a MemoryRecord properly."""
    provider = MockEmbeddingProvider(dimension=64)
    service = EmbeddingService(provider=provider, expected_dimension=64)

    record = MemoryRecord(
        memory_type=MemoryType.EPISODIC,
        content="EVA executed scheduled maintenance successfully.",
        provenance=ProvenanceRecord(source_type=MemorySource.SYSTEM_EVENT),
    )
    assert record.embedding is None
    assert record.embedding_ref is None

    updated = await service.attach_embedding(record)
    assert updated.embedding is not None
    assert len(updated.embedding) == 64
    assert updated.embedding_ref is not None
    assert updated.embedding_ref.dimensions == 64
    assert updated.embedding_ref.model_name == "mock-embedding-v1"
