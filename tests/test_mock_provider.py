"""Tests for MockProvider text and structured output generation."""

import asyncio
import pytest
from pydantic import BaseModel, Field

from eva.providers.base import ModelProvider
from eva.providers.mock import MockProvider
from eva.providers.models import ModelRequest, ModelResponse


class UserProfileSchema(BaseModel):
    username: str
    age: int
    is_active: bool = True


class ComplexSchema(BaseModel):
    intent: str
    confidence: float
    tags: list[str] = Field(default_factory=list)


class TestMockProviderProtocol:
    """MockProvider conforms to ModelProvider protocol."""

    def test_protocol_conformance(self):
        provider = MockProvider()
        assert isinstance(provider, ModelProvider)
        assert provider.name == "mock"

    @pytest.mark.asyncio
    async def test_is_available(self):
        provider = MockProvider()
        assert await provider.is_available() is True


class TestMockProviderTypedGeneration:
    """MockProvider handling of ModelRequest."""

    @pytest.fixture
    def provider(self) -> MockProvider:
        return MockProvider()

    @pytest.mark.asyncio
    async def test_generate_with_model_request(self, provider: MockProvider):
        request = ModelRequest(prompt="hello", model="test-mock")
        response = await provider.generate(request)

        assert isinstance(response, ModelResponse)
        assert response.success is True
        assert "EVA is online and ready" in response.content
        assert response.model == "test-mock"
        assert response.provider == "mock"
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_deterministic_patterns(self, provider: MockProvider):
        r1 = await provider.generate(ModelRequest(prompt="status"))
        assert "normally" in r1.content

        r2 = await provider.generate(ModelRequest(prompt="help me"))
        assert "assist" in r2.content

        r3 = await provider.generate(ModelRequest(prompt="random prompt"))
        assert "operating with a mock provider" in r3.content

    @pytest.mark.asyncio
    async def test_legacy_string_invocation(self, provider: MockProvider):
        # Must return str for backward compatibility with 1A tests
        text = await provider.generate("hello")
        assert isinstance(text, str)
        assert "EVA" in text


class TestMockProviderStructuredGeneration:
    """MockProvider structured output generation and validation."""

    @pytest.fixture
    def provider(self) -> MockProvider:
        return MockProvider()

    @pytest.mark.asyncio
    async def test_structured_generation_with_mock_data_dict(self, provider: MockProvider):
        request = ModelRequest(
            prompt="Extract user profile",
            metadata={"mock_data": {"username": "eva_user", "age": 28, "is_active": True}},
        )
        response = await provider.generate_structured(request, UserProfileSchema)

        assert isinstance(response, ModelResponse)
        assert response.success is True
        assert response.error is None
        assert isinstance(response.structured_output, UserProfileSchema)
        assert response.structured_output.username == "eva_user"
        assert response.structured_output.age == 28
        assert response.structured_output.is_active is True

    @pytest.mark.asyncio
    async def test_structured_generation_with_raw_json(self, provider: MockProvider):
        raw_json_str = '{"intent": "query", "confidence": 0.95, "tags": ["search", "local"]}'
        request = ModelRequest(
            prompt="Parse intent",
            metadata={"raw_json": raw_json_str},
        )
        response = await provider.generate_structured(request, ComplexSchema)

        assert response.success is True
        assert isinstance(response.structured_output, ComplexSchema)
        assert response.structured_output.intent == "query"
        assert response.structured_output.confidence == 0.95
        assert response.structured_output.tags == ["search", "local"]

    @pytest.mark.asyncio
    async def test_structured_generation_with_json_prompt(self, provider: MockProvider):
        request = ModelRequest(
            prompt='{"username": "direct_json", "age": 35}',
        )
        response = await provider.generate_structured(request, UserProfileSchema)

        assert response.success is True
        assert isinstance(response.structured_output, UserProfileSchema)
        assert response.structured_output.username == "direct_json"
        assert response.structured_output.age == 35

    @pytest.mark.asyncio
    async def test_structured_generation_synthetic_fallback(self, provider: MockProvider):
        request = ModelRequest(prompt="Synthesize dummy schema")
        response = await provider.generate_structured(request, UserProfileSchema)

        assert response.success is True
        assert isinstance(response.structured_output, UserProfileSchema)
        assert isinstance(response.structured_output.username, str)
        assert isinstance(response.structured_output.age, int)

    @pytest.mark.asyncio
    async def test_malformed_structured_data_returns_typed_failure(self, provider: MockProvider):
        # Invalid data: age is not a valid int
        request = ModelRequest(
            prompt="Extract profile",
            metadata={"mock_data": {"username": "eva_user", "age": "not-a-number"}},
        )
        response = await provider.generate_structured(request, UserProfileSchema)

        # Must return typed failure rather than raising uncaught exception
        assert isinstance(response, ModelResponse)
        assert response.success is False
        assert response.structured_output is None
        assert response.finish_reason == "error"
        assert response.error is not None
        assert "validation failed" in response.error.lower()

    @pytest.mark.asyncio
    async def test_malformed_raw_json_returns_typed_failure(self, provider: MockProvider):
        request = ModelRequest(
            prompt="Parse broken json",
            metadata={"raw_json": "{broken json text..."},
        )
        response = await provider.generate_structured(request, UserProfileSchema)

        assert response.success is False
        assert response.structured_output is None
        assert response.finish_reason == "error"
        assert response.error is not None

    @pytest.mark.asyncio
    async def test_concurrent_async_generation(self, provider: MockProvider):
        tasks = [
            provider.generate(ModelRequest(prompt=f"hello {i}"))
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for r in results:
            assert isinstance(r, ModelResponse)
            assert r.success is True
