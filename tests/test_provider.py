"""Tests for EVA mock provider."""

import pytest

from eva.providers import ModelProvider
from eva.providers.mock import MockProvider


class TestMockProviderProtocol:
    """Mock provider implements the ModelProvider protocol."""

    def test_is_model_provider(self):
        provider = MockProvider()
        assert isinstance(provider, ModelProvider)

    def test_name(self):
        provider = MockProvider()
        assert provider.name == "mock"


class TestMockProviderResponses:
    """Mock provider returns deterministic output."""

    @pytest.mark.asyncio
    async def test_hello_response(self):
        provider = MockProvider()
        response = await provider.generate("hello")
        assert "EVA" in response
        assert "online" in response.lower() or "ready" in response.lower()

    @pytest.mark.asyncio
    async def test_default_response(self):
        provider = MockProvider()
        response = await provider.generate("something random")
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_custom_default_response(self):
        provider = MockProvider(default_response="custom")
        response = await provider.generate("anything")
        assert response == "custom"

    @pytest.mark.asyncio
    async def test_status_response(self):
        provider = MockProvider()
        response = await provider.generate("status")
        assert "mock" in response.lower()

    @pytest.mark.asyncio
    async def test_help_response(self):
        provider = MockProvider()
        response = await provider.generate("help me")
        assert "mock" in response.lower()


class TestMockProviderAvailability:
    """Mock provider availability."""

    @pytest.mark.asyncio
    async def test_always_available(self):
        provider = MockProvider()
        assert await provider.is_available() is True
