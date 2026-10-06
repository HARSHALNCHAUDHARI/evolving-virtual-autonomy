"""Tests for ModelRouter provider selection and routing."""

import pytest
from pydantic import BaseModel

from eva.providers.mock import MockProvider
from eva.providers.models import (
    ModelRequest,
    ModelResponse,
    ProviderNotFoundError,
    RouteRequest,
    TaskType,
)
from eva.providers.router import ModelRouter


class OutputModel(BaseModel):
    message: str


class TestModelRouterRegistry:
    """Provider registration and inspection in ModelRouter."""

    def test_default_initialization(self):
        router = ModelRouter()
        assert router.default_provider_name == "mock"
        assert router.default_model_name == "mock-v1"
        assert router.has_provider("mock") is True
        assert "mock" in router.list_providers()

    def test_register_and_unregister_provider(self):
        router = ModelRouter()
        p2 = MockProvider(name="mock_fast")
        router.register_provider(p2)

        assert router.has_provider("mock_fast") is True
        assert router.get_provider("mock_fast").name == "mock_fast"

        router.unregister_provider("mock_fast")
        assert router.has_provider("mock_fast") is False

    def test_get_unsupported_provider_raises_error(self):
        router = ModelRouter()
        with pytest.raises(ProviderNotFoundError) as exc_info:
            router.get_provider("unsupported_ai")
        assert "unsupported_ai" in str(exc_info.value)


class TestModelRouterSelection:
    """Deterministic routing and provider selection."""

    @pytest.fixture
    def router(self) -> ModelRouter:
        r = ModelRouter(default_provider="primary")
        r.register_provider(MockProvider(name="primary"))
        r.register_provider(MockProvider(name="specialized"))
        return r

    def test_route_default_fallback(self, router: ModelRouter):
        provider = router.route()
        assert provider.name == "primary"

    def test_route_by_explicit_name(self, router: ModelRouter):
        provider = router.route("specialized")
        assert provider.name == "specialized"

    def test_route_by_route_request_provider(self, router: ModelRouter):
        req = RouteRequest(provider="specialized")
        provider = router.route(req)
        assert provider.name == "specialized"

    def test_route_by_model_request_provider(self, router: ModelRouter):
        req = ModelRequest(prompt="hello", provider="specialized")
        provider = router.route(req)
        assert provider.name == "specialized"

    def test_route_by_task_type_default_mapping(self, router: ModelRouter):
        provider = router.route(TaskType.CONVERSATION)
        assert provider.name == "primary"

    def test_route_by_custom_task_route(self, router: ModelRouter):
        router.register_route(TaskType.REASONING, "specialized")
        provider = router.route(TaskType.REASONING)
        assert provider.name == "specialized"

        # General task still uses primary
        assert router.route(TaskType.GENERAL).name == "primary"

    def test_route_unsupported_provider_raises_error(self, router: ModelRouter):
        req = RouteRequest(provider="openai_unregistered")
        with pytest.raises(ProviderNotFoundError):
            router.route(req)


class TestModelRouterExecution:
    """Execution dispatch through ModelRouter."""

    @pytest.fixture
    def router(self) -> ModelRouter:
        r = ModelRouter()
        r.register_provider(MockProvider(name="mock"))
        return r

    @pytest.mark.asyncio
    async def test_router_generate(self, router: ModelRouter):
        req = ModelRequest(prompt="hello")
        response = await router.generate(req, task_type=TaskType.CONVERSATION)

        assert isinstance(response, ModelResponse)
        assert response.success is True
        assert "EVA" in response.content

    @pytest.mark.asyncio
    async def test_router_generate_structured(self, router: ModelRouter):
        req = ModelRequest(
            prompt="output json",
            metadata={"mock_data": {"message": "routed success"}},
        )
        response = await router.generate_structured(req, OutputModel)

        assert isinstance(response, ModelResponse)
        assert response.success is True
        assert isinstance(response.structured_output, OutputModel)
        assert response.structured_output.message == "routed success"
