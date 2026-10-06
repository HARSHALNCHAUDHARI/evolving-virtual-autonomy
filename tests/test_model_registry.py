"""Tests for model capability definitions, specifications, and the model registry."""

import pytest

from eva.providers.mock import MockProvider
from eva.providers.models import (
    ContextLimits,
    DuplicateModelError,
    ModelCapability,
    ModelCost,
    ModelNotFoundError,
    ModelSpec,
    RouteRequest,
)
from eva.providers.registry import ModelRegistry
from eva.providers.router import ModelRouter
from eva.runtime import EVARuntime


# ===========================================================================
# 1. Capability Definitions
# ===========================================================================


class TestModelCapabilityDefinitions:
    """Validation of extensible ModelCapability enum."""

    def test_standard_capabilities_exist(self):
        expected = {
            "text",
            "reasoning",
            "vision",
            "audio",
            "structured_output",
            "tool_calling",
            "streaming",
            "long_context",
        }
        actual = {cap.value for cap in ModelCapability}
        assert expected.issubset(actual)

    def test_capability_string_comparison(self):
        assert ModelCapability.TEXT == "text"
        assert ModelCapability.REASONING == "reasoning"
        assert ModelCapability.STRUCTURED_OUTPUT == "structured_output"


# ===========================================================================
# 2. Model Specification (ModelSpec)
# ===========================================================================


class TestModelSpec:
    """Validation and behavior of ModelSpec."""

    def test_minimal_spec(self):
        spec = ModelSpec(
            provider="mock",
            model="mock-v1",
            capabilities={ModelCapability.TEXT},
        )
        assert spec.provider == "mock"
        assert spec.model == "mock-v1"
        assert spec.model_id == "mock:mock-v1"
        assert spec.has_capability(ModelCapability.TEXT) is True
        assert spec.has_capability("text") is True
        assert spec.has_capability("TEXT") is True
        assert spec.has_capability(ModelCapability.VISION) is False

    def test_extensible_custom_capabilities(self):
        spec = ModelSpec(
            provider="custom_ai",
            model="code-expert",
            capabilities={
                ModelCapability.TEXT,
                "code_synthesis",
                "repo_search",
            },
        )
        assert spec.has_capability("code_synthesis") is True
        assert spec.has_capability("repo_search") is True
        assert spec.has_capability("unsupported") is False

    def test_full_metadata_spec(self):
        spec = ModelSpec(
            provider="openai",
            model="gpt-4o",
            capabilities={
                ModelCapability.TEXT,
                ModelCapability.VISION,
                ModelCapability.TOOL_CALLING,
                ModelCapability.STREAMING,
            },
            context_limits=ContextLimits(
                max_input_tokens=128000,
                max_output_tokens=4096,
                context_window=128000,
            ),
            latency_tier="low",
            cost=ModelCost(
                input_cost_per_million=5.0,
                output_cost_per_million=15.0,
                currency="USD",
            ),
            priority=1,
            available=True,
            modalities=["text", "image"],
            metadata={"description": "Omni multimodal model"},
        )
        assert spec.context_limits.context_window == 128000
        assert spec.cost.input_cost_per_million == 5.0
        assert spec.cost.currency == "USD"
        assert spec.latency_tier == "low"
        assert spec.modalities == ["text", "image"]
        assert spec.metadata["description"] == "Omni multimodal model"


# ===========================================================================
# 3. Model Registry Lifecycle & Operations
# ===========================================================================


class TestModelRegistryLifecycle:
    """Registration, removal, and lookup in ModelRegistry."""

    def test_default_registry_contains_mock(self):
        registry = ModelRegistry()
        assert registry.has_provider("mock") is True
        assert registry.has_model("mock", "mock-v1") is True
        spec = registry.get_model("mock", "mock-v1")
        assert spec is not None
        assert spec.has_capability(ModelCapability.TEXT) is True
        assert spec.has_capability(ModelCapability.REASONING) is True

    def test_empty_registry_creation(self):
        registry = ModelRegistry(include_defaults=False)
        assert len(registry.list_models()) == 0
        assert registry.has_provider("mock") is False

    def test_register_and_lookup_model(self):
        registry = ModelRegistry(include_defaults=False)
        spec = ModelSpec(
            provider="gemini",
            model="gemini-1.5-pro",
            capabilities={ModelCapability.TEXT, ModelCapability.VISION, ModelCapability.LONG_CONTEXT},
        )
        registry.register_model(spec)

        assert registry.has_model("gemini", "gemini-1.5-pro") is True
        # Case insensitive lookup
        assert registry.has_model("GEMINI", "GEMINI-1.5-PRO") is True

        retrieved = registry.get_model("gemini", "gemini-1.5-pro")
        assert retrieved == spec

        lookup_res = registry.lookup("gemini", "gemini-1.5-pro")
        assert lookup_res == spec

    def test_duplicate_registration_handling(self):
        registry = ModelRegistry(include_defaults=False)
        spec1 = ModelSpec(provider="mock", model="m1", priority=1)
        spec2 = ModelSpec(provider="mock", model="m1", priority=2)

        # Overwrite=True (default) updates the model
        registry.register_model(spec1)
        registry.register_model(spec2, overwrite=True)
        assert registry.get_model("mock", "m1").priority == 2

        # Overwrite=False raises DuplicateModelError
        with pytest.raises(DuplicateModelError) as exc_info:
            registry.register_model(spec1, overwrite=False)
        assert "already registered" in str(exc_info.value)

    def test_unregister_model(self):
        registry = ModelRegistry(include_defaults=False)
        spec = ModelSpec(provider="mock", model="temp-model")
        registry.register_model(spec)

        assert registry.has_model("mock", "temp-model") is True
        removed = registry.unregister_model("mock", "temp-model")
        assert removed is True
        assert registry.has_model("mock", "temp-model") is False

        # Unregistering nonexistent returns False
        assert registry.unregister_model("mock", "temp-model") is False

    def test_lookup_unknown_model_raises_error(self):
        registry = ModelRegistry(include_defaults=False)
        assert registry.get_model("unknown", "unknown") is None

        with pytest.raises(ModelNotFoundError) as exc_info:
            registry.lookup("unknown_provider", "nonexistent_model")
        assert "unknown_provider:nonexistent_model" in str(exc_info.value)

    def test_clear_registry(self):
        registry = ModelRegistry()
        assert len(registry.list_models()) > 0
        registry.clear()
        assert len(registry.list_models()) == 0


# ===========================================================================
# 4. Capability Filtering and Queries
# ===========================================================================


class TestModelRegistryFiltering:
    """Capability and provider filtering in ModelRegistry."""

    @pytest.fixture
    def populated_registry(self) -> ModelRegistry:
        reg = ModelRegistry(include_defaults=False)
        reg.register_model(
            ModelSpec(
                provider="mock",
                model="text-only",
                capabilities={ModelCapability.TEXT},
            )
        )
        reg.register_model(
            ModelSpec(
                provider="mock",
                model="reasoning-model",
                capabilities={ModelCapability.TEXT, ModelCapability.REASONING},
            )
        )
        reg.register_model(
            ModelSpec(
                provider="vision_ai",
                model="vision-v1",
                capabilities={ModelCapability.TEXT, ModelCapability.VISION, ModelCapability.STREAMING},
            )
        )
        reg.register_model(
            ModelSpec(
                provider="vision_ai",
                model="audio-v1",
                capabilities={ModelCapability.TEXT, ModelCapability.AUDIO},
            )
        )
        return reg

    def test_list_models_by_provider(self, populated_registry: ModelRegistry):
        all_models = populated_registry.list_models()
        assert len(all_models) == 4

        mock_models = populated_registry.list_models(provider="mock")
        assert len(mock_models) == 2
        assert {m.model for m in mock_models} == {"text-only", "reasoning-model"}

        vision_models = populated_registry.list_models(provider="vision_ai")
        assert len(vision_models) == 2

        empty_provider = populated_registry.list_models(provider="nonexistent")
        assert empty_provider == []

    def test_filter_by_standard_capability(self, populated_registry: ModelRegistry):
        vision_models = populated_registry.filter_by_capability(ModelCapability.VISION)
        assert len(vision_models) == 1
        assert vision_models[0].model == "vision-v1"

        reasoning_models = populated_registry.filter_by_capability(ModelCapability.REASONING)
        assert len(reasoning_models) == 1
        assert reasoning_models[0].model == "reasoning-model"

        # Case-insensitive string check
        streaming_models = populated_registry.filter_by_capability("streaming")
        assert len(streaming_models) == 1
        assert streaming_models[0].model == "vision-v1"

        # Capability shared across all models
        text_models = populated_registry.filter_by_capability(ModelCapability.TEXT)
        assert len(text_models) == 4

    def test_filter_by_capability_and_provider(self, populated_registry: ModelRegistry):
        text_in_mock = populated_registry.filter_by_capability(
            ModelCapability.TEXT, provider="mock"
        )
        assert len(text_in_mock) == 2

        text_in_vision = populated_registry.filter_by_capability(
            ModelCapability.TEXT, provider="vision_ai"
        )
        assert len(text_in_vision) == 2

        # Non-matching combination
        vision_in_mock = populated_registry.filter_by_capability(
            ModelCapability.VISION, provider="mock"
        )
        assert vision_in_mock == []

    def test_list_providers(self, populated_registry: ModelRegistry):
        providers = populated_registry.list_providers()
        assert providers == ["mock", "vision_ai"]
        assert populated_registry.has_provider("mock") is True
        assert populated_registry.has_provider("vision_ai") is True
        assert populated_registry.has_provider("unknown") is False


# ===========================================================================
# 5. Router and MockProvider Compatibility
# ===========================================================================


class TestRouterAndMockCompatibility:
    """Compatibility between ModelRouter, ModelRegistry, and MockProvider."""

    def test_router_exposes_registry(self):
        router = ModelRouter()
        assert isinstance(router.registry, ModelRegistry)
        assert router.registry.has_model("mock", "mock-v1") is True

    def test_mock_provider_declares_capabilities(self):
        provider = MockProvider()
        assert ModelCapability.TEXT in provider.capabilities
        assert ModelCapability.REASONING in provider.capabilities
        assert ModelCapability.STRUCTURED_OUTPUT in provider.capabilities

        spec = provider.get_spec()
        assert isinstance(spec, ModelSpec)
        assert spec.provider == "mock"
        assert spec.model == "mock-v1"
        assert spec.has_capability(ModelCapability.REASONING) is True

    def test_router_capability_based_routing(self):
        # Create a router with mock provider and a specialized vision mock provider
        custom_registry = ModelRegistry(include_defaults=False)
        router = ModelRouter(registry=custom_registry, default_provider="mock_standard")

        standard_p = MockProvider(name="mock_standard")
        vision_p = MockProvider(name="mock_vision")

        router.register_provider(
            standard_p,
            spec=ModelSpec(
                provider="mock_standard",
                model="standard-v1",
                capabilities={ModelCapability.TEXT, ModelCapability.REASONING},
            ),
        )
        router.register_provider(
            vision_p,
            spec=ModelSpec(
                provider="mock_vision",
                model="vision-v1",
                capabilities={ModelCapability.TEXT, ModelCapability.VISION},
            ),
        )

        # Route requesting vision capability
        req = RouteRequest(capability="vision")
        selected = router.route(req)
        assert selected.name == "mock_vision"

        # Route requesting reasoning capability
        req_reasoning = RouteRequest(capability="reasoning")
        selected_reasoning = router.route(req_reasoning)
        assert selected_reasoning.name == "mock_standard"

    @pytest.mark.asyncio
    async def test_runtime_model_registry_property(self):
        runtime = EVARuntime()
        await runtime.start()

        try:
            assert isinstance(runtime.model_registry, ModelRegistry)
            assert runtime.model_registry.has_model("mock", "mock-v1") is True
        finally:
            await runtime.stop()
