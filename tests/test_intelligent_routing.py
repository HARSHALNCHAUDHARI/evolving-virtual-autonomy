"""Tests for intelligent capability-aware model routing (Milestone 1D.3).

Verifies capability filtering, deterministic candidate scoring, preference handling,
availability handling, fallback selection, explainability, error states, and
provider compatibility.
"""

from __future__ import annotations

import pytest

from eva.providers.gemini import GeminiProvider
from eva.providers.mock import MockProvider
from eva.providers.models import (
    ContextLimits,
    ModelCapability,
    ModelCost,
    ModelRequest,
    ModelResponse,
    ModelSpec,
    NoEligibleModelError,
    ProviderNotFoundError,
    RouteRequest,
    RoutingDecision,
    TaskType,
)
from eva.providers.registry import ModelRegistry
from eva.providers.router import ModelRouter


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


def create_test_router() -> tuple[ModelRouter, ModelRegistry]:
    """Create a ModelRouter with a populated ModelRegistry containing diverse specs."""
    registry = ModelRegistry(include_defaults=False)

    # 1. Mock standard model: text + reasoning
    mock_std = MockProvider(name="mock")
    spec_mock_std = ModelSpec(
        provider="mock",
        model="mock-v1",
        capabilities={ModelCapability.TEXT, ModelCapability.REASONING},
        latency_tier="low",
        priority=1,
        available=True,
    )
    registry.register_model(spec_mock_std)

    # 2. Mock vision model: text + vision
    mock_vision = MockProvider(name="mock_vision")
    spec_mock_vision = ModelSpec(
        provider="mock_vision",
        model="vision-v1",
        capabilities={ModelCapability.TEXT, ModelCapability.VISION},
        latency_tier="medium",
        priority=2,
        available=True,
    )
    registry.register_model(spec_mock_vision)

    # 3. Gemini multi-capable model: text + reasoning + structured_output + vision + long_context
    def mock_gemini_transport(req, timeout):
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": '{"result": "success"}'}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 5,
                "totalTokenCount": 10,
            },
        }

    gemini_prov = GeminiProvider(
        api_key="fake-test-key",
        model="gemini-1.5-flash",
        transport=mock_gemini_transport,
    )
    spec_gemini = ModelSpec(
        provider="gemini",
        model="gemini-1.5-flash",
        capabilities={
            ModelCapability.TEXT,
            ModelCapability.REASONING,
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.VISION,
            ModelCapability.LONG_CONTEXT,
        },
        latency_tier="low",
        cost=ModelCost(input_cost_per_million=0.15, output_cost_per_million=0.60),
        priority=3,
        available=True,
    )
    registry.register_model(spec_gemini)

    # 4. Heavy reasoning model (unavailable)
    heavy_mock = MockProvider(name="heavy_mock")
    spec_heavy = ModelSpec(
        provider="heavy_mock",
        model="heavy-reasoning-v1",
        capabilities={ModelCapability.TEXT, ModelCapability.REASONING},
        latency_tier="high",
        priority=10,
        available=False,  # Intentionally unavailable
    )
    registry.register_model(spec_heavy)

    router = ModelRouter(
        providers={
            "mock": mock_std,
            "mock_vision": mock_vision,
            "gemini": gemini_prov,
            "heavy_mock": heavy_mock,
        },
        default_provider="mock",
        default_model="mock-v1",
        registry=registry,
    )

    return router, registry


# ===========================================================================
# 1. Capability Filtering
# ===========================================================================


class TestCapabilityFiltering:
    """Verifies that capability filtering strictly gates candidate selection."""

    def test_single_capability_filtering(self):
        router, _ = create_test_router()
        req = RouteRequest(required_capabilities={ModelCapability.VISION})

        decision = router.select_model(req)
        assert decision.success is True
        # Only mock_vision and gemini have vision; gemini has higher priority (3 vs 2)
        assert decision.selected_model.has_capability(ModelCapability.VISION) is True
        assert decision.provider_name in ("gemini", "mock_vision")

    def test_multiple_required_capabilities(self):
        router, _ = create_test_router()
        req = RouteRequest(
            required_capabilities={
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.LONG_CONTEXT,
            }
        )

        decision = router.select_model(req)
        assert decision.success is True
        # Only gemini has both structured_output and long_context
        assert decision.provider_name == "gemini"
        assert decision.model_name == "gemini-1.5-flash"

    def test_model_missing_required_capability_is_never_selected(self):
        router, _ = create_test_router()
        req = RouteRequest(
            required_capabilities={ModelCapability.STRUCTURED_OUTPUT},
            preferred_provider="mock",  # User strongly prefers mock, but mock lacks structured_output!
        )

        decision = router.select_model(req)
        assert decision.success is True
        # Must NOT select mock because mock lacks structured_output!
        assert decision.provider_name != "mock"
        assert decision.provider_name == "gemini"

        # Check candidate evaluations
        mock_eval = next(c for c in decision.candidates_evaluated if c.provider == "mock")
        assert mock_eval.eligible is False
        assert "Missing required capabilities" in mock_eval.elimination_reason

    def test_required_capability_mismatch_fails_cleanly(self):
        router, _ = create_test_router()
        # Request capability that NO candidate supports
        req = RouteRequest(required_capabilities={ModelCapability.AUDIO})

        decision = router.select_model(req)
        assert decision.success is False
        assert "No eligible model found" in decision.error
        assert "audio" in decision.error

        with pytest.raises(NoEligibleModelError) as exc_info:
            router.route(req)
        assert "audio" in str(exc_info.value)


# ===========================================================================
# 2. Deterministic Scoring
# ===========================================================================


class TestDeterministicScoring:
    """Verifies multi-factor deterministic candidate scoring."""

    def test_scoring_is_deterministic_across_runs(self):
        router, _ = create_test_router()
        req = RouteRequest(required_capabilities={ModelCapability.REASONING})

        decision_1 = router.select_model(req)
        decision_2 = router.select_model(req)

        assert decision_1.provider_name == decision_2.provider_name
        assert decision_1.model_name == decision_2.model_name
        assert decision_1.score == decision_2.score
        assert [c.score for c in decision_1.candidates_evaluated] == [
            c.score for c in decision_2.candidates_evaluated
        ]

    def test_score_breakdown_contains_all_components(self):
        router, _ = create_test_router()
        req = RouteRequest(
            required_capabilities={ModelCapability.TEXT},
            preferred_provider="gemini",
            latency_preference="low",
        )

        decision = router.select_model(req)
        winner_eval = next(
            c for c in decision.candidates_evaluated if c.model_id == decision.selected_model.model_id
        )

        assert "required_capabilities" in winner_eval.score_breakdown
        assert "preferred_provider" in winner_eval.score_breakdown
        assert "availability" in winner_eval.score_breakdown
        assert "latency" in winner_eval.score_breakdown
        assert winner_eval.score > 50.0


# ===========================================================================
# 3. Preference Handling (Provider, Model, Latency, Cost, Privacy)
# ===========================================================================


class TestPreferences:
    """Verifies preferred provider, model, latency, cost, and privacy filters."""

    def test_preferred_provider_bonus(self):
        router, _ = create_test_router()
        # Both mock and gemini support reasoning
        req = RouteRequest(
            required_capabilities={ModelCapability.REASONING},
            preferred_provider="mock",
        )

        decision = router.select_model(req)
        assert decision.provider_name == "mock"
        assert any("matched preferred provider 'mock'" in r for r in decision.reasons)

    def test_preferred_model_bonus(self):
        router, _ = create_test_router()
        req = RouteRequest(
            preferred_model="mock-v1",
        )

        decision = router.select_model(req)
        assert decision.model_name == "mock-v1"
        assert any("matched preferred model 'mock-v1'" in r for r in decision.reasons)

    def test_latency_preference_low(self):
        router, _ = create_test_router()
        req = RouteRequest(
            required_capabilities={ModelCapability.VISION},
            latency_preference="low",
        )
        decision = router.select_model(req)
        # Gemini is 'low' latency tier, mock_vision is 'medium'
        assert decision.selected_model.latency_tier == "low"
        assert decision.provider_name == "gemini"

    def test_privacy_preference_local_only(self):
        router, _ = create_test_router()
        req = RouteRequest(
            privacy_preference="local_only",
        )
        decision = router.select_model(req)
        # Gemini is a cloud provider, not local; mock is local/mock
        assert decision.provider_name in ("mock", "mock_vision")
        gemini_eval = next(c for c in decision.candidates_evaluated if c.provider == "gemini")
        assert gemini_eval.eligible is False
        assert "privacy_preference='local_only'" in gemini_eval.elimination_reason

    def test_min_priority_filter(self):
        router, _ = create_test_router()
        # Set min_priority to 3 -> excludes mock (priority 1) and mock_vision (priority 2)
        req = RouteRequest(min_priority=3)
        decision = router.select_model(req)
        assert decision.provider_name == "gemini"
        assert decision.selected_model.priority >= 3


# ===========================================================================
# 4. Availability & Fallback Selection
# ===========================================================================


class TestAvailabilityAndFallback:
    """Verifies availability checks and automatic candidate fallback."""

    def test_unavailable_primary_falls_back_to_next_eligible(self):
        router, _ = create_test_router()
        # heavy_mock has priority 10 (highest), but available=False
        req = RouteRequest(
            required_capabilities={ModelCapability.REASONING},
            preferred_provider="heavy_mock",
            fallback_allowed=True,
        )

        decision = router.select_model(req)
        assert decision.success is True
        assert decision.fallback_used is True
        assert decision.fallback_from == "heavy_mock:heavy-reasoning-v1"
        # Falls back to next highest-scoring eligible available candidate
        assert decision.provider_name in ("gemini", "mock")
        assert decision.selected_model.available is True

    def test_unavailable_primary_fails_when_fallback_disabled(self):
        router, _ = create_test_router()
        req = RouteRequest(
            required_capabilities={ModelCapability.REASONING},
            preferred_provider="heavy_mock",
            fallback_allowed=False,
        )

        decision = router.select_model(req)
        assert decision.success is False
        assert "fallback is disabled" in decision.error

    @pytest.mark.asyncio
    async def test_runtime_unavailability_dispatches_fallback(self):
        class UnavailableProvider(MockProvider):
            async def is_available(self) -> bool:
                return False

        router, registry = create_test_router()
        bad_prov = UnavailableProvider(name="bad_prov")
        router.register_provider(
            bad_prov,
            spec=ModelSpec(
                provider="bad_prov",
                model="bad-v1",
                capabilities={ModelCapability.TEXT},
                priority=100,  # Very high priority
                available=True,  # Spec says True, but runtime check returns False
            ),
        )

        req = ModelRequest(prompt="Hello", provider="bad_prov")
        resp = await router.generate(req, fallback_allowed=True)

        assert resp.success is True
        # Runtime check caught bad_prov was offline, fell back to mock
        assert resp.provider != "bad_prov"
        assert resp.provider == "mock"

    @pytest.mark.asyncio
    async def test_runtime_unavailability_generate_structured_dispatches_fallback(self):
        class UnavailableStructuredProvider(MockProvider):
            async def is_available(self) -> bool:
                return False

        router, registry = create_test_router()
        bad_prov = UnavailableStructuredProvider(name="bad_structured_prov")
        router.register_provider(
            bad_prov,
            spec=ModelSpec(
                provider="bad_structured_prov",
                model="bad-struct-v1",
                capabilities={ModelCapability.TEXT, ModelCapability.STRUCTURED_OUTPUT},
                priority=100,  # Higher than gemini (priority 3)
                available=True,
            ),
        )

        req = ModelRequest(prompt="Produce data", provider="bad_structured_prov")
        resp = await router.generate_structured(req, dict, fallback_allowed=True)

        assert resp.success is True
        # Runtime check caught bad_structured_prov was offline, fell back to gemini
        assert resp.provider == "gemini"

    @pytest.mark.asyncio
    async def test_runtime_unavailability_generate_structured_fails_without_fallback(self):
        class UnavailableStructuredProvider(MockProvider):
            async def is_available(self) -> bool:
                return False

        router, registry = create_test_router()
        bad_prov = UnavailableStructuredProvider(name="bad_structured_prov")
        router.register_provider(
            bad_prov,
            spec=ModelSpec(
                provider="bad_structured_prov",
                model="bad-struct-v1",
                capabilities={ModelCapability.TEXT, ModelCapability.STRUCTURED_OUTPUT},
                priority=100,
                available=True,
            ),
        )

        req = ModelRequest(prompt="Produce data", provider="bad_structured_prov")
        resp = await router.generate_structured(req, dict, fallback_allowed=False)

        assert resp.success is False
        assert "is unavailable and no fallback succeeded" in resp.error
        assert resp.provider == "bad_structured_prov"


# ===========================================================================
# 5. Explainable Routing Decisions
# ===========================================================================


class TestExplainableRouting:
    """Verifies that routing decisions provide clear human-readable explanations."""

    def test_decision_explain_text(self):
        router, _ = create_test_router()
        req = RouteRequest(
            required_capabilities={ModelCapability.REASONING, ModelCapability.STRUCTURED_OUTPUT},
            preferred_provider="gemini",
        )

        decision = router.select_model(req)
        assert decision.success is True
        explanation = decision.explain()

        assert "Selected gemini:gemini-1.5-flash because:" in explanation
        assert "required reasoning capability matched" in explanation
        assert "required structured_output capability matched" in explanation
        assert "provider available" in explanation
        assert "matched preferred provider 'gemini'" in explanation

    def test_failed_decision_explanation(self):
        router, _ = create_test_router()
        req = RouteRequest(required_capabilities={ModelCapability.AUDIO})
        decision = router.select_model(req)
        assert decision.success is False
        explanation = decision.explain()
        assert "Routing failed:" in explanation
        assert "audio" in explanation

    def test_last_decision_property_on_router(self):
        router, _ = create_test_router()
        assert router.last_decision is None

        router.route("mock")
        assert router.last_decision is not None
        assert router.last_decision.provider_name == "mock"


# ===========================================================================
# 6. Provider Error Handling & Compatibility
# ===========================================================================


class TestRouterErrorHandling:
    """Verifies clean error handling on unknown providers and unsupported tasks."""

    def test_unregistered_provider_raises_provider_not_found(self):
        router, _ = create_test_router()
        req = RouteRequest(provider="unknown_llm")

        with pytest.raises(ProviderNotFoundError):
            router.route(req)

    def test_no_eligible_candidate_raises_no_eligible_model_error(self):
        router, _ = create_test_router()
        req = RouteRequest(required_capabilities={ModelCapability.AUDIO})

        with pytest.raises(NoEligibleModelError):
            router.route(req)

    @pytest.mark.asyncio
    async def test_generate_structured_enforces_structured_output_capability(self):
        router, _ = create_test_router()
        # mock lacks structured_output, gemini has it
        req = ModelRequest(prompt="Provide data")
        resp = await router.generate_structured(req, dict)

        assert resp.provider == "gemini"
        assert "routing_decision" in resp.metadata.extra
