"""Tests for EVA response generator."""

import pytest

from eva.core.brain.models import (
    ReasoningResult,
    UnderstandingResult,
    VerificationResult,
)
from eva.core.brain.response import (
    DeterministicResponseGenerator,
    ResponseGenerator,
)
from eva.core.context import Context
from eva.core.models import Observation


class TestResponseGeneratorProtocol:
    """DeterministicResponseGenerator conforms to ResponseGenerator protocol."""

    def test_protocol_conformance(self):
        generator = DeterministicResponseGenerator()
        assert isinstance(generator, ResponseGenerator)


class TestDeterministicResponseGenerator:
    """Response generation for success and distinct failure classifications."""

    @pytest.fixture
    def generator(self) -> DeterministicResponseGenerator:
        return DeterministicResponseGenerator()

    @pytest.fixture
    def context(self) -> Context:
        return Context()

    @pytest.fixture
    def successful_verification(self) -> VerificationResult:
        return VerificationResult(success=True, message="Completed successfully")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "intent",
        [
            "greeting",
            "farewell",
            "identity_query",
            "capability_query",
            "status_query",
        ],
    )
    async def test_intent_response_templates(
        self,
        generator: DeterministicResponseGenerator,
        context: Context,
        successful_verification: VerificationResult,
        intent: str,
    ):
        understanding = UnderstandingResult(
            intent=intent,
            confidence=1.0,
            original_input=intent,
        )
        reasoning = ReasoningResult(
            reasoning_summary=f"Handle {intent}",
            recommended_action="respond",
        )

        response = await generator.generate(
            understanding, reasoning, successful_verification, context
        )

        assert isinstance(response, str)
        assert len(response) > 0
        if intent == "greeting":
            assert "Hello" in response or "EVA" in response
        elif intent == "identity_query":
            assert "Evolving Virtual Autonomy" in response
        elif intent == "capability_query":
            assert "cognitive pipeline" in response
            assert "do not yet have access to" in response
        elif intent == "status_query":
            assert "operating normally" in response

    @pytest.mark.asyncio
    async def test_fallback_response_for_general_intent(
        self,
        generator: DeterministicResponseGenerator,
        context: Context,
        successful_verification: VerificationResult,
    ):
        understanding = UnderstandingResult(
            intent="general",
            confidence=0.5,
            original_input="something novel",
        )
        reasoning = ReasoningResult(
            reasoning_summary="Handle general",
            recommended_action="respond",
        )

        response = await generator.generate(
            understanding, reasoning, successful_verification, context
        )

        assert "deterministic" in response.lower()

    @pytest.mark.asyncio
    async def test_safety_blocked_response(
        self,
        generator: DeterministicResponseGenerator,
        context: Context,
    ):
        understanding = UnderstandingResult(
            intent="general",
            confidence=1.0,
            original_input="run bash command",
        )
        reasoning = ReasoningResult(
            reasoning_summary="Attempt blocked command",
            recommended_action="shell_execute",
        )
        verification = VerificationResult(
            success=False,
            message="Action 'shell_execute' was blocked by safety policy.",
            error_type="safety_blocked",
        )

        response = await generator.generate(
            understanding, reasoning, verification, context
        )

        assert response == generator.SAFETY_BLOCKED_RESPONSE
        assert "blocked by my safety policy" in response

    @pytest.mark.asyncio
    async def test_execution_failure_response(
        self,
        generator: DeterministicResponseGenerator,
        context: Context,
    ):
        understanding = UnderstandingResult(
            intent="general",
            confidence=1.0,
            original_input="perform action",
        )
        reasoning = ReasoningResult(
            reasoning_summary="Perform action",
            recommended_action="custom_action",
        )
        verification = VerificationResult(
            success=False,
            message="Action failed during execution.",
            error_type="execution_failure",
        )

        response = await generator.generate(
            understanding, reasoning, verification, context
        )

        assert response == generator.EXECUTION_FAILURE_RESPONSE
        assert "execution" in response.lower()
        # CRITICAL requirement: Normal executor failure must NOT produce safety block message
        assert "blocked by my safety policy" not in response

    @pytest.mark.asyncio
    async def test_verification_failure_response(
        self,
        generator: DeterministicResponseGenerator,
        context: Context,
    ):
        understanding = UnderstandingResult(
            intent="general",
            confidence=1.0,
            original_input="perform action",
        )
        reasoning = ReasoningResult(
            reasoning_summary="Perform action",
            recommended_action="respond",
        )
        # Executor succeeded, but verification failed
        context.recent_observations.append(
            Observation(
                source="action_executor",
                data={"action": "respond", "success": True},
            )
        )
        verification = VerificationResult(
            success=False,
            message="Action verification failed.",
            error_type="verification_failure",
        )

        response = await generator.generate(
            understanding, reasoning, verification, context
        )

        assert response == generator.VERIFICATION_FAILURE_RESPONSE
        assert "verification" in response.lower()
        # CRITICAL requirement: Verification failure must NOT produce safety block message
        assert "blocked by my safety policy" not in response

    @pytest.mark.asyncio
    async def test_classify_failure_via_context_observation(
        self,
        generator: DeterministicResponseGenerator,
        context: Context,
    ):
        understanding = UnderstandingResult(
            intent="general",
            confidence=1.0,
            original_input="test",
        )
        reasoning = ReasoningResult(
            reasoning_summary="test",
            recommended_action="test",
        )

        # 1. Safety blocked observation in context
        context.recent_observations = [
            Observation(source="safety_policy", data={"success": False, "error": "Blocked"})
        ]
        resp_safety = await generator.generate(
            understanding, reasoning, VerificationResult(success=False), context
        )
        assert resp_safety == generator.SAFETY_BLOCKED_RESPONSE

        # 2. Execution failed observation in context
        context.recent_observations = [
            Observation(source="action_executor", data={"success": False, "error": "Crash"})
        ]
        resp_exec = await generator.generate(
            understanding, reasoning, VerificationResult(success=False), context
        )
        assert resp_exec == generator.EXECUTION_FAILURE_RESPONSE
        assert "blocked by my safety policy" not in resp_exec

        # 3. Execution succeeded observation in context, but verification failed
        context.recent_observations = [
            Observation(source="action_executor", data={"success": True})
        ]
        resp_verif = await generator.generate(
            understanding, reasoning, VerificationResult(success=False), context
        )
        assert resp_verif == generator.VERIFICATION_FAILURE_RESPONSE
        assert "blocked by my safety policy" not in resp_verif
