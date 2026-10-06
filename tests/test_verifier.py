"""Tests for EVA verifier component."""

import pytest

from eva.core.brain.models import VerificationResult
from eva.core.brain.verifier import DeterministicVerifier, Verifier
from eva.core.context import Context
from eva.core.models import Action, Observation


class TestVerifierProtocol:
    """DeterministicVerifier conforms to Verifier protocol."""

    def test_protocol_conformance(self):
        verifier = DeterministicVerifier()
        assert isinstance(verifier, Verifier)


class TestDeterministicVerifier:
    """Action outcome verification."""

    @pytest.fixture
    def verifier(self) -> DeterministicVerifier:
        return DeterministicVerifier()

    @pytest.fixture
    def context(self) -> Context:
        return Context()

    @pytest.mark.asyncio
    async def test_verify_successful_observation(
        self,
        verifier: DeterministicVerifier,
        context: Context,
    ):
        action = Action(name="respond", description="Respond to user")
        obs = Observation(
            source="action_executor",
            data={"action": "respond", "success": True},
            confidence=0.9,
        )

        result = await verifier.verify(action, obs, context)

        assert isinstance(result, VerificationResult)
        assert result.success is True
        assert result.confidence == 0.9
        assert "completed successfully" in result.message
        assert result.error_type is None

    @pytest.mark.asyncio
    async def test_verify_executor_failure(
        self,
        verifier: DeterministicVerifier,
        context: Context,
    ):
        action = Action(name="respond", description="Respond to user")
        obs = Observation(
            source="action_executor",
            data={"action": "respond", "success": False, "error": "Execution device error"},
            confidence=0.8,
        )

        result = await verifier.verify(action, obs, context)

        assert isinstance(result, VerificationResult)
        assert result.success is False
        assert result.error_type == "execution_failure"
        assert result.message == "Execution device error"

    @pytest.mark.asyncio
    async def test_verify_safety_blocked_observation(
        self,
        verifier: DeterministicVerifier,
        context: Context,
    ):
        action = Action(name="shell_execute", description="Dangerous action")
        obs = Observation(
            source="safety_policy",
            data={"action": "shell_execute", "success": False, "error": "Blocked by policy"},
            confidence=1.0,
        )

        result = await verifier.verify(action, obs, context)

        assert isinstance(result, VerificationResult)
        assert result.success is False
        assert result.error_type == "safety_blocked"
        assert result.message == "Blocked by policy"
