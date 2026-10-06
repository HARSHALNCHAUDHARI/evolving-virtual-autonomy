"""Tests for EVA reasoning engine."""

import pytest

from eva.core.brain.models import ReasoningResult, UnderstandingResult
from eva.core.context import Context
from eva.core.models import Goal
from eva.core.reasoning import DeterministicReasoning, ReasoningEngine


class TestReasoningProtocol:
    """DeterministicReasoning conforms to ReasoningEngine protocol."""

    def test_protocol_conformance(self):
        engine = DeterministicReasoning()
        assert isinstance(engine, ReasoningEngine)


class TestDeterministicReasoning:
    """Mapping understood intents to structured reasoning outcomes."""

    @pytest.fixture
    def engine(self) -> DeterministicReasoning:
        return DeterministicReasoning()

    @pytest.fixture
    def context(self) -> Context:
        return Context()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("intent", "action_expected", "requires_action_expected"),
        [
            ("greeting", "respond", False),
            ("farewell", "respond", False),
            ("identity_query", "respond", False),
            ("capability_query", "respond", False),
            ("status_query", "respond", False),
            ("general", "respond", False),
            ("unrecognized_intent", "respond", False),
        ],
    )
    async def test_intent_reasoning(
        self,
        engine: DeterministicReasoning,
        context: Context,
        intent: str,
        action_expected: str,
        requires_action_expected: bool,
    ):
        understanding = UnderstandingResult(
            intent=intent,
            confidence=0.9,
            original_input=intent,
        )
        result = await engine.reason(understanding, context, None)

        assert isinstance(result, ReasoningResult)
        assert result.recommended_action == action_expected
        assert result.requires_action is requires_action_expected
        assert result.confidence == 0.9
        assert len(result.reasoning_summary) > 0

    @pytest.mark.asyncio
    async def test_reasoning_with_active_goal(
        self,
        engine: DeterministicReasoning,
        context: Context,
    ):
        understanding = UnderstandingResult(
            intent="greeting",
            confidence=1.0,
            original_input="hello",
        )
        goal = Goal(title="Greet user")
        result = await engine.reason(understanding, context, goal)

        assert isinstance(result, ReasoningResult)
        assert result.recommended_action == "respond"
