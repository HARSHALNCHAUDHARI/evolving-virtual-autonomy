"""Tests for EVA understanding engine."""

import pytest

from eva.core.brain.models import UnderstandingResult
from eva.core.brain.understanding import (
    DeterministicUnderstanding,
    UnderstandingEngine,
)
from eva.core.context import Context


class TestUnderstandingProtocol:
    """DeterministicUnderstanding conforms to UnderstandingEngine protocol."""

    def test_protocol_conformance(self):
        engine = DeterministicUnderstanding()
        assert isinstance(engine, UnderstandingEngine)


class TestDeterministicUnderstanding:
    """Keyword pattern matching for intents."""

    @pytest.fixture
    def engine(self) -> DeterministicUnderstanding:
        return DeterministicUnderstanding()

    @pytest.fixture
    def context(self) -> Context:
        return Context()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("user_input", "expected_intent"),
        [
            ("hello", "greeting"),
            ("hi there", "greeting"),
            ("Hey EVA", "greeting"),
            ("greetings everyone", "greeting"),
            ("good morning", "greeting"),
            ("good evening", "greeting"),
            ("bye", "farewell"),
            ("goodbye for now", "farewell"),
            ("see you later", "farewell"),
            ("good night", "farewell"),
            ("who are you?", "identity_query"),
            ("what are you", "identity_query"),
            ("tell me your name", "identity_query"),
            ("please identify yourself", "identity_query"),
            ("what can you do?", "capability_query"),
            ("list your capabilities", "capability_query"),
            ("tell me about your abilities", "capability_query"),
            ("can you help me", "capability_query"),
            ("what is your status", "status_query"),
            ("how are you doing", "status_query"),
            ("what is your state", "status_query"),
            ("are you ok?", "status_query"),
        ],
    )
    async def test_pattern_matching_intents(
        self,
        engine: DeterministicUnderstanding,
        context: Context,
        user_input: str,
        expected_intent: str,
    ):
        result = await engine.understand(user_input, context)
        assert isinstance(result, UnderstandingResult)
        assert result.intent == expected_intent
        assert result.confidence == 1.0
        assert result.original_input == user_input
        assert result.requires_goal is False

    @pytest.mark.asyncio
    async def test_case_insensitivity_and_whitespace(
        self,
        engine: DeterministicUnderstanding,
        context: Context,
    ):
        result = await engine.understand("   HELLO THERE   ", context)
        assert result.intent == "greeting"
        assert result.original_input == "   HELLO THERE   "

    @pytest.mark.asyncio
    async def test_general_fallback_intent(
        self,
        engine: DeterministicUnderstanding,
        context: Context,
    ):
        result = await engine.understand("quantum teleportation theory", context)
        assert result.intent == "general"
        assert result.confidence == 0.5
        assert result.requires_goal is False
        assert result.original_input == "quantum teleportation theory"

    @pytest.mark.asyncio
    async def test_empty_input_fallback(
        self,
        engine: DeterministicUnderstanding,
        context: Context,
    ):
        result = await engine.understand("", context)
        assert result.intent == "general"
        assert result.confidence == 0.5
