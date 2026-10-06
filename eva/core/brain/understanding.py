"""Understanding engine — converts raw input into structured intent.

Provides a protocol for understanding engines and a deterministic
mock implementation for Milestone 1B.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from eva.core.brain.models import UnderstandingResult
from eva.core.context import Context
from eva.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class UnderstandingEngine(Protocol):
    """Interface for input understanding.

    Future implementations will delegate to a ModelProvider.
    """

    async def understand(
        self,
        user_input: str,
        context: Context,
    ) -> UnderstandingResult:
        """Convert raw user input into structured understanding."""
        ...


# ---------------------------------------------------------------------------
# Pattern definitions for deterministic understanding
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: list[tuple[str, list[str], bool]] = [
    # (intent, keyword_patterns, requires_goal)
    ("greeting", ["hello", "hi", "hey", "greetings", "good morning", "good evening"], False),
    ("farewell", ["bye", "goodbye", "see you", "good night"], False),
    ("identity_query", ["who are you", "what are you", "your name", "identify yourself"], False),
    ("capability_query", ["what can you do", "capabilities", "your abilities", "help me"], False),
    ("status_query", ["status", "how are you", "your state", "are you ok"], False),
]


class DeterministicUnderstanding:
    """Pattern-based understanding for Milestone 1B.

    Uses keyword matching to classify intents deterministically.
    Designed to be replaced by a ModelProvider-backed implementation
    in future milestones.
    """

    async def understand(
        self,
        user_input: str,
        context: Context,
    ) -> UnderstandingResult:
        """Classify user input using keyword patterns."""
        normalized = user_input.strip().lower()
        logger.debug("Understanding input: %r", user_input)

        for intent, patterns, requires_goal in _INTENT_PATTERNS:
            for pattern in patterns:
                if pattern in normalized:
                    result = UnderstandingResult(
                        intent=intent,
                        confidence=1.0,
                        original_input=user_input,
                        requires_goal=requires_goal,
                    )
                    logger.info("Understanding complete: intent=%s", intent)
                    return result

        # Default: general conversation
        result = UnderstandingResult(
            intent="general",
            confidence=0.5,
            original_input=user_input,
            requires_goal=False,
        )
        logger.info("Understanding complete: intent=general (fallback)")
        return result
