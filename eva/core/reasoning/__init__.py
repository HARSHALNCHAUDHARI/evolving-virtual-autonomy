"""Reasoning engine — structured reasoning over understood input.

Provides a protocol and a deterministic implementation for Milestone 1B.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from eva.core.brain.models import ReasoningResult, UnderstandingResult
from eva.core.context import Context
from eva.core.models import Goal
from eva.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class ReasoningEngine(Protocol):
    """Interface for reasoning engines.

    Future implementations will delegate to a ModelProvider.
    """

    async def reason(
        self,
        understanding: UnderstandingResult,
        context: Context,
        goal: Goal | None,
    ) -> ReasoningResult:
        """Produce a structured reasoning result."""
        ...


# ---------------------------------------------------------------------------
# Intent → reasoning mapping
# ---------------------------------------------------------------------------

_REASONING_MAP: dict[str, tuple[str, str, bool]] = {
    # intent: (summary, recommended_action, requires_action)
    "greeting": (
        "User is greeting EVA. Respond with a friendly acknowledgment.",
        "respond",
        False,
    ),
    "farewell": (
        "User is saying goodbye. Respond politely and indicate availability.",
        "respond",
        False,
    ),
    "identity_query": (
        "User is asking about EVA's identity. Provide operational self-description.",
        "respond",
        False,
    ),
    "capability_query": (
        "User is asking about EVA's capabilities. Describe current abilities accurately.",
        "respond",
        False,
    ),
    "status_query": (
        "User is asking about EVA's status. Report current operational state.",
        "respond",
        False,
    ),
}


class DeterministicReasoning:
    """Intent-based deterministic reasoning for Milestone 1B.

    Maps understood intents to reasoning outcomes without
    using an external model. Designed for replacement by a
    ModelProvider-backed implementation.
    """

    async def reason(
        self,
        understanding: UnderstandingResult,
        context: Context,
        goal: Goal | None,
    ) -> ReasoningResult:
        """Produce reasoning based on the understood intent."""
        intent = understanding.intent
        logger.debug("Reasoning over intent: %s", intent)

        if intent in _REASONING_MAP:
            summary, action, requires = _REASONING_MAP[intent]
        else:
            summary = (
                f"EVA received a general input. "
                f"Respond helpfully within current capabilities."
            )
            action = "respond"
            requires = False

        result = ReasoningResult(
            reasoning_summary=summary,
            confidence=understanding.confidence,
            recommended_action=action,
            requires_action=requires,
        )
        logger.info("Reasoning complete: action=%s", action)
        return result
