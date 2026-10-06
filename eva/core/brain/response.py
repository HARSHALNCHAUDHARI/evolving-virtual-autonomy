"""Response generator — produces user-facing responses.

Generates responses based on the cognitive pipeline results.
Does not print directly — returns text for the CLI layer to display.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from eva.core.brain.models import (
    ReasoningResult,
    UnderstandingResult,
    VerificationResult,
)
from eva.core.context import Context
from eva.core.self_model import EVASelfModel
from eva.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class ResponseGenerator(Protocol):
    """Interface for response generation.

    Future implementations will use a ModelProvider for
    natural language generation.
    """

    async def generate(
        self,
        understanding: UnderstandingResult,
        reasoning: ReasoningResult,
        verification: VerificationResult,
        context: Context,
    ) -> str:
        """Generate a user-facing response."""
        ...


# ---------------------------------------------------------------------------
# Deterministic response templates
# ---------------------------------------------------------------------------

_RESPONSE_MAP: dict[str, str] = {
    "greeting": (
        "Hello. I'm EVA — Evolving Virtual Autonomy. I'm online and ready."
    ),
    "farewell": (
        "Goodbye. I'll be here when you need me."
    ),
    "identity_query": (
        "I am EVA — Evolving Virtual Autonomy.\n"
        "I am an autonomous personal AI system designed to assist, reason, "
        "plan, and act on behalf of my user.\n"
        "I am currently running in foundation mode. "
        "My cognitive architecture is being initialized incrementally."
    ),
    "capability_query": (
        "Currently, I can:\n"
        "• Process natural language through a structured cognitive pipeline\n"
        "• Track conversation context\n"
        "• Evaluate actions for safety before execution\n"
        "• Manage goals and plans\n"
        "• Reason about user intent\n\n"
        "I do not yet have access to external tools, web search, "
        "file operations, voice, vision, or autonomous agents. "
        "Those capabilities will arrive in future milestones."
    ),
    "status_query": (
        "I am operating normally.\n"
        "All cognitive systems are active: "
        "understanding, reasoning, planning, decision, and safety."
    ),
}

_FALLBACK_RESPONSE = (
    "I understand your message. I'm currently operating with a deterministic "
    "cognitive pipeline. Once a real model provider is connected, "
    "I'll be able to provide more nuanced responses."
)

_SAFETY_BLOCKED_RESPONSE = (
    "I'm unable to perform that action — it was blocked by my safety policy. "
    "This is by design to ensure safe operation."
)

_EXECUTION_FAILURE_RESPONSE = (
    "I attempted to execute the action, but it failed during execution."
)

_VERIFICATION_FAILURE_RESPONSE = (
    "The action was executed, but verification of the result failed."
)


class DeterministicResponseGenerator:
    """Generates deterministic responses for Milestone 1B.

    Maps understood intents to predefined response templates.
    Distinguishes successful execution, safety blocks, execution failures,
    and verification failures.
    """

    SAFETY_BLOCKED_RESPONSE = _SAFETY_BLOCKED_RESPONSE
    EXECUTION_FAILURE_RESPONSE = _EXECUTION_FAILURE_RESPONSE
    VERIFICATION_FAILURE_RESPONSE = _VERIFICATION_FAILURE_RESPONSE

    def _classify_failure(
        self,
        verification: VerificationResult,
        context: Context,
    ) -> str:
        """Classify failure as safety_blocked, execution_failure, or verification_failure."""
        if verification.error_type:
            return verification.error_type

        # Inspect context's most recent observation if available
        if context.recent_observations:
            last_obs = context.recent_observations[-1]
            if last_obs.source == "safety_policy":
                return "safety_blocked"
            if not last_obs.data.get("success", False):
                return "execution_failure"
            return "verification_failure"

        # Fallback inspection of verification message
        msg = (verification.message or "").lower()
        if "safety" in msg or "blocked" in msg:
            return "safety_blocked"
        if "execut" in msg:
            return "execution_failure"
        return "verification_failure"

    async def generate(
        self,
        understanding: UnderstandingResult,
        reasoning: ReasoningResult,
        verification: VerificationResult,
        context: Context,
    ) -> str:
        """Generate a response based on intent and verification."""
        logger.debug("Generating response for intent: %s", understanding.intent)

        if not verification.success:
            failure_mode = self._classify_failure(verification, context)
            if failure_mode == "safety_blocked":
                logger.info("Response generated: safety-blocked")
                return self.SAFETY_BLOCKED_RESPONSE
            elif failure_mode == "execution_failure":
                logger.info("Response generated: execution failure")
                return self.EXECUTION_FAILURE_RESPONSE
            else:
                logger.info("Response generated: verification failure")
                return self.VERIFICATION_FAILURE_RESPONSE

        response = _RESPONSE_MAP.get(understanding.intent, _FALLBACK_RESPONSE)
        logger.info("Response generated for intent: %s", understanding.intent)
        return response
