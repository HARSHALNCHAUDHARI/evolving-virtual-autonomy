"""Verifier — validates that actions completed successfully.

Provides a protocol and deterministic implementation for Milestone 1B.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from eva.core.brain.models import VerificationResult
from eva.core.context import Context
from eva.core.models import Action, Observation
from eva.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class Verifier(Protocol):
    """Interface for action verification.

    Future implementations may perform deeper validation,
    retries, or self-correction.
    """

    async def verify(
        self,
        action: Action,
        observation: Observation,
        context: Context,
    ) -> VerificationResult:
        """Verify whether an action completed successfully."""
        ...


class DeterministicVerifier:
    """Verifies action outcomes for Milestone 1B.

    Checks the observation data for success/failure indicators.
    """

    async def verify(
        self,
        action: Action,
        observation: Observation,
        context: Context,
    ) -> VerificationResult:
        """Check the observation for success indicators."""
        logger.debug("Verifying action: %s", action.name)

        success = observation.data.get("success", False)
        error = observation.data.get("error")

        if success:
            result = VerificationResult(
                success=True,
                confidence=observation.confidence,
                message=f"Action '{action.name}' completed successfully.",
            )
        elif observation.source == "safety_policy":
            result = VerificationResult(
                success=False,
                confidence=observation.confidence,
                message=error or f"Action '{action.name}' was blocked by safety policy.",
                error_type="safety_blocked",
            )
        else:
            result = VerificationResult(
                success=False,
                confidence=observation.confidence,
                message=error or f"Action '{action.name}' failed during execution.",
                error_type="execution_failure",
            )

        logger.info("Verification: success=%s", result.success)
        return result
