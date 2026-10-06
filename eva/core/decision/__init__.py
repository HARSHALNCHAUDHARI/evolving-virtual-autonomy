"""Decision engine — selects the next permitted action.

Inspects reasoning, plan, and safety state to choose the action
that EVA should execute next.
"""

from __future__ import annotations

from eva.core.brain.models import DecisionResult, ReasoningResult
from eva.core.context import Context
from eva.core.enums import ActionStatus, RiskLevel
from eva.core.models import Action, Plan
from eva.logging import get_logger
from eva.safety import SafetyPolicy, SafetyVerdict

logger = get_logger(__name__)


class DecisionEngine:
    """Selects the next action and validates it against safety.

    The decision engine bridges reasoning/planning with execution
    by producing a concrete action and verifying it is permitted.
    """

    def __init__(self, safety: SafetyPolicy) -> None:
        self._safety = safety

    async def decide(
        self,
        reasoning: ReasoningResult,
        plan: Plan,
        context: Context,
    ) -> tuple[DecisionResult, SafetyVerdict]:
        """Select an action and evaluate its safety.

        Returns:
            A tuple of (DecisionResult, SafetyVerdict).
            The SafetyVerdict indicates whether execution is permitted.
        """
        logger.debug("Decision engine evaluating: %s", reasoning.recommended_action)

        # Build the action from the recommended step
        action = Action(
            name=reasoning.recommended_action,
            description=f"Execute: {reasoning.recommended_action}",
            risk_level=self._assess_risk(reasoning.recommended_action),
            status=ActionStatus.PENDING,
        )

        # Safety evaluation
        verdict = self._safety.evaluate(action)

        if verdict.allowed:
            action = action.model_copy(update={"status": ActionStatus.APPROVED})
        else:
            action = action.model_copy(update={"status": ActionStatus.REJECTED})

        decision = DecisionResult(
            decision=reasoning.recommended_action,
            selected_action=action,
            confidence=reasoning.confidence,
            requires_confirmation=False,
        )

        logger.info(
            "Decision: %s | safety=%s",
            decision.decision,
            "APPROVED" if verdict.allowed else "REJECTED",
        )
        return decision, verdict

    @staticmethod
    def _assess_risk(action_name: str) -> RiskLevel:
        """Assess the risk level of an action by name."""
        safe_actions = {"respond", "no_op"}
        if action_name in safe_actions:
            return RiskLevel.LOW
        return RiskLevel.MEDIUM
