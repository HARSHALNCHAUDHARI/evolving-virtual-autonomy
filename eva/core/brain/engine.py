"""Cognitive engine — central orchestrator of the EVA cognitive loop.

Composes independent components (understanding, goals, reasoning,
planning, decision, execution, verification, response) into a
structured pipeline. Each stage is a separate component injected
via constructor — no stage logic lives in this module.
"""

from __future__ import annotations

from eva.core.brain.executor import ActionExecutor, SafeActionExecutor
from eva.core.brain.models import (
    CognitiveResult,
    ReasoningResult,
    UnderstandingResult,
    VerificationResult,
)
from eva.core.brain.response import DeterministicResponseGenerator, ResponseGenerator
from eva.core.brain.understanding import DeterministicUnderstanding, UnderstandingEngine
from eva.core.brain.verifier import DeterministicVerifier, Verifier
from eva.core.context import Context, ContextManager
from eva.core.decision import DecisionEngine
from eva.core.enums import ActionStatus, RiskLevel
from eva.core.goals import GoalManager
from eva.core.models import Action, Goal, Observation
from eva.core.planner import DeterministicPlanner, Planner
from eva.core.reasoning import DeterministicReasoning, ReasoningEngine
from eva.logging import get_logger
from eva.safety import SafetyPolicy

logger = get_logger(__name__)


class CognitiveEngine:
    """Orchestrates the full cognitive cycle.

    Pipeline stages:
    1. Understanding  — parse intent from raw input
    2. Goal eval      — decide if a goal is needed
    3. Reasoning      — structured reasoning over intent + context
    4. Planning       — create an execution plan
    5. Decision       — select and safety-check the next action
    6. Execution      — execute the approved action
    7. Observation    — capture the result
    8. Verification   — confirm success or failure
    9. Response       — generate user-facing text

    Each component is injected via constructor, allowing future
    milestones to swap in ModelProvider-backed implementations
    without modifying this orchestrator.
    """

    def __init__(
        self,
        *,
        context_manager: ContextManager,
        safety: SafetyPolicy,
        understanding: UnderstandingEngine | None = None,
        goal_manager: GoalManager | None = None,
        reasoning: ReasoningEngine | None = None,
        planner: Planner | None = None,
        decision: DecisionEngine | None = None,
        executor: ActionExecutor | None = None,
        verifier: Verifier | None = None,
        response_generator: ResponseGenerator | None = None,
    ) -> None:
        self._context_manager = context_manager
        self._understanding = understanding or DeterministicUnderstanding()
        self._goal_manager = goal_manager or GoalManager()
        self._reasoning = reasoning or DeterministicReasoning()
        self._planner = planner or DeterministicPlanner()
        self._decision = decision or DecisionEngine(safety)
        self._executor = executor or SafeActionExecutor()
        self._verifier = verifier or DeterministicVerifier()
        self._response_gen = response_generator or DeterministicResponseGenerator()
        logger.debug("CognitiveEngine initialized")

    @property
    def goal_manager(self) -> GoalManager:
        """Expose goal manager for external inspection."""
        return self._goal_manager

    async def process(
        self,
        user_input: str,
        context: Context,
    ) -> CognitiveResult:
        """Execute the full cognitive cycle.

        Args:
            user_input: Raw text from the user.
            context: Current EVA context snapshot.

        Returns:
            A CognitiveResult capturing every stage of the cycle.
        """
        logger.info("Cognitive cycle started")

        try:
            # 1. Understanding
            understanding = await self._understanding.understand(user_input, context)

            # 2. Goal evaluation
            goal: Goal | None = None
            if self._goal_manager.evaluate_need(understanding):
                goal = self._goal_manager.create_goal(
                    title=f"Handle: {understanding.intent}",
                    description=understanding.original_input,
                )
                self._context_manager.set_active_goal(goal)
                logger.info("Goal evaluation: goal created")
            else:
                logger.info("Goal evaluation: no goal needed")

            # 3. Reasoning
            reasoning = await self._reasoning.reason(understanding, context, goal)

            # 4. Planning
            plan = await self._planner.create_plan(reasoning, goal, context)

            # 5. Decision + Safety
            decision, verdict = await self._decision.decide(reasoning, plan, context)

            # 6. Execution (only if approved)
            if verdict.allowed:
                observation = await self._executor.execute(
                    decision.selected_action, context
                )
            else:
                observation = Observation(
                    source="safety_policy",
                    data={
                        "action": decision.selected_action.name,
                        "success": False,
                        "error": verdict.reason,
                    },
                    confidence=1.0,
                )
                logger.warning(
                    "Action blocked by safety: %s", decision.selected_action.name
                )

            # Record observation in context
            self._context_manager.add_observation(observation)

            # 7. Verification
            verification = await self._verifier.verify(
                decision.selected_action, observation, context
            )

            # 8. Response generation
            response = await self._response_gen.generate(
                understanding, reasoning, verification, context
            )

            # Complete goal if one was created
            if goal is not None:
                if verification.success:
                    goal = self._goal_manager.complete_goal(goal.id)
                else:
                    goal = self._goal_manager.fail_goal(goal.id)
                self._context_manager.clear_active_goal()

            result = CognitiveResult(
                input=user_input,
                understanding=understanding,
                goal=goal,
                reasoning_summary=reasoning.reasoning_summary,
                plan=plan,
                decision=decision,
                action=decision.selected_action,
                observation=observation,
                verification=verification,
                response=response,
                success=verification.success,
                error=None if verification.success else (verification.message or "Pipeline step failed"),
            )

            logger.info("Cognitive cycle completed successfully")
            return result

        except Exception as exc:
            logger.exception("Cognitive cycle failed: %s", exc)

            if goal is not None:
                goal = self._goal_manager.fail_goal(goal.id)
                self._context_manager.clear_active_goal()

            # Build a graceful failure result
            error_understanding = UnderstandingResult(
                intent="error",
                confidence=0.0,
                original_input=user_input,
            )
            error_action = Action(
                name="no_op",
                description="Error recovery",
                risk_level=RiskLevel.LOW,
                status=ActionStatus.FAILED,
            )
            error_observation = Observation(
                source="cognitive_engine",
                data={"success": False, "error": str(exc)},
                confidence=0.0,
            )
            error_verification = VerificationResult(
                success=False,
                confidence=0.0,
                message=str(exc),
            )
            from eva.core.brain.models import DecisionResult

            error_decision = DecisionResult(
                decision="error",
                selected_action=error_action,
                confidence=0.0,
            )

            return CognitiveResult(
                input=user_input,
                understanding=error_understanding,
                goal=goal,
                reasoning_summary=f"Error during processing: {exc}",
                decision=error_decision,
                action=error_action,
                observation=error_observation,
                verification=error_verification,
                response=(
                    "I encountered an error while processing your request. "
                    "Please try again."
                ),
                success=False,
                error=str(exc),
            )
