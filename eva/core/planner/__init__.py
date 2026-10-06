"""Planner — creates action plans from reasoning results.

Provides a protocol and deterministic implementation for Milestone 1B.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import uuid4

from eva.core.brain.models import ReasoningResult
from eva.core.context import Context
from eva.core.enums import PlanStatus
from eva.core.models import Goal, Plan, PlanStep
from eva.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class Planner(Protocol):
    """Interface for planning engines.

    Future implementations may use a ModelProvider for
    complex multi-step plan generation.
    """

    async def create_plan(
        self,
        reasoning: ReasoningResult,
        goal: Goal | None,
        context: Context,
    ) -> Plan:
        """Create an execution plan from reasoning results."""
        ...


class DeterministicPlanner:
    """Creates simple plans for Milestone 1B.

    For conversational queries, produces a single-step RESPOND plan.
    The interface supports multi-step plans for future milestones.
    """

    async def create_plan(
        self,
        reasoning: ReasoningResult,
        goal: Goal | None,
        context: Context,
    ) -> Plan:
        """Create a plan based on the reasoning result."""
        logger.debug("Creating plan for action: %s", reasoning.recommended_action)

        # Use the goal ID if available, otherwise generate a transient ID
        goal_id = goal.id if goal else uuid4()

        steps = [
            PlanStep(
                index=0,
                description=reasoning.recommended_action,
            )
        ]

        plan = Plan(
            goal_id=goal_id,
            steps=steps,
            status=PlanStatus.ACTIVE,
        )
        logger.info("Plan created: %d step(s)", len(steps))
        return plan
