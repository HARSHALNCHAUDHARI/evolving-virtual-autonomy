"""Tests for EVA planner."""

from uuid import UUID

import pytest

from eva.core.brain.models import ReasoningResult
from eva.core.context import Context
from eva.core.enums import PlanStatus
from eva.core.models import Goal, Plan
from eva.core.planner import DeterministicPlanner, Planner


class TestPlannerProtocol:
    """DeterministicPlanner conforms to Planner protocol."""

    def test_protocol_conformance(self):
        planner = DeterministicPlanner()
        assert isinstance(planner, Planner)


class TestDeterministicPlanner:
    """Plan generation from reasoning."""

    @pytest.fixture
    def planner(self) -> DeterministicPlanner:
        return DeterministicPlanner()

    @pytest.fixture
    def context(self) -> Context:
        return Context()

    @pytest.mark.asyncio
    async def test_create_plan_with_goal(
        self,
        planner: DeterministicPlanner,
        context: Context,
    ):
        goal = Goal(title="Test Goal")
        reasoning = ReasoningResult(
            reasoning_summary="Need to respond to user",
            confidence=1.0,
            recommended_action="respond",
            requires_action=False,
        )

        plan = await planner.create_plan(reasoning, goal, context)

        assert isinstance(plan, Plan)
        assert plan.goal_id == goal.id
        assert plan.status == PlanStatus.ACTIVE
        assert len(plan.steps) == 1
        assert plan.steps[0].index == 0
        assert plan.steps[0].description == "respond"

    @pytest.mark.asyncio
    async def test_create_plan_without_goal(
        self,
        planner: DeterministicPlanner,
        context: Context,
    ):
        reasoning = ReasoningResult(
            reasoning_summary="Handle greeting",
            confidence=0.8,
            recommended_action="respond",
            requires_action=False,
        )

        plan = await planner.create_plan(reasoning, None, context)

        assert isinstance(plan, Plan)
        assert isinstance(plan.goal_id, UUID)
        assert plan.status == PlanStatus.ACTIVE
        assert len(plan.steps) == 1
        assert plan.steps[0].index == 0
        assert plan.steps[0].description == "respond"
