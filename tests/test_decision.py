"""Tests for EVA decision engine and safety integration."""

from uuid import uuid4

import pytest

from eva.core.brain.models import DecisionResult, ReasoningResult
from eva.core.context import Context
from eva.core.decision import DecisionEngine
from eva.core.enums import ActionStatus, RiskLevel
from eva.core.models import Plan, PlanStep
from eva.safety import SafetyPolicy, SafetyVerdict


class TestDecisionEngine:
    """Decision making and safety policy verification."""

    @pytest.fixture
    def safety(self) -> SafetyPolicy:
        return SafetyPolicy()

    @pytest.fixture
    def decision_engine(self, safety: SafetyPolicy) -> DecisionEngine:
        return DecisionEngine(safety)

    @pytest.fixture
    def context(self) -> Context:
        return Context()

    @pytest.fixture
    def sample_plan(self) -> Plan:
        return Plan(
            goal_id=uuid4(),
            steps=[PlanStep(index=0, description="sample step")],
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("safe_action", ["respond", "no_op"])
    async def test_safe_action_approved(
        self,
        decision_engine: DecisionEngine,
        context: Context,
        sample_plan: Plan,
        safe_action: str,
    ):
        reasoning = ReasoningResult(
            reasoning_summary=f"Recommend {safe_action}",
            confidence=0.95,
            recommended_action=safe_action,
        )

        decision, verdict = await decision_engine.decide(reasoning, sample_plan, context)

        assert isinstance(decision, DecisionResult)
        assert isinstance(verdict, SafetyVerdict)
        assert verdict.allowed is True
        assert verdict.risk_level == RiskLevel.LOW
        assert decision.selected_action.status == ActionStatus.APPROVED
        assert decision.selected_action.risk_level == RiskLevel.LOW
        assert decision.selected_action.name == safe_action
        assert decision.confidence == 0.95

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "blocked_action",
        [
            "shell_execute",
            "file_delete",
            "system_shutdown",
            "system_reboot",
            "browser_navigate",
            "device_control",
            "send_email",
            "network_request",
        ],
    )
    async def test_blocked_actions_rejected(
        self,
        decision_engine: DecisionEngine,
        context: Context,
        sample_plan: Plan,
        blocked_action: str,
    ):
        reasoning = ReasoningResult(
            reasoning_summary=f"Attempt {blocked_action}",
            confidence=1.0,
            recommended_action=blocked_action,
        )

        decision, verdict = await decision_engine.decide(reasoning, sample_plan, context)

        assert verdict.allowed is False
        assert decision.selected_action.status == ActionStatus.REJECTED
        assert decision.selected_action.name == blocked_action

    @pytest.mark.asyncio
    async def test_custom_action_risk_assessment(
        self,
        decision_engine: DecisionEngine,
        context: Context,
        sample_plan: Plan,
    ):
        reasoning = ReasoningResult(
            reasoning_summary="Attempt unknown custom action",
            confidence=0.8,
            recommended_action="custom_computation",
        )

        decision, verdict = await decision_engine.decide(reasoning, sample_plan, context)

        assert decision.selected_action.risk_level == RiskLevel.MEDIUM
        assert verdict.allowed is True
        assert decision.selected_action.status == ActionStatus.APPROVED
