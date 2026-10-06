"""Integration tests for EVA CognitiveEngine pipeline."""

from typing import Any
import pytest

from eva.core.brain import CognitiveEngine, CognitiveResult
from eva.core.brain.executor import ActionExecutor, SafeActionExecutor
from eva.core.brain.models import (
    ReasoningResult,
    UnderstandingResult,
    VerificationResult,
)
from eva.core.brain.response import DeterministicResponseGenerator
from eva.core.brain.understanding import UnderstandingEngine
from eva.core.brain.verifier import Verifier
from eva.core.context import Context, ContextManager
from eva.core.enums import ActionStatus, GoalStatus
from eva.core.models import Action, Goal, Observation, Plan
from eva.core.planner import Planner
from eva.core.reasoning import ReasoningEngine
from eva.safety import SafetyPolicy


class SpyActionExecutor:
    """Action executor test spy to verify execution call tracking."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_action: Action | None = None

    async def execute(self, action: Action, context: Context) -> Observation:
        self.call_count += 1
        self.last_action = action
        return Observation(
            source="action_executor",
            data={"action": action.name, "success": True, "response_generated": True},
            confidence=1.0,
        )


class FailingActionExecutor:
    """Action executor that simulates execution failure."""

    async def execute(self, action: Action, context: Context) -> Observation:
        return Observation(
            source="action_executor",
            data={
                "action": action.name,
                "success": False,
                "error": "I/O execution failure",
            },
            confidence=1.0,
        )


class FailingVerifier:
    """Verifier that rejects actions even after successful execution."""

    async def verify(
        self,
        action: Action,
        observation: Observation,
        context: Context,
    ) -> VerificationResult:
        return VerificationResult(
            success=False,
            confidence=1.0,
            message="Post-execution constraint check failed.",
            error_type="verification_failure",
        )


class CrashingPlanner:
    """Planner that raises an unexpected exception."""

    async def create_plan(
        self,
        reasoning: ReasoningResult,
        goal: Goal | None,
        context: Context,
    ) -> Plan:
        raise RuntimeError("Unexpected planner crash")


class GoalRequiringUnderstanding:
    """Understanding engine that triggers goal creation."""

    async def understand(
        self,
        user_input: str,
        context: Context,
    ) -> UnderstandingResult:
        return UnderstandingResult(
            intent="complex_task",
            confidence=1.0,
            original_input=user_input,
            requires_goal=True,
        )


class BlockedActionReasoning:
    """Reasoning engine that recommends an action prohibited by safety."""

    async def reason(
        self,
        understanding: UnderstandingResult,
        context: Context,
        goal: Goal | None,
    ) -> ReasoningResult:
        return ReasoningResult(
            reasoning_summary="Attempt dangerous command",
            recommended_action="shell_execute",
            confidence=1.0,
            requires_action=True,
        )


class TestCognitiveEngineIntegration:
    """Full-pipeline integration tests for CognitiveEngine."""

    @pytest.fixture
    def context_manager(self) -> ContextManager:
        cm = ContextManager()
        cm.start_conversation()
        return cm

    @pytest.fixture
    def safety(self) -> SafetyPolicy:
        return SafetyPolicy()

    @pytest.fixture
    def standard_engine(
        self, context_manager: ContextManager, safety: SafetyPolicy
    ) -> CognitiveEngine:
        return CognitiveEngine(
            context_manager=context_manager,
            safety=safety,
        )

    @pytest.mark.asyncio
    async def test_greeting_completes_successfully(
        self,
        standard_engine: CognitiveEngine,
        context_manager: ContextManager,
    ):
        result = await standard_engine.process("hello", context_manager.current)

        assert isinstance(result, CognitiveResult)
        assert result.success is True
        assert result.error is None
        assert result.understanding.intent == "greeting"
        assert result.goal is None
        assert result.decision.selected_action.name == "respond"
        assert result.decision.selected_action.status == ActionStatus.APPROVED
        assert result.observation.data.get("success") is True
        assert result.verification.success is True
        assert "EVA" in result.response
        assert len(context_manager.current.recent_observations) == 1

    @pytest.mark.asyncio
    async def test_identity_query_completes_successfully(
        self,
        standard_engine: CognitiveEngine,
        context_manager: ContextManager,
    ):
        result = await standard_engine.process("who are you", context_manager.current)

        assert result.success is True
        assert result.understanding.intent == "identity_query"
        assert "Evolving Virtual Autonomy" in result.response

    @pytest.mark.asyncio
    async def test_capability_query_completes_successfully(
        self,
        standard_engine: CognitiveEngine,
        context_manager: ContextManager,
    ):
        result = await standard_engine.process("what can you do", context_manager.current)

        assert result.success is True
        assert result.understanding.intent == "capability_query"
        assert "cognitive pipeline" in result.response
        assert "do not yet have access to" in result.response

    @pytest.mark.asyncio
    async def test_status_query_completes_successfully(
        self,
        standard_engine: CognitiveEngine,
        context_manager: ContextManager,
    ):
        result = await standard_engine.process("status", context_manager.current)

        assert result.success is True
        assert result.understanding.intent == "status_query"
        assert "operating normally" in result.response

    @pytest.mark.asyncio
    async def test_general_input_completes_successfully(
        self,
        standard_engine: CognitiveEngine,
        context_manager: ContextManager,
    ):
        result = await standard_engine.process(
            "the sky appears blue today", context_manager.current
        )

        assert result.success is True
        assert result.understanding.intent == "general"
        assert result.understanding.confidence == 0.5
        assert "deterministic" in result.response.lower()

    @pytest.mark.asyncio
    async def test_goal_creation_and_lifecycle(
        self,
        context_manager: ContextManager,
        safety: SafetyPolicy,
    ):
        engine = CognitiveEngine(
            context_manager=context_manager,
            safety=safety,
            understanding=GoalRequiringUnderstanding(),
        )

        result = await engine.process("organize my workspace", context_manager.current)

        assert result.success is True
        assert result.goal is not None
        assert result.goal.title == "Handle: complex_task"
        # Completed goal lifecycle:
        assert result.goal.status == GoalStatus.COMPLETED
        # Context active_goal should be cleared after completion:
        assert context_manager.current.active_goal is None
        # Stored goal in engine's goal_manager:
        stored = engine.goal_manager.get_goal(result.goal.id)
        assert stored is not None
        assert stored.status == GoalStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_safety_blocked_action_does_not_reach_executor(
        self,
        context_manager: ContextManager,
        safety: SafetyPolicy,
    ):
        spy_executor = SpyActionExecutor()
        engine = CognitiveEngine(
            context_manager=context_manager,
            safety=safety,
            reasoning=BlockedActionReasoning(),
            executor=spy_executor,
        )

        result = await engine.process("run rm -rf /", context_manager.current)

        # 1. Action was rejected
        assert result.decision.selected_action.status == ActionStatus.REJECTED
        # 2. Blocked action NEVER reached the executor
        assert spy_executor.call_count == 0
        assert spy_executor.last_action is None
        # 3. Observation came from safety_policy
        assert result.observation.source == "safety_policy"
        assert result.observation.data.get("success") is False
        # 4. Result is failed with safety-blocked response
        assert result.success is False
        assert result.verification.success is False
        assert result.response == DeterministicResponseGenerator.SAFETY_BLOCKED_RESPONSE
        assert "blocked by my safety policy" in result.response

    @pytest.mark.asyncio
    async def test_executor_failure_produces_failed_cognitive_result(
        self,
        context_manager: ContextManager,
        safety: SafetyPolicy,
    ):
        engine = CognitiveEngine(
            context_manager=context_manager,
            safety=safety,
            executor=FailingActionExecutor(),
        )

        result = await engine.process("hello", context_manager.current)

        assert result.success is False
        assert result.observation.data.get("success") is False
        assert result.verification.success is False
        assert result.error is not None
        # Response must NOT claim safety block
        assert result.response == DeterministicResponseGenerator.EXECUTION_FAILURE_RESPONSE
        assert "blocked by my safety policy" not in result.response

    @pytest.mark.asyncio
    async def test_verification_failure_produces_failed_cognitive_result(
        self,
        context_manager: ContextManager,
        safety: SafetyPolicy,
    ):
        engine = CognitiveEngine(
            context_manager=context_manager,
            safety=safety,
            verifier=FailingVerifier(),
        )

        result = await engine.process("hello", context_manager.current)

        assert result.success is False
        # Executor succeeded
        assert result.observation.data.get("success") is True
        # But verification failed
        assert result.verification.success is False
        assert result.error is not None
        # Response must NOT claim safety block
        assert result.response == DeterministicResponseGenerator.VERIFICATION_FAILURE_RESPONSE
        assert "blocked by my safety policy" not in result.response

    @pytest.mark.asyncio
    async def test_unexpected_exception_produces_structured_failure_result(
        self,
        context_manager: ContextManager,
        safety: SafetyPolicy,
    ):
        engine = CognitiveEngine(
            context_manager=context_manager,
            safety=safety,
            planner=CrashingPlanner(),
        )

        result = await engine.process("hello", context_manager.current)

        assert isinstance(result, CognitiveResult)
        assert result.success is False
        assert result.error == "Unexpected planner crash"
        assert result.understanding.intent == "error"
        assert result.action.status == ActionStatus.FAILED
        assert result.verification.success is False
        assert "encountered an error" in result.response
