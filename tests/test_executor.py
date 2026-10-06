"""Tests for EVA action executor."""

import pytest

from eva.core.brain.executor import ActionExecutor, SafeActionExecutor
from eva.core.context import Context
from eva.core.enums import RiskLevel
from eva.core.models import Action, Observation


class TestActionExecutorProtocol:
    """SafeActionExecutor conforms to ActionExecutor protocol."""

    def test_protocol_conformance(self):
        executor = SafeActionExecutor()
        assert isinstance(executor, ActionExecutor)


class TestSafeActionExecutor:
    """Execution of permitted and refused actions."""

    @pytest.fixture
    def executor(self) -> SafeActionExecutor:
        return SafeActionExecutor()

    @pytest.fixture
    def context(self) -> Context:
        return Context()

    @pytest.mark.asyncio
    async def test_execute_respond(
        self,
        executor: SafeActionExecutor,
        context: Context,
    ):
        action = Action(name="respond", description="Respond to user")
        obs = await executor.execute(action, context)

        assert isinstance(obs, Observation)
        assert obs.source == "action_executor"
        assert obs.data.get("action") == "respond"
        assert obs.data.get("success") is True
        assert obs.data.get("response_generated") is True

    @pytest.mark.asyncio
    async def test_execute_no_op(
        self,
        executor: SafeActionExecutor,
        context: Context,
    ):
        action = Action(name="no_op", description="Do nothing")
        obs = await executor.execute(action, context)

        assert isinstance(obs, Observation)
        assert obs.source == "action_executor"
        assert obs.data.get("action") == "no_op"
        assert obs.data.get("success") is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "unsupported_name",
        ["shell_execute", "open_browser", "delete_file", "custom_tool"],
    )
    async def test_unsupported_actions_refused(
        self,
        executor: SafeActionExecutor,
        context: Context,
        unsupported_name: str,
    ):
        action = Action(
            name=unsupported_name,
            description="Attempt unsupported action",
            risk_level=RiskLevel.MEDIUM,
        )
        obs = await executor.execute(action, context)

        assert isinstance(obs, Observation)
        assert obs.source == "action_executor"
        assert obs.data.get("success") is False
        assert "not supported in Milestone 1B" in obs.data.get("error", "")
