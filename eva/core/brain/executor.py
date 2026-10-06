"""Action executor — executes approved actions safely.

Provides a protocol and a safe internal executor for Milestone 1B.
Only RESPOND and NO_OP actions are implemented.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from eva.core.context import Context
from eva.core.enums import ActionStatus
from eva.core.models import Action, Observation
from eva.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class ActionExecutor(Protocol):
    """Interface for action execution.

    Future implementations will support tool calls,
    filesystem operations, browser control, etc.
    """

    async def execute(
        self,
        action: Action,
        context: Context,
    ) -> Observation:
        """Execute an action and return an observation."""
        ...


class SafeActionExecutor:
    """Executes only safe internal actions for Milestone 1B.

    Supported actions:
    - respond: signals that a text response should be generated
    - no_op: does nothing (used when no action is required)
    """

    _SAFE_ACTIONS = frozenset({"respond", "no_op"})

    async def execute(
        self,
        action: Action,
        context: Context,
    ) -> Observation:
        """Execute a safe action and return an observation."""
        logger.debug("Executing action: %s", action.name)

        if action.name not in self._SAFE_ACTIONS:
            logger.warning("Unsupported action refused: %s", action.name)
            return Observation(
                source="action_executor",
                data={
                    "action": action.name,
                    "success": False,
                    "error": f"Action '{action.name}' is not supported in Milestone 1B.",
                },
                confidence=1.0,
            )

        if action.name == "no_op":
            logger.info("Action executed: no_op")
            return Observation(
                source="action_executor",
                data={"action": "no_op", "success": True},
                confidence=1.0,
            )

        # respond
        logger.info("Action executed: respond")
        return Observation(
            source="action_executor",
            data={
                "action": "respond",
                "success": True,
                "response_generated": True,
            },
            confidence=1.0,
        )
