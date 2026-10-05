"""EVA safety foundation.

Provides risk evaluation and permission management to ensure
EVA operates within safe boundaries. All potentially consequential
actions must be evaluated before execution.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from eva.core.enums import RiskLevel
from eva.core.models import Action
from eva.logging import get_logger

logger = get_logger(__name__)


class Permission(str, Enum):
    """Permissions that can be granted or denied."""

    EXECUTE = "execute"
    READ = "read"
    WRITE = "write"
    NETWORK = "network"
    SYSTEM = "system"


class SafetyVerdict(BaseModel):
    """Result of a safety evaluation."""

    allowed: bool
    risk_level: RiskLevel
    reason: str
    required_permissions: list[Permission] = Field(default_factory=list)


class SafetyPolicy:
    """Evaluates actions against safety boundaries.

    Milestone 1A policy is conservative:
    - LOW risk actions are allowed.
    - MEDIUM risk actions are allowed with logging.
    - HIGH risk actions are denied by default.
    - CRITICAL risk actions are always denied.
    """

    # Actions that are never allowed in Milestone 1A
    _BLOCKED_ACTIONS: frozenset[str] = frozenset(
        {
            "shell_execute",
            "file_delete",
            "system_shutdown",
            "system_reboot",
            "browser_navigate",
            "device_control",
            "send_email",
            "network_request",
        }
    )

    def evaluate(self, action: Action) -> SafetyVerdict:
        """Evaluate whether an action is safe to execute.

        Args:
            action: The action to evaluate.

        Returns:
            A SafetyVerdict indicating whether the action is allowed.
        """
        # Check blocked actions first
        if action.name in self._BLOCKED_ACTIONS:
            logger.warning(
                "Action BLOCKED by safety policy: %s (explicitly prohibited)",
                action.name,
            )
            return SafetyVerdict(
                allowed=False,
                risk_level=action.risk_level,
                reason=f"Action '{action.name}' is explicitly blocked in Milestone 1A.",
            )

        # Evaluate by risk level
        if action.risk_level == RiskLevel.CRITICAL:
            logger.warning("Action DENIED: %s (CRITICAL risk)", action.name)
            return SafetyVerdict(
                allowed=False,
                risk_level=RiskLevel.CRITICAL,
                reason="CRITICAL risk actions are not permitted.",
            )

        if action.risk_level == RiskLevel.HIGH:
            logger.warning("Action DENIED: %s (HIGH risk)", action.name)
            return SafetyVerdict(
                allowed=False,
                risk_level=RiskLevel.HIGH,
                reason="HIGH risk actions require explicit approval (not implemented in 1A).",
            )

        if action.risk_level == RiskLevel.MEDIUM:
            logger.info("Action ALLOWED with caution: %s (MEDIUM risk)", action.name)
            return SafetyVerdict(
                allowed=True,
                risk_level=RiskLevel.MEDIUM,
                reason="MEDIUM risk — allowed with logging.",
            )

        # LOW risk
        logger.debug("Action ALLOWED: %s (LOW risk)", action.name)
        return SafetyVerdict(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="LOW risk — allowed.",
        )
