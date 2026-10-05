"""EVA self-model.

An operational model of EVA's identity, capabilities, limitations,
and current status. This is NOT a claim of sentience — it is a
structured representation that allows EVA to reason about what
it can and cannot do.
"""

from __future__ import annotations

from eva.core.enums import EVAStatus
from eva.core.models import EVAState
from eva.logging import get_logger

logger = get_logger(__name__)


class EVASelfModel:
    """EVA's operational self-representation.

    Provides structured access to identity, capabilities,
    limitations, and runtime status.
    """

    _IDENTITY = (
        "EVA — Evolving Virtual Autonomy. "
        "An autonomous personal AI system designed to assist, reason, "
        "plan, and act on behalf of its user."
    )

    _CAPABILITIES: list[str] = [
        "Natural language interaction",
        "Context tracking",
        "Goal and plan management",
        "Safety-aware action evaluation",
        "Multi-provider model abstraction",
    ]

    _LIMITATIONS: list[str] = [
        "No persistent memory (Milestone 1A)",
        "No external tool execution (Milestone 1A)",
        "No autonomous agent execution (Milestone 1A)",
        "No voice or vision processing (Milestone 1A)",
        "No browser or device control (Milestone 1A)",
        "Requires explicit user interaction for decisions",
    ]

    def __init__(self, state: EVAState | None = None) -> None:
        self._state = state or EVAState()
        logger.debug("SelfModel initialized")

    @property
    def state(self) -> EVAState:
        """Return the current EVA state."""
        return self._state

    @state.setter
    def state(self, new_state: EVAState) -> None:
        self._state = new_state

    def identity(self) -> str:
        """Return EVA's identity description."""
        return self._IDENTITY

    def capabilities(self) -> list[str]:
        """Return the list of current capabilities."""
        return list(self._CAPABILITIES)

    def limitations(self) -> list[str]:
        """Return the list of current limitations."""
        return list(self._LIMITATIONS)

    def status(self) -> EVAStatus:
        """Return the current operational status."""
        return self._state.status

    def update_status(self, status: EVAStatus) -> None:
        """Update the operational status."""
        previous = self._state.status
        self._state = self._state.model_copy(update={"status": status})
        logger.info("EVA status: %s -> %s", previous.value, status.value)
