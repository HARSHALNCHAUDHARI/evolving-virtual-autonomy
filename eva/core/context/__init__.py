"""EVA context management.

The context system provides a structured representation of what EVA
currently knows about the active interaction: the user, conversation state,
environment, active goals, and recent events.

This is an in-memory implementation for Milestone 1A.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from eva.core.enums import GoalStatus
from eva.core.models import Event, Goal, Observation, UserContext
from eva.logging import get_logger

logger = get_logger(__name__)


class ConversationState(BaseModel):
    """Tracks the current conversation."""

    active: bool = False
    turn_count: int = 0
    started_at: datetime | None = None


class EnvironmentState(BaseModel):
    """Representation of the operating environment."""

    platform: str | None = None
    hostname: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Context(BaseModel):
    """The full context snapshot available to EVA at any point in time."""

    user: UserContext = Field(default_factory=UserContext)
    conversation: ConversationState = Field(default_factory=ConversationState)
    environment: EnvironmentState = Field(default_factory=EnvironmentState)
    active_goal: Goal | None = None
    recent_events: list[Event] = Field(default_factory=list)
    recent_observations: list[Observation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextManager:
    """Manages the in-memory EVA context.

    Provides operations to create, update, retrieve, and reset
    the current interaction context.
    """

    _MAX_RECENT_ITEMS = 50

    def __init__(self) -> None:
        self._context = Context()
        logger.debug("ContextManager initialized")

    @property
    def current(self) -> Context:
        """Return the current context snapshot."""
        return self._context

    def update_user(self, **fields: Any) -> None:
        """Update user context fields."""
        updated = self._context.user.model_copy(update=fields)
        self._context.user = updated
        logger.debug("User context updated: %s", list(fields.keys()))

    def set_active_goal(self, goal: Goal) -> None:
        """Set the currently active goal."""
        self._context.active_goal = goal
        logger.debug("Active goal set: %s", goal.title)

    def clear_active_goal(self) -> None:
        """Clear the active goal."""
        self._context.active_goal = None
        logger.debug("Active goal cleared")

    def add_event(self, event: Event) -> None:
        """Record an event in the context window."""
        self._context.recent_events.append(event)
        if len(self._context.recent_events) > self._MAX_RECENT_ITEMS:
            self._context.recent_events = self._context.recent_events[-self._MAX_RECENT_ITEMS :]
        logger.debug("Event added: %s", event.event_type)

    def add_observation(self, observation: Observation) -> None:
        """Record an observation in the context window."""
        self._context.recent_observations.append(observation)
        if len(self._context.recent_observations) > self._MAX_RECENT_ITEMS:
            self._context.recent_observations = self._context.recent_observations[
                -self._MAX_RECENT_ITEMS :
            ]
        logger.debug("Observation added: %s", observation.source)

    def start_conversation(self) -> None:
        """Mark the beginning of a conversation."""
        self._context.conversation = ConversationState(
            active=True,
            turn_count=0,
            started_at=datetime.now(timezone.utc),
        )
        logger.debug("Conversation started")

    def increment_turn(self) -> None:
        """Increment the conversation turn counter."""
        self._context.conversation.turn_count += 1

    def update_environment(self, **fields: Any) -> None:
        """Update environment state fields."""
        updated = self._context.environment.model_copy(update=fields)
        self._context.environment = updated

    def reset(self) -> None:
        """Reset all context to defaults."""
        self._context = Context()
        logger.info("Context reset")
