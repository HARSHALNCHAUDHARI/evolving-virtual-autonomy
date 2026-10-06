"""Goal manager — lifecycle management for EVA goals.

Handles creation, retrieval, status updates, and completion of goals.
In-memory storage for Milestone 1B.
"""

from __future__ import annotations

from uuid import UUID

from eva.core.brain.models import UnderstandingResult
from eva.core.context import Context
from eva.core.enums import GoalStatus, Priority
from eva.core.models import Goal
from eva.logging import get_logger

logger = get_logger(__name__)


class GoalManager:
    """Manages EVA goals in memory.

    Determines whether interactions require goals, creates them,
    and tracks their lifecycle.
    """

    def __init__(self) -> None:
        self._goals: dict[UUID, Goal] = {}

    def evaluate_need(self, understanding: UnderstandingResult) -> bool:
        """Determine whether the current interaction requires a goal.

        Simple conversational intents (greetings, queries) do not.
        Complex multi-step requests will in future milestones.
        """
        return understanding.requires_goal

    def create_goal(self, title: str, description: str = "") -> Goal:
        """Create and store a new goal."""
        goal = Goal(
            title=title,
            description=description,
            status=GoalStatus.ACTIVE,
        )
        self._goals[goal.id] = goal
        logger.info("Goal created: %s (%s)", title, goal.id)
        return goal

    def get_goal(self, goal_id: UUID) -> Goal | None:
        """Retrieve a goal by ID."""
        return self._goals.get(goal_id)

    def get_active_goals(self) -> list[Goal]:
        """Return all goals with ACTIVE status."""
        return [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]

    def complete_goal(self, goal_id: UUID) -> Goal | None:
        """Mark a goal as completed."""
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        updated = goal.model_copy(update={"status": GoalStatus.COMPLETED})
        self._goals[goal_id] = updated
        logger.info("Goal completed: %s", updated.title)
        return updated

    def fail_goal(self, goal_id: UUID) -> Goal | None:
        """Mark a goal as failed."""
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        updated = goal.model_copy(update={"status": GoalStatus.FAILED})
        self._goals[goal_id] = updated
        logger.info("Goal failed: %s", updated.title)
        return updated

    def cancel_goal(self, goal_id: UUID) -> Goal | None:
        """Mark a goal as cancelled."""
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        updated = goal.model_copy(update={"status": GoalStatus.CANCELLED})
        self._goals[goal_id] = updated
        logger.info("Goal cancelled: %s", updated.title)
        return updated

    @property
    def all_goals(self) -> list[Goal]:
        """Return all tracked goals."""
        return list(self._goals.values())
