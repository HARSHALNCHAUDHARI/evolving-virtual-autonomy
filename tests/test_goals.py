"""Tests for EVA goal manager."""

from uuid import uuid4

import pytest

from eva.core.brain.models import UnderstandingResult
from eva.core.enums import GoalStatus
from eva.core.goals import GoalManager
from eva.core.models import Goal


class TestGoalManager:
    """In-memory goal manager lifecycle testing."""

    @pytest.fixture
    def manager(self) -> GoalManager:
        return GoalManager()

    def test_evaluate_need(self, manager: GoalManager):
        no_goal = UnderstandingResult(
            intent="greeting",
            confidence=1.0,
            original_input="hi",
            requires_goal=False,
        )
        assert manager.evaluate_need(no_goal) is False

        needs_goal = UnderstandingResult(
            intent="complex_task",
            confidence=1.0,
            original_input="organize my workspace",
            requires_goal=True,
        )
        assert manager.evaluate_need(needs_goal) is True

    def test_create_and_get_goal(self, manager: GoalManager):
        goal = manager.create_goal(
            title="Organize files",
            description="Sort files in downloads",
        )
        assert isinstance(goal, Goal)
        assert goal.title == "Organize files"
        assert goal.description == "Sort files in downloads"
        assert goal.status == GoalStatus.ACTIVE

        retrieved = manager.get_goal(goal.id)
        assert retrieved is not None
        assert retrieved.id == goal.id
        assert retrieved.title == "Organize files"

    def test_get_nonexistent_goal(self, manager: GoalManager):
        assert manager.get_goal(uuid4()) is None

    def test_get_active_goals(self, manager: GoalManager):
        g1 = manager.create_goal("Goal 1")
        g2 = manager.create_goal("Goal 2")

        active = manager.get_active_goals()
        assert len(active) == 2

        manager.complete_goal(g1.id)
        active_after = manager.get_active_goals()
        assert len(active_after) == 1
        assert active_after[0].id == g2.id

    def test_complete_goal(self, manager: GoalManager):
        goal = manager.create_goal("Task to complete")
        completed = manager.complete_goal(goal.id)

        assert completed is not None
        assert completed.status == GoalStatus.COMPLETED
        assert manager.get_goal(goal.id).status == GoalStatus.COMPLETED

    def test_complete_nonexistent_goal(self, manager: GoalManager):
        assert manager.complete_goal(uuid4()) is None

    def test_fail_goal(self, manager: GoalManager):
        goal = manager.create_goal("Task to fail")
        failed = manager.fail_goal(goal.id)

        assert failed is not None
        assert failed.status == GoalStatus.FAILED
        assert manager.get_goal(goal.id).status == GoalStatus.FAILED

    def test_fail_nonexistent_goal(self, manager: GoalManager):
        assert manager.fail_goal(uuid4()) is None

    def test_cancel_goal(self, manager: GoalManager):
        goal = manager.create_goal("Task to cancel")
        cancelled = manager.cancel_goal(goal.id)

        assert cancelled is not None
        assert cancelled.status == GoalStatus.CANCELLED
        assert manager.get_goal(goal.id).status == GoalStatus.CANCELLED

    def test_cancel_nonexistent_goal(self, manager: GoalManager):
        assert manager.cancel_goal(uuid4()) is None

    def test_all_goals(self, manager: GoalManager):
        assert len(manager.all_goals) == 0
        g1 = manager.create_goal("G1")
        g2 = manager.create_goal("G2")
        manager.fail_goal(g2.id)

        all_g = manager.all_goals
        assert len(all_g) == 2
        assert {g.id for g in all_g} == {g1.id, g2.id}
