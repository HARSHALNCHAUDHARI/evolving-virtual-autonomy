"""Tests for EVA core domain models."""

from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from eva.core.enums import (
    ActionStatus,
    EVAMode,
    EVAStatus,
    FocusState,
    GoalStatus,
    PlanStatus,
    Priority,
    RiskLevel,
)
from eva.core.models import (
    Action,
    EVAState,
    Event,
    Goal,
    Observation,
    Plan,
    PlanStep,
    UserContext,
)


class TestUserContext:
    """UserContext model tests."""

    def test_defaults(self):
        ctx = UserContext()
        assert ctx.user_id == "default"
        assert ctx.display_name == "User"
        assert ctx.timezone == "UTC"
        assert ctx.focus_state == FocusState.UNKNOWN
        assert ctx.preferences == {}

    def test_custom_values(self):
        ctx = UserContext(
            user_id="u123",
            display_name="Harshal",
            timezone="Asia/Kolkata",
            current_activity="coding",
            focus_state=FocusState.ACTIVE,
        )
        assert ctx.user_id == "u123"
        assert ctx.display_name == "Harshal"
        assert ctx.current_activity == "coding"


class TestEVAState:
    """EVAState model tests."""

    def test_defaults(self):
        state = EVAState()
        assert state.status == EVAStatus.OFFLINE
        assert state.current_goal_id is None
        assert state.current_mode == EVAMode.INTERACTIVE
        assert isinstance(state.timestamp, datetime)

    def test_custom_status(self):
        state = EVAState(status=EVAStatus.ONLINE)
        assert state.status == EVAStatus.ONLINE


class TestGoal:
    """Goal model tests."""

    def test_creation_with_title(self):
        goal = Goal(title="Test Goal")
        assert isinstance(goal.id, UUID)
        assert goal.title == "Test Goal"
        assert goal.status == GoalStatus.PENDING
        assert goal.priority == Priority.MEDIUM

    def test_parent_goal(self):
        parent = Goal(title="Parent")
        child = Goal(title="Child", parent_goal_id=parent.id)
        assert child.parent_goal_id == parent.id

    def test_requires_title(self):
        with pytest.raises(ValidationError):
            Goal()  # type: ignore[call-arg]


class TestPlan:
    """Plan model tests."""

    def test_creation(self):
        goal = Goal(title="Goal")
        plan = Plan(
            goal_id=goal.id,
            steps=[
                PlanStep(index=0, description="Step 1"),
                PlanStep(index=1, description="Step 2"),
            ],
        )
        assert plan.goal_id == goal.id
        assert len(plan.steps) == 2
        assert plan.status == PlanStatus.DRAFT

    def test_requires_goal_id(self):
        with pytest.raises(ValidationError):
            Plan()  # type: ignore[call-arg]


class TestAction:
    """Action model tests."""

    def test_creation(self):
        action = Action(name="read_file")
        assert action.name == "read_file"
        assert action.risk_level == RiskLevel.LOW
        assert action.status == ActionStatus.PENDING

    def test_high_risk(self):
        action = Action(name="delete_all", risk_level=RiskLevel.CRITICAL)
        assert action.risk_level == RiskLevel.CRITICAL

    def test_requires_name(self):
        with pytest.raises(ValidationError):
            Action()  # type: ignore[call-arg]


class TestObservation:
    """Observation model tests."""

    def test_creation(self):
        obs = Observation(source="sensor_1", data={"temp": 22})
        assert obs.source == "sensor_1"
        assert obs.data["temp"] == 22
        assert obs.confidence == 1.0

    def test_confidence_bounds(self):
        obs = Observation(source="test", confidence=0.5)
        assert obs.confidence == 0.5

        with pytest.raises(ValidationError):
            Observation(source="test", confidence=1.5)

        with pytest.raises(ValidationError):
            Observation(source="test", confidence=-0.1)


class TestEvent:
    """Event model tests."""

    def test_creation(self):
        event = Event(event_type="user_input", source="cli")
        assert event.event_type == "user_input"
        assert event.source == "cli"
        assert event.priority == Priority.MEDIUM
        assert isinstance(event.timestamp, datetime)

    def test_with_payload(self):
        event = Event(
            event_type="notification",
            source="system",
            payload={"message": "hello"},
            priority=Priority.HIGH,
        )
        assert event.payload["message"] == "hello"
        assert event.priority == Priority.HIGH
