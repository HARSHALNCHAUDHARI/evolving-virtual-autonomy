"""Tests for EVA context system."""

import pytest

from eva.core.context import Context, ContextManager, ConversationState
from eva.core.enums import FocusState, GoalStatus, Priority
from eva.core.models import Event, Goal, Observation, UserContext


class TestContextCreation:
    """Context can be created with defaults."""

    def test_default_context(self):
        ctx = Context()
        assert isinstance(ctx.user, UserContext)
        assert isinstance(ctx.conversation, ConversationState)
        assert ctx.active_goal is None
        assert ctx.recent_events == []
        assert ctx.recent_observations == []

    def test_context_manager_creation(self):
        cm = ContextManager()
        assert isinstance(cm.current, Context)


class TestContextUpdate:
    """Context can be updated."""

    def test_update_user(self):
        cm = ContextManager()
        cm.update_user(display_name="Test User", focus_state=FocusState.ACTIVE)
        assert cm.current.user.display_name == "Test User"
        assert cm.current.user.focus_state == FocusState.ACTIVE

    def test_set_active_goal(self):
        cm = ContextManager()
        goal = Goal(title="Test Goal")
        cm.set_active_goal(goal)
        assert cm.current.active_goal is not None
        assert cm.current.active_goal.title == "Test Goal"

    def test_clear_active_goal(self):
        cm = ContextManager()
        cm.set_active_goal(Goal(title="Goal"))
        cm.clear_active_goal()
        assert cm.current.active_goal is None

    def test_add_event(self):
        cm = ContextManager()
        event = Event(event_type="test", source="unit_test")
        cm.add_event(event)
        assert len(cm.current.recent_events) == 1
        assert cm.current.recent_events[0].event_type == "test"

    def test_add_observation(self):
        cm = ContextManager()
        obs = Observation(source="test_sensor")
        cm.add_observation(obs)
        assert len(cm.current.recent_observations) == 1

    def test_start_conversation(self):
        cm = ContextManager()
        cm.start_conversation()
        assert cm.current.conversation.active is True
        assert cm.current.conversation.turn_count == 0
        assert cm.current.conversation.started_at is not None

    def test_increment_turn(self):
        cm = ContextManager()
        cm.start_conversation()
        cm.increment_turn()
        cm.increment_turn()
        assert cm.current.conversation.turn_count == 2

    def test_update_environment(self):
        cm = ContextManager()
        cm.update_environment(platform="Darwin", hostname="test-host")
        assert cm.current.environment.platform == "Darwin"
        assert cm.current.environment.hostname == "test-host"


class TestContextReset:
    """Context can be reset."""

    def test_reset_clears_all(self):
        cm = ContextManager()
        cm.update_user(display_name="Someone")
        cm.set_active_goal(Goal(title="Goal"))
        cm.add_event(Event(event_type="e", source="s"))
        cm.start_conversation()

        cm.reset()

        assert cm.current.user.display_name == "User"
        assert cm.current.active_goal is None
        assert cm.current.recent_events == []
        assert cm.current.conversation.active is False


class TestContextBounds:
    """Recent items lists are bounded."""

    def test_events_bounded(self):
        cm = ContextManager()
        for i in range(60):
            cm.add_event(Event(event_type=f"e{i}", source="test"))
        assert len(cm.current.recent_events) == 50

    def test_observations_bounded(self):
        cm = ContextManager()
        for i in range(60):
            cm.add_observation(Observation(source=f"s{i}"))
        assert len(cm.current.recent_observations) == 50
