"""EVA core domain models.

These models represent the fundamental domain concepts of the EVA system.
They are provider-agnostic and must not depend on any external SDK.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> UUID:
    return uuid4()


# ---------------------------------------------------------------------------
# User context
# ---------------------------------------------------------------------------


class UserContext(BaseModel):
    """Representation of the current user and their state."""

    user_id: str = "default"
    display_name: str = "User"
    timezone: str = "UTC"
    current_activity: str | None = None
    current_application: str | None = None
    focus_state: FocusState = FocusState.UNKNOWN
    preferences: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# EVA state
# ---------------------------------------------------------------------------


class EVAState(BaseModel):
    """Runtime state snapshot of the EVA system."""

    status: EVAStatus = EVAStatus.OFFLINE
    current_goal_id: UUID | None = None
    current_task: str | None = None
    active_context: dict[str, Any] = Field(default_factory=dict)
    last_event: str | None = None
    current_mode: EVAMode = EVAMode.INTERACTIVE
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------


class Goal(BaseModel):
    """An objective EVA may work toward."""

    id: UUID = Field(default_factory=_new_id)
    title: str
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING
    priority: Priority = Priority.MEDIUM
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    parent_goal_id: UUID | None = None


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """A single step within a Plan."""

    index: int
    description: str
    status: ActionStatus = ActionStatus.PENDING


class Plan(BaseModel):
    """A sequence of intended steps toward a Goal."""

    id: UUID = Field(default_factory=_new_id)
    goal_id: UUID
    steps: list[PlanStep] = Field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class Action(BaseModel):
    """An action EVA intends to execute."""

    id: UUID = Field(default_factory=_new_id)
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    status: ActionStatus = ActionStatus.PENDING


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """The result or state observed after an action or from a sensor."""

    id: UUID = Field(default_factory=_new_id)
    source: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


class Event(BaseModel):
    """An event entering the EVA system."""

    id: UUID = Field(default_factory=_new_id)
    event_type: str
    source: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)
    priority: Priority = Priority.MEDIUM
