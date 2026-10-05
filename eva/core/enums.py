"""Shared enumerations used across EVA domain models."""

from __future__ import annotations

from enum import Enum


class EVAStatus(str, Enum):
    """Runtime status of the EVA system."""

    OFFLINE = "offline"
    INITIALIZING = "initializing"
    ONLINE = "online"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


class GoalStatus(str, Enum):
    """Lifecycle status of a Goal."""

    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlanStatus(str, Enum):
    """Lifecycle status of a Plan."""

    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActionStatus(str, Enum):
    """Lifecycle status of an Action."""

    PENDING = "pending"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class RiskLevel(str, Enum):
    """Risk classification for actions and operations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Priority(int, Enum):
    """Priority levels (lower number = higher priority)."""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class FocusState(str, Enum):
    """User focus/attention state."""

    ACTIVE = "active"
    IDLE = "idle"
    AWAY = "away"
    DO_NOT_DISTURB = "do_not_disturb"
    UNKNOWN = "unknown"


class EVAMode(str, Enum):
    """Operating mode of EVA."""

    INTERACTIVE = "interactive"
    AUTONOMOUS = "autonomous"
    MONITORING = "monitoring"
    STANDBY = "standby"
