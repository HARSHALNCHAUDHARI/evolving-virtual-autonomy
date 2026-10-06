"""Cognitive loop data models.

Structured results for each stage of the EVA cognitive pipeline.
These models are provider-agnostic and extend the existing domain models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from eva.core.enums import RiskLevel
from eva.core.models import Action, Goal, Observation, Plan


# ---------------------------------------------------------------------------
# Understanding
# ---------------------------------------------------------------------------


class UnderstandingResult(BaseModel):
    """Structured interpretation of raw user input."""

    intent: str
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    original_input: str
    requires_goal: bool = False


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


class ReasoningResult(BaseModel):
    """Outcome of the reasoning stage."""

    reasoning_summary: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    recommended_action: str = "respond"
    requires_action: bool = False


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


class DecisionResult(BaseModel):
    """Outcome of the decision stage."""

    decision: str
    selected_action: Action
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    requires_confirmation: bool = False


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class VerificationResult(BaseModel):
    """Outcome of verifying an action's observation."""

    success: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    message: str = ""
    error_type: str | None = None


# ---------------------------------------------------------------------------
# Cognitive result (complete pipeline output)
# ---------------------------------------------------------------------------


class CognitiveResult(BaseModel):
    """Complete output of one cognitive cycle.

    Captures every stage so that future systems can inspect,
    log, or replay the full reasoning trace.
    """

    input: str
    understanding: UnderstandingResult
    goal: Goal | None = None
    reasoning_summary: str
    plan: Plan | None = None
    decision: DecisionResult
    action: Action
    observation: Observation
    verification: VerificationResult
    response: str
    success: bool = True
    error: str | None = None
