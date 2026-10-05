"""Tests for EVA safety foundation."""

import pytest

from eva.core.enums import RiskLevel
from eva.core.models import Action
from eva.safety import Permission, SafetyPolicy, SafetyVerdict


class TestLowRiskActions:
    """Low-risk actions are allowed."""

    def test_low_risk_passes(self):
        policy = SafetyPolicy()
        action = Action(name="read_context", risk_level=RiskLevel.LOW)
        verdict = policy.evaluate(action)
        assert verdict.allowed is True
        assert verdict.risk_level == RiskLevel.LOW

    def test_medium_risk_passes(self):
        policy = SafetyPolicy()
        action = Action(name="update_note", risk_level=RiskLevel.MEDIUM)
        verdict = policy.evaluate(action)
        assert verdict.allowed is True
        assert verdict.risk_level == RiskLevel.MEDIUM


class TestUnsafeActions:
    """Unsafe or unapproved actions are rejected."""

    def test_high_risk_denied(self):
        policy = SafetyPolicy()
        action = Action(name="risky_op", risk_level=RiskLevel.HIGH)
        verdict = policy.evaluate(action)
        assert verdict.allowed is False

    def test_critical_risk_denied(self):
        policy = SafetyPolicy()
        action = Action(name="nuclear_op", risk_level=RiskLevel.CRITICAL)
        verdict = policy.evaluate(action)
        assert verdict.allowed is False

    def test_blocked_action_denied(self):
        policy = SafetyPolicy()
        action = Action(name="shell_execute", risk_level=RiskLevel.LOW)
        verdict = policy.evaluate(action)
        assert verdict.allowed is False

    def test_file_delete_blocked(self):
        policy = SafetyPolicy()
        action = Action(name="file_delete", risk_level=RiskLevel.LOW)
        verdict = policy.evaluate(action)
        assert verdict.allowed is False

    def test_system_shutdown_blocked(self):
        policy = SafetyPolicy()
        action = Action(name="system_shutdown", risk_level=RiskLevel.LOW)
        verdict = policy.evaluate(action)
        assert verdict.allowed is False

    def test_browser_navigate_blocked(self):
        policy = SafetyPolicy()
        action = Action(name="browser_navigate", risk_level=RiskLevel.LOW)
        verdict = policy.evaluate(action)
        assert verdict.allowed is False

    def test_device_control_blocked(self):
        policy = SafetyPolicy()
        action = Action(name="device_control", risk_level=RiskLevel.LOW)
        verdict = policy.evaluate(action)
        assert verdict.allowed is False


class TestSafetyVerdict:
    """SafetyVerdict model tests."""

    def test_verdict_structure(self):
        verdict = SafetyVerdict(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Safe",
        )
        assert verdict.allowed is True
        assert verdict.reason == "Safe"
        assert verdict.required_permissions == []

    def test_verdict_with_permissions(self):
        verdict = SafetyVerdict(
            allowed=False,
            risk_level=RiskLevel.HIGH,
            reason="Needs approval",
            required_permissions=[Permission.EXECUTE, Permission.SYSTEM],
        )
        assert len(verdict.required_permissions) == 2
