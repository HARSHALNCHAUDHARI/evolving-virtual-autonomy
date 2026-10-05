"""Tests for EVA self-model."""

import pytest

from eva.core.enums import EVAStatus
from eva.core.models import EVAState
from eva.core.self_model import EVASelfModel


class TestIdentity:
    """Self-model identity is available."""

    def test_identity_string(self):
        sm = EVASelfModel()
        identity = sm.identity()
        assert isinstance(identity, str)
        assert "EVA" in identity
        assert len(identity) > 10


class TestCapabilities:
    """Self-model capabilities are available."""

    def test_capabilities_list(self):
        sm = EVASelfModel()
        caps = sm.capabilities()
        assert isinstance(caps, list)
        assert len(caps) > 0
        assert all(isinstance(c, str) for c in caps)

    def test_capabilities_returns_copy(self):
        sm = EVASelfModel()
        caps1 = sm.capabilities()
        caps2 = sm.capabilities()
        assert caps1 == caps2
        caps1.append("new_cap")
        assert "new_cap" not in sm.capabilities()


class TestLimitations:
    """Self-model limitations are available."""

    def test_limitations_list(self):
        sm = EVASelfModel()
        lims = sm.limitations()
        assert isinstance(lims, list)
        assert len(lims) > 0

    def test_limitations_returns_copy(self):
        sm = EVASelfModel()
        lims = sm.limitations()
        lims.append("fake")
        assert "fake" not in sm.limitations()


class TestStatus:
    """Self-model status tracking."""

    def test_default_status_is_offline(self):
        sm = EVASelfModel()
        assert sm.status() == EVAStatus.OFFLINE

    def test_update_status(self):
        sm = EVASelfModel()
        sm.update_status(EVAStatus.ONLINE)
        assert sm.status() == EVAStatus.ONLINE

    def test_status_from_provided_state(self):
        state = EVAState(status=EVAStatus.INITIALIZING)
        sm = EVASelfModel(state=state)
        assert sm.status() == EVAStatus.INITIALIZING

    def test_state_property(self):
        sm = EVASelfModel()
        assert isinstance(sm.state, EVAState)
