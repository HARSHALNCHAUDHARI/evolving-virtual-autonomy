"""Shared test fixtures for EVA."""

import pytest

from eva.config import EVASettings
from eva.logging import reset_logging


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset logging state between tests to avoid handler accumulation."""
    yield
    reset_logging()


@pytest.fixture
def settings() -> EVASettings:
    """Create test settings with safe defaults."""
    return EVASettings(
        env="testing",
        log_level="DEBUG",
        debug=False,
        model_provider="mock",
        model_name="mock-v1",
    )
