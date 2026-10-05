"""Tests for EVA runtime lifecycle."""

import pytest

from eva.config import EVASettings
from eva.core.enums import EVAStatus
from eva.runtime import EVARuntime


@pytest.fixture
def runtime() -> EVARuntime:
    """Create a runtime with test settings."""
    settings = EVASettings(
        env="testing",
        log_level="DEBUG",
        model_provider="mock",
    )
    return EVARuntime(settings=settings)


class TestRuntimeInitialization:
    """EVA initializes successfully."""

    @pytest.mark.asyncio
    async def test_starts_successfully(self, runtime):
        await runtime.start()
        assert runtime.state.status == EVAStatus.ONLINE
        await runtime.stop()

    @pytest.mark.asyncio
    async def test_reaches_online_state(self, runtime):
        await runtime.start()
        assert runtime.state.status == EVAStatus.ONLINE
        await runtime.stop()

    @pytest.mark.asyncio
    async def test_provider_available_after_start(self, runtime):
        await runtime.start()
        provider = runtime.provider
        assert provider.name == "mock"
        assert await provider.is_available() is True
        await runtime.stop()

    @pytest.mark.asyncio
    async def test_context_initialized(self, runtime):
        await runtime.start()
        ctx = runtime.context.current
        assert ctx.conversation.active is True
        assert ctx.environment.platform is not None
        await runtime.stop()


class TestRuntimeShutdown:
    """Runtime shuts down cleanly."""

    @pytest.mark.asyncio
    async def test_shutdown_reaches_offline(self, runtime):
        await runtime.start()
        await runtime.stop()
        assert runtime.state.status == EVAStatus.OFFLINE

    @pytest.mark.asyncio
    async def test_context_reset_on_shutdown(self, runtime):
        await runtime.start()
        await runtime.stop()
        assert runtime.context.current.conversation.active is False


class TestRuntimeProperties:
    """Runtime property access."""

    def test_settings_accessible(self, runtime):
        assert runtime.settings.env.value == "testing"

    def test_self_model_accessible(self, runtime):
        sm = runtime.self_model
        assert sm.identity() is not None

    def test_safety_accessible(self, runtime):
        assert runtime.safety is not None

    def test_provider_raises_before_start(self, runtime):
        with pytest.raises(RuntimeError):
            _ = runtime.provider
