"""Tests for EVARuntime integration with ModelRouter."""

import pytest

from eva.config import EVASettings
from eva.providers.router import ModelRouter
from eva.runtime import EVARuntime


class TestRuntimeModelRouter:
    """EVARuntime model router initialization and lifecycle."""

    def test_router_raises_before_start(self):
        runtime = EVARuntime()
        with pytest.raises(RuntimeError) as exc_info:
            _ = runtime.router
        assert "not been started" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_router_initialized_on_start(self):
        runtime = EVARuntime()
        await runtime.start()

        try:
            assert isinstance(runtime.router, ModelRouter)
            assert runtime.router.has_provider("mock") is True
            assert runtime.cognitive_engine.model_router is runtime.router
        finally:
            await runtime.stop()

    @pytest.mark.asyncio
    async def test_router_cleared_on_stop(self):
        runtime = EVARuntime()
        await runtime.start()
        await runtime.stop()

        with pytest.raises(RuntimeError):
            _ = runtime.router
