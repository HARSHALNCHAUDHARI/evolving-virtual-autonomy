"""Tests for EVA CLI logic (non-interactive)."""

import pytest

from eva.cli import BANNER
from eva.config import EVASettings
from eva.core.enums import EVAStatus
from eva.runtime import EVARuntime


class TestCLIBanner:
    """CLI banner and branding."""

    def test_banner_contains_eva(self):
        assert "E . V . A" in BANNER or "EVA" in BANNER

    def test_banner_contains_tagline(self):
        assert "Evolving Virtual Autonomy" in BANNER


class TestCLIRuntime:
    """CLI uses runtime correctly."""

    @pytest.mark.asyncio
    async def test_runtime_lifecycle_for_cli(self):
        """Simulate the runtime lifecycle that the CLI performs."""
        settings = EVASettings(env="testing", model_provider="mock")
        runtime = EVARuntime(settings=settings)

        await runtime.start()
        assert runtime.state.status == EVAStatus.ONLINE

        # Simulate a conversation turn
        runtime.context.increment_turn()
        response = await runtime.provider.generate("hello")
        assert isinstance(response, str)
        assert len(response) > 0

        await runtime.stop()
        assert runtime.state.status == EVAStatus.OFFLINE
