"""Mock model provider for development and testing.

Returns deterministic responses without requiring any external API.
This enables the project to run, tests to pass, and the CLI to
provide basic interaction during Milestone 1A.
"""

from __future__ import annotations

from typing import Any


class MockProvider:
    """A deterministic mock provider for testing and development.

    Implements the ModelProvider protocol without external dependencies.
    """

    def __init__(self, *, default_response: str | None = None) -> None:
        self._default_response = default_response or (
            "I understand your message. EVA is currently operating with a mock provider. "
            "Connect a real model provider for full capabilities."
        )

    @property
    def name(self) -> str:
        return "mock"

    async def generate(
        self,
        prompt: str,
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Return a deterministic response.

        If the prompt matches known patterns, return a contextual
        response. Otherwise return the default.
        """
        lower = prompt.strip().lower()

        if lower in ("hello", "hi", "hey"):
            return "Hello. EVA is online and ready."

        if lower in ("status", "state"):
            return "EVA is operating normally with mock provider."

        if "help" in lower:
            return (
                "EVA can assist you once a real model provider is connected. "
                "Currently running in mock mode."
            )

        return self._default_response

    async def is_available(self) -> bool:
        """Mock provider is always available."""
        return True
