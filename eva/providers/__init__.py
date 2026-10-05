"""EVA model provider abstraction.

Defines the protocol that all model providers must implement,
ensuring EVA is never hard-coded to a single AI provider.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelProvider(Protocol):
    """Interface for AI model providers.

    All providers must implement this protocol. EVA interacts
    with language models exclusively through this abstraction.
    """

    @property
    def name(self) -> str:
        """Return the provider name."""
        ...

    async def generate(
        self,
        prompt: str,
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response from the model.

        Args:
            prompt: The user/system prompt to send.
            context: Optional context dictionary for the model.
            system_prompt: Optional system-level instruction.

        Returns:
            The generated text response.
        """
        ...

    async def is_available(self) -> bool:
        """Check whether the provider is currently available.

        Returns:
            True if the provider can accept requests.
        """
        ...
