"""Base interface for AI model providers.

Defines the protocol that all model providers must implement.
EVA interacts with language models exclusively through this abstraction.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from eva.providers.models import ModelRequest, ModelResponse

T = TypeVar("T")


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
        request: ModelRequest | str,
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> ModelResponse | str:
        """Generate a response from the model.

        Accepts either a ModelRequest (returning ModelResponse)
        or a legacy string prompt (returning str for 1A backward compatibility).
        """
        ...

    async def generate_structured(
        self,
        request: ModelRequest,
        schema: type[T],
    ) -> ModelResponse:
        """Generate structured output adhering to a Pydantic schema."""
        ...

    async def is_available(self) -> bool:
        """Check whether the provider is currently available."""
        ...
