"""Mock model provider for development and testing.

Returns deterministic responses without requiring any external API.
This enables the project to run, tests to pass, and the CLI to
provide basic interaction without network dependencies.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from eva.providers.models import (
    ModelMetadata,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)

T = TypeVar("T")


class MockProvider:
    """A deterministic mock provider for testing and development.

    Implements the ModelProvider protocol without external dependencies.
    Supports both legacy string calls and typed ModelRequest/ModelResponse interactions.
    """

    def __init__(
        self,
        *,
        default_response: str | None = None,
        name: str = "mock",
    ) -> None:
        self._name = name
        self._default_response = default_response or (
            "I understand your message. EVA is currently operating with a mock provider. "
            "Connect a real model provider for full capabilities."
        )

    @property
    def name(self) -> str:
        return self._name

    def _determine_text(self, prompt: str) -> str:
        """Produce deterministic text for a given prompt string."""
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

    async def generate(
        self,
        request: ModelRequest | str,
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> ModelResponse | str:
        """Return a deterministic response.

        When passed a string (Milestone 1A signature), returns a string.
        When passed a ModelRequest, returns a typed ModelResponse.
        """
        if isinstance(request, str):
            return self._determine_text(request)

        # Handle ModelRequest
        prompt_text = request.get_prompt_text()
        content = self._determine_text(prompt_text)
        prompt_tokens = max(1, len(prompt_text.split())) if prompt_text else 0
        completion_tokens = max(1, len(content.split()))

        return ModelResponse(
            content=content,
            model=request.model or "mock-v1",
            provider=self._name,
            finish_reason="stop",
            usage=ModelUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            metadata=ModelMetadata(latency_ms=0.5, finish_reason="stop"),
            request_id=request.request_id,
            success=True,
        )

    async def generate_structured(
        self,
        request: ModelRequest,
        schema: type[T],
    ) -> ModelResponse:
        """Generate structured output validated against a schema.

        If mock data is provided in request metadata or prompt, validates it.
        If structured output cannot be validated or is malformed, returns
        a typed failure (success=False, error=...) instead of raising.
        """
        raw_text = ""
        try:
            # 1. Check metadata for raw mock data or raw JSON
            if "mock_data" in request.metadata:
                mock_input = request.metadata["mock_data"]
                if issubclass(schema, BaseModel) and isinstance(mock_input, dict):
                    validated = schema.model_validate(mock_input)
                    raw_text = json.dumps(mock_input)
                elif issubclass(schema, BaseModel) and isinstance(mock_input, str):
                    validated = schema.model_validate_json(mock_input)
                    raw_text = mock_input
                else:
                    raise ValueError(f"Unsupported mock_data format for schema {schema}")

            elif "raw_json" in request.metadata:
                raw_json = request.metadata["raw_json"]
                raw_text = raw_json
                if issubclass(schema, BaseModel):
                    validated = schema.model_validate_json(raw_json)
                else:
                    raise ValueError(f"Schema must be a Pydantic model: {schema}")

            elif request.prompt and request.prompt.strip().startswith(("{", "[")):
                raw_text = request.prompt.strip()
                if issubclass(schema, BaseModel):
                    validated = schema.model_validate_json(raw_text)
                else:
                    raise ValueError(f"Schema must be a Pydantic model: {schema}")

            else:
                # 2. Synthesize default mock instance if schema is a Pydantic model
                if issubclass(schema, BaseModel):
                    try:
                        validated = schema()
                        raw_text = validated.model_dump_json()
                    except ValidationError:
                        # Construct a mock dictionary matching field types
                        dummy: dict[str, Any] = {}
                        for field_name, field_info in schema.model_fields.items():
                            annotation = field_info.annotation
                            if annotation is str:
                                dummy[field_name] = f"mock_{field_name}"
                            elif annotation is int:
                                dummy[field_name] = 1
                            elif annotation is float:
                                dummy[field_name] = 1.0
                            elif annotation is bool:
                                dummy[field_name] = True
                            elif annotation is list:
                                dummy[field_name] = []
                            elif annotation is dict:
                                dummy[field_name] = {}
                            else:
                                dummy[field_name] = None
                        validated = schema.model_validate(dummy)
                        raw_text = json.dumps(dummy)
                else:
                    raise ValueError(f"Schema must be a Pydantic model: {schema}")

            return ModelResponse(
                content=raw_text,
                model=request.model or "mock-v1",
                provider=self._name,
                structured_output=validated,
                finish_reason="stop",
                usage=ModelUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
                metadata=ModelMetadata(latency_ms=0.5, finish_reason="stop"),
                request_id=request.request_id,
                success=True,
            )

        except Exception as exc:
            # Return typed failure on invalid / malformed data
            return ModelResponse(
                content=raw_text or request.get_prompt_text(),
                model=request.model or "mock-v1",
                provider=self._name,
                structured_output=None,
                finish_reason="error",
                usage=ModelUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                metadata=ModelMetadata(latency_ms=0.5, finish_reason="error"),
                request_id=request.request_id,
                success=False,
                error=f"Structured output validation failed: {exc}",
            )

    async def is_available(self) -> bool:
        """Mock provider is always available."""
        return True
