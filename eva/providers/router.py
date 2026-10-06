"""Model router — selects and routes requests to AI model providers.

Provides deterministic routing based on task type, requested provider,
model, or capability. Extensible for future multi-provider orchestration.
"""

from __future__ import annotations

from typing import Any, TypeVar

from eva.logging import get_logger
from eva.providers.base import ModelProvider
from eva.providers.mock import MockProvider
from eva.providers.models import (
    ModelRequest,
    ModelResponse,
    ProviderNotFoundError,
    RouteRequest,
    TaskType,
)

logger = get_logger(__name__)

T = TypeVar("T")


class ModelRouter:
    """Routes model requests to appropriate provider implementations.

    Decouples the cognitive core from individual providers.
    Supports deterministic routing based on task type, explicit provider
    selection, and configured defaults.
    """

    def __init__(
        self,
        providers: dict[str, ModelProvider] | None = None,
        *,
        default_provider: str = "mock",
        default_model: str = "mock-v1",
    ) -> None:
        self._providers: dict[str, ModelProvider] = {}
        self._default_provider = default_provider
        self._default_model = default_model
        self._task_routes: dict[TaskType, str] = {
            TaskType.CONVERSATION: default_provider,
            TaskType.UNDERSTANDING: default_provider,
            TaskType.REASONING: default_provider,
            TaskType.STRUCTURED_OUTPUT: default_provider,
            TaskType.GENERAL: default_provider,
        }

        if providers:
            for p in providers.values():
                self.register_provider(p)
        else:
            # Register a default mock provider if none supplied
            self.register_provider(MockProvider(name=default_provider))

        logger.debug(
            "ModelRouter initialized (default_provider=%s, default_model=%s)",
            self._default_provider,
            self._default_model,
        )

    @property
    def default_provider_name(self) -> str:
        """Return the default provider name."""
        return self._default_provider

    @property
    def default_model_name(self) -> str:
        """Return the default model name."""
        return self._default_model

    def register_provider(self, provider: ModelProvider) -> None:
        """Register a model provider instance."""
        self._providers[provider.name.lower()] = provider
        logger.debug("Registered provider: %s", provider.name)

    def unregister_provider(self, name: str) -> None:
        """Unregister a provider by name."""
        self._providers.pop(name.lower(), None)

    def get_provider(self, name: str) -> ModelProvider:
        """Retrieve a registered provider by name.

        Raises:
            ProviderNotFoundError: If the provider is not registered.
        """
        normalized = name.lower()
        provider = self._providers.get(normalized)
        if provider is None:
            raise ProviderNotFoundError(f"Provider '{name}' is not registered or supported.")
        return provider

    def has_provider(self, name: str) -> bool:
        """Check if a provider is registered."""
        return name.lower() in self._providers

    def list_providers(self) -> list[str]:
        """List names of all registered providers."""
        return list(self._providers.keys())

    def register_route(self, task_type: TaskType | str, provider_name: str) -> None:
        """Map a task type to a specific provider name."""
        if isinstance(task_type, str):
            task_type = TaskType(task_type.lower())
        self._task_routes[task_type] = provider_name.lower()
        logger.debug("Mapped task %s -> provider %s", task_type.value, provider_name)

    def route(
        self,
        criteria: RouteRequest | ModelRequest | TaskType | str | None = None,
    ) -> ModelProvider:
        """Select a ModelProvider based on routing criteria.

        Priority:
        1. Explicitly requested provider in criteria
        2. Task-specific provider mapping
        3. Default provider

        Raises:
            ProviderNotFoundError: If the target provider is not registered.
        """
        target_name: str | None = None

        if criteria is None:
            target_name = self._default_provider

        elif isinstance(criteria, str):
            # Could be a provider name or a task type
            lowered = criteria.lower()
            if lowered in self._providers:
                target_name = lowered
            else:
                try:
                    tt = TaskType(lowered)
                    target_name = self._task_routes.get(tt, self._default_provider)
                except ValueError:
                    target_name = lowered

        elif isinstance(criteria, TaskType):
            target_name = self._task_routes.get(criteria, self._default_provider)

        elif isinstance(criteria, RouteRequest):
            if criteria.provider:
                target_name = criteria.provider
            else:
                target_name = self._task_routes.get(criteria.task_type, self._default_provider)

        elif isinstance(criteria, ModelRequest):
            if criteria.provider:
                target_name = criteria.provider
            else:
                target_name = self._default_provider

        if not target_name:
            target_name = self._default_provider

        return self.get_provider(target_name)

    async def generate(
        self,
        request: ModelRequest,
        *,
        task_type: TaskType = TaskType.GENERAL,
    ) -> ModelResponse:
        """Route and execute a text generation request."""
        criteria = RouteRequest(
            task_type=task_type,
            provider=request.provider,
            model=request.model or self._default_model,
        )
        provider = self.route(criteria)
        response = await provider.generate(request)
        if isinstance(response, str):
            # In case a legacy provider returns a str
            return ModelResponse(
                content=response,
                model=request.model or self._default_model,
                provider=provider.name,
                request_id=request.request_id,
            )
        return response

    async def generate_structured(
        self,
        request: ModelRequest,
        schema: type[T],
        *,
        task_type: TaskType = TaskType.STRUCTURED_OUTPUT,
    ) -> ModelResponse:
        """Route and execute a structured generation request."""
        criteria = RouteRequest(
            task_type=task_type,
            provider=request.provider,
            model=request.model or self._default_model,
        )
        provider = self.route(criteria)
        return await provider.generate_structured(request, schema)
