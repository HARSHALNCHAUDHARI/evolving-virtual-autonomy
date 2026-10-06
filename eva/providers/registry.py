"""Model registry — central repository for model specifications and capabilities.

Tracks available models, their capabilities, modality constraints,
and performance parameters across all providers.
"""

from __future__ import annotations

from typing import Sequence

from eva.logging import get_logger
from eva.providers.models import (
    DuplicateModelError,
    ModelCapability,
    ModelNotFoundError,
    ModelSpec,
)

logger = get_logger(__name__)


class ModelRegistry:
    """Central registry of known models and their capabilities.

    Maintains model specifications per provider and facilitates
    capability-based discovery and query.
    """

    def __init__(self, *, include_defaults: bool = True) -> None:
        self._models: dict[tuple[str, str], ModelSpec] = {}

        if include_defaults:
            self.register_default_models()

        logger.debug("ModelRegistry initialized with %d model(s)", len(self._models))

    def register_default_models(self) -> None:
        """Register default models for baseline operation."""
        self.register_model(
            ModelSpec(
                provider="mock",
                model="mock-v1",
                capabilities={
                    ModelCapability.TEXT,
                    ModelCapability.REASONING,
                    ModelCapability.STRUCTURED_OUTPUT,
                },
                modalities=["text"],
                available=True,
                metadata={"default": True},
            ),
            overwrite=True,
        )

    def register_model(self, spec: ModelSpec, *, overwrite: bool = True) -> None:
        """Register a model specification.

        Args:
            spec: The model specification to register.
            overwrite: If False, raises DuplicateModelError on collision.

        Raises:
            DuplicateModelError: If model is already registered and overwrite=False.
        """
        key = (spec.provider.lower(), spec.model.lower())
        if key in self._models and not overwrite:
            raise DuplicateModelError(
                f"Model '{spec.provider}:{spec.model}' is already registered."
            )

        self._models[key] = spec
        logger.debug("Registered model in registry: %s:%s", spec.provider, spec.model)

    def unregister_model(self, provider: str, model: str) -> bool:
        """Unregister a model by provider and model name.

        Returns:
            True if the model was found and removed, False otherwise.
        """
        key = (provider.lower(), model.lower())
        if key in self._models:
            del self._models[key]
            logger.debug("Unregistered model: %s:%s", provider, model)
            return True
        return False

    def get_model(self, provider: str, model: str) -> ModelSpec | None:
        """Retrieve a model specification by provider and model name."""
        key = (provider.lower(), model.lower())
        return self._models.get(key)

    def lookup(self, provider: str, model: str) -> ModelSpec:
        """Retrieve a model specification or raise ModelNotFoundError.

        Raises:
            ModelNotFoundError: If the model is not found in the registry.
        """
        model_spec = self.get_model(provider, model)
        if model_spec is None:
            raise ModelNotFoundError(
                f"Model '{provider}:{model}' is not registered in the ModelRegistry."
            )
        return model_spec

    def list_models(self, provider: str | None = None) -> list[ModelSpec]:
        """List all registered models, optionally filtered by provider."""
        if provider is not None:
            prov_lower = provider.lower()
            return [spec for (p, _), spec in self._models.items() if p == prov_lower]
        return list(self._models.values())

    def filter_by_capability(
        self,
        capability: ModelCapability | str,
        provider: str | None = None,
    ) -> list[ModelSpec]:
        """Return all models that possess the requested capability."""
        models = self.list_models(provider=provider)
        return [m for m in models if m.has_capability(capability)]

    def has_model(self, provider: str, model: str) -> bool:
        """Check if a specific model is registered."""
        key = (provider.lower(), model.lower())
        return key in self._models

    def has_provider(self, provider: str) -> bool:
        """Check if any models are registered under the given provider."""
        prov_lower = provider.lower()
        return any(p == prov_lower for (p, _) in self._models.keys())

    def list_providers(self) -> list[str]:
        """List all unique providers that have registered models."""
        return sorted(list({p for (p, _) in self._models.keys()}))

    def clear(self) -> None:
        """Clear all registered models."""
        self._models.clear()
        logger.debug("ModelRegistry cleared")
