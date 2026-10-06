"""EVA runtime — bootstrap and lifecycle management.

The runtime initializes all EVA subsystems and manages the
system lifecycle from startup through shutdown.
"""

from __future__ import annotations

import platform
import socket

from eva.config import EVASettings, get_settings
from eva.core.brain import CognitiveEngine, CognitiveResult
from eva.core.context import ContextManager
from eva.core.enums import EVAStatus
from eva.core.models import EVAState
from eva.core.self_model import EVASelfModel
from eva.logging import get_logger, setup_logging
from eva.providers import ModelProvider
from eva.providers.mock import MockProvider
from eva.safety import SafetyPolicy

logger = get_logger(__name__)


class EVARuntime:
    """Manages the full lifecycle of the EVA system.

    Initializes configuration, logging, context, self-model,
    provider, safety, and cognitive engine subsystems.
    Provides start/stop lifecycle and state access.
    """

    def __init__(self, settings: EVASettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._self_model = EVASelfModel()
        self._context = ContextManager()
        self._safety = SafetyPolicy()
        self._provider: ModelProvider | None = None
        self._cognitive_engine: CognitiveEngine | None = None

    @property
    def settings(self) -> EVASettings:
        return self._settings

    @property
    def state(self) -> EVAState:
        return self._self_model.state

    @property
    def context(self) -> ContextManager:
        return self._context

    @property
    def self_model(self) -> EVASelfModel:
        return self._self_model

    @property
    def safety(self) -> SafetyPolicy:
        return self._safety

    @property
    def provider(self) -> ModelProvider:
        if self._provider is None:
            raise RuntimeError("EVA runtime has not been started. Call start() first.")
        return self._provider

    @property
    def cognitive_engine(self) -> CognitiveEngine:
        """Return the cognitive engine. Requires start() to have been called."""
        if self._cognitive_engine is None:
            raise RuntimeError("EVA runtime has not been started. Call start() first.")
        return self._cognitive_engine

    async def start(self) -> None:
        """Initialize and start the EVA system."""
        # 1. Logging
        setup_logging(level=self._settings.log_level, debug=self._settings.debug)
        logger.info("EVA starting...")

        # 2. Status → INITIALIZING
        self._self_model.update_status(EVAStatus.INITIALIZING)

        # 3. Initialize provider
        self._provider = self._create_provider()
        available = await self._provider.is_available()
        logger.info("Provider '%s' available: %s", self._provider.name, available)

        # 4. Initialize cognitive engine
        self._cognitive_engine = CognitiveEngine(
            context_manager=self._context,
            safety=self._safety,
        )
        logger.info("Cognitive engine initialized")

        # 5. Populate environment context
        self._context.update_environment(
            platform=platform.system(),
            hostname=socket.gethostname(),
        )

        # 6. Start conversation context
        self._context.start_conversation()

        # 7. Status → ONLINE
        self._self_model.update_status(EVAStatus.ONLINE)
        logger.info("EVA is online and ready.")

    async def stop(self) -> None:
        """Gracefully shut down the EVA system."""
        logger.info("EVA shutting down...")
        self._self_model.update_status(EVAStatus.SHUTTING_DOWN)
        self._context.reset()
        self._cognitive_engine = None
        self._self_model.update_status(EVAStatus.OFFLINE)
        logger.info("EVA shutdown complete.")

    async def process_input(self, user_input: str) -> CognitiveResult:
        """Process user input through the full cognitive pipeline.

        Args:
            user_input: Raw text from the user.

        Returns:
            A CognitiveResult capturing the full cognitive cycle.
        """
        self._context.increment_turn()
        return await self.cognitive_engine.process(
            user_input, self._context.current
        )

    def _create_provider(self) -> ModelProvider:
        """Create the appropriate model provider based on configuration."""
        provider_name = self._settings.model_provider.lower()

        if provider_name == "mock":
            return MockProvider()

        # Future providers will be added here:
        # if provider_name == "openai": ...
        # if provider_name == "gemini": ...
        # if provider_name == "anthropic": ...

        logger.warning(
            "Unknown provider '%s', falling back to mock.", provider_name
        )
        return MockProvider()
