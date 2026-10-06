"""EVA configuration management.

Environment-based configuration using pydantic-settings.
All settings have sensible defaults for local development.
No external services are required to start EVA.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Runtime environment."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class EVASettings(BaseSettings):
    """Central configuration for EVA.

    Values are loaded from environment variables (prefixed with EVA_)
    and .env files. All fields have defaults suitable for local development.
    """

    model_config = SettingsConfigDict(
        env_prefix="EVA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core
    env: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    debug: bool = False

    # Database (optional — not required for Milestone 1A)
    database_url: str | None = None

    # Redis (optional — not required for Milestone 1A)
    redis_url: str | None = None

    # Model provider & router defaults
    model_provider: str = "mock"
    model_name: str = "mock-v1"
    default_provider: str = "mock"
    default_model: str = "mock-v1"

    @model_validator(mode="after")
    def _sync_provider_defaults(self) -> EVASettings:
        """Keep model_provider/name and default_provider/model in sync."""
        if self.default_provider != "mock" and self.model_provider == "mock":
            self.model_provider = self.default_provider
        elif self.model_provider != "mock" and self.default_provider == "mock":
            self.default_provider = self.model_provider

        if self.default_model != "mock-v1" and self.model_name == "mock-v1":
            self.model_name = self.default_model
        elif self.model_name != "mock-v1" and self.default_model == "mock-v1":
            self.default_model = self.model_name
        return self

    # API keys (never logged, never committed)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    @property
    def is_development(self) -> bool:
        return self.env == Environment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        return self.env == Environment.TESTING

    @property
    def is_production(self) -> bool:
        return self.env == Environment.PRODUCTION

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent


def get_settings(**overrides: object) -> EVASettings:
    """Create an EVASettings instance, optionally overriding values.

    This is the primary entry point for obtaining configuration.
    """
    return EVASettings(**overrides)  # type: ignore[arg-type]
