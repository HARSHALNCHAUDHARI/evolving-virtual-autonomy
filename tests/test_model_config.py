"""Tests for model provider and router configuration."""

import os
from unittest.mock import patch

from eva.config import EVASettings, get_settings


class TestModelConfiguration:
    """Configuration defaults and environment variable behavior."""

    def test_default_model_settings(self):
        settings = EVASettings()
        assert settings.model_provider == "mock"
        assert settings.model_name == "mock-v1"
        assert settings.default_provider == "mock"
        assert settings.default_model == "mock-v1"

    def test_override_default_provider(self):
        settings = EVASettings(default_provider="custom_p", default_model="custom_m")
        assert settings.default_provider == "custom_p"
        assert settings.default_model == "custom_m"
        assert settings.model_provider == "custom_p"
        assert settings.model_name == "custom_m"

    def test_override_model_provider(self):
        settings = EVASettings(model_provider="legacy_p", model_name="legacy_m")
        assert settings.model_provider == "legacy_p"
        assert settings.default_provider == "legacy_p"
        assert settings.model_name == "legacy_m"
        assert settings.default_model == "legacy_m"

    def test_env_var_default_provider(self):
        with patch.dict(os.environ, {"EVA_DEFAULT_PROVIDER": "env_provider", "EVA_DEFAULT_MODEL": "env_model"}):
            settings = get_settings()
            assert settings.default_provider == "env_provider"
            assert settings.default_model == "env_model"
            assert settings.model_provider == "env_provider"
            assert settings.model_name == "env_model"
