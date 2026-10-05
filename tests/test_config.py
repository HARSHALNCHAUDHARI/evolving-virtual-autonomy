"""Tests for EVA configuration system."""

import os

import pytest

from eva.config import EVASettings, Environment, get_settings


class TestDefaults:
    """Configuration defaults load correctly."""

    def test_default_env_is_development(self):
        settings = EVASettings()
        assert settings.env == Environment.DEVELOPMENT

    def test_default_log_level(self):
        settings = EVASettings()
        assert settings.log_level == "INFO"

    def test_default_debug_is_false(self):
        settings = EVASettings()
        assert settings.debug is False

    def test_default_provider_is_mock(self):
        settings = EVASettings()
        assert settings.model_provider == "mock"

    def test_default_model_name(self):
        settings = EVASettings()
        assert settings.model_name == "mock-v1"

    def test_no_database_url_by_default(self):
        settings = EVASettings()
        assert settings.database_url is None

    def test_no_redis_url_by_default(self):
        settings = EVASettings()
        assert settings.redis_url is None

    def test_no_api_keys_by_default(self):
        settings = EVASettings()
        assert settings.openai_api_key is None
        assert settings.anthropic_api_key is None
        assert settings.google_api_key is None


class TestEnvironmentOverrides:
    """Environment variables override defaults."""

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("EVA_ENV", "testing")
        settings = EVASettings()
        assert settings.env == Environment.TESTING

    def test_log_level_override(self, monkeypatch):
        monkeypatch.setenv("EVA_LOG_LEVEL", "DEBUG")
        settings = EVASettings()
        assert settings.log_level == "DEBUG"

    def test_debug_override(self, monkeypatch):
        monkeypatch.setenv("EVA_DEBUG", "true")
        settings = EVASettings()
        assert settings.debug is True

    def test_provider_override(self, monkeypatch):
        monkeypatch.setenv("EVA_MODEL_PROVIDER", "openai")
        settings = EVASettings()
        assert settings.model_provider == "openai"

    def test_database_url_override(self, monkeypatch):
        monkeypatch.setenv("EVA_DATABASE_URL", "postgresql://localhost/eva")
        settings = EVASettings()
        assert settings.database_url == "postgresql://localhost/eva"


class TestConvenienceProperties:
    """Environment convenience properties work correctly."""

    def test_is_development(self):
        settings = EVASettings(env="development")
        assert settings.is_development is True
        assert settings.is_testing is False
        assert settings.is_production is False

    def test_is_testing(self):
        settings = EVASettings(env="testing")
        assert settings.is_testing is True

    def test_is_production(self):
        settings = EVASettings(env="production")
        assert settings.is_production is True

    def test_project_root_exists(self):
        settings = EVASettings()
        assert settings.project_root.exists()


class TestGetSettings:
    """get_settings() factory works."""

    def test_returns_settings_instance(self):
        s = get_settings()
        assert isinstance(s, EVASettings)

    def test_accepts_overrides(self):
        s = get_settings(env="testing", debug=True)
        assert s.env == Environment.TESTING
        assert s.debug is True
