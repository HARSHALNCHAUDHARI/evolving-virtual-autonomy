"""Tests for the Google Gemini model provider (Milestone 1D.2).

Verifies protocol conformance, request building, response normalization,
token usage, latency tracking, error handling, API key sanitization,
structured output validation, timeout handling, configuration, and
coexistence with MockProvider.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from eva.config import EVASettings, get_settings
from eva.providers.base import ModelProvider
from eva.providers.gemini import GeminiProvider
from eva.providers.mock import MockProvider
from eva.providers.models import (
    ChatMessage,
    ModelCapability,
    ModelRequest,
    ModelResponse,
    RouteRequest,
    TaskType,
)
from eva.providers.registry import ModelRegistry
from eva.providers.router import ModelRouter
from eva.runtime import EVARuntime


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


class SampleStructuredOutput(BaseModel):
    summary: str
    confidence: float
    tags: list[str] = Field(default_factory=list)


def create_gemini_response_json(
    text: str = "Hello from Gemini!",
    finish_reason: str = "STOP",
    prompt_tokens: int = 12,
    completion_tokens: int = 8,
) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}],
                    "role": "model",
                },
                "finishReason": finish_reason,
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
            "totalTokenCount": prompt_tokens + completion_tokens,
        },
    }


# ===========================================================================
# 1. Protocol & Interface Conformance
# ===========================================================================


class TestGeminiProviderInterface:
    """Verifies that GeminiProvider strictly conforms to the ModelProvider protocol."""

    def test_implements_model_provider_protocol(self):
        provider = GeminiProvider(api_key="test-key")
        assert isinstance(provider, ModelProvider)
        assert provider.name == "gemini"
        assert provider.model_name == "gemini-1.5-flash"

    def test_custom_model_initialization(self):
        provider = GeminiProvider(api_key="test-key", model="gemini-1.5-pro", timeout=45.0)
        assert provider.model_name == "gemini-1.5-pro"
        assert provider._timeout == 45.0

    def test_capabilities_set(self):
        provider = GeminiProvider(api_key="test-key")
        caps = provider.capabilities
        assert ModelCapability.TEXT in caps
        assert ModelCapability.REASONING in caps
        assert ModelCapability.STRUCTURED_OUTPUT in caps
        assert ModelCapability.VISION in caps
        assert ModelCapability.STREAMING in caps
        assert ModelCapability.LONG_CONTEXT in caps

    def test_get_spec(self):
        provider = GeminiProvider(api_key="test-key", model="gemini-1.5-flash")
        spec = provider.get_spec()
        assert spec.provider == "gemini"
        assert spec.model == "gemini-1.5-flash"
        assert spec.available is True
        assert spec.context_limits is not None
        assert spec.context_limits.context_window == 1048576
        assert spec.has_capability(ModelCapability.TEXT) is True
        assert spec.has_capability(ModelCapability.STRUCTURED_OUTPUT) is True


# ===========================================================================
# 2. Availability Checks
# ===========================================================================


class TestGeminiAvailability:
    """Tests provider availability based on API key configuration."""

    @pytest.mark.asyncio
    async def test_unavailable_when_api_key_is_none(self):
        provider = GeminiProvider(api_key=None)
        assert await provider.is_available() is False
        assert provider.get_spec().available is False

    @pytest.mark.asyncio
    async def test_unavailable_when_api_key_is_empty(self):
        provider = GeminiProvider(api_key="   ")
        assert await provider.is_available() is False
        assert provider.get_spec().available is False

    @pytest.mark.asyncio
    async def test_available_when_api_key_provided(self):
        provider = GeminiProvider(api_key="actual-api-key")
        assert await provider.is_available() is True
        assert provider.get_spec().available is True


# ===========================================================================
# 3. Request Construction & Execution (Mocked Transport)
# ===========================================================================


class TestGeminiTextGeneration:
    """Tests normal text generation and request serialization."""

    @pytest.mark.asyncio
    async def test_generate_with_model_request(self):
        captured_requests = []

        def mock_transport(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
            captured_requests.append((req, timeout))
            return create_gemini_response_json(text="Gemini answer", finish_reason="STOP")

        provider = GeminiProvider(api_key="secret-api-key-123", transport=mock_transport)
        req = ModelRequest(
            prompt="What is the speed of light?",
            system_prompt="Be concise.",
            temperature=0.3,
            max_tokens=100,
        )

        resp = await provider.generate(req)

        assert isinstance(resp, ModelResponse)
        assert resp.success is True
        assert resp.content == "Gemini answer"
        assert resp.provider == "gemini"
        assert resp.model == "gemini-1.5-flash"
        assert resp.finish_reason == "stop"
        assert resp.usage is not None
        assert resp.usage.prompt_tokens == 12
        assert resp.usage.completion_tokens == 8
        assert resp.usage.total_tokens == 20
        assert resp.metadata.latency_ms >= 0.0

        # Verify outgoing HTTP request inspection
        assert len(captured_requests) == 1
        http_req, timeout = captured_requests[0]
        assert timeout == 30.0
        assert "secret-api-key-123" not in http_req.full_url  # No key in URL!
        assert http_req.get_header("X-goog-api-key") == "secret-api-key-123"

        payload = json.loads(http_req.data.decode("utf-8"))
        assert payload["systemInstruction"]["parts"][0]["text"] == "Be concise."
        assert payload["generationConfig"]["temperature"] == 0.3
        assert payload["generationConfig"]["maxOutputTokens"] == 100
        assert payload["contents"][0]["role"] == "user"
        assert payload["contents"][0]["parts"][0]["text"] == "What is the speed of light?"

    @pytest.mark.asyncio
    async def test_generate_with_chat_messages(self):
        captured_payloads = []

        def mock_transport(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
            captured_payloads.append(json.loads(req.data.decode("utf-8")))
            return create_gemini_response_json("Chat response")

        provider = GeminiProvider(api_key="key", transport=mock_transport)
        req = ModelRequest(
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there"),
                ChatMessage(role="user", content="How are you?"),
            ]
        )

        resp = await provider.generate(req)
        assert isinstance(resp, ModelResponse)
        assert resp.content == "Chat response"

        payload = captured_payloads[0]
        assert len(payload["contents"]) == 3
        assert payload["contents"][0]["role"] == "user"
        assert payload["contents"][1]["role"] == "model"
        assert payload["contents"][2]["role"] == "user"

    @pytest.mark.asyncio
    async def test_generate_with_string_prompt(self):
        def mock_transport(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
            return create_gemini_response_json("String prompt response")

        provider = GeminiProvider(api_key="key", transport=mock_transport)
        raw_result = await provider.generate("Tell me a fact")
        assert isinstance(raw_result, str)
        assert raw_result == "String prompt response"


# ===========================================================================
# 4. Structured Output Generation
# ===========================================================================


class TestGeminiStructuredOutput:
    """Tests schema enforcement and Pydantic validation."""

    @pytest.mark.asyncio
    async def test_generate_structured_success(self):
        valid_json = json.dumps({
            "summary": "Plan completed successfully.",
            "confidence": 0.98,
            "tags": ["planning", "eva"],
        })

        def mock_transport(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["generationConfig"]["responseMimeType"] == "application/json"
            assert "responseSchema" in payload["generationConfig"]
            return create_gemini_response_json(text=valid_json)

        provider = GeminiProvider(api_key="key", transport=mock_transport)
        req = ModelRequest(prompt="Summarize the plan")
        resp = await provider.generate_structured(req, SampleStructuredOutput)

        assert resp.success is True
        assert resp.finish_reason == "stop"
        assert isinstance(resp.structured_output, SampleStructuredOutput)
        assert resp.structured_output.summary == "Plan completed successfully."
        assert resp.structured_output.confidence == 0.98
        assert resp.structured_output.tags == ["planning", "eva"]

    @pytest.mark.asyncio
    async def test_generate_structured_malformed_json_returns_typed_failure(self):
        def mock_transport(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
            return create_gemini_response_json(text="This is not valid JSON at all")

        provider = GeminiProvider(api_key="key", transport=mock_transport)
        req = ModelRequest(prompt="Summarize")
        resp = await provider.generate_structured(req, SampleStructuredOutput)

        assert resp.success is False
        assert resp.structured_output is None
        assert resp.finish_reason == "error"
        assert "Structured output validation failed" in resp.error


# ===========================================================================
# 5. Security & API Key Masking
# ===========================================================================


class TestGeminiSecurityAndSanitization:
    """Verifies that API keys are never exposed in errors, URLs, or logs."""

    @pytest.mark.asyncio
    async def test_api_key_redacted_from_http_error(self):
        sensitive_key = "AIzaSySensitiveSecretKey999"

        def failing_transport(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
            error_json = json.dumps({
                "error": {
                    "code": 400,
                    "message": f"Invalid argument involving key {sensitive_key}",
                }
            }).encode("utf-8")
            fp = urllib.error.HTTPError(
                url=req.full_url,
                code=400,
                msg="Bad Request",
                hdrs={},
                fp=None,
            )
            # Patch read to simulate urllib error stream
            fp.read = lambda: error_json
            raise fp

        provider = GeminiProvider(api_key=sensitive_key, transport=failing_transport)
        req = ModelRequest(prompt="Test error")
        resp = await provider.generate(req)

        assert resp.success is False
        assert sensitive_key not in resp.error
        assert "***REDACTED***" in resp.error
        assert "400" in resp.error

    @pytest.mark.asyncio
    async def test_api_key_redacted_from_generic_exception(self):
        sensitive_key = "AIzaSyAnotherSecretKey888"

        def crash_transport(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
            raise RuntimeError(f"Connection pool crash with token: {sensitive_key}")

        provider = GeminiProvider(api_key=sensitive_key, transport=crash_transport)
        req = ModelRequest(prompt="Test crash")
        resp = await provider.generate(req)

        assert resp.success is False
        assert sensitive_key not in resp.error
        assert "***REDACTED***" in resp.error


# ===========================================================================
# 6. Error & Timeout Handling
# ===========================================================================


class TestGeminiErrorAndTimeoutHandling:
    """Verifies provider behavior on network failure, missing key, and timeouts."""

    @pytest.mark.asyncio
    async def test_unconfigured_api_key_returns_structured_failure(self):
        provider = GeminiProvider(api_key=None)
        req = ModelRequest(prompt="Hello")
        resp = await provider.generate(req)

        assert resp.success is False
        assert resp.error == "Gemini API key is not configured."
        assert resp.finish_reason == "error"

    @pytest.mark.asyncio
    async def test_timeout_returns_structured_failure(self):
        def timeout_transport(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
            raise TimeoutError("Socket read timed out")

        provider = GeminiProvider(api_key="key", timeout=12.5, transport=timeout_transport)
        req = ModelRequest(prompt="Slow query")
        resp = await provider.generate(req)

        assert resp.success is False
        assert resp.finish_reason == "timeout"
        assert "timed out after 12.5s" in resp.error

    @pytest.mark.asyncio
    async def test_finish_reason_mapping(self):
        test_cases = [
            ("STOP", "stop"),
            ("MAX_TOKENS", "length"),
            ("SAFETY", "safety"),
            ("RECITATION", "recitation"),
            ("OTHER", "stop"),
        ]
        for gemini_reason, expected in test_cases:
            def transport_for_case(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
                return create_gemini_response_json("Response", finish_reason=gemini_reason)

            provider = GeminiProvider(api_key="key", transport=transport_for_case)
            resp = await provider.generate(ModelRequest(prompt="test"))
            assert resp.finish_reason == expected


# ===========================================================================
# 7. Configuration & Environment Integration
# ===========================================================================


class TestGeminiConfiguration:
    """Verifies settings loading and environment variable mapping."""

    def test_default_gemini_settings(self):
        settings = EVASettings()
        assert settings.gemini_model == "gemini-1.5-flash"
        assert settings.gemini_timeout == 30.0
        assert settings.gemini_api_key is None

    def test_gemini_settings_from_env(self):
        env_vars = {
            "EVA_GEMINI_API_KEY": "env-gemini-key-xyz",
            "EVA_GEMINI_MODEL": "gemini-1.5-pro",
            "EVA_GEMINI_TIMEOUT": "45.0",
        }
        with patch.dict(os.environ, env_vars):
            settings = get_settings()
            assert settings.gemini_api_key == "env-gemini-key-xyz"
            assert settings.gemini_model == "gemini-1.5-pro"
            assert settings.gemini_timeout == 45.0

    def test_bidirectional_google_gemini_api_key_sync(self):
        settings_google = EVASettings(google_api_key="google-key-1")
        assert settings_google.gemini_api_key == "google-key-1"

        settings_gemini = EVASettings(gemini_api_key="gemini-key-2")
        assert settings_gemini.google_api_key == "gemini-key-2"


# ===========================================================================
# 8. Runtime & MockProvider Compatibility
# ===========================================================================


class TestRuntimeAndMockCompatibility:
    """Ensures MockProvider remains intact and coexists with GeminiProvider."""

    def test_mock_provider_remains_default(self):
        runtime = EVARuntime()
        provider = runtime._create_provider()
        assert isinstance(provider, MockProvider)
        assert provider.name == "mock"

    def test_runtime_creates_gemini_provider_when_configured(self):
        settings = EVASettings(
            default_provider="gemini",
            gemini_api_key="test-key",
            gemini_model="gemini-1.5-flash",
        )
        runtime = EVARuntime(settings=settings)
        provider = runtime._create_provider()
        assert isinstance(provider, GeminiProvider)
        assert provider.name == "gemini"
        assert provider.model_name == "gemini-1.5-flash"

    def test_model_router_coexistence(self):
        gemini = GeminiProvider(api_key="key", model="gemini-1.5-flash")
        mock = MockProvider(name="mock")
        registry = ModelRegistry()

        router = ModelRouter(
            providers={"gemini": gemini, "mock": mock},
            default_provider="mock",
            registry=registry,
        )

        assert router.has_provider("mock") is True
        assert router.has_provider("gemini") is True
        assert registry.has_model("mock", "mock-v1") is True
        assert registry.has_model("gemini", "gemini-1.5-flash") is True

    @pytest.mark.asyncio
    async def test_model_router_routes_to_gemini(self):
        def mock_transport(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
            return create_gemini_response_json("Routed to Gemini")

        gemini = GeminiProvider(api_key="key", transport=mock_transport)
        mock = MockProvider(name="mock")
        router = ModelRouter(
            providers={"gemini": gemini, "mock": mock},
            default_provider="mock",
        )

        # Route explicit to gemini
        route_req = RouteRequest(
            request=ModelRequest(prompt="Hello"),
            provider="gemini",
            task_type=TaskType.GENERAL,
        )
        selected_provider = router.route(route_req)
        assert selected_provider.name == "gemini"

        resp = await router.generate(
            ModelRequest(prompt="Hello", provider="gemini"),
            task_type=TaskType.GENERAL,
        )
        assert resp.content == "Routed to Gemini"
        assert resp.provider == "gemini"

        # Route default to mock
        default_route = RouteRequest(
            request=ModelRequest(prompt="Hello"),
            task_type=TaskType.GENERAL,
        )
        default_provider = router.route(default_route)
        assert default_provider.name == "mock"

        mock_resp = await router.generate(
            ModelRequest(prompt="Hello"),
            task_type=TaskType.GENERAL,
        )
        assert mock_resp.provider == "mock"


# ===========================================================================
# 9. Optional Real Integration Test
# ===========================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("EVA_GEMINI_API_KEY"),
    reason="Requires EVA_GEMINI_API_KEY environment variable for real network test.",
)
class TestGeminiRealIntegration:
    """Optional live integration test against Google Gemini API.

    Only executed if EVA_GEMINI_API_KEY is explicitly configured in the environment.
    """

    @pytest.mark.asyncio
    async def test_live_text_generation(self):
        api_key = os.environ["EVA_GEMINI_API_KEY"]
        provider = GeminiProvider(api_key=api_key)
        assert await provider.is_available() is True

        req = ModelRequest(prompt="Reply with exactly the word PONG.")
        resp = await provider.generate(req)

        assert resp.success is True
        assert "PONG" in resp.content.upper()
        assert resp.provider == "gemini"
