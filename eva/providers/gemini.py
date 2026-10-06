"""Google Gemini model provider implementation.

Integrates Gemini models via REST API with asynchronous execution,
strict API key masking, structured generation, token usage normalization,
and timeout handling.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from eva.logging import get_logger
from eva.providers.models import (
    ChatMessage,
    ContextLimits,
    ModelCapability,
    ModelMetadata,
    ModelRequest,
    ModelResponse,
    ModelSpec,
    ModelUsage,
    ProviderError,
)

logger = get_logger(__name__)

T = TypeVar("T")


class GeminiProvider:
    """Real AI model provider for Google Gemini models.

    Conforms to the ModelProvider protocol while communicating with
    the Gemini REST API. Ensures no API keys are logged or leaked.
    """

    _FINISH_REASON_MAP: dict[str, str] = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "safety",
        "RECITATION": "recitation",
        "OTHER": "stop",
    }

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "gemini-1.5-flash",
        timeout: float = 30.0,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        transport: Callable[[urllib.request.Request, float], dict[str, Any]] | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        logger.debug("GeminiProvider initialized with model: %s", self._model)

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "gemini"

    @property
    def model_name(self) -> str:
        """Configured default model identifier."""
        return self._model

    @property
    def capabilities(self) -> set[ModelCapability]:
        """Capabilities provided by Gemini models."""
        return {
            ModelCapability.TEXT,
            ModelCapability.REASONING,
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.VISION,
            ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT,
        }

    def get_spec(self, model: str | None = None) -> ModelSpec:
        """Return the model specification for this Gemini provider."""
        target_model = model or self._model
        return ModelSpec(
            provider="gemini",
            model=target_model,
            capabilities=self.capabilities,
            context_limits=ContextLimits(
                max_input_tokens=1048576,
                max_output_tokens=8192,
                context_window=1048576,
            ),
            latency_tier="low",
            modalities=["text", "image", "audio"],
            available=self._has_valid_api_key(),
            metadata={"endpoint": self._base_url},
        )

    def _has_valid_api_key(self) -> bool:
        """Check if an API key is present and non-empty."""
        return bool(self._api_key and self._api_key.strip())

    async def is_available(self) -> bool:
        """Check whether the Gemini provider is available.

        Requires an API key to be configured.
        """
        return self._has_valid_api_key()

    def _sanitize(self, text: str) -> str:
        """Mask the API key if it appears anywhere in an error message or string."""
        if not self._api_key:
            return text
        sanitized = text.replace(self._api_key, "***REDACTED***")
        return sanitized

    def _build_payload(
        self,
        request: ModelRequest,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Construct the Gemini REST API JSON payload."""
        contents: list[dict[str, Any]] = []

        if request.messages:
            for msg in request.messages:
                role = "user" if msg.role in ("user", "system") else "model"
                contents.append({"role": role, "parts": [{"text": msg.content}]})
        else:
            prompt_text = request.get_prompt_text()
            contents.append({"role": "user", "parts": [{"text": prompt_text}]})

        payload: dict[str, Any] = {"contents": contents}

        # System instruction
        if request.system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": request.system_prompt}]
            }

        # Generation config
        gen_config: dict[str, Any] = {}
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            gen_config["maxOutputTokens"] = request.max_tokens
        if request.stop:
            gen_config["stopSequences"] = request.stop

        if response_schema is not None:
            gen_config["responseMimeType"] = "application/json"
            gen_config["responseSchema"] = response_schema

        if gen_config:
            payload["generationConfig"] = gen_config

        return payload

    def _execute_http(
        self,
        endpoint_url: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        """Perform HTTP request using configured transport or urllib."""
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint_url,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key or "",
            },
            method="POST",
        )

        if self._transport is not None:
            return self._transport(req, timeout)

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)

    async def _send_request(
        self,
        model: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, float, str | None, str]:
        """Send asynchronous request to Gemini and return (data, latency_ms, error, finish_reason)."""
        if not self._has_valid_api_key():
            return None, 0.0, "Gemini API key is not configured.", "error"

        url = f"{self._base_url}/models/{model}:generateContent"
        start_time = time.perf_counter()

        try:
            raw_response = await asyncio.to_thread(
                self._execute_http, url, payload, self._timeout
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return raw_response, latency_ms, None, "stop"

        except TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"Gemini request timed out after {self._timeout}s."
            logger.warning("Gemini timeout: %s", err_msg)
            return None, latency_ms, err_msg, "timeout"

        except urllib.error.HTTPError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            try:
                error_body = exc.read().decode("utf-8")
                parsed = json.loads(error_body)
                server_msg = parsed.get("error", {}).get("message", str(exc))
            except Exception:
                server_msg = str(exc)
            safe_msg = self._sanitize(f"Gemini API error ({exc.code}): {server_msg}")
            logger.warning("Gemini HTTP error: %s", safe_msg)
            return None, latency_ms, safe_msg, "error"

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            safe_msg = self._sanitize(f"Gemini request failed: {exc}")
            logger.warning("Gemini request exception: %s", safe_msg)
            return None, latency_ms, safe_msg, "error"

    def _normalize_response(
        self,
        raw_data: dict[str, Any] | None,
        model: str,
        request_id: str,
        latency_ms: float,
        error: str | None,
        finish_reason_fallback: str,
    ) -> ModelResponse:
        """Convert raw Gemini JSON response into typed ModelResponse."""
        if error is not None or raw_data is None:
            return ModelResponse(
                content="",
                model=model,
                provider="gemini",
                request_id=request_id,
                success=False,
                error=error or "Unknown error",
                finish_reason=finish_reason_fallback,
                metadata=ModelMetadata(
                    latency_ms=latency_ms,
                    finish_reason=finish_reason_fallback,
                ),
            )

        # Parse text candidate
        candidates = raw_data.get("candidates", [])
        content_text = ""
        gemini_finish = "STOP"

        if candidates:
            first = candidates[0]
            gemini_finish = first.get("finishReason", "STOP")
            parts = first.get("content", {}).get("parts", [])
            if parts:
                content_text = parts[0].get("text", "")

        finish_reason = self._FINISH_REASON_MAP.get(gemini_finish, "stop")

        # Parse token usage
        usage_meta = raw_data.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        completion_tokens = usage_meta.get("candidatesTokenCount", 0)
        total_tokens = usage_meta.get("totalTokenCount", prompt_tokens + completion_tokens)

        return ModelResponse(
            content=content_text,
            model=model,
            provider="gemini",
            request_id=request_id,
            success=True,
            finish_reason=finish_reason,
            usage=ModelUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            metadata=ModelMetadata(
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                raw={"finishReason": gemini_finish},
            ),
        )

    async def generate(
        self,
        request: ModelRequest | str,
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> ModelResponse | str:
        """Generate response from Gemini model."""
        if isinstance(request, str):
            model_req = ModelRequest(prompt=request, system_prompt=system_prompt)
            resp = await self.generate(model_req)
            if isinstance(resp, ModelResponse):
                return resp.content
            return str(resp)

        target_model = request.model or self._model
        payload = self._build_payload(request)

        raw_data, latency_ms, error, finish_reason = await self._send_request(
            target_model, payload
        )

        return self._normalize_response(
            raw_data,
            model=target_model,
            request_id=request.request_id,
            latency_ms=latency_ms,
            error=error,
            finish_reason_fallback=finish_reason,
        )

    async def generate_structured(
        self,
        request: ModelRequest,
        schema: type[T],
    ) -> ModelResponse:
        """Generate structured output validated against a Pydantic schema."""
        target_model = request.model or self._model

        # Extract JSON schema if Pydantic model
        schema_dict: dict[str, Any] | None = None
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                schema_dict = schema.model_json_schema()
            except Exception as exc:
                logger.warning(
                    "Failed to extract JSON schema from %s: %s",
                    schema,
                    exc,
                )
                schema_dict = None

        payload = self._build_payload(request, response_schema=schema_dict)

        raw_data, latency_ms, error, finish_reason = await self._send_request(
            target_model, payload
        )

        normalized = self._normalize_response(
            raw_data,
            model=target_model,
            request_id=request.request_id,
            latency_ms=latency_ms,
            error=error,
            finish_reason_fallback=finish_reason,
        )

        if not normalized.success:
            return normalized

        # Validate structured output against Pydantic schema
        try:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                validated = schema.model_validate_json(normalized.content)
                normalized.structured_output = validated
            else:
                parsed = json.loads(normalized.content)
                normalized.structured_output = parsed
            return normalized

        except (ValidationError, json.JSONDecodeError, Exception) as exc:
            safe_err = f"Structured output validation failed: {exc}"
            logger.warning("Gemini structured validation error: %s", safe_err)
            return ModelResponse(
                content=normalized.content,
                model=target_model,
                provider="gemini",
                request_id=request.request_id,
                success=False,
                error=safe_err,
                finish_reason="error",
                usage=normalized.usage,
                metadata=normalized.metadata,
            )
