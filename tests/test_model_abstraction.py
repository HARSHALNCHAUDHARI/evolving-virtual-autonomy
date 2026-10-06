"""Tests for model provider request and response abstractions."""

import pytest
from pydantic import BaseModel

from eva.providers.models import (
    ChatMessage,
    ModelMetadata,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ProviderError,
    ProviderNotFoundError,
    RouteRequest,
    StructuredOutputError,
    TaskType,
)


class TestModelRequest:
    """Validation and behavior of ModelRequest."""

    def test_create_with_prompt_string(self):
        req = ModelRequest(prompt="Hello EVA", model="test-model")
        assert req.prompt == "Hello EVA"
        assert req.get_prompt_text() == "Hello EVA"
        assert req.model == "test-model"
        assert req.request_id is not None
        assert req.messages == []

    def test_create_with_messages(self):
        msgs = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is the weather?"),
        ]
        req = ModelRequest(messages=msgs)
        assert len(req.messages) == 2
        assert req.get_prompt_text() == "What is the weather?"

    def test_empty_request_prompt_text(self):
        req = ModelRequest()
        assert req.get_prompt_text() == ""

    def test_parameters_and_metadata(self):
        req = ModelRequest(
            prompt="Analyze input",
            provider="mock",
            temperature=0.7,
            max_tokens=256,
            stop=["\n\n"],
            system_prompt="System instructions",
            metadata={"priority": "high"},
            request_id="custom-req-123",
        )
        assert req.provider == "mock"
        assert req.temperature == 0.7
        assert req.max_tokens == 256
        assert req.stop == ["\n\n"]
        assert req.system_prompt == "System instructions"
        assert req.metadata["priority"] == "high"
        assert req.request_id == "custom-req-123"


class TestModelUsage:
    """Token metrics calculation in ModelUsage."""

    def test_explicit_tokens(self):
        usage = ModelUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40)
        assert usage.prompt_tokens == 15
        assert usage.completion_tokens == 25
        assert usage.total_tokens == 40

    def test_auto_compute_total_tokens(self):
        usage = ModelUsage(prompt_tokens=10, completion_tokens=20)
        assert usage.total_tokens == 30

    def test_default_zeros(self):
        usage = ModelUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestModelMetadata:
    """ModelMetadata representation."""

    def test_metadata_defaults_and_fields(self):
        meta = ModelMetadata(
            latency_ms=12.5,
            finish_reason="stop",
            raw={"raw_key": "raw_val"},
            extra={"cached": True},
        )
        assert meta.latency_ms == 12.5
        assert meta.finish_reason == "stop"
        assert meta.raw["raw_key"] == "raw_val"
        assert meta.extra["cached"] is True


class DummySchema(BaseModel):
    title: str
    count: int


class TestModelResponse:
    """ModelResponse structure and validation."""

    def test_default_response_creation(self):
        resp = ModelResponse(
            content="Generated answer",
            model="mock-v1",
            provider="mock",
        )
        assert resp.content == "Generated answer"
        assert resp.model == "mock-v1"
        assert resp.provider == "mock"
        assert resp.success is True
        assert resp.error is None
        assert resp.structured_output is None
        assert resp.response_id is not None
        assert resp.finish_reason == "stop"

    def test_structured_response_creation(self):
        dummy = DummySchema(title="Test", count=42)
        resp = ModelResponse(
            content='{"title": "Test", "count": 42}',
            model="mock-v1",
            provider="mock",
            structured_output=dummy,
            success=True,
        )
        assert resp.structured_output == dummy
        assert resp.structured_output.title == "Test"
        assert resp.structured_output.count == 42

    def test_failure_response(self):
        resp = ModelResponse(
            model="mock-v1",
            provider="mock",
            success=False,
            error="Model rate limited",
            finish_reason="error",
        )
        assert resp.success is False
        assert resp.error == "Model rate limited"
        assert resp.finish_reason == "error"


class TestRoutingTypesAndExceptions:
    """TaskType, RouteRequest, and exception hierarchies."""

    def test_task_type_values(self):
        assert TaskType.CONVERSATION.value == "conversation"
        assert TaskType.UNDERSTANDING.value == "understanding"
        assert TaskType.REASONING.value == "reasoning"
        assert TaskType.STRUCTURED_OUTPUT.value == "structured_output"
        assert TaskType.GENERAL.value == "general"

    def test_route_request_model(self):
        req = RouteRequest(
            task_type=TaskType.REASONING,
            provider="mock",
            model="mock-v1",
            capability="fast",
        )
        assert req.task_type == TaskType.REASONING
        assert req.provider == "mock"
        assert req.model == "mock-v1"
        assert req.capability == "fast"

    def test_provider_exceptions(self):
        assert issubclass(ProviderNotFoundError, ProviderError)
        assert issubclass(StructuredOutputError, ProviderError)
