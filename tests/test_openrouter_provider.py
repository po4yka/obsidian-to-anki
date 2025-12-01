"""Tests for OpenRouterProvider structured output fallbacks."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from obsidian_anki_sync.agents.json_schemas import get_qa_extraction_schema
from obsidian_anki_sync.providers.openrouter import OpenRouterProvider

BASE_URL = "https://mock.openrouter.ai/api/v1"


@pytest.fixture()
def openrouter_provider() -> OpenRouterProvider:
    """Provide an OpenRouterProvider instance configured for tests."""
    return OpenRouterProvider(
        api_key="test-api-key",
        base_url=BASE_URL,
        timeout=5.0,
        max_tokens=4096,
    )


def _build_openrouter_response(
    *,
    content: str,
    finish_reason: str = "stop",
) -> dict[str, object]:
    """Helper to craft OpenRouter-style responses."""
    return {
        "id": "mock-response",
        "model": "qwen/qwen-2.5-72b-instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning": None,
                    "refusal": None,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100},
    }


@respx.mock
def test_generate_json_falls_back_when_model_returns_empty_completion(
    openrouter_provider: OpenRouterProvider,
) -> None:
    """Provider retries without schema when structured output returns empty content."""
    responses = [
        httpx.Response(200, json=_build_openrouter_response(content="")),
        httpx.Response(
            200,
            json=_build_openrouter_response(
                content=json.dumps(
                    {
                        "qa_pairs": [
                            {
                                "card_index": 1,
                                "question_en": "Q?",
                                "question_ru": "",
                                "answer_en": "Answer",
                                "answer_ru": "",
                                "context": "",
                                "followups": "",
                                "references": "",
                                "related": "",
                            }
                        ],
                        "extraction_notes": "",
                        "total_pairs": 1,
                    }
                )
            ),
        ),
    ]

    route = respx.post(f"{BASE_URL}/chat/completions")
    route.mock(side_effect=responses)

    schema = get_qa_extraction_schema()
    result = openrouter_provider.generate_json(
        model="qwen/qwen-2.5-72b-instruct",
        prompt="Extract QAs",
        system="system prompt",
        temperature=0.0,
        json_schema=schema,
    )

    assert result["total_pairs"] == 1
    assert route.call_count == 2

    first_payload = json.loads(route.calls[0].request.content.decode())
    second_payload = json.loads(route.calls[1].request.content.decode())

    # For models that support response_format, verify it's used correctly
    # Some models (like Qwen) may skip response_format entirely
    if "response_format" in first_payload:
        assert first_payload["response_format"]["type"] == "json_schema"
    if "response_format" in second_payload:
        assert second_payload["response_format"]["type"] == "json_object"


@respx.mock
def test_generate_json_raises_when_fallback_also_returns_empty(
    openrouter_provider: OpenRouterProvider,
) -> None:
    """Provider surfaces an error if both structured and fallback calls return empty content."""
    route = respx.post(f"{BASE_URL}/chat/completions")
    route.mock(
        return_value=httpx.Response(200, json=_build_openrouter_response(content=""))
    )

    schema = get_qa_extraction_schema()

    with pytest.raises(ValueError, match="returned empty completion"):
        openrouter_provider.generate_json(
            model="qwen/qwen-2.5-72b-instruct",
            prompt="Extract QAs",
            system="system prompt",
            temperature=0.0,
            json_schema=schema,
        )

    # Verify it made at least 2 calls (structured + fallback)
    assert route.call_count >= 2


@respx.mock
def test_generate_retries_on_rate_limit_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    openrouter_provider: OpenRouterProvider,
) -> None:
    """Provider waits according to Retry-After and retries 429 responses."""

    recorded_sleeps: list[float] = []
    monkeypatch.setattr(
        "obsidian_anki_sync.providers.openrouter.provider.time.sleep",
        lambda seconds: recorded_sleeps.append(seconds),
    )

    responses = [
        httpx.Response(
            429,
            json={
                "error": {
                    "type": "rate_limit",
                    "message": "Too many requests",
                }
            },
            headers={"Retry-After": "0.1"},
        ),
        httpx.Response(
            200,
            json=_build_openrouter_response(
                content=json.dumps(
                    {
                        "qa_pairs": [
                            {
                                "card_index": 1,
                                "question_en": "Q?",
                                "question_ru": "",
                                "answer_en": "Answer",
                                "answer_ru": "",
                                "context": "",
                                "followups": "",
                                "references": "",
                                "related": "",
                            }
                        ],
                        "extraction_notes": "",
                        "total_pairs": 1,
                    }
                )
            ),
        ),
    ]

    route = respx.post(f"{BASE_URL}/chat/completions")
    route.mock(side_effect=responses)

    schema = get_qa_extraction_schema()
    result = openrouter_provider.generate_json(
        model="qwen/qwen-2.5-72b-instruct",
        prompt="Extract QAs",
        system="system prompt",
        temperature=0.0,
        json_schema=schema,
    )

    assert result["total_pairs"] == 1
    assert route.call_count == 2
    assert recorded_sleeps == [pytest.approx(0.1, rel=1e-2)]


@respx.mock
def test_grok_reasoning_enabled_for_complex_schema(
    openrouter_provider: OpenRouterProvider,
) -> None:
    """Grok models enable reasoning for complex JSON schemas."""
    route = respx.post(f"{BASE_URL}/chat/completions")
    route.respond(
        200,
        json=_build_openrouter_response(
            content=json.dumps(
                {
                    "qa_pairs": [
                        {
                            "card_index": 1,
                            "question_en": "What is Python?",
                            "question_ru": "",
                            "answer_en": "A programming language",
                            "answer_ru": "",
                            "context": "Python is a high-level programming language",
                            "followups": "What are Python's key features?",
                            "references": "Python documentation",
                            "related": "JavaScript, Java",
                        }
                    ],
                    "extraction_notes": "Complex extraction completed",
                    "total_pairs": 1,
                }
            )
        ),
    )

    # Complex schema with many properties and nested structures
    complex_schema = {
        "name": "qa_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "qa_pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "card_index": {"type": "integer"},
                            "question_en": {"type": "string"},
                            "question_ru": {"type": "string"},
                            "answer_en": {"type": "string"},
                            "answer_ru": {"type": "string"},
                            "context": {"type": "string"},
                            "followups": {"type": "string"},
                            "references": {"type": "string"},
                            "related": {"type": "string"},
                        },
                        "required": ["card_index", "question_en", "answer_en"],
                    },
                },
                "extraction_notes": {"type": "string"},
                "total_pairs": {"type": "integer"},
            },
            "required": ["qa_pairs", "total_pairs"],
        },
    }

    result = openrouter_provider.generate_json(
        model="x-ai/grok-4.1-fast",
        prompt="Extract complex QAs from this text",
        system="You are an expert at QA extraction",
        temperature=0.0,
        json_schema=complex_schema,
        reasoning_enabled=False,  # Should be auto-enabled due to complexity
    )

    assert result["total_pairs"] == 1
    assert route.call_count == 1

    payload = json.loads(route.calls[0].request.content.decode())
    # Should be enabled for complex schema
    assert payload["reasoning"]["enabled"] is True


@respx.mock
def test_grok_reasoning_disabled_for_simple_schema(
    openrouter_provider: OpenRouterProvider,
) -> None:
    """Grok models disable reasoning for simple JSON schemas."""
    route = respx.post(f"{BASE_URL}/chat/completions")
    route.respond(
        200,
        json=_build_openrouter_response(
            content=json.dumps(
                {"answer": "The capital of France is Paris.", "confidence": 0.95}
            )
        ),
    )

    # Simple schema with few properties
    simple_schema = {
        "name": "simple_answer",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer"],
        },
    }

    result = openrouter_provider.generate_json(
        model="x-ai/grok-4.1-fast",
        prompt="What is the capital of France?",
        system="Answer questions directly",
        temperature=0.0,
        json_schema=simple_schema,
        reasoning_enabled=False,
    )

    assert result["answer"] == "The capital of France is Paris."
    assert route.call_count == 1

    payload = json.loads(route.calls[0].request.content.decode())
    # Should be disabled for simple schema
    assert payload["reasoning"]["enabled"] is False


@respx.mock
def test_grok_explicit_reasoning_enabled_parameter(
    openrouter_provider: OpenRouterProvider,
) -> None:
    """Grok models respect explicit reasoning_enabled parameter."""
    route = respx.post(f"{BASE_URL}/chat/completions")
    route.respond(
        200,
        json=_build_openrouter_response(
            content=json.dumps({"result": "Analysis complete"})
        ),
    )

    result = openrouter_provider.generate(
        model="x-ai/grok-4.1-fast",
        prompt="Analyze this text",
        system="Be analytical",
        temperature=0.0,
        reasoning_enabled=True,  # Explicitly enabled
    )

    assert result["response"] == '{"result": "Analysis complete"}'
    assert route.call_count == 1

    payload = json.loads(route.calls[0].request.content.decode())
    # Should respect explicit parameter
    assert payload["reasoning"]["enabled"] is True
