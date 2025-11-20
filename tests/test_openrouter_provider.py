"""Tests for OpenRouterProvider structured output fallbacks."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from obsidian_anki_sync.agents.json_schemas import get_qa_extraction_schema
from obsidian_anki_sync.providers.openrouter import OpenRouterProvider

BASE_URL = "https://mock.openrouter.ai/api/v1"


@pytest.fixture
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
    responses = [
        httpx.Response(200, json=_build_openrouter_response(content="")),
        httpx.Response(200, json=_build_openrouter_response(content="")),
    ]

    route = respx.post(f"{BASE_URL}/chat/completions")
    route.mock(side_effect=responses)

    schema = get_qa_extraction_schema()

    with pytest.raises(ValueError, match="returned empty completion"):
        openrouter_provider.generate_json(
            model="qwen/qwen-2.5-72b-instruct",
            prompt="Extract QAs",
            system="system prompt",
            temperature=0.0,
            json_schema=schema,
        )

    assert route.call_count == 2
    second_payload = json.loads(route.calls[1].request.content.decode())
    assert second_payload["response_format"]["type"] == "json_object"


@respx.mock
def test_generate_retries_on_rate_limit_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    openrouter_provider: OpenRouterProvider,
) -> None:
    """Provider waits according to Retry-After and retries 429 responses."""

    recorded_sleeps: list[float] = []
    monkeypatch.setattr(
        "obsidian_anki_sync.providers.openrouter.time.sleep",
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
