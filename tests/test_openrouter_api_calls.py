"""Tests for OpenRouter atomic API calls.

These tests verify each API call function individually with mocked responses.
They cover success cases, error cases, and edge cases for:
- check_connection
- list_models
- fetch_key_status
- chat_completion
- chat_completion_with_tools
- chat_completion_structured
"""

import json

import httpx
import pytest
import respx

from obsidian_anki_sync.providers.openrouter.api_calls import (
    APICallResult,
    ChatCompletionResult,
    chat_completion,
    chat_completion_structured,
    chat_completion_with_tools,
    check_connection,
    create_openrouter_client,
    fetch_key_status,
    list_models,
)

BASE_URL = "https://mock.openrouter.ai/api/v1"


@pytest.fixture
def http_client() -> httpx.Client:
    """Create a test HTTP client."""
    return httpx.Client(timeout=httpx.Timeout(10.0))


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Create test authorization headers."""
    return {
        "Authorization": "Bearer test-api-key",
        "Content-Type": "application/json",
    }


# =============================================================================
# Tests for check_connection
# =============================================================================


class TestCheckConnection:
    """Tests for check_connection atomic API call."""

    @respx.mock
    def test_success(self, http_client: httpx.Client) -> None:
        """check_connection returns success when API is accessible."""
        respx.get(f"{BASE_URL}/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        result = check_connection(http_client, BASE_URL)

        assert result.success is True
        assert result.status_code == 200
        assert result.data == {"accessible": True}
        assert result.latency_ms > 0

    @respx.mock
    def test_server_error(self, http_client: httpx.Client) -> None:
        """check_connection returns failure on server error."""
        respx.get(f"{BASE_URL}/models").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        result = check_connection(http_client, BASE_URL)

        assert result.success is False
        assert result.status_code == 500

    @respx.mock
    def test_timeout(self, http_client: httpx.Client) -> None:
        """check_connection handles timeout gracefully."""
        respx.get(f"{BASE_URL}/models").mock(side_effect=httpx.TimeoutException("timeout"))

        result = check_connection(http_client, BASE_URL, timeout=1.0)

        assert result.success is False
        assert "timeout" in result.error.lower()

    @respx.mock
    def test_connection_error(self, http_client: httpx.Client) -> None:
        """check_connection handles connection errors."""
        respx.get(f"{BASE_URL}/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        result = check_connection(http_client, BASE_URL)

        assert result.success is False
        assert result.error is not None

    def test_result_boolean_conversion(self) -> None:
        """APICallResult can be used in boolean context."""
        success_result = APICallResult(success=True)
        failure_result = APICallResult(success=False)

        assert bool(success_result) is True
        assert bool(failure_result) is False


# =============================================================================
# Tests for list_models
# =============================================================================


class TestListModels:
    """Tests for list_models atomic API call."""

    @respx.mock
    def test_success(self, http_client: httpx.Client) -> None:
        """list_models returns model list on success."""
        mock_response = {
            "data": [
                {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini"},
                {"id": "anthropic/claude-3-haiku", "name": "Claude 3 Haiku"},
                {"id": "google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash"},
            ]
        }
        respx.get(f"{BASE_URL}/models").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = list_models(http_client, BASE_URL)

        assert result.success is True
        assert result.data["count"] == 3
        assert "openai/gpt-4o-mini" in result.data["models"]
        assert "anthropic/claude-3-haiku" in result.data["models"]

    @respx.mock
    def test_empty_models(self, http_client: httpx.Client) -> None:
        """list_models handles empty model list."""
        respx.get(f"{BASE_URL}/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        result = list_models(http_client, BASE_URL)

        assert result.success is True
        assert result.data["count"] == 0
        assert result.data["models"] == []

    @respx.mock
    def test_server_error(self, http_client: httpx.Client) -> None:
        """list_models returns failure on server error."""
        respx.get(f"{BASE_URL}/models").mock(
            return_value=httpx.Response(503, text="Service Unavailable")
        )

        result = list_models(http_client, BASE_URL)

        assert result.success is False
        assert result.status_code == 503

    @respx.mock
    def test_invalid_json(self, http_client: httpx.Client) -> None:
        """list_models handles invalid JSON response."""
        respx.get(f"{BASE_URL}/models").mock(
            return_value=httpx.Response(200, text="not valid json")
        )

        result = list_models(http_client, BASE_URL)

        assert result.success is False
        assert result.error is not None


# =============================================================================
# Tests for fetch_key_status
# =============================================================================


class TestFetchKeyStatus:
    """Tests for fetch_key_status atomic API call."""

    @respx.mock
    def test_success(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """fetch_key_status returns key info on success."""
        mock_response = {
            "data": {
                "label": "Test Key",
                "usage": 1000,
                "limit": 10000,
                "is_free_tier": False,
                "rate_limit": {"requests": 100, "interval": "minute"},
            }
        }
        respx.get(f"{BASE_URL}/auth/key").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = fetch_key_status(http_client, BASE_URL, auth_headers)

        assert result.success is True
        assert result.data["label"] == "Test Key"
        assert result.data["usage"] == 1000

    @respx.mock
    def test_unauthorized(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """fetch_key_status handles unauthorized response."""
        respx.get(f"{BASE_URL}/auth/key").mock(
            return_value=httpx.Response(401, json={"error": "Invalid API key"})
        )

        result = fetch_key_status(http_client, BASE_URL, auth_headers)

        assert result.success is False
        assert result.status_code == 401

    @respx.mock
    def test_free_tier(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """fetch_key_status correctly identifies free tier."""
        mock_response = {
            "data": {
                "label": "Free Tier Key",
                "usage": 0,
                "limit": 1000,
                "is_free_tier": True,
            }
        }
        respx.get(f"{BASE_URL}/auth/key").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = fetch_key_status(http_client, BASE_URL, auth_headers)

        assert result.success is True
        assert result.data["is_free_tier"] is True


# =============================================================================
# Tests for chat_completion
# =============================================================================


class TestChatCompletion:
    """Tests for chat_completion atomic API call."""

    @respx.mock
    def test_success(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """chat_completion returns content on success."""
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = chat_completion(
            client=http_client,
            base_url=BASE_URL,
            headers=auth_headers,
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result.success is True
        assert result.content == "Hello! How can I help you today?"
        assert result.finish_reason == "stop"
        assert result.model == "openai/gpt-4o-mini"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 8

    @respx.mock
    def test_with_system_message(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """chat_completion handles system + user messages."""
        mock_response = {
            "id": "chatcmpl-123",
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "I am a helpful assistant."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        }
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = chat_completion(
            client=http_client,
            base_url=BASE_URL,
            headers=auth_headers,
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who are you?"},
            ],
        )

        assert result.success is True
        assert "helpful assistant" in result.content.lower()

    @respx.mock
    def test_rate_limit_error(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """chat_completion handles rate limit errors."""
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                429,
                json={"error": {"message": "Rate limit exceeded", "code": "rate_limit"}},
            )
        )

        result = chat_completion(
            client=http_client,
            base_url=BASE_URL,
            headers=auth_headers,
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result.success is False
        assert result.status_code == 429
        assert "429" in result.error

    @respx.mock
    def test_empty_choices(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """chat_completion handles empty choices array."""
        mock_response = {
            "id": "chatcmpl-123",
            "model": "openai/gpt-4o-mini",
            "choices": [],
            "usage": {},
        }
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = chat_completion(
            client=http_client,
            base_url=BASE_URL,
            headers=auth_headers,
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result.success is False
        assert "No choices" in result.error

    @respx.mock
    def test_finish_reason_length(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """chat_completion captures length finish reason (truncation)."""
        mock_response = {
            "id": "chatcmpl-123",
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Truncated response..."},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110},
        }
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = chat_completion(
            client=http_client,
            base_url=BASE_URL,
            headers=auth_headers,
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Tell me a long story"}],
            max_tokens=100,
        )

        assert result.success is True
        assert result.finish_reason == "length"

    @respx.mock
    def test_json_format(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """chat_completion handles JSON response format."""
        json_content = json.dumps({"answer": "42", "confidence": 0.95})
        mock_response = {
            "id": "chatcmpl-123",
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "message": {"role": "assistant", "content": json_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
        }
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = chat_completion(
            client=http_client,
            base_url=BASE_URL,
            headers=auth_headers,
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Answer in JSON"}],
            response_format={"type": "json_object"},
        )

        assert result.success is True
        parsed = json.loads(result.content)
        assert parsed["answer"] == "42"


# =============================================================================
# Tests for chat_completion_with_tools
# =============================================================================


class TestChatCompletionWithTools:
    """Tests for chat_completion_with_tools atomic API call."""

    @respx.mock
    def test_tool_call_success(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """chat_completion_with_tools returns tool calls correctly."""
        mock_response = {
            "id": "chatcmpl-123",
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        }
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        result = chat_completion_with_tools(
            client=http_client,
            base_url=BASE_URL,
            headers=auth_headers,
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
            tools=tools,
        )

        assert result.success is True
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["function"]["arguments"])
        assert args["location"] == "San Francisco"

    @respx.mock
    def test_null_arguments_sanitization(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """chat_completion_with_tools sanitizes null function arguments."""
        # This simulates the malformed response from some models
        mock_response = {
            "id": "chatcmpl-123",
            "model": "qwen/qwen3-next-80b-a3b-instruct",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "final_result",
                                    "arguments": None,  # Malformed - should be string
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        }
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "final_result",
                    "description": "Return the final result",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = chat_completion_with_tools(
            client=http_client,
            base_url=BASE_URL,
            headers=auth_headers,
            model="qwen/qwen3-next-80b-a3b-instruct",
            messages=[{"role": "user", "content": "Generate output"}],
            tools=tools,
            tool_choice="required",
        )

        assert result.success is True
        # Arguments should be sanitized from None to "{}"
        assert result.tool_calls[0]["function"]["arguments"] == "{}"

    @respx.mock
    def test_no_tool_called(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """chat_completion_with_tools handles no tool being called."""
        mock_response = {
            "id": "chatcmpl-123",
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I don't need to use any tools for this.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        }
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = chat_completion_with_tools(
            client=http_client,
            base_url=BASE_URL,
            headers=auth_headers,
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=tools,
            tool_choice="auto",
        )

        assert result.success is True
        assert result.content is not None
        assert len(result.tool_calls) == 0

    @respx.mock
    def test_multiple_tool_calls(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """chat_completion_with_tools handles multiple parallel tool calls."""
        mock_response = {
            "id": "chatcmpl-123",
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "New York"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "London"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 60, "completion_tokens": 40, "total_tokens": 100},
        }
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        result = chat_completion_with_tools(
            client=http_client,
            base_url=BASE_URL,
            headers=auth_headers,
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Weather in NY and London?"}],
            tools=tools,
        )

        assert result.success is True
        assert len(result.tool_calls) == 2
        locations = [
            json.loads(tc["function"]["arguments"])["location"]
            for tc in result.tool_calls
        ]
        assert "New York" in locations
        assert "London" in locations


# =============================================================================
# Tests for chat_completion_structured
# =============================================================================


class TestChatCompletionStructured:
    """Tests for chat_completion_structured atomic API call."""

    @respx.mock
    def test_success(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """chat_completion_structured returns valid JSON."""
        json_content = json.dumps({"name": "John", "age": 30, "active": True})
        mock_response = {
            "id": "chatcmpl-123",
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "message": {"role": "assistant", "content": json_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
        }
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        json_schema = {
            "name": "user_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "active": {"type": "boolean"},
                },
                "required": ["name", "age", "active"],
            },
        }

        result = chat_completion_structured(
            client=http_client,
            base_url=BASE_URL,
            headers=auth_headers,
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Give me user info for John, 30, active"}],
            json_schema=json_schema,
        )

        assert result.success is True
        parsed = json.loads(result.content)
        assert parsed["name"] == "John"
        assert parsed["age"] == 30
        assert parsed["active"] is True

    @respx.mock
    def test_invalid_json_response(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """chat_completion_structured handles invalid JSON in response."""
        mock_response = {
            "id": "chatcmpl-123",
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is not valid JSON {broken",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
        }
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        json_schema = {
            "name": "test",
            "schema": {"type": "object", "properties": {}},
        }

        result = chat_completion_structured(
            client=http_client,
            base_url=BASE_URL,
            headers=auth_headers,
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Return JSON"}],
            json_schema=json_schema,
        )

        assert result.success is False
        assert "Invalid JSON" in result.error

    @respx.mock
    def test_nested_json_schema(
        self, http_client: httpx.Client, auth_headers: dict[str, str]
    ) -> None:
        """chat_completion_structured handles nested JSON schemas."""
        json_content = json.dumps(
            {
                "qa_pairs": [
                    {
                        "card_index": 1,
                        "question_en": "What is Python?",
                        "answer_en": "A programming language",
                    }
                ],
                "total_pairs": 1,
            }
        )
        mock_response = {
            "id": "chatcmpl-123",
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "message": {"role": "assistant", "content": json_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        }
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        json_schema = {
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
                                "answer_en": {"type": "string"},
                            },
                        },
                    },
                    "total_pairs": {"type": "integer"},
                },
            },
        }

        result = chat_completion_structured(
            client=http_client,
            base_url=BASE_URL,
            headers=auth_headers,
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Extract QA pairs"}],
            json_schema=json_schema,
        )

        assert result.success is True
        parsed = json.loads(result.content)
        assert len(parsed["qa_pairs"]) == 1
        assert parsed["qa_pairs"][0]["question_en"] == "What is Python?"


# =============================================================================
# Tests for create_openrouter_client
# =============================================================================


class TestCreateOpenrouterClient:
    """Tests for create_openrouter_client factory function."""

    def test_creates_client_and_headers(self) -> None:
        """create_openrouter_client returns configured client and headers."""
        client, headers = create_openrouter_client(
            api_key="test-key-123",
            site_url="https://example.com",
            site_name="Test App",
        )

        try:
            assert isinstance(client, httpx.Client)
            assert headers["Authorization"] == "Bearer test-key-123"
            assert headers["HTTP-Referer"] == "https://example.com"
            assert headers["X-Title"] == "Test App"
        finally:
            client.close()

    def test_minimal_configuration(self) -> None:
        """create_openrouter_client works with minimal config."""
        client, headers = create_openrouter_client(api_key="test-key")

        try:
            assert "Authorization" in headers
            assert "HTTP-Referer" not in headers
            assert "X-Title" not in headers
        finally:
            client.close()


# =============================================================================
# Tests for ChatCompletionResult dataclass
# =============================================================================


class TestChatCompletionResult:
    """Tests for ChatCompletionResult dataclass."""

    def test_default_values(self) -> None:
        """ChatCompletionResult has sensible defaults."""
        result = ChatCompletionResult(success=True)

        assert result.success is True
        assert result.content == ""
        assert result.finish_reason == ""
        assert result.model == ""
        assert result.usage == {}
        assert result.tool_calls == []

    def test_full_initialization(self) -> None:
        """ChatCompletionResult accepts all parameters."""
        result = ChatCompletionResult(
            success=True,
            data={"key": "value"},
            status_code=200,
            latency_ms=150.5,
            content="Hello world",
            finish_reason="stop",
            model="openai/gpt-4o-mini",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            tool_calls=[{"id": "call_1", "function": {"name": "test"}}],
        )

        assert result.content == "Hello world"
        assert result.latency_ms == 150.5
        assert result.usage["prompt_tokens"] == 10
        assert len(result.tool_calls) == 1
