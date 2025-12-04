"""Comprehensive unit tests for LLM providers.

Tests cover:
- Provider factory creation
- Ollama provider behavior
- Base provider interface
- Error handling scenarios
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.providers.base import BaseLLMProvider
from obsidian_anki_sync.providers.factory import ProviderFactory
from obsidian_anki_sync.providers.ollama import OllamaProvider


@pytest.fixture
def temp_dir(tmp_path):
    """Alias for pytest's tmp_path fixture."""
    return tmp_path


# Test Provider Factory


def test_factory_creates_ollama_provider():
    """Factory creates OllamaProvider for 'ollama' type."""
    provider = ProviderFactory.create_provider(
        "ollama", base_url="http://localhost:11434"
    )
    assert isinstance(provider, OllamaProvider)
    assert provider.base_url == "http://localhost:11434"


def test_factory_creates_ollama_provider_case_insensitive():
    """Factory handles case-insensitive provider type."""
    provider = ProviderFactory.create_provider(
        "OLLAMA", base_url="http://localhost:11434"
    )
    assert isinstance(provider, OllamaProvider)


def test_factory_creates_ollama_with_api_key():
    """Factory creates Ollama provider with API key for cloud deployment."""
    provider = ProviderFactory.create_provider(
        "ollama", base_url="https://api.ollama.com", api_key="test-key"
    )
    assert isinstance(provider, OllamaProvider)
    assert provider.api_key == "test-key"


def test_factory_raises_error_for_invalid_provider():
    """Factory raises ValueError for unsupported provider type."""
    with pytest.raises(ValueError, match="Unsupported provider type: invalid_provider"):
        ProviderFactory.create_provider("invalid_provider")


def test_factory_create_from_config_ollama(temp_dir):
    """Factory extracts correct kwargs from config for Ollama."""
    vault_path = temp_dir / "vault"
    vault_path.mkdir(parents=True, exist_ok=True)
    source_dir = vault_path / "questions"
    source_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        vault_path=vault_path,
        source_dir=source_dir,
        anki_connect_url="http://localhost:8765",
        anki_deck_name="Test",
        anki_note_type="APF::Simple",
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_api_key="test-key",
        llm_timeout=120.0,
        db_path=temp_dir / "test.db",
    )

    provider = ProviderFactory.create_from_config(config, verify_connectivity=False)

    assert isinstance(provider, OllamaProvider)
    assert provider.base_url == "http://localhost:11434"
    assert provider.api_key == "test-key"
    assert provider.timeout == 120.0


def test_factory_list_supported_providers():
    """Factory lists all supported provider types."""
    providers = ProviderFactory.list_supported_providers()

    assert "ollama" in providers
    assert "openrouter" in providers
    assert "openai" in providers
    assert "anthropic" in providers
    assert "claude" in providers
    assert "lm_studio" in providers
    assert "lmstudio" in providers


# Test Base Provider Interface


def test_base_provider_safe_config_redacts_api_key():
    """Base provider redacts sensitive fields in config for logging."""

    class TestProvider(BaseLLMProvider):
        def generate(self, model, prompt, **kwargs):
            return {"response": "test"}

        def check_connection(self):
            return True

        def list_models(self):
            return []

    provider = TestProvider(api_key="secret-key-123", base_url="http://test.com")

    safe_config = provider._safe_config_for_logging()

    assert safe_config["api_key"] == "***REDACTED***"
    assert safe_config["base_url"] == "http://test.com"


def test_base_provider_safe_config_redacts_token():
    """Base provider redacts token field."""

    class TestProvider(BaseLLMProvider):
        def generate(self, model, prompt, **kwargs):
            return {"response": "test"}

        def check_connection(self):
            return True

        def list_models(self):
            return []

    provider = TestProvider(token="secret-token-456")

    safe_config = provider._safe_config_for_logging()

    assert safe_config["token"] == "***REDACTED***"


def test_base_provider_safe_config_redacts_password():
    """Base provider redacts password field."""

    class TestProvider(BaseLLMProvider):
        def generate(self, model, prompt, **kwargs):
            return {"response": "test"}

        def check_connection(self):
            return True

        def list_models(self):
            return []

    provider = TestProvider(password="secret-pass-789")

    safe_config = provider._safe_config_for_logging()

    assert safe_config["password"] == "***REDACTED***"


def test_base_provider_get_provider_name():
    """Base provider returns human-readable name."""

    class TestProvider(BaseLLMProvider):
        def generate(self, model, prompt, **kwargs):
            return {"response": "test"}

        def check_connection(self):
            return True

        def list_models(self):
            return []

    provider = TestProvider()

    assert provider.get_provider_name() == "Test"


def test_base_provider_generate_json_success():
    """Base provider generate_json parses valid JSON response."""

    class TestProvider(BaseLLMProvider):
        def generate(self, model, prompt, **kwargs):
            return {"response": '{"key": "value", "count": 42}'}

        def check_connection(self):
            return True

        def list_models(self):
            return []

    provider = TestProvider()
    result = provider.generate_json(model="test", prompt="test")

    assert result == {"key": "value", "count": 42}


def test_base_provider_generate_json_empty_response_raises_error():
    """Base provider generate_json raises error for empty JSON."""

    class TestProvider(BaseLLMProvider):
        def generate(self, model, prompt, **kwargs):
            return {"response": "{}"}

        def check_connection(self):
            return True

        def list_models(self):
            return []

    provider = TestProvider()

    with pytest.raises(ValueError, match="returned empty JSON response"):
        provider.generate_json(model="test", prompt="test")


def test_base_provider_generate_json_invalid_json_raises_error():
    """Base provider generate_json raises JSONDecodeError for malformed JSON."""

    class TestProvider(BaseLLMProvider):
        def generate(self, model, prompt, **kwargs):
            return {"response": "not valid json {"}

        def check_connection(self):
            return True

        def list_models(self):
            return []

    provider = TestProvider()

    with pytest.raises(json.JSONDecodeError):
        provider.generate_json(model="test", prompt="test")


# Test Ollama Provider


@pytest.fixture
def ollama_provider():
    """Provide an OllamaProvider instance for tests."""
    return OllamaProvider(
        base_url="http://localhost:11434", timeout=5.0, enable_safety=False
    )


@respx.mock
def test_ollama_generate_success(ollama_provider):
    """Ollama provider successfully generates completion."""
    mock_response = {
        "model": "llama2",
        "response": "Generated text response",
        "done": True,
        "context": [1, 2, 3],
        "eval_count": 50,
        "eval_duration": 1000000000,
        "prompt_eval_count": 20,
        "prompt_eval_duration": 500000000,
    }

    route = respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = ollama_provider.generate(model="llama2", prompt="Hello world")

    assert result["response"] == "Generated text response"
    assert result["done"] is True
    assert result["context"] == [1, 2, 3]
    assert result["_token_usage"]["completion_tokens"] == 50
    assert result["_token_usage"]["prompt_tokens"] == 20
    assert result["_token_usage"]["total_tokens"] == 70
    assert route.call_count == 1


@respx.mock
def test_ollama_generate_with_system_prompt(ollama_provider):
    """Ollama provider includes system prompt in request."""
    mock_response = {
        "response": "System-guided response",
        "done": True,
        "eval_count": 30,
        "eval_duration": 1000000000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 500000000,
    }

    route = respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = ollama_provider.generate(
        model="llama2", prompt="User prompt", system="System prompt"
    )

    assert result["response"] == "System-guided response"
    assert route.call_count == 1

    request_payload = json.loads(route.calls[0].request.content.decode())
    assert request_payload["system"] == "System prompt"


@respx.mock
def test_ollama_generate_with_json_format(ollama_provider):
    """Ollama provider requests JSON format when specified."""
    mock_response = {
        "response": '{"key": "value"}',
        "done": True,
        "eval_count": 10,
        "eval_duration": 1000000000,
        "prompt_eval_count": 5,
        "prompt_eval_duration": 500000000,
    }

    route = respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = ollama_provider.generate(
        model="llama2", prompt="Generate JSON", format="json"
    )

    assert result["response"] == '{"key": "value"}'
    assert route.call_count == 1

    request_payload = json.loads(route.calls[0].request.content.decode())
    assert request_payload["format"] == "json"


@respx.mock
def test_ollama_generate_with_temperature(ollama_provider):
    """Ollama provider includes temperature in request."""
    mock_response = {
        "response": "Creative response",
        "done": True,
        "eval_count": 10,
        "eval_duration": 1000000000,
        "prompt_eval_count": 5,
        "prompt_eval_duration": 500000000,
    }

    route = respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = ollama_provider.generate(model="llama2", prompt="Test", temperature=0.9)

    assert result["response"] == "Creative response"
    assert route.call_count == 1

    request_payload = json.loads(route.calls[0].request.content.decode())
    assert request_payload["options"]["temperature"] == 0.9


@respx.mock
def test_ollama_check_connection_success(ollama_provider):
    """Ollama provider check_connection returns True when accessible."""
    mock_response = {"models": []}

    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = ollama_provider.check_connection()

    assert result is True


@respx.mock
def test_ollama_check_connection_failure(ollama_provider):
    """Ollama provider check_connection returns False on error."""
    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    result = ollama_provider.check_connection()

    assert result is False


@respx.mock
def test_ollama_list_models_success(ollama_provider):
    """Ollama provider lists available models."""
    mock_response = {
        "models": [
            {"name": "llama2:7b"},
            {"name": "qwen:14b"},
            {"name": "mistral:latest"},
        ]
    }

    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = ollama_provider.list_models()

    assert len(result) == 3
    assert "llama2:7b" in result
    assert "qwen:14b" in result
    assert "mistral:latest" in result


@respx.mock
def test_ollama_list_models_failure(ollama_provider):
    """Ollama provider returns empty list on list_models error."""
    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    result = ollama_provider.list_models()

    assert result == []


# Test Ollama Error Handling


@respx.mock
def test_ollama_generate_http_500_error(ollama_provider):
    """Ollama provider raises HTTPStatusError on server error."""
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(500, json={"error": "Internal server error"})
    )

    with pytest.raises(httpx.HTTPStatusError):
        ollama_provider.generate(model="llama2", prompt="Test")


@respx.mock
def test_ollama_generate_http_404_error(ollama_provider):
    """Ollama provider raises HTTPStatusError on not found error."""
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(404, json={"error": "Model not found"})
    )

    with pytest.raises(httpx.HTTPStatusError):
        ollama_provider.generate(model="nonexistent-model", prompt="Test")


@respx.mock
def test_ollama_generate_connection_error(ollama_provider):
    """Ollama provider raises RequestError on connection failure."""
    respx.post("http://localhost:11434/api/generate").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    with pytest.raises(httpx.RequestError):
        ollama_provider.generate(model="llama2", prompt="Test")


@respx.mock
def test_ollama_generate_timeout_error(ollama_provider):
    """Ollama provider raises TimeoutException on request timeout."""
    respx.post("http://localhost:11434/api/generate").mock(
        side_effect=httpx.TimeoutException("Request timed out")
    )

    with pytest.raises(httpx.TimeoutException):
        ollama_provider.generate(model="llama2", prompt="Test")


@respx.mock
def test_ollama_generate_malformed_json_response(ollama_provider):
    """Ollama provider handles malformed JSON gracefully."""
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(200, content=b"This is not valid JSON {")
    )

    with pytest.raises(json.JSONDecodeError):
        ollama_provider.generate(model="llama2", prompt="Test")


@respx.mock
def test_ollama_pull_model_success(ollama_provider):
    """Ollama provider successfully pulls model."""
    respx.post("http://localhost:11434/api/pull").mock(
        return_value=httpx.Response(200, json={"status": "success"})
    )

    result = ollama_provider.pull_model("llama2:7b")

    assert result is True


@respx.mock
def test_ollama_pull_model_failure(ollama_provider):
    """Ollama provider returns False on pull_model error."""
    respx.post("http://localhost:11434/api/pull").mock(
        return_value=httpx.Response(404, json={"error": "Model not found"})
    )

    result = ollama_provider.pull_model("nonexistent-model")

    assert result is False


def test_ollama_generate_streaming_not_implemented(ollama_provider):
    """Ollama provider raises NotImplementedError for streaming."""
    with pytest.raises(NotImplementedError, match="Streaming is not yet supported"):
        ollama_provider.generate(model="llama2", prompt="Test", stream=True)


# Test Ollama Context Manager


@respx.mock
def test_ollama_context_manager_closes_client():
    """Ollama provider closes HTTP client when used as context manager."""
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": []})
    )

    with OllamaProvider(
        base_url="http://localhost:11434", enable_safety=False
    ) as provider:
        assert provider.check_connection() is True
        assert hasattr(provider, "client")

    # After exiting context, client should be closed
    # (Note: we can't directly test if client.is_closed since it's internal)


# Test Ollama Performance Metrics


@respx.mock
def test_ollama_generate_includes_token_usage_metrics(ollama_provider):
    """Ollama provider includes token usage metrics in response."""
    mock_response = {
        "response": "Test response",
        "done": True,
        "eval_count": 100,
        "eval_duration": 2000000000,
        "prompt_eval_count": 50,
        "prompt_eval_duration": 1000000000,
    }

    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = ollama_provider.generate(model="llama2", prompt="Test")

    assert "_token_usage" in result
    assert result["_token_usage"]["prompt_tokens"] == 50
    assert result["_token_usage"]["completion_tokens"] == 100
    assert result["_token_usage"]["total_tokens"] == 150


# Test Factory Edge Cases


def test_factory_create_provider_with_extra_kwargs():
    """Factory passes through extra kwargs to provider."""
    provider = ProviderFactory.create_provider(
        "ollama",
        base_url="http://localhost:11434",
        timeout=300.0,
        enable_safety=False,
    )

    assert isinstance(provider, OllamaProvider)
    assert provider.timeout == 300.0
    assert provider.enable_safety is False


def test_factory_create_from_config_uses_default_provider_ollama(temp_dir):
    """Factory uses default provider (ollama) when not explicitly specified."""
    vault_path = temp_dir / "vault"
    vault_path.mkdir(parents=True, exist_ok=True)
    source_dir = vault_path / "questions"
    source_dir.mkdir(parents=True, exist_ok=True)

    # Create config without explicitly setting llm_provider (uses default "ollama")
    config = Config(
        vault_path=vault_path,
        source_dir=source_dir,
        anki_connect_url="http://localhost:8765",
        anki_deck_name="Test",
        anki_note_type="APF::Simple",
        db_path=temp_dir / "test.db",
    )

    # Verify default is ollama
    assert config.llm_provider == "ollama"

    provider = ProviderFactory.create_from_config(config, verify_connectivity=False)

    assert isinstance(provider, OllamaProvider)
