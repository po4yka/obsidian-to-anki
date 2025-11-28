"""Mock implementation of ILLMProvider for testing."""

from typing import Any, Dict, List, Optional

from obsidian_anki_sync.domain.interfaces.llm_provider import ILLMProvider


class MockLLMProvider(ILLMProvider):
    """Mock implementation of LLM provider for testing.

    Provides controllable responses for testing agent operations
    without requiring real LLM API calls.
    """

    def __init__(self):
        """Initialize mock provider."""
        self.responses = {}  # prompt -> response mapping
        self.call_history = []  # List of (method, args) tuples
        self.should_fail = False
        self.fail_message = "Mock LLM provider failure"
        self.default_response = None  # Default response when prompt not found

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        format: str = "",
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        reasoning_enabled: bool = False,
    ) -> Dict[str, Any]:
        """Generate a completion."""
        self.call_history.append(("generate", {
            "model": model, "prompt": prompt, "system": system,
            "temperature": temperature, "format": format
        }))

        if self.should_fail:
            raise Exception(self.fail_message)

        # Return mock response
        response_text = self.responses.get(
            prompt, f"Mock response for: {prompt[:50]}...")
        return {
            "response": response_text,
            "model": model,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "finish_reason": "stop",
        }

    def generate_json(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        json_schema: Optional[Dict[str, Any]] = None,
        reasoning_enabled: bool = False,
    ) -> Dict[str, Any]:
        """Generate a JSON response."""
        self.call_history.append(("generate_json", {
            "model": model, "prompt": prompt, "system": system,
            "temperature": temperature, "json_schema": json_schema
        }))

        if self.should_fail:
            raise Exception(self.fail_message)

        # Return mock JSON response - check for exact match first, then default
        mock_json = self.responses.get(prompt)
        if mock_json is None and self.default_response is not None:
            mock_json = self.default_response
        if mock_json is None:
            mock_json = '{"mock": "response", "status": "success"}'

        # Try to parse as JSON, fallback to dict
        try:
            import json
            parsed = json.loads(mock_json)
        except Exception:
            parsed = {"mock": "response", "status": "success"}

        return parsed

    async def generate_async(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        format: str = "",
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        reasoning_enabled: bool = False,
    ) -> Dict[str, Any]:
        """Generate a completion asynchronously."""
        # Reuse synchronous implementation for mock
        return self.generate(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            format=format,
            json_schema=json_schema,
            stream=stream,
            reasoning_enabled=reasoning_enabled,
        )

    def check_connection(self) -> bool:
        """Check connection."""
        return not self.should_fail

    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status."""
        return {
            "connected": self.check_connection(),
            "latency": 0.0,
            "error": self.fail_message if self.should_fail else None,
        }

    def validate_credentials(self) -> bool:
        """Validate credentials."""
        return True

    def list_models(self) -> List[str]:
        """List available models."""
        return ["gpt-4", "gpt-3.5-turbo", "claude-3", "mock-model"]

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "MockLLMProvider"

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "name": self.get_provider_name(),
            "type": "mock",
            "models": self.list_models(),
            "connected": self.check_connection(),
        }

    def get_model_info(self, model: str) -> Dict[str, Any] | None:
        """Get model info."""
        if model in self.list_models():
            return {"id": model, "name": model, "context_length": 4096}
        return None

    def list_models_with_info(self) -> List[Dict[str, Any]]:
        """List models with info."""
        return [
            {"id": m, "name": m, "context_length": 4096}
            for m in self.list_models()
        ]

    def supports_model(self, model: str) -> bool:
        """Check if model is supported."""
        return model in self.list_models()

    def get_default_model(self) -> str:
        """Get default model."""
        return "gpt-3.5-turbo"

    # Test helper methods

    def set_response(self, prompt: str, response: str) -> None:
        """Set mock response for a prompt."""
        self.responses[prompt] = response

    def set_json_response(self, prompt: str, response_data: Dict[str, Any]) -> None:
        """Set mock JSON response for a prompt."""
        import json
        self.responses[prompt] = json.dumps(response_data)

    def set_failure(self, message: str = "Mock failure") -> None:
        """Make the provider fail on next calls."""
        self.should_fail = True
        self.fail_message = message

    def clear_failure(self) -> None:
        """Clear failure state."""
        self.should_fail = False

    def get_call_history(self) -> List[tuple]:
        """Get call history."""
        return self.call_history.copy()

    def clear_call_history(self) -> None:
        """Clear call history."""
        self.call_history.clear()

    def set_default_response(self, response_data: Dict[str, Any]) -> None:
        """Set default JSON response for any prompt."""
        import json
        self.default_response = json.dumps(response_data)

    def reset(self) -> None:
        """Reset mock state."""
        self.responses.clear()
        self.call_history.clear()
        self.should_fail = False
        self.fail_message = "Mock LLM provider failure"
        self.default_response = None
