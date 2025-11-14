"""OpenRouter provider implementation."""

import json
import os
from typing import Any, cast

import httpx

from ..utils.logging import get_logger
from .base import BaseLLMProvider

logger = get_logger(__name__)


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter LLM provider using OpenAI-compatible API.

    OpenRouter provides access to multiple LLM providers through a unified API.
    Requires an API key from https://openrouter.ai/

    Configuration:
        api_key: OpenRouter API key (required, can use OPENROUTER_API_KEY env var)
        base_url: API endpoint URL (default: https://openrouter.ai/api/v1)
        timeout: Request timeout in seconds (default: 180.0)
        max_tokens: Maximum tokens in response (default: 2048)
        site_url: Your site URL for OpenRouter rankings (optional)
        site_name: Your site name for OpenRouter rankings (optional)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 180.0,
        max_tokens: int = 2048,
        site_url: str | None = None,
        site_name: str | None = None,
        **kwargs: Any,
    ):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key (if not provided, uses OPENROUTER_API_KEY env var)
            base_url: Base URL for OpenRouter API
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
            site_url: Your site URL (optional, for rankings)
            site_name: Your site name (optional, for rankings)
            **kwargs: Additional configuration options

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        # Try to get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Provide it via the api_key parameter "
                "or set the OPENROUTER_API_KEY environment variable."
            )

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            site_url=site_url,
            site_name=site_name,
            **kwargs,
        )

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.site_url = site_url
        self.site_name = site_name

        # Set up headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Add optional site information for OpenRouter rankings
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        # Initialize HTTP client
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers=headers,
        )

        logger.info(
            "openrouter_provider_initialized",
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            has_site_info=bool(site_url and site_name),
        )

    def __del__(self) -> None:
        """Clean up client resources."""
        if hasattr(self, "client"):
            self.client.close()

    def _clean_json_response(self, text: str, model: str) -> str:
        """Clean JSON response from DeepSeek and similar models.

        DeepSeek often wraps JSON in markdown fences, adds reasoning text,
        or appends special tokens. This method extracts the clean JSON.

        Args:
            text: Raw response text
            model: Model identifier for logging

        Returns:
            Cleaned JSON text
        """
        cleaned = text.strip()

        # Remove markdown code fences if present (must be done first)
        if cleaned.startswith("```json"):
            end_fence = cleaned.rfind("```")
            if end_fence > 6:  # Found closing fence after opening
                cleaned = cleaned[7:end_fence].strip()
                logger.debug(
                    "removed_json_markdown_fence",
                    model=model,
                    original_length=len(text),
                    cleaned_length=len(cleaned),
                )
        elif cleaned.startswith("```"):
            end_fence = cleaned.rfind("```")
            if end_fence > 3:
                cleaned = cleaned[3:end_fence].strip()
                logger.debug(
                    "removed_generic_markdown_fence",
                    model=model,
                    original_length=len(text),
                    cleaned_length=len(cleaned),
                )

        # Remove DeepSeek special tokens like <｜begin▁of▁sentence｜>
        if "<｜" in cleaned:
            token_pos = cleaned.find("<｜")
            if token_pos > 0:
                cleaned = cleaned[:token_pos].strip()
                logger.debug(
                    "removed_special_tokens",
                    model=model,
                    cleaned_length=len(cleaned),
                )

        # Extract JSON if response contains reasoning/explanatory text before JSON
        if not cleaned.startswith("{"):
            first_brace = cleaned.find("{")
            if first_brace != -1:
                cleaned = cleaned[first_brace:]
                logger.debug(
                    "extracted_json_from_text",
                    model=model,
                    original_length=len(text),
                    cleaned_length=len(cleaned),
                )

        # Find the actual end of the JSON object by counting braces
        # This handles cases where there's extra text after the JSON closes
        if cleaned.startswith("{"):
            brace_count = 0
            in_string = False
            escape_next = False

            for i, char in enumerate(cleaned):
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found the end of the JSON object
                            if i + 1 < len(cleaned):
                                cleaned = cleaned[:i + 1]
                                logger.debug(
                                    "trimmed_extra_data_after_json",
                                    model=model,
                                    original_length=len(text),
                                    cleaned_length=len(cleaned),
                                )
                            break

        return cleaned

    def check_connection(self) -> bool:
        """Check if OpenRouter is accessible.

        Returns:
            True if OpenRouter is accessible, False otherwise
        """
        try:
            # OpenRouter doesn't have a dedicated health endpoint,
            # so we try to list models as a health check
            response = self.client.get(f"{self.base_url}/models")
            return bool(response.status_code == 200)
        except Exception as e:
            logger.error(
                "openrouter_connection_check_failed",
                base_url=self.base_url,
                error=str(e),
            )
            return False

    def list_models(self) -> list[str]:
        """List available models from OpenRouter.

        Returns:
            List of model identifiers (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
        """
        try:
            response = self.client.get(f"{self.base_url}/models")
            response.raise_for_status()
            data = response.json()

            # OpenRouter API format: {"data": [{"id": "model-name"}, ...]}
            models = [model["id"] for model in data.get("data", [])]
            logger.info("openrouter_list_models_success", model_count=len(models))
            return models
        except Exception as e:
            logger.error("openrouter_list_models_failed", error=str(e))
            return []

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        format: str = "",
        json_schema: dict[str, Any] | None = None,
        stream: bool = False,
        reasoning_enabled: bool = False,
    ) -> dict[str, Any]:
        """Generate completion from OpenRouter.

        Args:
            model: Model identifier (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            format: Response format ("json" for basic JSON mode, ignored if json_schema provided)
            json_schema: JSON schema for structured output (recommended over format="json")
            stream: Enable streaming (not implemented)
            reasoning_enabled: Enable reasoning mode for DeepSeek models (default: False)

        Returns:
            Response dictionary with 'response' key containing the completion

        Raises:
            NotImplementedError: If streaming is requested
            httpx.HTTPError: If request fails after retries
        """
        if stream:
            raise NotImplementedError("Streaming is not yet supported")

        # Build messages in OpenAI format
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Build payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        # Enable reasoning mode for DeepSeek models if requested
        # Note: Reasoning mode is incompatible with strict JSON schema mode
        # If JSON schema is provided, we disable reasoning mode
        if reasoning_enabled and not json_schema:
            payload["reasoning_enabled"] = True
            logger.debug(
                "reasoning_mode_enabled",
                model=model,
                note="Reasoning mode disabled when using JSON schema",
            )
        elif reasoning_enabled and json_schema:
            logger.warning(
                "reasoning_mode_disabled_for_json_schema",
                model=model,
                reason="Reasoning mode is incompatible with strict JSON schema structured output",
            )

        # Handle structured output with JSON schema (preferred method)
        if json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("name", "response"),
                    "strict": json_schema.get("strict", True),
                    "schema": json_schema.get("schema", {}),
                },
            }
        # Fallback to basic JSON mode if format="json" and no schema
        elif format == "json":
            payload["response_format"] = {"type": "json_object"}

        logger.info(
            "openrouter_generate_request",
            model=model,
            prompt_length=len(prompt),
            system_length=len(system),
            temperature=temperature,
            format=format,
            has_json_schema=bool(json_schema),
        )

        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()

            # Extract completion from OpenAI format
            message = result["choices"][0]["message"]
            completion = message.get("content")

            # Handle reasoning models (like DeepSeek) that may return content in different fields
            # When using reasoning mode with structured output, the content might be null
            # but the actual response might be in refusal, reasoning, or other fields
            if completion is None or completion == "":
                # Check for reasoning field (DeepSeek V3.1 puts structured output here)
                if "reasoning" in message and message["reasoning"]:
                    completion = message["reasoning"]
                    logger.debug(
                        "extracted_from_reasoning_field",
                        model=model,
                        reasoning_length=len(completion),
                    )
                # Check for refusal field (some models use this)
                elif "refusal" in message and message["refusal"]:
                    completion = message["refusal"]
                # Check for tool_calls or function_call (structured output alternative)
                elif "tool_calls" in message and message["tool_calls"]:
                    # Extract from tool calls if present
                    tool_call = message["tool_calls"][0]
                    if "function" in tool_call and "arguments" in tool_call["function"]:
                        completion = tool_call["function"]["arguments"]
                else:
                    # Log the full response for debugging
                    logger.warning(
                        "empty_completion_from_openrouter",
                        model=model,
                        message_keys=list(message.keys()),
                        finish_reason=result["choices"][0].get("finish_reason"),
                        full_message=message,
                    )
                    completion = ""

            # Clean up completion text (especially important for DeepSeek responses)
            # DeepSeek may add markdown fences, reasoning text, or special tokens
            if completion and json_schema:
                completion = self._clean_json_response(completion, model)

            # Convert to standard format
            standardized_result = {
                "response": completion if completion else "",
                "model": result.get("model"),
                "finish_reason": result["choices"][0].get("finish_reason"),
                "usage": result.get("usage", {}),
            }

            logger.info(
                "openrouter_generate_success",
                model=model,
                response_length=len(completion),
                tokens_used=result.get("usage", {}).get("total_tokens", 0),
            )

            return standardized_result

        except httpx.HTTPStatusError as e:
            # Parse response for structured error details if available
            error_details = {}
            try:
                error_json = e.response.json()
                error_details = error_json
            except Exception:
                error_details = {"raw_response": e.response.text}

            logger.error(
                "openrouter_http_error",
                status_code=e.response.status_code,
                model=model,
                prompt_length=len(prompt),
                error=str(e),
                error_details=error_details,
                response_headers=dict(e.response.headers),
            )
            raise
        except httpx.RequestError as e:
            logger.error("openrouter_request_error", error=str(e))
            raise
        except Exception as e:
            logger.error("openrouter_unexpected_error", error=str(e))
            raise

    def generate_json(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        json_schema: dict[str, Any] | None = None,
        reasoning_enabled: bool = False,
    ) -> dict[str, Any]:
        """Generate a JSON response from OpenRouter.

        Overrides base implementation to use OpenAI's response_format parameter.
        Supports structured outputs via JSON schema for guaranteed valid responses.

        Args:
            model: Model identifier
            prompt: User prompt (should request JSON format)
            system: System prompt (optional)
            temperature: Sampling temperature
            json_schema: JSON schema for structured output (recommended for reliability)
            reasoning_enabled: Enable reasoning mode for models that support it

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        result = self.generate(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            format="json" if not json_schema else "",
            json_schema=json_schema,
            reasoning_enabled=reasoning_enabled,
        )

        response_text = result.get("response", "{}")

        # Clean up response text before parsing (already cleaned in generate() if using json_schema)
        # But also clean here for cases where format="json" is used without schema
        cleaned_text = self._clean_json_response(response_text, model) if response_text else "{}"

        try:
            return cast(dict[str, Any], json.loads(cleaned_text))
        except json.JSONDecodeError as e:
            logger.error(
                "openrouter_json_parse_error",
                model=model,
                error=str(e),
                error_position=f"line {e.lineno}, col {e.colno}" if hasattr(e, "lineno") else "unknown",
                response_text=response_text,
                cleaned_text=cleaned_text,
                response_length=len(response_text),
                cleaned_length=len(cleaned_text),
                had_schema=bool(json_schema),
            )
            raise
