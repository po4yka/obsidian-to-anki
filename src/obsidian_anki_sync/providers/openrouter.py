"""OpenRouter provider implementation."""

import json
import os
from typing import Any, cast

import httpx

from ..utils.logging import get_logger
from .base import BaseLLMProvider

logger = get_logger(__name__)

# Models known to have issues with strict structured outputs
MODELS_WITH_STRUCTURED_OUTPUT_ISSUES = {
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-chat-v3.1:free",
    "moonshotai/kimi-k2-thinking",
}

# Models that support structured outputs well
MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS = {
    "openai/gpt-4",
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
    "openai/gpt-3.5-turbo",
    "google/gemini-pro",
    "google/gemini-2.0-flash-exp",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-haiku",
    "qwen/qwen3-max",
}


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
        # Use synchronous httpx.Client for compatibility with existing sync code
        # This provider is used in sync contexts, so async client is not needed
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

    def _optimize_schema_for_request(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Optimize JSON schema for token efficiency.

        Removes verbose descriptions and metadata that don't affect validation
        but consume tokens. Keeps essential validation rules.

        Args:
            schema: Original JSON schema dictionary

        Returns:
            Optimized schema dictionary
        """
        if not isinstance(schema, dict):
            return schema

        optimized = {}

        # Keep essential fields
        essential_fields = {
            "type", "properties", "items", "required", "enum",
            "minimum", "maximum", "minLength", "maxLength",
            "pattern", "format", "additionalProperties", "anyOf", "oneOf", "allOf"
        }

        for key, value in schema.items():
            if key in essential_fields:
                if key == "properties" and isinstance(value, dict):
                    # Recursively optimize nested properties
                    optimized[key] = {
                        k: self._optimize_schema_for_request(v)
                        for k, v in value.items()
                    }
                elif key == "items" and isinstance(value, dict):
                    # Optimize array items schema
                    optimized[key] = self._optimize_schema_for_request(value)
                elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
                    # Optimize union types
                    optimized[key] = [
                        self._optimize_schema_for_request(item)
                        if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    optimized[key] = value

        return optimized

    def _repair_truncated_json(self, text: str) -> str:
        """Attempt to repair truncated JSON by closing open structures.

        Args:
            text: Potentially truncated JSON text

        Returns:
            Repaired JSON text (may still be invalid if too corrupted)
        """
        if not text.strip():
            return "{}"

        repaired = text.rstrip()

        # Track state
        brace_count = 0
        bracket_count = 0
        in_string = False
        escape_next = False
        last_valid_pos = len(repaired)

        for i, char in enumerate(repaired):
            if escape_next:
                escape_next = False
                last_valid_pos = i + 1
                continue

            if char == '\\':
                escape_next = True
                last_valid_pos = i + 1
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                last_valid_pos = i + 1
                continue

            if not in_string:
                if char == '{':
                    brace_count += 1
                    last_valid_pos = i + 1
                elif char == '}':
                    brace_count -= 1
                    last_valid_pos = i + 1
                elif char == '[':
                    bracket_count += 1
                    last_valid_pos = i + 1
                elif char == ']':
                    bracket_count -= 1
                    last_valid_pos = i + 1
                elif char not in (' ', '\n', '\t', '\r'):
                    # Valid JSON character (including ',' and ':')
                    last_valid_pos = i + 1

        # Truncate to last valid position
        repaired = repaired[:last_valid_pos]

        # If we're still in a string, close it
        if in_string:
            repaired += '"'

        # Close any open brackets
        while bracket_count > 0:
            repaired += ']'
            bracket_count -= 1

        # Close any open braces
        while brace_count > 0:
            repaired += '}'
            brace_count -= 1

        return repaired

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
            bracket_count = 0
            in_string = False
            escape_next = False
            last_valid_pos = 0

            for i, char in enumerate(cleaned):
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    last_valid_pos = i + 1
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                        last_valid_pos = i + 1
                    elif char == '}':
                        brace_count -= 1
                        last_valid_pos = i + 1
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
                    elif char == '[':
                        bracket_count += 1
                        last_valid_pos = i + 1
                    elif char == ']':
                        bracket_count -= 1
                        last_valid_pos = i + 1
                    else:
                        last_valid_pos = i + 1

            # If we ended while still in a string or with unclosed braces, try to repair
            if in_string or brace_count > 0 or bracket_count > 0:
                logger.warning(
                    "detected_truncated_json",
                    model=model,
                    in_string=in_string,
                    brace_count=brace_count,
                    bracket_count=bracket_count,
                    original_length=len(text),
                    cleaned_length=len(cleaned),
                )
                # Try to repair the truncated JSON
                cleaned = self._repair_truncated_json(cleaned[:last_valid_pos])

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
            schema_dict = json_schema.get("schema", {})
            schema_name = json_schema.get("name", "response")

            # Validate schema structure before sending
            if not schema_dict or not isinstance(schema_dict, dict):
                logger.warning(
                    "invalid_json_schema_structure",
                    model=model,
                    schema_name=schema_name,
                    schema_type=type(schema_dict).__name__,
                )
                # Fallback to basic JSON mode if schema is invalid
                payload["response_format"] = {"type": "json_object"}
            else:
                # Determine strict mode based on model compatibility
                default_strict = json_schema.get("strict", True)

                # Some models have issues with strict mode, use non-strict as fallback
                if model in MODELS_WITH_STRUCTURED_OUTPUT_ISSUES:
                    use_strict = False
                    logger.debug(
                        "using_non_strict_mode_for_model",
                        model=model,
                        reason="Model known to have structured output issues",
                    )
                else:
                    use_strict = default_strict

                # Optimize schema: remove unnecessary metadata for token efficiency
                optimized_schema = self._optimize_schema_for_request(schema_dict)

                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": use_strict,
                        "schema": optimized_schema,
                    },
                }

                logger.debug(
                    "structured_output_configured",
                    model=model,
                    schema_name=schema_name,
                    strict=use_strict,
                    schema_size=len(json.dumps(optimized_schema)),
                )
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

            # Extract detailed usage information
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Check finish reason for potential issues
            finish_reason = result["choices"][0].get("finish_reason", "unknown")

            logger.info(
                "openrouter_generate_success",
                model=model,
                response_length=len(completion),
                tokens_used=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                had_json_schema=bool(json_schema),
                schema_name=json_schema.get("name", None) if json_schema else None,
            )

            # Warn if response was truncated (length limit reached)
            if finish_reason == "length" and json_schema:
                logger.warning(
                    "structured_output_truncated",
                    model=model,
                    response_length=len(completion),
                    max_tokens=self.max_tokens,
                    suggestion="Consider increasing max_tokens or simplifying schema",
                )

            return standardized_result

        except httpx.HTTPStatusError as e:
            # Parse response for structured error details if available
            error_details = {}
            try:
                error_json = e.response.json()
                error_details = error_json

                # Check for structured output specific errors
                if "error" in error_json:
                    error_msg = error_json.get("error", {})
                    if isinstance(error_msg, dict):
                        error_type = error_msg.get("type", "")
                        error_message = error_msg.get("message", "")

                        # If it's a schema validation error, provide helpful context
                        if "schema" in error_type.lower() or "validation" in error_type.lower():
                            logger.error(
                                "openrouter_schema_validation_error",
                                status_code=e.response.status_code,
                                model=model,
                                error_type=error_type,
                                error_message=error_message,
                                had_json_schema=bool(json_schema),
                                schema_name=json_schema.get("name", "unknown") if json_schema else None,
                            )
                        elif "rate_limit" in error_type.lower():
                            logger.error(
                                "openrouter_rate_limit_error",
                                status_code=e.response.status_code,
                                model=model,
                                error_message=error_message,
                            )
            except Exception:
                error_details = {"raw_response": e.response.text[:500]}

            logger.error(
                "openrouter_http_error",
                status_code=e.response.status_code,
                model=model,
                prompt_length=len(prompt),
                error=str(e),
                error_details=error_details,
                response_headers=dict(e.response.headers),
                had_json_schema=bool(json_schema),
            )

            # If structured output failed, suggest fallback
            if json_schema and e.response.status_code == 400:
                logger.warning(
                    "structured_output_failed_suggestion",
                    model=model,
                    suggestion="Consider using non-strict mode or basic JSON format",
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
            # Try to repair truncated JSON before giving up
            if "Unterminated string" in str(e) or "Expecting" in str(e):
                logger.warning(
                    "attempting_json_repair",
                    model=model,
                    original_error=str(e),
                    error_position=f"line {e.lineno}, col {e.colno}" if hasattr(e, "lineno") else "unknown",
                )
                try:
                    repaired_text = self._repair_truncated_json(cleaned_text)
                    repaired_json = json.loads(repaired_text)
                    logger.info(
                        "json_repair_success",
                        model=model,
                        original_length=len(cleaned_text),
                        repaired_length=len(repaired_text),
                    )
                    return cast(dict[str, Any], repaired_json)
                except json.JSONDecodeError as repair_error:
                    logger.error(
                        "json_repair_failed",
                        model=model,
                        repair_error=str(repair_error),
                        original_error=str(e),
                    )

            logger.error(
                "openrouter_json_parse_error",
                model=model,
                error=str(e),
                error_position=f"line {e.lineno}, col {e.colno}" if hasattr(e, "lineno") else "unknown",
                response_text=response_text[:500] if len(response_text) > 500 else response_text,
                cleaned_text=cleaned_text[:500] if len(cleaned_text) > 500 else cleaned_text,
                response_length=len(response_text),
                cleaned_length=len(cleaned_text),
                had_schema=bool(json_schema),
            )
            raise
