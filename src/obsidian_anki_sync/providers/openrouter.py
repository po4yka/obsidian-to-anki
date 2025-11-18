"""OpenRouter provider implementation."""

import json
import os
import time
from typing import Any, cast

import httpx

from ..utils.logging import get_logger
from ..utils.llm_logging import (
    log_llm_error,
    log_llm_request,
    log_llm_retry,
    log_llm_success,
)
from .base import BaseLLMProvider

logger = get_logger(__name__)

# Models known to have issues with strict structured outputs
MODELS_WITH_STRUCTURED_OUTPUT_ISSUES = {
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-chat-v3.1:free",
    "moonshotai/kimi-k2-thinking",
    # Qwen models may have issues with strict structured outputs
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-32b-instruct",
    "qwen/qwen3-max",
}

# Model context window sizes (in tokens)
# Default: 131072 (128k) for most modern models
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "moonshotai/kimi-k2": 131072,
    "moonshotai/kimi-k2-thinking": 131072,
    "qwen/qwen-2.5-72b-instruct": 131072,
    "qwen/qwen-2.5-32b-instruct": 131072,
    "deepseek/deepseek-chat-v3.1": 131072,
    "deepseek/deepseek-chat": 131072,
    "minimax/minimax-m2": 131072,
}
DEFAULT_CONTEXT_WINDOW = 131072  # 128k tokens
CONTEXT_SAFETY_MARGIN = 1000  # Reserve tokens for safety

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
        # Configure timeout with separate values:
        # - connect: 10s (quick connection establishment)
        # - read: timeout (long for large model responses)
        # - write: 30s (sufficient for request sending)
        # - pool: 5s (connection pool timeout)
        timeout_config = httpx.Timeout(
            connect=10.0,
            read=timeout,
            write=30.0,
            pool=5.0,
        )
        self.client = httpx.Client(
            timeout=timeout_config,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers=headers,
        )
        # Async client for async operations (lazy initialization)
        self._async_client: httpx.AsyncClient | None = None
        self._headers = headers
        self._timeout = timeout

        logger.info(
            "openrouter_provider_initialized",
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            has_site_info=bool(site_url and site_name),
        )

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_client is None:
            # Use same timeout configuration as sync client
            timeout_config = httpx.Timeout(
                connect=10.0,
                read=self._timeout,
                write=30.0,
                pool=5.0,
            )
            self._async_client = httpx.AsyncClient(
                timeout=timeout_config,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                headers=self._headers,
            )
        return self._async_client

    async def generate_async(
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
        """Generate a completion asynchronously using AsyncClient.

        Args:
            model: Model identifier
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            format: Response format
            json_schema: JSON schema for structured output
            stream: Enable streaming
            reasoning_enabled: Enable reasoning mode

        Returns:
            Response dictionary
        """
        if stream:
            raise NotImplementedError("Streaming is not yet supported in async mode")

        # Use async client for better performance
        async_client = self._get_async_client()

        # Build messages (same as sync version)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Calculate max_tokens (same logic as sync)
        prompt_tokens_estimate = (len(prompt) + len(system)) // 4
        schema_overhead = len(json.dumps(json_schema.get("schema", {}))) // 4 if json_schema else 0

        # Get model's context window (default to 128k)
        context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)

        if json_schema:
            # Use same improved logic as sync version
            multiplier = 4.5 if prompt_tokens_estimate > 3000 else 4.0
            estimated_needed = int(prompt_tokens_estimate * multiplier) + schema_overhead

            schema_name = json_schema.get("name", "")
            if "qa_extraction" in schema_name.lower() or "extraction" in schema_name.lower():
                min_tokens_for_schema = 16000
            elif "validation" in schema_name.lower():
                min_tokens_for_schema = 8000
            else:
                min_tokens_for_schema = 12000

            desired_max_tokens = max(
                self.max_tokens,
                estimated_needed,
                min_tokens_for_schema
            )

            max_allowed_by_context = context_window - prompt_tokens_estimate - CONTEXT_SAFETY_MARGIN
            effective_max_tokens = min(desired_max_tokens, max_allowed_by_context)

            if effective_max_tokens < desired_max_tokens:
                logger.warning(
                    "reduced_max_tokens_for_context_window_async",
                    model=model,
                    desired_max_tokens=desired_max_tokens,
                    effective_max_tokens=effective_max_tokens,
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    context_window=context_window,
                )

            if effective_max_tokens < min_tokens_for_schema:
                logger.warning(
                    "max_tokens_below_recommended_minimum_async",
                    model=model,
                    effective_max_tokens=effective_max_tokens,
                    recommended_minimum=min_tokens_for_schema,
                    prompt_tokens_estimate=prompt_tokens_estimate,
                )
        else:
            max_allowed_by_context = context_window - prompt_tokens_estimate - CONTEXT_SAFETY_MARGIN
            effective_max_tokens = min(self.max_tokens, max_allowed_by_context)

        # Build payload (same as sync version)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
            "stream": False,
        }

        if reasoning_enabled and not json_schema:
            payload["reasoning_enabled"] = True

        # Handle structured output
        if json_schema:
            schema_dict = json_schema.get("schema", {})
            schema_name = json_schema.get("name", "response")
            if schema_dict and isinstance(schema_dict, dict):
                default_strict = json_schema.get("strict", True)
                use_strict = False if model in MODELS_WITH_STRUCTURED_OUTPUT_ISSUES else default_strict
                optimized_schema = self._optimize_schema_for_request(schema_dict)
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": use_strict,
                        "schema": optimized_schema,
                    },
                }
            else:
                payload["response_format"] = {"type": "json_object"}
        elif format == "json":
            payload["response_format"] = {"type": "json_object"}

        # Use enhanced logging for async request
        request_start_time = log_llm_request(
            model=model,
            operation="openrouter_generate_async",
            prompt_length=len(prompt),
            system_length=len(system),
            prompt_tokens_estimate=prompt_tokens_estimate,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            has_json_schema=bool(json_schema),
            format=format,
        )

        try:
            response = await async_client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            message = result["choices"][0]["message"]
            completion = message.get("content")

            # Handle reasoning models
            if completion is None or completion == "":
                if "reasoning" in message and message["reasoning"]:
                    completion = message["reasoning"]
                elif "refusal" in message and message["refusal"]:
                    completion = message["refusal"]
                else:
                    completion = ""

            # Clean up completion
            if json_schema or format == "json":
                completion = self._clean_json_response(completion, model)

            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            finish_reason = result["choices"][0].get("finish_reason", "stop")

            # Use enhanced logging for async success
            context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
            log_llm_success(
                model=model,
                operation="openrouter_generate_async",
                start_time=request_start_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                response_length=len(completion),
                finish_reason=finish_reason,
                context_window=context_window,
                estimate_cost_flag=True,
            )

            return {
                "response": completion,
                "tokens": completion_tokens,
                "finish_reason": finish_reason,
            }

        except Exception as e:
            log_llm_error(
                model=model,
                operation="openrouter_generate_async",
                start_time=request_start_time,
                error=e,
                error_type=type(e).__name__,
                retryable=isinstance(e, httpx.HTTPStatusError)
                and e.response.status_code in (429, 500, 502, 503, 504),
            )
            raise

    def __del__(self) -> None:
        """Clean up client resources."""
        if hasattr(self, "client"):
            self.client.close()
        if hasattr(self, "_async_client") and self._async_client:
            # Note: Can't await in __del__, so we'll close sync
            # Async client should be closed explicitly via async context manager
            pass

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
        last_valid_pos = 0
        last_key_pos = -1  # Track where we last saw a key (before ':')

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
                if not in_string:
                    # String closed, this is a valid position
                    last_valid_pos = i + 1
                else:
                    # String opened
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
                elif char == ':':
                    # Found a key-value separator, mark this position
                    last_key_pos = i
                    last_valid_pos = i + 1
                elif char == ',':
                    # Found a separator, this is a valid position
                    last_valid_pos = i + 1
                elif char not in (' ', '\n', '\t', '\r'):
                    # Valid JSON character
                    last_valid_pos = i + 1

        # Truncate to last valid position
        repaired = repaired[:last_valid_pos]

        # If we're in a string, we need to close it
        if in_string:
            # Just close the string
            repaired += '"'

        # Check for incomplete values after colons (common truncation pattern)
        # Look for patterns like "key": "" or "key": "incomplete
        if last_key_pos >= 0 and last_key_pos < len(repaired):
            # Find the colon position
            colon_pos = repaired.find(':', last_key_pos)
            if colon_pos != -1:
                # Get everything after the colon
                after_colon = repaired[colon_pos + 1:].lstrip()

                # Check if we have an incomplete string value
                if after_colon.startswith('"'):
                    # Count quotes in the value part
                    quote_count = after_colon.count('"')
                    # If odd number of quotes, string is not closed
                    if quote_count % 2 == 1:
                        # String is not closed, close it
                        if not repaired.endswith('"'):
                            repaired += '"'
                    # If we have "" but nothing after, it's complete (empty string)
                    elif after_colon == '""':
                        pass  # Complete empty string
                    # If we have "something but no closing quote
                    elif quote_count == 1:
                        # Only opening quote, close it
                        if not repaired.endswith('"'):
                            repaired += '"'
                elif not after_colon:
                    # No value after colon, add empty string
                    repaired += ' ""'
                elif after_colon and not (after_colon.startswith('"') or after_colon.startswith('[') or after_colon.startswith('{') or (after_colon and after_colon[0].isdigit()) or after_colon in ('true', 'false', 'null')):
                    # Invalid value start, add empty string
                    repaired += ' ""'

        # Before closing structures, handle the case where we have patterns like "key": ""]}}
        # This means an object inside an array is incomplete - we need to close the object first
        # Check if the text ends with premature closing brackets/braces
        trimmed_end = repaired.rstrip()
        if trimmed_end.endswith(']') and brace_count > 0:
            # We have a ] but still open braces - this means object wasn't closed before array
            # Close the object(s) first
            while brace_count > 0:
                repaired += '}'
                brace_count -= 1

        # Close any open braces (objects) first - objects must be closed before arrays
        while brace_count > 0:
            repaired += '}'
            brace_count -= 1

        # Then close any open brackets (arrays)
        while bracket_count > 0:
            repaired += ']'
            bracket_count -= 1

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
        # Track warnings per call to avoid duplicate warnings for the same text
        # Use a simple identifier based on text start and length
        text_id = f"{hash(text[:100])}:{len(text)}"
        if not hasattr(self, '_warned_text_ids'):
            self._warned_text_ids = set()

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
                        if brace_count == 0 and bracket_count == 0:
                            # Found the end of the JSON object (all structures closed)
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
                        # If we close a bracket but still have open braces, check if we're near the end
                        # This could indicate truncation, but only warn if we're close to the end of text
                        # (Arrays can legitimately close before their containing objects in valid JSON)
                        if bracket_count == 0 and brace_count > 0:
                            # Only warn if we're in the last 10% of the text (likely truncation)
                            # or if there's very little text remaining
                            remaining_chars = len(cleaned) - i - 1
                            is_near_end = remaining_chars < max(100, len(cleaned) * 0.1)

                            if is_near_end:
                                # Only log warning once per unique text (avoid duplicates)
                                warning_key = f"premature_array_close:{text_id}"
                                if warning_key not in self._warned_text_ids:
                                    # Array closed but object(s) still open near end of text - likely truncation
                                    logger.warning(
                                        "detected_premature_array_close",
                                        model=model,
                                        brace_count=brace_count,
                                        position=i,
                                        remaining_chars=remaining_chars,
                                        total_length=len(cleaned),
                                        note="Array closed before containing object near end of text - likely truncation",
                                    )
                                    self._warned_text_ids.add(warning_key)
                            # Don't break here - let repair handle it if needed
                    else:
                        last_valid_pos = i + 1

            # If we ended while still in a string or with unclosed braces/brackets, try to repair
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
                # Use text up to last_valid_pos, but if that's 0, use the whole cleaned text
                text_to_repair = cleaned[:last_valid_pos] if last_valid_pos > 0 else cleaned
                cleaned = self._repair_truncated_json(text_to_repair)

        # Clear warning tracking for this text after processing
        # This allows the same text to be processed again in future calls if needed
        if hasattr(self, '_warned_text_ids'):
            self._warned_text_ids.discard(f"premature_array_close:{text_id}")

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

        # Calculate appropriate max_tokens based on input size and schema complexity
        # For structured outputs with JSON schema, we need more tokens to ensure completion
        # Estimate: input tokens + (input tokens * 3) for bilingual content + schema overhead
        prompt_tokens_estimate = (len(prompt) + len(system)) // 4  # Rough estimate: 1 token ≈ 4 chars
        schema_overhead = len(json.dumps(json_schema.get("schema", {}))) // 4 if json_schema else 0

        # Get model's context window (default to 128k)
        context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)

        # For structured outputs, ensure we have enough tokens for complete responses
        # Use more generous multipliers to prevent truncation
        if json_schema:
            # For structured outputs, be more generous with tokens
            # Use 4-5x multiplier for complex bilingual content with structured output
            # This accounts for:
            # - Bilingual responses (2x)
            # - Structured JSON overhead (1x)
            # - Code examples and formatting (1x)
            # - Safety margin (0.5-1x)
            multiplier = 4.5 if prompt_tokens_estimate > 3000 else 4.0
            estimated_needed = int(prompt_tokens_estimate * multiplier) + schema_overhead

            # For structured outputs, ensure minimum floor to prevent truncation
            # Complex schemas (like QA extraction) need more tokens
            schema_name = json_schema.get("name", "")
            if "qa_extraction" in schema_name.lower() or "extraction" in schema_name.lower():
                min_tokens_for_schema = 16000  # QA extraction needs more tokens
            elif "validation" in schema_name.lower():
                min_tokens_for_schema = 8000  # Validation schemas are simpler
            else:
                min_tokens_for_schema = 12000  # Default for other structured outputs

            # Use the larger of: configured max, estimated needed, or minimum floor
            desired_max_tokens = max(
                self.max_tokens,
                estimated_needed,
                min_tokens_for_schema
            )

            # But ensure we don't exceed the model's context window
            # Reserve space for input tokens and safety margin
            max_allowed_by_context = context_window - prompt_tokens_estimate - CONTEXT_SAFETY_MARGIN
            effective_max_tokens = min(desired_max_tokens, max_allowed_by_context)

            # Log if we had to reduce max_tokens due to context window limits
            if effective_max_tokens < desired_max_tokens:
                logger.warning(
                    "reduced_max_tokens_for_context_window",
                    model=model,
                    desired_max_tokens=desired_max_tokens,
                    effective_max_tokens=effective_max_tokens,
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    context_window=context_window,
                    max_allowed_by_context=max_allowed_by_context,
                    suggestion="Consider reducing input size or using a model with larger context window",
                )

            # Ensure we don't go below a reasonable minimum (but respect context limits)
            if effective_max_tokens < min_tokens_for_schema:
                logger.warning(
                    "max_tokens_below_recommended_minimum",
                    model=model,
                    effective_max_tokens=effective_max_tokens,
                    recommended_minimum=min_tokens_for_schema,
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    context_window=context_window,
                    suggestion="Consider reducing input size or using a model with larger context window",
                )

            # Log proactive token allocation
            if effective_max_tokens >= desired_max_tokens:
                logger.debug(
                    "proactive_max_tokens_allocation",
                    model=model,
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    estimated_needed=estimated_needed,
                    min_tokens_for_schema=min_tokens_for_schema,
                    effective_max_tokens=effective_max_tokens,
                    multiplier=multiplier,
                    schema_name=schema_name,
                )
        else:
            # For non-structured outputs, still respect context window
            max_allowed_by_context = context_window - prompt_tokens_estimate - CONTEXT_SAFETY_MARGIN
            effective_max_tokens = min(self.max_tokens, max_allowed_by_context)

        # Build payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
            "stream": False,
        }

        if json_schema and effective_max_tokens > self.max_tokens:
            logger.debug(
                "adjusted_max_tokens_for_structured_output",
                model=model,
                original_max_tokens=self.max_tokens,
                adjusted_max_tokens=effective_max_tokens,
                prompt_tokens_estimate=prompt_tokens_estimate,
                schema_overhead=schema_overhead,
            )

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
            # Only log at debug level since this is expected behavior
            # Reasoning mode is automatically disabled when JSON schema is used
            logger.debug(
                "reasoning_mode_disabled_for_json_schema",
                model=model,
                reason="Reasoning mode is incompatible with strict JSON schema structured output",
                note="This is expected - reasoning is automatically disabled when using JSON schemas",
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

        # Use enhanced logging for request start
        request_start_time = log_llm_request(
            model=model,
            operation="openrouter_generate",
            prompt_length=len(prompt),
            system_length=len(system),
            prompt_tokens_estimate=prompt_tokens_estimate,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            has_json_schema=bool(json_schema),
            format=format,
            schema_name=json_schema.get("name") if json_schema else None,
        )

        # Retry logic for network errors
        max_retries = 3
        last_exception: Exception | None = None

        for attempt in range(max_retries):
            try:
                # Client already has timeout configured, don't override it
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                break  # Success, exit retry loop
            except httpx.RequestError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = 2 ** attempt
                    log_llm_retry(
                        model=model,
                        operation="openrouter_generate",
                        attempt=attempt + 1,
                        max_attempts=max_retries,
                        reason="network_error",
                        error=str(e),
                        wait_seconds=wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                # Last attempt failed, will raise below
                break
            except httpx.HTTPStatusError:
                # Don't retry HTTP status errors here, let the existing handler deal with them
                raise

        # If we exhausted retries, raise the last exception
        if last_exception and isinstance(last_exception, httpx.RequestError):
            log_llm_error(
                model=model,
                operation="openrouter_generate",
                start_time=request_start_time,
                error=last_exception,
                error_type="RequestError",
                retryable=False,  # Already retried max times
            )
            raise last_exception

        # Continue with response processing
        try:
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
                    finish_reason = result["choices"][0].get("finish_reason", "unknown")
                    logger.warning(
                        "empty_completion_from_openrouter",
                        model=model,
                        message_keys=list(message.keys()),
                        finish_reason=finish_reason,
                        full_message=message,
                        has_json_schema=bool(json_schema),
                        schema_name=json_schema.get("name") if json_schema else None,
                        note="Model returned empty content. This may indicate structured output compatibility issues.",
                    )
                    # Raise error instead of silently continuing with empty string
                    raise ValueError(
                        f"Model {model} returned empty completion. "
                        f"Finish reason: {finish_reason}. "
                        f"This may indicate the model doesn't support structured outputs properly. "
                        f"Consider using a different model or non-strict mode."
                    )

            # Extract detailed usage information
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Check finish reason for potential issues
            finish_reason = result["choices"][0].get("finish_reason", "unknown")

            # Clean up completion text (especially important for DeepSeek responses)
            # DeepSeek may add markdown fences, reasoning text, or special tokens
            json_truncated = False
            if completion and json_schema:
                original_completion = completion
                completion = self._clean_json_response(completion, model)

                # Check if JSON appears truncated by validating it
                # If completion_tokens is suspiciously low or JSON parsing fails, mark as truncated
                if json_schema:
                    try:
                        json.loads(completion)
                    except (json.JSONDecodeError, ValueError):
                        # JSON is invalid - likely truncated
                        json_truncated = True
                        logger.warning(
                            "json_validation_failed_likely_truncated",
                            model=model,
                            completion_tokens=completion_tokens,
                            max_tokens=effective_max_tokens,
                            completion_preview=completion[:200],
                        )
                    # Also check if completion_tokens is suspiciously low (< 5% of max_tokens)
                    # This indicates the model stopped early, possibly due to truncation
                    if completion_tokens > 0 and effective_max_tokens > 0:
                        token_usage_ratio = completion_tokens / effective_max_tokens
                        if token_usage_ratio < 0.05 and completion_tokens < 100:
                            json_truncated = True
                            logger.warning(
                                "suspiciously_low_completion_tokens",
                                model=model,
                                completion_tokens=completion_tokens,
                                max_tokens=effective_max_tokens,
                                usage_ratio=round(token_usage_ratio, 4),
                            )

            # Initialize retry variables (will be set if retry happens)
            retry_completion: str | None = None
            retry_result: dict[str, Any] | None = None
            retry_usage: dict[str, Any] | None = None
            retry_finish_reason: str | None = None

            # Use enhanced logging for success
            context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
            log_llm_success(
                model=model,
                operation="openrouter_generate",
                start_time=request_start_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                response_length=len(completion),
                finish_reason=finish_reason,
                context_window=context_window,
                estimate_cost_flag=True,
                had_json_schema=bool(json_schema),
                schema_name=json_schema.get("name") if json_schema else None,
                max_tokens_used=effective_max_tokens,
            )

            # Check if response was truncated (either by finish_reason or JSON validation)
            # For structured outputs, truncation is critical - retry with increased max_tokens
            is_truncated = finish_reason == "length" or json_truncated
            if is_truncated and json_schema and effective_max_tokens < 128000:
                # Increase max_tokens significantly and retry once
                retry_max_tokens = min(effective_max_tokens * 2, 128000)
                truncation_reason = "finish_reason_length" if finish_reason == "length" else "json_truncated"
                log_llm_retry(
                    model=model,
                    operation="openrouter_generate",
                    attempt=1,
                    max_attempts=2,
                    reason=f"response_truncated_{truncation_reason}",
                    original_max_tokens=effective_max_tokens,
                    retry_max_tokens=retry_max_tokens,
                    finish_reason=finish_reason,
                    json_truncated=json_truncated,
                )
                # Update payload and retry
                payload["max_tokens"] = retry_max_tokens
                # Client already has timeout configured, don't override it
                retry_response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
                retry_response.raise_for_status()
                retry_result = retry_response.json()
                retry_message = retry_result["choices"][0]["message"]
                retry_completion = retry_message.get("content")

                # Handle reasoning field if needed
                if retry_completion is None or retry_completion == "":
                    if "reasoning" in retry_message and retry_message["reasoning"]:
                        retry_completion = retry_message["reasoning"]

                # Clean up retry completion
                if retry_completion and json_schema:
                    retry_completion = self._clean_json_response(retry_completion, model)

                retry_finish_reason = retry_result["choices"][0].get("finish_reason", "unknown")
                retry_usage = retry_result.get("usage", {})
                retry_completion_tokens = retry_usage.get("completion_tokens", 0)

                if retry_finish_reason == "length":
                    # Still truncated even with increased tokens - this is a problem
                    raise ValueError(
                        f"Response still truncated at {retry_completion_tokens} tokens (max: {retry_max_tokens}). "
                        f"The response is too large. Consider simplifying the request or splitting into smaller parts."
                    )

                # Log retry success with enhanced metrics
                retry_total_tokens = retry_usage.get("total_tokens", 0)
                retry_prompt_tokens = retry_usage.get("prompt_tokens", prompt_tokens)
                log_llm_success(
                    model=model,
                    operation="openrouter_generate_retry",
                    start_time=request_start_time,
                    prompt_tokens=retry_prompt_tokens,
                    completion_tokens=retry_completion_tokens,
                    total_tokens=retry_total_tokens,
                    response_length=len(retry_completion or ""),
                    finish_reason=retry_finish_reason or "unknown",
                    context_window=context_window,
                    estimate_cost_flag=True,
                    is_retry=True,
                    original_max_tokens=effective_max_tokens,
                    retry_max_tokens=retry_max_tokens,
                )
            elif json_schema and is_truncated:
                # Truncated but already at max, can't retry
                truncation_details = []
                if finish_reason == "length":
                    truncation_details.append("finish_reason=length")
                if json_truncated:
                    truncation_details.append("json_validation_failed")
                truncation_info = ", ".join(truncation_details) if truncation_details else "unknown"
                raise ValueError(
                    f"Response truncated at {completion_tokens} tokens (max: {effective_max_tokens}, {truncation_info}). "
                    f"Cannot increase max_tokens further (already at context window limit). "
                    f"Consider simplifying the request, reducing input size, or splitting into smaller parts."
                )

            # Convert to standard format - use retry values if available
            final_completion = retry_completion if retry_completion else completion
            final_result = retry_result if retry_result else result
            final_usage = retry_usage if retry_usage else usage
            final_finish_reason = retry_finish_reason if retry_finish_reason else finish_reason

            standardized_result = {
                "response": final_completion if final_completion else "",
                "model": final_result.get("model"),
                "finish_reason": final_finish_reason,
                "usage": final_usage,
            }

            return standardized_result

        except httpx.HTTPStatusError as e:
            # Parse response for structured error details if available
            error_details = {}
            error_type = None
            error_message = str(e)

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

            # Use enhanced error logging
            log_llm_error(
                model=model,
                operation="openrouter_generate",
                start_time=request_start_time,
                error=e,
                error_type=error_type or "HTTPStatusError",
                status_code=e.response.status_code,
                retryable=e.response.status_code in (429, 500, 502, 503, 504),
                prompt_length=len(prompt),
                system_length=len(system),
                error_details=error_details,
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
            # This should not happen here since RequestError is handled in retry loop above
            # But keep as fallback for any edge cases
            log_llm_error(
                model=model,
                operation="openrouter_generate",
                start_time=request_start_time,
                error=e,
                error_type="RequestError",
                retryable=False,  # Already handled in retry loop
            )
            raise
        except Exception as e:
            log_llm_error(
                model=model,
                operation="openrouter_generate",
                start_time=request_start_time,
                error=e,
                error_type=type(e).__name__,
                retryable=False,
            )
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
            # Check for common truncation errors
            error_str = str(e)
            is_truncation_error = (
                "Unterminated string" in error_str
                or "Expecting" in error_str
                or "Unterminated" in error_str
                or "Invalid" in error_str
            )

            if is_truncation_error:
                logger.warning(
                    "attempting_json_repair",
                    model=model,
                    original_error=str(e),
                    error_position=f"line {e.lineno}, col {e.colno}" if hasattr(e, "lineno") else "unknown",
                    cleaned_text_preview=cleaned_text[:200],
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
                        repaired_text_preview=repaired_text[:200] if 'repaired_text' in locals() else None,
                    )
                    # If repair failed, try one more time with a more aggressive approach
                    # Remove incomplete trailing structures and return partial result
                    try:
                        # Try to extract valid JSON up to the error position
                        error_pos = getattr(e, "pos", None)
                        if error_pos and error_pos > 0:
                            partial_text = cleaned_text[:error_pos]
                            # Try to close any open structures
                            partial_repaired = self._repair_truncated_json(partial_text)
                            partial_json = json.loads(partial_repaired)
                            logger.warning(
                                "partial_json_recovery",
                                model=model,
                                recovered_keys=list(partial_json.keys()) if isinstance(partial_json, dict) else "array",
                            )
                            return cast(dict[str, Any], partial_json)
                    except Exception:
                        pass  # Give up on partial recovery

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
