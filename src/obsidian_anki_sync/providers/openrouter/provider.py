"""OpenRouter provider implementation."""

import json
import os
import time
from types import TracebackType
from typing import Any, Literal, cast

import httpx

from ...utils.llm_logging import (
    log_llm_error,
    log_llm_request,
    log_llm_retry,
    log_llm_success,
)
from ...utils.logging import get_logger
from ..base import BaseLLMProvider
from .error_handler import parse_api_error_response, should_fallback_to_basic_json
from .json_utils import clean_json_response, repair_truncated_json
from .models import (
    CONTEXT_SAFETY_MARGIN,
    DEFAULT_CONTEXT_WINDOW,
    MODEL_CONTEXT_WINDOWS,
    MODELS_WITH_STRUCTURED_OUTPUT_ISSUES,
)
from .payload_builder import build_messages, build_payload
from .retry_handler import calculate_retry_backoff, is_retryable_status
from .token_calculator import (
    calculate_effective_max_tokens,
    calculate_prompt_tokens_estimate,
    calculate_schema_overhead,
    get_model_context_window,
    get_model_max_output,
)

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
        max_tokens: int | None = 2048,
        site_url: str | None = None,
        site_name: str | None = None,
        **kwargs: Any,
    ):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key (if not provided, uses OPENROUTER_API_KEY env var)
            base_url: Base URL for OpenRouter API
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response (default: 2048)
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

        # Ensure max_tokens has a default value
        if max_tokens is None:
            max_tokens = 2048

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
        timeout_config = httpx.Timeout(
            connect=10.0,
            read=timeout,
            write=30.0,
            pool=5.0,
        )
        self.client = httpx.Client(
            timeout=timeout_config,
            limits=httpx.Limits(max_keepalive_connections=5,
                                max_connections=10),
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
            timeout_config = httpx.Timeout(
                connect=10.0,
                read=self._timeout,
                write=30.0,
                pool=5.0,
            )
            self._async_client = httpx.AsyncClient(
                timeout=timeout_config,
                limits=httpx.Limits(
                    max_keepalive_connections=5, max_connections=10),
                headers=self._headers,
            )
        return self._async_client

    def __enter__(self) -> "OpenRouterProvider":
        """Enter context manager for synchronous usage."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Exit context manager and cleanup resources."""
        self.close()
        return False

    async def __aenter__(self) -> "OpenRouterProvider":
        """Enter async context manager for asynchronous usage."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Exit async context manager and cleanup resources."""
        await self.close_async()
        return False

    def close(self) -> None:
        """Close synchronous HTTP client."""
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
                logger.debug(
                    "openrouter_client_closed",
                    base_url=self.base_url,
                )
            except Exception as e:
                logger.warning(
                    "openrouter_client_cleanup_failed",
                    base_url=self.base_url,
                    error=str(e),
                )

    async def close_async(self) -> None:
        """Close asynchronous HTTP client."""
        if hasattr(self, "_async_client") and self._async_client:
            try:
                await self._async_client.aclose()
                logger.debug(
                    "openrouter_async_client_closed",
                    base_url=self.base_url,
                )
            except Exception as e:
                logger.warning(
                    "openrouter_async_client_cleanup_failed",
                    base_url=self.base_url,
                    error=str(e),
                )
        # Also close sync client
        self.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass

    def check_connection(self) -> bool:
        """Check if OpenRouter is accessible.

        Returns:
            True if OpenRouter is accessible, False otherwise
        """
        try:
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
            List of model identifiers
        """
        try:
            response = self.client.get(f"{self.base_url}/models")
            response.raise_for_status()
            data = response.json()

            models = [model["id"] for model in data.get("data", [])]
            logger.info("openrouter_list_models_success",
                        model_count=len(models))
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
            model: Model identifier
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            format: Response format
            json_schema: JSON schema for structured output
            stream: Enable streaming (not implemented)
            reasoning_enabled: Enable reasoning mode

        Returns:
            Response dictionary

        Raises:
            NotImplementedError: If streaming is requested
            httpx.HTTPError: If request fails after retries
        """
        if stream:
            raise NotImplementedError("Streaming is not yet supported")

        # Build messages
        messages = build_messages(prompt, system)

        # Calculate tokens
        prompt_tokens_estimate = calculate_prompt_tokens_estimate(
            prompt, system)
        schema_overhead = calculate_schema_overhead(json_schema)
        effective_max_tokens = calculate_effective_max_tokens(
            model=model,
            prompt_tokens_estimate=prompt_tokens_estimate,
            schema_overhead=schema_overhead,
            json_schema=json_schema,
            configured_max_tokens=self.max_tokens,
        )

        # Log token calculations
        self._log_token_calculations(
            model=model,
            prompt_tokens_estimate=prompt_tokens_estimate,
            effective_max_tokens=effective_max_tokens,
            json_schema=json_schema,
        )

        # Build payload (skip response_format for problematic models)
        if json_schema and model in MODELS_WITH_STRUCTURED_OUTPUT_ISSUES:
            logger.info(
                "skipping_response_format_for_model",
                model=model,
                reason="Model has known structured output issues",
                schema_name=json_schema.get("name", "unknown"),
            )
            # Build payload without response_format
            payload = build_payload(
                model=model,
                messages=messages,
                temperature=temperature,
                effective_max_tokens=effective_max_tokens,
                json_schema=None,  # Skip response_format
                format=format,
                reasoning_enabled=reasoning_enabled,
            )
        else:
            payload = build_payload(
                model=model,
                messages=messages,
                temperature=temperature,
                effective_max_tokens=effective_max_tokens,
                json_schema=json_schema,
                format=format,
                reasoning_enabled=reasoning_enabled,
            )

        # Start request timing
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

        # Retry logic for HTTP errors
        response = self._execute_with_retry(
            payload=payload,
            model=model,
            request_start_time=request_start_time,
        )

        # Process response with retry for empty completions
        max_empty_retries = 2
        for attempt in range(max_empty_retries + 1):
            try:
                return self._process_response(
                    response=response,
                    model=model,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    format=format,
                    json_schema=json_schema,
                    stream=stream,
                    reasoning_enabled=reasoning_enabled,
                    request_start_time=request_start_time,
                    effective_max_tokens=effective_max_tokens,
                    payload=payload,
                )
            except ValueError as e:
                if "empty completion" in str(e).lower() and attempt < max_empty_retries:
                    logger.warning(
                        "retrying_empty_completion",
                        model=model,
                        attempt=attempt + 1,
                        max_attempts=max_empty_retries + 1,
                    )
                    time.sleep(1.0 * (attempt + 1))  # Backoff
                    # Re-execute the request
                    request_start_time = time.time()
                    response = self._execute_with_retry(
                        payload=payload,
                        model=model,
                        request_start_time=request_start_time,
                    )
                    continue
                raise

        # Should not reach here, but satisfy type checker
        raise RuntimeError("Unexpected retry loop exit")

    def _log_token_calculations(
        self,
        model: str,
        prompt_tokens_estimate: int,
        effective_max_tokens: int,
        json_schema: dict[str, Any] | None,
    ) -> None:
        """Log token calculation details."""
        if not json_schema:
            return

        context_window = get_model_context_window(model)
        model_max_output = get_model_max_output(model)
        max_allowed_by_context = (
            context_window - prompt_tokens_estimate - CONTEXT_SAFETY_MARGIN
        )

        # Handle case where max_tokens might be None (shouldn't happen but be defensive)
        configured_max = self.max_tokens if self.max_tokens is not None else 2048
        desired_max_tokens = max(
            configured_max,
            int(prompt_tokens_estimate * 4.0),
        )

        if effective_max_tokens < desired_max_tokens:
            reduction_reason = []
            if effective_max_tokens == max_allowed_by_context:
                reduction_reason.append("context_window_limit")
            if effective_max_tokens == model_max_output:
                reduction_reason.append("model_output_limit")

            if "context_window_limit" in reduction_reason:
                logger.warning(
                    "reduced_max_tokens_for_context_window",
                    model=model,
                    desired_max_tokens=desired_max_tokens,
                    effective_max_tokens=effective_max_tokens,
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    context_window=context_window,
                    model_max_output=model_max_output,
                )
            else:
                logger.debug(
                    "max_tokens_limited_by_model_output",
                    model=model,
                    desired_max_tokens=desired_max_tokens,
                    effective_max_tokens=effective_max_tokens,
                    model_max_output=model_max_output,
                )

    def _execute_with_retry(
        self,
        payload: dict[str, Any],
        model: str,
        request_start_time: float,
    ) -> httpx.Response:
        """Execute request with retry logic."""
        max_retries = 3
        last_exception: Exception | None = None

        for attempt in range(max_retries):
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers,
                )
                response.raise_for_status()
                return response
            except httpx.RequestError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = float(2**attempt)
                    log_llm_retry(
                        model=model,
                        operation="openrouter_generate",
                        attempt=attempt + 1,
                        max_attempts=max_retries,
                        reason="network_error",
                        error=e,
                        wait_seconds=wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                break
            except httpx.HTTPStatusError as e:
                last_exception = e
                status_code = e.response.status_code

                if status_code == 400:
                    self._log_400_error(e, model, payload)

                if is_retryable_status(status_code) and attempt < max_retries - 1:
                    wait_time = calculate_retry_backoff(
                        status_code=status_code,
                        attempt=attempt,
                        response=e.response,
                    )
                    log_llm_retry(
                        model=model,
                        operation="openrouter_generate",
                        attempt=attempt + 1,
                        max_attempts=max_retries,
                        reason=f"http_status_{status_code}",
                        error=e,
                        status_code=status_code,
                        wait_seconds=wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                raise

        if last_exception:
            log_llm_error(
                model=model,
                operation="openrouter_generate",
                start_time=request_start_time,
                error=last_exception,
                error_type=type(last_exception).__name__,
                retryable=isinstance(last_exception, httpx.HTTPStatusError)
                and is_retryable_status(
                    getattr(last_exception.response, "status_code", 0)
                ),
            )
            raise last_exception

        raise RuntimeError("Unexpected retry loop exit")

    def _log_400_error(
        self, error: httpx.HTTPStatusError, model: str, payload: dict[str, Any]
    ) -> None:
        """Log 400 error details."""
        raw_response_text = error.response.text[:1000]
        error_details = {}
        error_type = None
        error_message = str(error)

        try:
            error_json = error.response.json()
            error_details = error_json
            if "error" in error_json:
                error_msg = error_json.get("error", {})
                if isinstance(error_msg, dict):
                    error_type = error_msg.get("type", "")
                    error_message = error_msg.get("message", "")
        except Exception:
            error_details = {"raw_response": raw_response_text}

        logger.error(
            "openrouter_400_error_in_retry_loop",
            model=model,
            error_type=error_type,
            error_message=error_message,
            raw_response=raw_response_text,
            error_json=error_details,
            payload_preview=str(payload)[:500],
        )

    def _process_response(
        self,
        response: httpx.Response,
        model: str,
        prompt: str,
        system: str,
        temperature: float,
        format: str,
        json_schema: dict[str, Any] | None,
        stream: bool,
        reasoning_enabled: bool,
        request_start_time: float,
        effective_max_tokens: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Process API response and handle errors."""
        try:
            result = response.json()

            # Validate response structure
            choices = result.get("choices", [])
            if not choices:
                raise ValueError(
                    f"OpenRouter returned empty choices array. Response: {str(result)[:500]}"
                )

            first_choice = choices[0]
            message = first_choice.get("message", {})
            completion = self._extract_completion(
                message, model, result, json_schema)

            # Clean JSON if needed
            if json_schema or format == "json":
                completion = clean_json_response(completion, model)

            # Get usage stats
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            finish_reason = first_choice.get("finish_reason", "stop")

            # Log success
            context_window = MODEL_CONTEXT_WINDOWS.get(
                model, DEFAULT_CONTEXT_WINDOW)
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

            # Handle truncation
            if (
                finish_reason == "length"
                or self._is_json_truncated(completion, json_schema)
            ) and json_schema:
                completion = self._retry_with_more_tokens(
                    payload=payload,
                    model=model,
                    effective_max_tokens=effective_max_tokens,
                    json_schema=json_schema,
                    finish_reason=finish_reason,
                    request_start_time=request_start_time,
                )

            return {
                "response": completion,
                "model": result.get("model"),
                "finish_reason": finish_reason,
                "usage": usage,
            }

        except httpx.HTTPStatusError as e:
            self._handle_http_error(
                error=e,
                model=model,
                prompt=prompt,
                system=system,
                temperature=temperature,
                format=format,
                json_schema=json_schema,
                stream=stream,
                reasoning_enabled=reasoning_enabled,
                request_start_time=request_start_time,
            )
            raise
        except ValueError as e:
            error_msg = str(e)
            # Check if this is an empty completion error (retryable)
            is_empty_completion = "empty completion" in error_msg.lower()
            log_llm_error(
                model=model,
                operation="openrouter_generate",
                start_time=request_start_time,
                error=e,
                error_type=type(e).__name__,
                retryable=is_empty_completion,
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

    def _extract_completion(
        self,
        message: dict[str, Any],
        model: str,
        result: dict[str, Any],
        json_schema: dict[str, Any] | None,
    ) -> str:
        """Extract completion from response message."""
        completion = message.get("content")

        if completion is None or completion == "":
            # Try alternative fields
            if "reasoning" in message and message["reasoning"]:
                completion = message["reasoning"]
            elif "refusal" in message and message["refusal"]:
                completion = message["refusal"]
            else:
                finish_reason = result["choices"][0].get(
                    "finish_reason", "unknown")
                logger.warning(
                    "empty_completion_from_openrouter",
                    model=model,
                    message_keys=list(message.keys()),
                    finish_reason=finish_reason,
                    has_json_schema=bool(json_schema),
                )

                if json_schema:
                    # Retry without schema
                    raise ValueError(
                        f"Model {model} returned empty completion. "
                        f"This may indicate structured output issues."
                    )
                completion = ""

        return completion

    def _is_json_truncated(
        self, completion: str, json_schema: dict[str, Any] | None
    ) -> bool:
        """Check if JSON response is truncated."""
        if not json_schema:
            return False

        try:
            json.loads(completion)
            return False
        except (json.JSONDecodeError, ValueError):
            return True

    def _retry_with_more_tokens(
        self,
        payload: dict[str, Any],
        model: str,
        effective_max_tokens: int,
        json_schema: dict[str, Any] | None,
        finish_reason: str,
        request_start_time: float,
    ) -> str:
        """Retry request with increased max_tokens."""
        model_max_output = get_model_max_output(model)
        max_retry_tokens = model_max_output

        if effective_max_tokens >= max_retry_tokens:
            raise ValueError(
                f"Response truncated at {effective_max_tokens} tokens. "
                f"Cannot increase further."
            )

        retry_max_tokens = min(effective_max_tokens * 2, max_retry_tokens)
        log_llm_retry(
            model=model,
            operation="openrouter_generate",
            attempt=1,
            max_attempts=2,
            reason="response_truncated",
            original_max_tokens=effective_max_tokens,
            retry_max_tokens=retry_max_tokens,
        )

        payload["max_tokens"] = retry_max_tokens
        retry_response = self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers,
        )
        retry_response.raise_for_status()

        retry_result = retry_response.json()

        # Validate retry response structure
        retry_choices = retry_result.get("choices", [])
        if not retry_choices:
            raise ValueError(
                f"OpenRouter retry returned empty choices. Response: {str(retry_result)[:500]}"
            )

        retry_message = retry_choices[0].get("message", {})
        retry_completion: str = retry_message.get("content", "")

        if retry_completion and json_schema:
            retry_completion = clean_json_response(retry_completion, model)

        return retry_completion

    def _handle_http_error(
        self,
        error: httpx.HTTPStatusError,
        model: str,
        prompt: str,
        system: str,
        temperature: float,
        format: str,
        json_schema: dict[str, Any] | None,
        stream: bool,
        reasoning_enabled: bool,
        request_start_time: float,
    ) -> None:
        """Handle HTTP errors with fallback logic."""
        error_info = parse_api_error_response(error, model, json_schema)

        log_llm_error(
            model=model,
            operation="openrouter_generate",
            start_time=request_start_time,
            error=error,
            error_type=error_info["error_type"] or "HTTPStatusError",
            status_code=error.response.status_code,
            retryable=is_retryable_status(error.response.status_code),
            prompt_length=len(prompt),
            system_length=len(system),
            error_details=error_info["error_details"],
            had_json_schema=bool(json_schema),
        )

        # Enhanced fallback for Grok models: try with reasoning enabled first
        if (
            should_fallback_to_basic_json(
                model, error.response.status_code, json_schema)
            and "grok" in model.lower()
            and json_schema
            and not reasoning_enabled
        ):
            logger.info(
                "grok_structured_output_failed_trying_with_reasoning",
                model=model,
                error_message=error_info["error_message"],
            )
            try:
                # Try again with reasoning enabled
                return self.generate(
                    model=model,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    format=format,
                    json_schema=json_schema,
                    stream=stream,
                    reasoning_enabled=True,  # Enable reasoning for fallback
                )
            except Exception as retry_error:
                logger.warning(
                    "grok_reasoning_fallback_failed",
                    model=model,
                    retry_error=str(retry_error),
                )
                # Continue to basic JSON fallback

        # Fallback to basic JSON for problematic models
        if should_fallback_to_basic_json(
            model, error.response.status_code, json_schema
        ):
            logger.warning(
                "structured_output_failed_fallback_to_json",
                model=model,
                error_message=error_info["error_message"],
            )
            try:
                self.generate(
                    model=model,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    format="json",
                    json_schema=None,
                    stream=stream,
                    reasoning_enabled=reasoning_enabled,
                )
            except Exception:
                pass  # Re-raise original error

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

        Args:
            model: Model identifier
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            json_schema: JSON schema for structured output
            reasoning_enabled: Enable reasoning mode

        Returns:
            Parsed JSON response

        Raises:
            json.JSONDecodeError: If response is not valid JSON
            ValueError: If model returns empty completion after fallback
        """
        try:
            result = self.generate(
                model=model,
                prompt=prompt,
                system=system,
                temperature=temperature,
                format="json" if not json_schema else "",
                json_schema=json_schema,
                reasoning_enabled=reasoning_enabled,
            )
        except ValueError as e:
            # Enhanced fallback for Grok models: try with reasoning enabled first
            if (
                "empty completion" in str(e).lower()
                and json_schema
                and "grok" in model.lower()
                and not reasoning_enabled
            ):
                logger.info(
                    "generate_json_grok_fallback_with_reasoning",
                    model=model,
                    reason="Empty completion with json_schema, trying with reasoning enabled",
                )
                try:
                    result = self.generate(
                        model=model,
                        prompt=prompt,
                        system=system,
                        temperature=temperature,
                        format="",  # Use schema-based format
                        json_schema=json_schema,
                        reasoning_enabled=True,  # Enable reasoning for fallback
                    )
                    # Check if reasoning fallback also returned empty
                    response_text = result.get("response", "")
                    if not response_text or response_text.strip() == "":
                        raise ValueError(
                            f"Model {model} returned empty completion even with reasoning enabled."
                        ) from e
                except Exception as reasoning_error:
                    logger.warning(
                        "grok_reasoning_fallback_failed_in_generate_json",
                        model=model,
                        error=str(reasoning_error),
                    )
                    # Continue to basic JSON fallback

            # If structured output failed (empty completion), try fallback without schema
            if "empty completion" in str(e).lower() and json_schema:
                logger.info(
                    "generate_json_fallback_without_schema",
                    model=model,
                    reason="Empty completion with json_schema, retrying with json_object",
                )
                result = self.generate(
                    model=model,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    format="json",  # Use json_object format instead
                    json_schema=None,  # No schema
                    reasoning_enabled=reasoning_enabled,
                )
                # Check if fallback also returned empty
                response_text = result.get("response", "")
                if not response_text or response_text.strip() == "":
                    raise ValueError(
                        f"Model {model} returned empty completion. "
                        f"This may indicate structured output issues."
                    ) from e
            else:
                raise

        response_text = result.get("response", "{}")
        cleaned_text = clean_json_response(response_text, model)

        try:
            return cast(dict[str, Any], json.loads(cleaned_text))
        except json.JSONDecodeError as e:
            # Try repair
            if "Unterminated" in str(e) or "Expecting" in str(e):
                repaired_text = repair_truncated_json(cleaned_text)
                try:
                    return cast(dict[str, Any], json.loads(repaired_text))
                except json.JSONDecodeError:
                    pass

            logger.error(
                "openrouter_json_parse_error",
                model=model,
                error=str(e),
                response_preview=response_text[:500],
            )
            raise

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
        """Generate a completion asynchronously.

        Args:
            model: Model identifier
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            format: Response format
            json_schema: JSON schema for structured output
            stream: Enable streaming (not implemented)
            reasoning_enabled: Enable reasoning mode

        Returns:
            Response dictionary
        """
        if stream:
            raise NotImplementedError(
                "Streaming is not yet supported in async mode")

        async_client = self._get_async_client()
        messages = build_messages(prompt, system)

        prompt_tokens_estimate = calculate_prompt_tokens_estimate(
            prompt, system)
        schema_overhead = calculate_schema_overhead(json_schema)
        effective_max_tokens = calculate_effective_max_tokens(
            model=model,
            prompt_tokens_estimate=prompt_tokens_estimate,
            schema_overhead=schema_overhead,
            json_schema=json_schema,
            configured_max_tokens=self.max_tokens,
        )

        # Skip response_format for problematic models
        if json_schema and model in MODELS_WITH_STRUCTURED_OUTPUT_ISSUES:
            payload = build_payload(
                model=model,
                messages=messages,
                temperature=temperature,
                effective_max_tokens=effective_max_tokens,
                json_schema=None,
                format=format,
                reasoning_enabled=reasoning_enabled,
            )
        else:
            payload = build_payload(
                model=model,
                messages=messages,
                temperature=temperature,
                effective_max_tokens=effective_max_tokens,
                json_schema=json_schema,
                format=format,
                reasoning_enabled=reasoning_enabled,
            )

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
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()

            # Validate response structure
            choices = result.get("choices", [])
            if not choices:
                raise ValueError(
                    f"OpenRouter async returned empty choices. Response: {str(result)[:500]}"
                )

            first_choice = choices[0]
            message = first_choice.get("message", {})
            completion = message.get("content", "")

            if json_schema or format == "json":
                completion = clean_json_response(completion, model)

            usage = result.get("usage", {})
            finish_reason = first_choice.get("finish_reason", "stop")

            log_llm_success(
                model=model,
                operation="openrouter_generate_async",
                start_time=request_start_time,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                response_length=len(completion),
                finish_reason=finish_reason,
                context_window=get_model_context_window(model),
                estimate_cost_flag=True,
            )

            return {
                "response": completion,
                "tokens": usage.get("completion_tokens", 0),
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
                and is_retryable_status(getattr(e.response, "status_code", 0)),
            )
            raise
