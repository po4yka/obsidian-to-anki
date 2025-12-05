"""Ollama provider implementation (local and cloud)."""

import contextlib
import time
from types import TracebackType
from typing import Any, Literal, cast

import httpx

from obsidian_anki_sync.utils.logging import get_logger

from .base import BaseLLMProvider
from .safety import OllamaSafetyWrapper, SafetyConfig

logger = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider supporting both local and cloud deployments.

    Supports:
    - Local Ollama: http://localhost:11434
    - Ollama Cloud: https://api.ollama.com (requires API key)

    Configuration:
        base_url: API endpoint URL (default: http://localhost:11434)
        api_key: API key for cloud deployments (optional, for cloud only)
        timeout: Request timeout in seconds (default: 2700.0 - 45 minutes)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
        timeout: float = 2700.0,
        enable_safety: bool = True,
        safety_config: SafetyConfig | None = None,
        verbose_logging: bool = False,
        **kwargs: Any,
    ):
        """Initialize Ollama provider.

        Args:
            base_url: Base URL for Ollama API (local or cloud)
            api_key: API key for Ollama Cloud (optional, not needed for local)
            timeout: Request timeout in seconds
            enable_safety: Enable safety controls
            safety_config: Custom safety configuration
            verbose_logging: Whether to log detailed initialization info
            **kwargs: Additional configuration options
        """
        super().__init__(
            verbose_logging=verbose_logging,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            **kwargs,
        )

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        # Initialize safety wrapper
        self.enable_safety = enable_safety
        self.safety: OllamaSafetyWrapper | None
        if self.enable_safety:
            self.safety = OllamaSafetyWrapper(config=safety_config)
        else:
            self.safety = None

        # Set up headers
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Initialize HTTP client with connection pooling
        # Use synchronous httpx.Client for compatibility with existing sync code
        # This provider is used in sync contexts, so async client is not needed
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers=headers,
        )

        deployment_type = "cloud" if self.api_key else "local"

        # Track first request per model (to measure model loading time)
        self._model_first_request: dict[str, bool] = {}

        logger.info(
            "ollama_provider_initialized",
            base_url=base_url,
            deployment_type=deployment_type,
            timeout=timeout,
            safety_enabled=enable_safety,
        )

    def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
                logger.debug(
                    "ollama_client_closed",
                    base_url=self.base_url,
                )
            except Exception as e:
                logger.warning(
                    "ollama_client_cleanup_failed",
                    base_url=self.base_url,
                    error=str(e),
                )

    def __enter__(self) -> "OllamaProvider":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Context manager exit with cleanup."""
        self.close()
        return False

    def __del__(self) -> None:
        """Clean up client resources on deletion."""
        with contextlib.suppress(Exception):
            self.close()

    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible.

        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return bool(response.status_code == 200)
        except Exception as e:
            logger.error(
                "ollama_connection_check_failed",
                base_url=self.base_url,
                error=str(e),
            )
            return False

    def list_models(self) -> list[str]:
        """List available models in Ollama.

        Returns:
            List of model names
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            logger.info("ollama_list_models_success", model_count=len(models))
            return models
        except Exception as e:
            logger.error("ollama_list_models_failed", error=str(e))
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
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Generate completion from Ollama.

        Args:
            model: Model name (e.g., "qwen3:8b", "qwen2.5:32b")
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            format: Response format ("json" for structured output)
            stream: Enable streaming (not implemented)

        Returns:
            Response dictionary with 'response' and 'context' keys

        Raises:
            NotImplementedError: If streaming is requested
            httpx.HTTPError: If request fails after retries
        """
        if stream:
            msg = "Streaming is not yet supported"
            raise NotImplementedError(msg)

        # Safety controls: validate and sanitize input
        if self.enable_safety and self.safety:
            validation = self.safety.validate_input(prompt, system)

            # Use sanitized versions
            prompt = validation["prompt"]
            system = validation["system"]
            estimated_tokens = validation["estimated_tokens"]

            if validation["warnings"]:
                logger.warning(
                    "input_sanitized",
                    warnings_count=len(validation["warnings"]),
                    model=model,
                )

            # Rate limiting and concurrency control
            rate_info = self.safety.rate_limiter.check_and_wait(estimated_tokens)
            if rate_info["wait_time"] > 0:
                logger.info(
                    "rate_limit_wait",
                    wait_seconds=round(rate_info["wait_time"], 2),
                    reason=rate_info["reason"],
                )

            # Acquire concurrency slot
            concurrency_wait = self.safety.concurrency_limiter.acquire()
            if concurrency_wait > 0.5:
                logger.info(
                    "concurrency_wait",
                    wait_seconds=round(concurrency_wait, 2),
                    status=self.safety.concurrency_limiter.get_status(),
                )
        else:
            estimated_tokens = (len(prompt) + len(system)) // 4

        try:
            payload: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            }

            if system:
                payload["system"] = system

            if format == "json":
                payload["format"] = "json"

            request_start_time = time.time()

            # Check if this is the first request to this model
            is_first_request = model not in self._model_first_request
            if is_first_request:
                self._model_first_request[model] = True

            logger.info(
                "ollama_generate_request",
                model=model,
                prompt_length=len(prompt),
                system_length=len(system),
                temperature=temperature,
                format=format,
                timeout=self.timeout,
                estimated_tokens=estimated_tokens,
                first_request_to_model=is_first_request,
            )

            response = self.client.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            request_duration = time.time() - request_start_time

            # Extract performance metrics from Ollama response
            eval_count = result.get("eval_count", 0)
            eval_duration = result.get("eval_duration", 0) / 1e9  # Convert to seconds
            prompt_eval_count = result.get("prompt_eval_count", 0)
            prompt_eval_duration = result.get("prompt_eval_duration", 0) / 1e9

            tokens_per_sec = eval_count / eval_duration if eval_duration > 0 else 0
            total_tokens = prompt_eval_count + eval_count

            # Store token usage in result for caller access
            result["_token_usage"] = {
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
                "total_tokens": total_tokens,
            }

            # Safety controls: validate output
            if self.enable_safety and self.safety:
                response_text = result.get("response", "")
                expected_format = format if format else None
                output_validation = self.safety.validate_output(
                    response_text, expected_format=expected_format
                )

                if output_validation["warnings"]:
                    logger.warning(
                        "output_validation_warnings",
                        warnings_count=len(output_validation["warnings"]),
                        model=model,
                    )

                if not output_validation["is_valid"]:
                    logger.error(
                        "output_validation_failed",
                        model=model,
                        warnings=output_validation["warnings"],
                    )

            # Log model loading time if this was the first request
            if is_first_request and request_duration > 30:
                logger.info(
                    "model_loading_detected",
                    model=model,
                    duration=round(request_duration, 2),
                    note="First request to this model may include loading time",
                )

            # Warn about slow operations
            if request_duration > 600:  # 10 minutes
                logger.warning(
                    "very_slow_operation_detected",
                    model=model,
                    duration=round(request_duration, 2),
                    threshold=600,
                    tokens_per_second=round(tokens_per_sec, 2),
                    recommendation="Consider using a smaller/faster model",
                )
            elif request_duration > 300:  # 5 minutes
                logger.warning(
                    "slow_operation_detected",
                    model=model,
                    duration=round(request_duration, 2),
                    threshold=300,
                    tokens_per_second=round(tokens_per_sec, 2),
                )

            logger.info(
                "ollama_generate_success",
                model=model,
                response_length=len(result.get("response", "")),
                request_duration=round(request_duration, 2),
                prompt_tokens=prompt_eval_count,
                completion_tokens=eval_count,
                total_tokens=total_tokens,
                prompt_eval_duration=round(prompt_eval_duration, 2),
                eval_duration=round(eval_duration, 2),
                tokens_per_second=round(tokens_per_sec, 2),
            )

            return cast("dict[str, Any]", result)

        except httpx.HTTPStatusError as e:
            request_duration = time.time() - request_start_time
            logger.error(
                "ollama_http_error",
                status_code=e.response.status_code,
                error=str(e),
                response_text=e.response.text[:500],
                request_duration=round(request_duration, 2),
            )
            raise
        except httpx.RequestError as e:
            request_duration = time.time() - request_start_time
            logger.error(
                "ollama_request_error",
                error=str(e),
                request_duration=round(request_duration, 2),
                timeout_configured=self.timeout,
            )
            raise
        except Exception as e:
            request_duration = time.time() - request_start_time
            logger.error(
                "ollama_unexpected_error",
                error=str(e),
                request_duration=round(request_duration, 2),
            )
            raise
        finally:
            # Always release concurrency slot
            if self.enable_safety and self.safety:
                self.safety.concurrency_limiter.release()

    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry.

        Args:
            model: Model name to pull (e.g., "qwen3:8b")

        Returns:
            True if successful, False otherwise

        Note:
            This is a blocking call that may take several minutes for large models.
        """
        logger.info("ollama_pull_model_start", model=model)

        try:
            # Use a longer timeout for model pulls
            response = self.client.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=httpx.Timeout(1800.0),  # 30 minutes
            )
            response.raise_for_status()

            logger.info("ollama_pull_model_success", model=model)
            return True

        except Exception as e:
            logger.error("ollama_pull_model_failed", model=model, error=str(e))
            return False
