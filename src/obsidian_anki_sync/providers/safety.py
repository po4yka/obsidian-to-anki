"""Safety controls for LLM operations."""

import re
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

from limits import RateLimitItemPerMinute, RateLimitItemPerSecond
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SafetyConfig:
    """Configuration for safety controls."""

    # Resource limits
    max_prompt_length: int = 100000  # ~25K tokens
    max_response_length: int = 50000  # ~12.5K tokens
    max_concurrent_requests: int = 3  # Prevent resource exhaustion
    request_timeout_seconds: float = 600.0  # 10 minutes

    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 500000  # ~125K tokens/min

    # Memory protection
    max_memory_mb: int = 8192  # 8GB safety limit for Ollama
    warn_memory_mb: int = 6144  # Warn at 75%

    # Content safety
    block_sensitive_patterns: bool = True
    sanitize_inputs: bool = True
    validate_outputs: bool = True

    # Logging safety
    redact_sensitive_logs: bool = True
    max_log_content_length: int = 500


class RateLimiter:
    """Rate limiter for API calls using limits library."""

    def __init__(self, max_requests: int, max_tokens: int, window_seconds: int = 60):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window
            max_tokens: Maximum tokens per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.window_seconds = window_seconds

        # Initialize limits library with memory storage
        storage_backend = MemoryStorage()
        self.request_limiter = FixedWindowRateLimiter(storage_backend)
        self.token_limiter = FixedWindowRateLimiter(storage_backend)

        # Create rate limit items for limits library
        # Convert window_seconds to appropriate RateLimitItem
        if window_seconds == 60:
            self.request_limit = RateLimitItemPerMinute(max_requests)
            self.token_limit = RateLimitItemPerMinute(max_tokens)
        elif window_seconds == 1:
            self.request_limit = RateLimitItemPerSecond(max_requests)
            self.token_limit = RateLimitItemPerSecond(max_tokens)
        else:
            # For custom periods, approximate using per-minute with adjusted count
            # This is a limitation - limits library doesn't support arbitrary windows
            # We'll use per-minute and adjust the count proportionally
            adjusted_requests = int(max_requests * (60 / window_seconds))
            adjusted_tokens = int(max_tokens * (60 / window_seconds))
            self.request_limit = RateLimitItemPerMinute(adjusted_requests)
            self.token_limit = RateLimitItemPerMinute(adjusted_tokens)
            logger.warning(
                "custom_rate_limit_window",
                window_seconds=window_seconds,
                note="Using per-minute approximation",
            )

    def check_and_wait(self, estimated_tokens: int = 0) -> dict[str, Any]:
        """Check rate limits and wait if necessary.

        Args:
            estimated_tokens: Estimated tokens for this request

        Returns:
            Dictionary with wait time and limit info
        """
        wait_time = 0.0
        reason = None
        current_requests = 0
        current_tokens = 0

        # Check request limit
        try:
            # Use a unique key for this limiter instance
            request_key = "llm_requests"
            # hit() returns True if within limit, False if exceeded
            if not self.request_limiter.hit(self.request_limit, request_key):
                # Rate limit exceeded, need to wait
                # Calculate wait time (approximate - limits doesn't provide exact wait time)
                wait_time = self.window_seconds
                reason = "request_limit"
                logger.warning(
                    "rate_limit_triggered",
                    reason=reason,
                    wait_seconds=wait_time,
                    max_requests=self.max_requests,
                )
                time.sleep(wait_time)
                # Retry after waiting
                self.request_limiter.hit(self.request_limit, request_key)
        except Exception as e:
            logger.warning("rate_limit_check_error", error=str(e))
            # Continue on error

        # Check token limit
        # Note: Token tracking is simplified - we check if we can make a "request"
        # for the estimated tokens. This is approximate since limits doesn't track
        # custom token values directly.
        if estimated_tokens > 0:
            token_key = "llm_tokens"
            try:
                # Approximate: treat each token batch as a separate "request"
                # This is a limitation - for precise token tracking, we'd need
                # a custom implementation or a different library
                if not self.token_limiter.hit(self.token_limit, token_key):
                    token_wait = self.window_seconds
                    if token_wait > wait_time:
                        wait_time = token_wait
                        reason = "token_limit"
                        logger.warning(
                            "token_limit_triggered",
                            wait_seconds=token_wait,
                            estimated_tokens=estimated_tokens,
                            max_tokens=self.max_tokens,
                        )
                        time.sleep(token_wait - wait_time)
            except Exception as e:
                logger.warning("token_limit_check_error", error=str(e))

        return {
            "wait_time": wait_time,
            "reason": reason,
            "current_requests": current_requests,
            "current_tokens": current_tokens,
        }


class ConcurrencyLimiter:
    """Limits concurrent requests to prevent resource exhaustion."""

    def __init__(self, max_concurrent: int):
        """Initialize concurrency limiter.

        Args:
            max_concurrent: Maximum concurrent requests
        """
        self.max_concurrent = max_concurrent
        self.current_count = 0
        self.lock = Lock()
        self.waiting = 0

    def acquire(self) -> float:
        """Acquire a concurrency slot, waiting if necessary.

        Returns:
            Wait time in seconds
        """
        start_wait = time.time()

        with self.lock:
            self.waiting += 1

        while True:
            with self.lock:
                if self.current_count < self.max_concurrent:
                    self.current_count += 1
                    self.waiting -= 1
                    wait_time = time.time() - start_wait

                    if wait_time > 0.1:
                        logger.info(
                            "concurrency_slot_acquired",
                            wait_seconds=round(wait_time, 2),
                            active_requests=self.current_count,
                            max_concurrent=self.max_concurrent,
                        )

                    return wait_time

            # Wait briefly before checking again
            time.sleep(0.1)

    def release(self) -> None:
        """Release a concurrency slot."""
        with self.lock:
            self.current_count = max(0, self.current_count - 1)

    def get_status(self) -> dict[str, int]:
        """Get current concurrency status.

        Returns:
            Dictionary with current and waiting counts
        """
        with self.lock:
            return {
                "active": self.current_count,
                "waiting": self.waiting,
                "max": self.max_concurrent,
            }


class InputValidator:
    """Validates and sanitizes inputs to LLM."""

    # Patterns that might indicate sensitive data
    SENSITIVE_PATTERNS = [
        r"\b(?:password|passwd|pwd)\s*[:=]\s*\S+",  # Passwords
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Emails
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  # Credit cards
        r"(?:api[_-]?key|token|secret)[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_\-]+)",  # API keys
        r"sk-[A-Za-z0-9]{48}",  # OpenAI API key pattern
    ]

    @classmethod
    def validate_prompt(
        cls, prompt: str, max_length: int, sanitize: bool = True
    ) -> tuple[str, list[str]]:
        """Validate and optionally sanitize a prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum allowed length
            sanitize: Whether to sanitize sensitive patterns

        Returns:
            Tuple of (sanitized_prompt, list of warnings)
        """
        warnings = []

        # Check length
        if len(prompt) > max_length:
            warnings.append(f"Prompt exceeds max length ({len(prompt)} > {max_length})")
            logger.warning(
                "prompt_too_long",
                length=len(prompt),
                max_length=max_length,
                action="truncating",
            )
            prompt = prompt[:max_length]

        # Check for sensitive patterns
        if sanitize:
            for pattern in cls.SENSITIVE_PATTERNS:
                matches = re.findall(pattern, prompt, re.IGNORECASE)
                if matches:
                    warnings.append(f"Detected sensitive pattern: {pattern[:50]}...")
                    logger.warning(
                        "sensitive_pattern_detected",
                        pattern_preview=pattern[:50],
                        matches_count=len(matches),
                    )
                    # Redact matches
                    prompt = re.sub(pattern, "[REDACTED]", prompt, flags=re.IGNORECASE)

        # Check for excessive repetition (potential attack)
        if cls._detect_repetition(prompt):
            warnings.append(
                "Excessive repetition detected (potential prompt injection)"
            )
            logger.warning("excessive_repetition_detected", prompt_length=len(prompt))

        return prompt, warnings

    @staticmethod
    def _detect_repetition(text: str, threshold: float = 0.7) -> bool:
        """Detect excessive repetition in text.

        Args:
            text: Text to check
            threshold: Repetition threshold (0-1)

        Returns:
            True if excessive repetition detected
        """
        if len(text) < 100:
            return False

        # Sample text for efficiency
        sample = text[:1000]
        words = sample.split()

        if len(words) < 10:
            return False

        # Check word repetition
        unique_words = len(set(words))
        total_words = len(words)
        uniqueness = unique_words / total_words

        return uniqueness < (1 - threshold)


class OutputValidator:
    """Validates outputs from LLM."""

    @staticmethod
    def validate_response(
        response: str, max_length: int, expected_format: str | None = None
    ) -> tuple[bool, list[str]]:
        """Validate LLM response.

        Args:
            response: LLM response
            max_length: Maximum allowed length
            expected_format: Expected format ("json", "html", etc.)

        Returns:
            Tuple of (is_valid, list of warnings)
        """
        warnings = []
        is_valid = True

        # Check if response is empty
        if not response or not response.strip():
            warnings.append("Empty response")
            is_valid = False
            return is_valid, warnings

        # Check length
        if len(response) > max_length:
            warnings.append(
                f"Response exceeds max length ({len(response)} > {max_length})"
            )
            logger.warning(
                "response_too_long", length=len(response), max_length=max_length
            )

        # Format-specific validation
        if expected_format == "json":
            if not response.strip().startswith(("{", "[")):
                warnings.append("Expected JSON but response doesn't start with { or [")
                is_valid = False

        elif expected_format == "html":
            if "<!--" not in response:
                warnings.append("Expected HTML but no comment markers found")
                is_valid = False

        # Check for common error indicators
        error_indicators = [
            "error:",
            "exception:",
            "failed to",
            "cannot process",
            "invalid request",
        ]

        response_lower = response.lower()
        for indicator in error_indicators:
            if indicator in response_lower:
                warnings.append(f"Potential error in response: {indicator}")
                logger.warning("error_indicator_in_response", indicator=indicator)

        return is_valid, warnings


class OllamaSafetyWrapper:
    """Safety wrapper for Ollama operations."""

    def __init__(self, config: SafetyConfig | None = None):
        """Initialize safety wrapper.

        Args:
            config: Safety configuration
        """
        self.config = config or SafetyConfig()

        self.rate_limiter = RateLimiter(
            max_requests=self.config.max_requests_per_minute,
            max_tokens=self.config.max_tokens_per_minute,
        )

        self.concurrency_limiter = ConcurrencyLimiter(
            max_concurrent=self.config.max_concurrent_requests
        )

        self.input_validator = InputValidator()
        self.output_validator = OutputValidator()

        logger.info(
            "safety_wrapper_initialized",
            max_concurrent=self.config.max_concurrent_requests,
            max_requests_per_min=self.config.max_requests_per_minute,
            max_prompt_length=self.config.max_prompt_length,
        )

    def validate_input(self, prompt: str, system: str = "") -> dict[str, Any]:
        """Validate and sanitize input before sending to LLM.

        Args:
            prompt: User prompt
            system: System prompt

        Returns:
            Dictionary with sanitized prompts and validation info
        """
        # Validate user prompt
        sanitized_prompt, prompt_warnings = self.input_validator.validate_prompt(
            prompt,
            max_length=self.config.max_prompt_length,
            sanitize=self.config.sanitize_inputs,
        )

        # Validate system prompt
        sanitized_system, system_warnings = self.input_validator.validate_prompt(
            system,
            max_length=self.config.max_prompt_length // 2,
            sanitize=self.config.sanitize_inputs,
        )

        all_warnings = prompt_warnings + system_warnings

        if all_warnings:
            logger.warning(
                "input_validation_warnings",
                warnings_count=len(all_warnings),
                warnings=all_warnings[:5],  # Log first 5
            )

        # Estimate tokens
        total_chars = len(sanitized_prompt) + len(sanitized_system)
        estimated_tokens = total_chars // 4

        return {
            "prompt": sanitized_prompt,
            "system": sanitized_system,
            "warnings": all_warnings,
            "estimated_tokens": estimated_tokens,
            "sanitized": len(all_warnings) > 0,
        }

    def validate_output(
        self, response: str, expected_format: str | None = None
    ) -> dict[str, Any]:
        """Validate output from LLM.

        Args:
            response: LLM response
            expected_format: Expected format

        Returns:
            Dictionary with validation results
        """
        is_valid, warnings = self.output_validator.validate_response(
            response,
            max_length=self.config.max_response_length,
            expected_format=expected_format,
        )

        if warnings:
            logger.warning(
                "output_validation_warnings",
                warnings_count=len(warnings),
                warnings=warnings[:5],
            )

        return {"is_valid": is_valid, "warnings": warnings}

    def redact_for_logging(self, text: str) -> str:
        """Redact sensitive information for logging.

        Args:
            text: Text to redact

        Returns:
            Redacted text
        """
        if not self.config.redact_sensitive_logs:
            return text

        # Truncate if too long
        if len(text) > self.config.max_log_content_length:
            text = text[: self.config.max_log_content_length] + "...[truncated]"

        # Redact sensitive patterns
        for pattern in InputValidator.SENSITIVE_PATTERNS:
            text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)

        return text
