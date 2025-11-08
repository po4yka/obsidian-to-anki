"""Safety controls for LLM operations."""

import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any

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
    """Token bucket rate limiter for API calls."""

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

        self.requests: deque[float] = deque()
        self.tokens: deque[tuple[float, int]] = deque()
        self.lock = Lock()

    def check_and_wait(self, estimated_tokens: int = 0) -> dict[str, Any]:
        """Check rate limits and wait if necessary.

        Args:
            estimated_tokens: Estimated tokens for this request

        Returns:
            Dictionary with wait time and limit info
        """
        with self.lock:
            now = time.time()
            cutoff = now - self.window_seconds

            # Clean old entries
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            while self.tokens and self.tokens[0][0] < cutoff:
                self.tokens.popleft()

            # Calculate current usage
            current_requests = len(self.requests)
            current_tokens = sum(t[1] for t in self.tokens)

            # Check if we need to wait
            wait_time = 0.0
            reason = None

            if current_requests >= self.max_requests:
                # Need to wait for oldest request to expire
                wait_time = self.requests[0] + self.window_seconds - now
                reason = "request_limit"
            elif current_tokens + estimated_tokens > self.max_tokens:
                # Need to wait for tokens to become available
                wait_time = self.tokens[0][0] + self.window_seconds - now
                reason = "token_limit"

            if wait_time > 0:
                logger.warning(
                    "rate_limit_triggered",
                    reason=reason,
                    wait_seconds=round(wait_time, 2),
                    current_requests=current_requests,
                    max_requests=self.max_requests,
                    current_tokens=current_tokens,
                    max_tokens=self.max_tokens,
                )
                time.sleep(wait_time)

            # Record this request
            self.requests.append(time.time())
            if estimated_tokens > 0:
                self.tokens.append((time.time(), estimated_tokens))

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
