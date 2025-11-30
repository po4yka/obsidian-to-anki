"""Resilience patterns for specialized agent system.

Provides retry with backoff (via tenacity), rate limiting, bulkhead isolation,
and confidence validation to prevent cascading failures and ensure system reliability.

Note: Circuit breaker pattern is implemented using tenacity's retry_error_callback
and custom state tracking. For simpler use cases, use create_retry_decorator().
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# ============================================================================
# Circuit Breaker Pattern (Tenacity-based)
# ============================================================================


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, reject requests
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str = "Circuit breaker is OPEN"):
        self.message = message
        super().__init__(self.message)


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        self.message = message
        super().__init__(self.message)


class ResourceExhaustedError(Exception):
    """Raised when bulkhead resources are exhausted."""

    def __init__(self, message: str = "Resource exhausted"):
        self.message = message
        super().__init__(self.message)


class LowConfidenceError(Exception):
    """Raised when confidence validation fails."""

    def __init__(self, message: str = "Confidence below threshold"):
        self.message = message
        super().__init__(self.message)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 1


class CircuitBreaker:
    """Circuit breaker pattern implementation using tenacity.

    Wraps tenacity retry logic with circuit breaker state management.
    Prevents cascading failures by stopping requests to failing services.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        """Initialize circuit breaker.

        Args:
            name: Name identifier for this circuit breaker
            config: Configuration for circuit breaker behavior
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state (thread-safe)."""
        with self._lock:
            return self._state

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitBreakerState.OPEN:
                msg = f"Circuit breaker '{self.name}' is OPEN"
                raise CircuitBreakerError(msg)

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    msg = f"Circuit breaker '{self.name}' HALF_OPEN call limit reached"
                    raise CircuitBreakerError(msg)
                self._half_open_calls += 1

        # Execute with tenacity retry
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(1),  # No retry, circuit breaker handles it
                reraise=True,
            ):
                with attempt:
                    result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception:
            self._record_failure()
            raise

    def _check_state_transition(self) -> None:
        """Check if circuit should transition states (must hold lock)."""
        if self._state == CircuitBreakerState.OPEN:
            if self._last_failure_time and (
                time.time() - self._last_failure_time > self.config.recovery_timeout
            ):
                self._state = CircuitBreakerState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(
                    "circuit_breaker_half_open",
                    name=self.name,
                    recovery_timeout=self.config.recovery_timeout,
                )

    def _record_success(self) -> None:
        """Record successful call."""
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                logger.info("circuit_breaker_closed", name=self.name)
            elif self._state == CircuitBreakerState.CLOSED:
                self._failure_count = 0

    def _record_failure(self) -> None:
        """Record failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
                logger.warning(
                    "circuit_breaker_opened_from_half_open",
                    name=self.name,
                    failure_count=self._failure_count,
                )
            elif (
                self._state == CircuitBreakerState.CLOSED
                and self._failure_count >= self.config.failure_threshold
            ):
                self._state = CircuitBreakerState.OPEN
                logger.warning(
                    "circuit_breaker_opened",
                    name=self.name,
                    failure_count=self._failure_count,
                    threshold=self.config.failure_threshold,
                )

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == CircuitBreakerState.OPEN

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0
            logger.info("circuit_breaker_reset", name=self.name)


# ============================================================================
# Tenacity-based Retry Helpers
# ============================================================================


def create_retry_decorator(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 1.0,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Create a tenacity retry decorator with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (tenacity 'min' parameter)
        max_delay: Maximum delay cap in seconds
        jitter: Maximum jitter to add (in seconds)
        retry_exceptions: Tuple of exceptions to retry on

    Returns:
        Configured retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential_jitter(
            initial=initial_delay,
            max=max_delay,
            jitter=jitter,
        ),
        retry=retry_if_exception_type(retry_exceptions),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log retry attempts for observability."""
    logger.warning(
        "tenacity_retry",
        attempt=retry_state.attempt_number,
        wait_seconds=round(retry_state.next_action.sleep if retry_state.next_action else 0, 2),
        error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
    )


class RetryWithJitter:
    """Retry logic with exponential backoff and jitter using tenacity.

    This is a wrapper class for backward compatibility. For new code,
    prefer using create_retry_decorator() directly.
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,  # Kept for API compatibility, not used
        jitter: bool = True,
        jitter_range: float = 0.1,  # Kept for API compatibility
    ):
        """Initialize retry with jitter.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay cap in seconds
            exponential_base: Base for exponential backoff (ignored, tenacity uses 2)
            jitter: Whether to add jitter to delays
            jitter_range: Jitter range as fraction of delay (converted to seconds)
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        # Convert jitter_range (fraction) to seconds for tenacity
        self._jitter_seconds = max_delay * jitter_range if jitter else 0

    def execute(
        self,
        func: Callable[..., T],
        exceptions: tuple[type[Exception], ...] = (Exception,),
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function with retry and jitter.

        Args:
            func: Function to execute
            exceptions: Tuple of exceptions to catch and retry
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            Last exception if all retries exhausted
        """
        retryer = Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential_jitter(
                initial=self.initial_delay,
                max=self.max_delay,
                jitter=self._jitter_seconds,
            ),
            retry=retry_if_exception_type(exceptions),
            before_sleep=_log_retry_attempt,
            reraise=True,
        )

        try:
            for attempt in retryer:
                with attempt:
                    return func(*args, **kwargs)
        except RetryError as e:
            logger.error(
                "retry_exhausted",
                func=func.__name__,
                attempts=self.max_retries,
                error=str(e.last_attempt.exception()) if e.last_attempt else "unknown",
            )
            raise

        # Should never reach here, but satisfy type checker
        return func(*args, **kwargs)


class RateLimiter:
    """Rate limiter using token bucket algorithm.

    Prevents overwhelming the system or external services.
    """

    def __init__(self, max_calls_per_minute: int = 60):
        """Initialize rate limiter.

        Args:
            max_calls_per_minute: Maximum number of calls allowed per minute
        """
        self.max_calls = max_calls_per_minute
        self.call_times: list[float] = []
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        """Acquire permission to make a call.

        Returns:
            True if permission granted, False if rate limit exceeded
        """
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.call_times = [t for t in self.call_times if now - t < 60]

            if len(self.call_times) < self.max_calls:
                self.call_times.append(now)
                return True

            return False

    def wait_if_needed(self, timeout: float = 30.0) -> bool:
        """Wait if rate limit is exceeded.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if permission acquired, False if timeout
        """
        start_time = time.time()
        while not self.acquire():
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.1)
        return True


class Bulkhead:
    """Bulkhead pattern for resource isolation.

    Prevents resource exhaustion by limiting concurrent executions per domain.
    """

    def __init__(self, max_concurrent: int = 3, timeout: float = 30.0):
        """Initialize bulkhead.

        Args:
            max_concurrent: Maximum concurrent executions allowed
            timeout: Timeout for acquiring resources in seconds
        """
        self.semaphore = threading.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.active_count = 0
        self.lock = threading.Lock()

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with bulkhead isolation.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            ResourceExhaustedError: If resources unavailable within timeout
        """
        if not self.semaphore.acquire(timeout=self.timeout):
            msg = f"Bulkhead resources exhausted (max: {self.max_concurrent})"
            raise ResourceExhaustedError(msg)

        try:
            with self.lock:
                self.active_count += 1

            try:
                return func(*args, **kwargs)
            finally:
                with self.lock:
                    self.active_count -= 1
        finally:
            self.semaphore.release()

    def get_active_count(self) -> int:
        """Get current number of active executions."""
        with self.lock:
            return self.active_count


@dataclass
class ConfidenceValidationResult:
    """Result of confidence validation."""

    is_valid: bool
    reason: str = ""
    suspicious_patterns: list[str] = field(default_factory=list)


class ConfidenceValidator:
    """Validates agent outputs based on confidence scores and patterns.

    Prevents low-quality repairs from propagating through the system.
    """

    def __init__(self, min_confidence: float = 0.7, validate_patterns: bool = True):
        """Initialize confidence validator.

        Args:
            min_confidence: Minimum confidence threshold
            validate_patterns: Whether to check for suspicious patterns
        """
        self.min_confidence = min_confidence
        self.validate_patterns = validate_patterns
        self.validation_history: list[dict[str, Any]] = []

    def validate(self, result: Any) -> ConfidenceValidationResult:
        """Validate agent result based on confidence and patterns.

        Args:
            result: AgentResult or similar object with confidence and content

        Returns:
            ConfidenceValidationResult with validation outcome
        """
        # Extract confidence
        confidence = getattr(result, "confidence", 0.0)

        # Check confidence threshold
        if confidence < self.min_confidence:
            self._record_validation(
                confidence=confidence,
                reason="Below threshold",
                is_valid=False,
            )
            return ConfidenceValidationResult(
                is_valid=False,
                reason=f"Confidence {confidence} below threshold {self.min_confidence}",
            )

        # Check for suspicious patterns if enabled
        if self.validate_patterns:
            content = getattr(result, "content", None) or getattr(
                result, "repaired_content", ""
            )
            if content:
                suspicious = self._detect_suspicious_patterns(content)
                if suspicious:
                    self._record_validation(
                        confidence=confidence,
                        reason="Suspicious patterns detected",
                        is_valid=False,
                        patterns=suspicious,
                    )
                    return ConfidenceValidationResult(
                        is_valid=False,
                        reason="Suspicious patterns detected",
                        suspicious_patterns=suspicious,
                    )

        # Validation passed
        self._record_validation(confidence=confidence, reason="Valid", is_valid=True)
        return ConfidenceValidationResult(is_valid=True, reason="Valid")

    def _detect_suspicious_patterns(self, content: str) -> list[str]:
        """Detect suspicious patterns in content.

        Args:
            content: Content to analyze

        Returns:
            List of detected suspicious patterns
        """
        suspicious = []

        # Check for excessive placeholders
        placeholder_count = content.count("[PLACEHOLDER]") + content.count("[TODO]")
        if placeholder_count > len(content) / 100:  # More than 1% placeholders
            suspicious.append("excessive_placeholders")

        # Check for incomplete repairs
        if content.count("...") > 5:
            suspicious.append("excessive_ellipsis")

        # Check for empty or very short content
        if len(content.strip()) < 50:
            suspicious.append("content_too_short")

        # Check for repetitive patterns (possible corruption)
        if self._has_repetitive_corruption(content):
            suspicious.append("repetitive_corruption")

        return suspicious

    def _has_repetitive_corruption(self, content: str) -> bool:
        """Check for repetitive corruption patterns."""
        import re

        corruption_patterns = [
            r"(.)\1{10,}",  # Same character repeated 10+ times
            r"[a-zA-Z]\d{1,2}[a-zA-Z]\d{1,2}",  # Mixed alphanumeric patterns
        ]

        return any(re.search(pattern, content) for pattern in corruption_patterns)

    def _record_validation(
        self,
        confidence: float,
        reason: str,
        is_valid: bool,
        patterns: list[str] | None = None,
    ) -> None:
        """Record validation decision for audit trail."""
        self.validation_history.append(
            {
                "timestamp": time.time(),
                "confidence": confidence,
                "reason": reason,
                "is_valid": is_valid,
                "patterns": patterns or [],
            }
        )

        # Keep only last 1000 entries
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
