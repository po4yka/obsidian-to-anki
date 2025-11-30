"""Application service for handling retry logic with exponential backoff."""

import time
from collections.abc import Callable
from typing import Any

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        backoff_multiplier: float = 2.0,
        retry_on_exceptions: tuple | None = None,
    ):
        """Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for first retry
            max_delay: Maximum delay between retries
            backoff_multiplier: Multiplier for exponential backoff
            retry_on_exceptions: Tuple of exception types to retry on
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.retry_on_exceptions = retry_on_exceptions or (Exception,)

    def get_retry_delays(self) -> list[float]:
        """Get list of delays for each retry attempt.

        Returns:
            List of delay times in seconds
        """
        delays = []
        delay = self.base_delay

        for _ in range(self.max_retries):
            delays.append(min(delay, self.max_delay))
            delay *= self.backoff_multiplier

        return delays


class RetryResult:
    """Result of a retry operation."""

    def __init__(
        self,
        success: bool,
        result: Any = None,
        error: Exception | None = None,
        attempts: int = 0,
        total_time: float = 0.0,
        retry_delays: list[float] | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize retry result.

        Args:
            success: Whether the operation ultimately succeeded
            result: The result if successful
            error: The final error if failed
            attempts: Number of attempts made (including initial)
            total_time: Total time spent on all attempts
            retry_delays: List of delays used between retries
        """
        self.success = success
        self.result = result
        self.error = error
        self.attempts = attempts
        self.total_time = total_time
        self.retry_delays = retry_delays or []
        self.context = context or {}


class RetryHandler:
    """Service for handling retry logic with exponential backoff.

    This service encapsulates retry behavior, making it reusable across
    different parts of the application following the Single Responsibility
    Principle.
    """

    def __init__(self, config: RetryConfig | None = None):
        """Initialize retry handler.

        Args:
            config: Retry configuration, uses defaults if None
        """
        self.config = config or RetryConfig()
        self._retry_delays = self.config.get_retry_delays()

        logger.debug(
            "retry_handler_initialized",
            max_retries=self.config.max_retries,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
        )

    def execute_with_retry(
        self,
        operation: Callable[[], Any],
        context: dict[str, Any | None] | None = None,
    ) -> RetryResult:
        """Execute an operation with retry logic.

        Args:
            operation: Callable that performs the operation
            context: Optional context for logging

        Returns:
            RetryResult with outcome and metadata
        """
        context = context or {}
        start_time = time.time()
        last_error = None

        logger.debug("retry_operation_start", **context)

        # Initial attempt (attempt 0)
        try:
            result = operation()
            total_time = time.time() - start_time

            logger.debug(
                "retry_operation_success_first_attempt",
                attempts=1,
                total_time=total_time,
                **context,
            )

            return RetryResult(
                success=True,
                result=result,
                attempts=1,
                total_time=total_time,
                context=context,
            )

        except self.config.retry_on_exceptions as e:
            last_error = e
            logger.debug(
                "retry_operation_failed_first_attempt", error=str(e), **context
            )

        # Retry attempts
        for attempt in range(self.config.max_retries):
            delay = (
                self._retry_delays[attempt]
                if attempt < len(self._retry_delays)
                else self.config.max_delay
            )

            logger.debug(
                "retry_attempt",
                attempt=attempt + 1,
                max_attempts=self.config.max_retries,
                delay=delay,
                **context,
            )

            # Wait before retry
            if delay > 0:
                time.sleep(delay)

            try:
                result = operation()
                total_time = time.time() - start_time

                logger.info(
                    "retry_operation_success_after_retries",
                    attempts=attempt + 2,  # +1 for initial attempt, +1 for 0-indexing
                    total_time=total_time,
                    **context,
                )

                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 2,
                    total_time=total_time,
                    retry_delays=self._retry_delays[: attempt + 1],
                    context=context,
                )

            except self.config.retry_on_exceptions as e:
                last_error = e
                logger.warning(
                    "retry_attempt_failed",
                    attempt=attempt + 1,
                    max_attempts=self.config.max_retries,
                    error=str(e),
                    **context,
                )

        # All attempts failed
        total_time = time.time() - start_time

        logger.error(
            "retry_operation_failed_all_attempts",
            attempts=self.config.max_retries + 1,
            total_time=total_time,
            final_error=str(last_error),
            **context,
        )

        return RetryResult(
            success=False,
            error=last_error,
            attempts=self.config.max_retries + 1,
            total_time=total_time,
            retry_delays=self._retry_delays,
            context=context,
        )

    def get_adaptive_config(
        self,
        error_type: str,
        base_config: dict[str, int | None] | None = None,
    ) -> RetryConfig:
        """Get adaptive retry config based on error type.

        Args:
            error_type: Type of error encountered
            base_config: Base configuration mapping error types to retry counts

        Returns:
            Adapted RetryConfig for this error type
        """
        base_config = base_config or {}

        # Default retry counts by error type
        defaults = {
            "syntax": 5,  # More retries for fixable syntax errors
            "template": 4,  # Template errors are fairly fixable
            "html": 4,  # HTML errors are fixable
            "manifest": 3,  # Manifest errors are moderate
            "factual": 2,  # Factual errors are harder to fix
            "semantic": 2,  # Semantic errors are harder to fix
        }

        max_retries = base_config.get(
            error_type) or defaults.get(error_type, 3)
        if max_retries is None:
            max_retries = 3

        return RetryConfig(
            max_retries=max_retries,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
            backoff_multiplier=self.config.backoff_multiplier,
            retry_on_exceptions=self.config.retry_on_exceptions,
        )

    def execute_with_adaptive_retry(
        self,
        operation: Callable[[], Any],
        error_type_getter: Callable[[Exception], str],
        context: dict[str, Any | None] | None = None,
        base_retry_config: dict[str, int | None] | None = None,
    ) -> RetryResult:
        """Execute operation with adaptive retry based on error type.

        Args:
            operation: The operation to retry
            error_type_getter: Function that extracts error type from exception
            context: Optional context for logging
            base_retry_config: Base retry counts by error type

        Returns:
            RetryResult with outcome
        """
        context = context or {}
        start_time = time.time()

        # First attempt
        try:
            result = operation()
            return RetryResult(
                success=True,
                result=result,
                attempts=1,
                total_time=time.time() - start_time,
            )
        except Exception as e:
            error_type = error_type_getter(e)

            # Get adaptive config for this error type
            adaptive_config = self.get_adaptive_config(
                error_type, base_retry_config)

            logger.debug(
                "adaptive_retry_start",
                error_type=error_type,
                adaptive_max_retries=adaptive_config.max_retries,
                **context,
            )

            # Create temporary handler with adaptive config
            adaptive_handler = RetryHandler(adaptive_config)
            return adaptive_handler.execute_with_retry(operation, context)
