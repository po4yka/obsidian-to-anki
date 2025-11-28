"""Performance metrics tracking for agent operations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a single operation type."""

    operation_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    total_duration: float = 0.0
    total_llm_duration: float = 0.0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    errors_by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class SessionMetrics:
    """Aggregated metrics for an entire session."""

    session_start: datetime = field(default_factory=datetime.now)
    operations: dict[str, OperationMetrics] = field(default_factory=dict)

    def record_operation(
        self,
        operation: str,
        success: bool,
        duration: float,
        llm_duration: float = 0.0,
        tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        retried: bool = False,
        error_type: str | None = None,
    ) -> None:
        """Record metrics for an operation.

        Args:
            operation: Operation name (e.g., "card_generation", "validation")
            success: Whether the operation succeeded
            duration: Total duration in seconds
            llm_duration: Time spent in LLM call
            tokens: Total tokens used
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            retried: Whether the operation was retried
            error_type: Type of error if failed
        """
        if operation not in self.operations:
            self.operations[operation] = OperationMetrics(operation_name=operation)

        metrics = self.operations[operation]
        metrics.total_calls += 1

        if success:
            metrics.successful_calls += 1
        else:
            metrics.failed_calls += 1
            if error_type:
                metrics.errors_by_type[error_type] = (
                    metrics.errors_by_type.get(error_type, 0) + 1
                )

        if retried:
            metrics.retried_calls += 1

        metrics.total_duration += duration
        metrics.total_llm_duration += llm_duration
        metrics.total_tokens += tokens
        metrics.total_prompt_tokens += prompt_tokens
        metrics.total_completion_tokens += completion_tokens

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all metrics.

        Returns:
            Dictionary with aggregated metrics
        """
        session_duration = (datetime.now() - self.session_start).total_seconds()

        summary = {
            "session_duration_seconds": round(session_duration, 2),
            "total_operations": sum(m.total_calls for m in self.operations.values()),
            "successful_operations": sum(
                m.successful_calls for m in self.operations.values()
            ),
            "failed_operations": sum(m.failed_calls for m in self.operations.values()),
            "retry_rate_pct": 0.0,
            "total_tokens": sum(m.total_tokens for m in self.operations.values()),
            "total_llm_time_seconds": round(
                sum(m.total_llm_duration for m in self.operations.values()), 2
            ),
            "by_operation": {},
        }

        total_calls = cast("int", summary["total_operations"])
        if total_calls > 0:
            total_retried = sum(m.retried_calls for m in self.operations.values())
            summary["retry_rate_pct"] = round((total_retried / total_calls) * 100, 1)
            successful_ops = cast("int", summary["successful_operations"])
            summary["success_rate_pct"] = round((successful_ops / total_calls) * 100, 1)

        # Per-operation details
        by_operation: dict[str, Any] = {}
        for op_name, metrics in self.operations.items():
            if metrics.total_calls == 0:
                continue

            success_rate = (metrics.successful_calls / metrics.total_calls) * 100
            avg_duration = metrics.total_duration / metrics.total_calls
            avg_llm_duration = metrics.total_llm_duration / metrics.total_calls
            avg_tokens = (
                metrics.total_tokens / metrics.successful_calls
                if metrics.successful_calls > 0
                else 0
            )

            by_operation[op_name] = {
                "calls": metrics.total_calls,
                "success_rate_pct": round(success_rate, 1),
                "failed": metrics.failed_calls,
                "retried": metrics.retried_calls,
                "avg_duration_seconds": round(avg_duration, 2),
                "avg_llm_duration_seconds": round(avg_llm_duration, 2),
                "total_tokens": metrics.total_tokens,
                "avg_tokens_per_call": round(avg_tokens, 0),
                "errors_by_type": metrics.errors_by_type,
            }

        summary["by_operation"] = by_operation
        return summary

    def log_summary(self) -> None:
        """Log metrics summary."""
        summary = self.get_summary()

        logger.info(
            "session_metrics_summary",
            session_duration=summary["session_duration_seconds"],
            total_operations=summary["total_operations"],
            success_rate_pct=summary.get("success_rate_pct", 0),
            retry_rate_pct=summary["retry_rate_pct"],
            total_tokens=summary["total_tokens"],
            total_llm_time=summary["total_llm_time_seconds"],
        )

        # Log per-operation details
        for op_name, op_metrics in summary["by_operation"].items():
            logger.info(
                "operation_metrics",
                operation=op_name,
                calls=op_metrics["calls"],
                success_rate_pct=op_metrics["success_rate_pct"],
                avg_duration=op_metrics["avg_duration_seconds"],
                avg_tokens=op_metrics["avg_tokens_per_call"],
                errors=op_metrics["errors_by_type"],
            )


# Global session metrics instance
_global_metrics: SessionMetrics | None = None


def get_session_metrics() -> SessionMetrics:
    """Get the global session metrics instance.

    Returns:
        Global SessionMetrics instance
    """
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = SessionMetrics()
    return _global_metrics


def reset_session_metrics() -> None:
    """Reset the global session metrics."""
    global _global_metrics
    _global_metrics = SessionMetrics()


def record_operation_metric(
    operation: str,
    success: bool,
    duration: float,
    llm_duration: float = 0.0,
    tokens: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    retried: bool = False,
    error_type: str | None = None,
) -> None:
    """Convenience function to record operation metrics.

    Args:
        operation: Operation name
        success: Whether operation succeeded
        duration: Total duration
        llm_duration: LLM call duration
        tokens: Total tokens
        prompt_tokens: Input tokens
        completion_tokens: Output tokens
        retried: Whether operation was retried
        error_type: Error type if failed
    """
    metrics = get_session_metrics()
    metrics.record_operation(
        operation=operation,
        success=success,
        duration=duration,
        llm_duration=llm_duration,
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        retried=retried,
        error_type=error_type,
    )


def log_session_summary() -> None:
    """Log summary of session metrics."""
    metrics = get_session_metrics()
    metrics.log_summary()
