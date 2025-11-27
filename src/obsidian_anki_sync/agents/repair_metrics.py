"""Repair quality metrics and feedback loop tracking.

This module tracks repair attempts, successes, failure modes, and quality
improvements to enable learning and optimization.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RepairAttempt:
    """Single repair attempt record."""

    timestamp: float
    error_category: str
    error_type: str
    strategy_used: str
    success: bool
    quality_before: float | None = None
    quality_after: float | None = None
    repair_time: float = 0.0
    error_message: str = ""


@dataclass
class RepairMetrics:
    """Aggregated repair metrics."""

    total_attempts: int = 0
    total_successes: int = 0
    total_failures: int = 0
    attempts_by_category: dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    successes_by_category: dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    failures_by_category: dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    attempts_by_strategy: dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    successes_by_strategy: dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    failures_by_strategy: dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    average_repair_time: float = 0.0
    average_quality_improvement: float = 0.0
    repair_patterns: list[dict[str, Any]] = field(default_factory=list)


class RepairMetricsCollector:
    """Collects and aggregates repair metrics for analysis and learning."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = RepairMetrics()
        self.recent_attempts: list[RepairAttempt] = []
        self.max_recent_attempts = 1000  # Keep last 1000 attempts

    def record_attempt(
        self,
        error_category: str,
        error_type: str,
        strategy_used: str,
        success: bool,
        quality_before: float | None = None,
        quality_after: float | None = None,
        repair_time: float = 0.0,
        error_message: str = "",
    ) -> None:
        """Record a repair attempt.

        Args:
            error_category: Error category (syntax/structure/content/etc.)
            error_type: Specific error type
            strategy_used: Repair strategy used
            success: Whether repair succeeded
            quality_before: Quality score before repair
            quality_after: Quality score after repair
            repair_time: Time taken for repair
            error_message: Error message if failed
        """
        attempt = RepairAttempt(
            timestamp=time.time(),
            error_category=error_category,
            error_type=error_type,
            strategy_used=strategy_used,
            success=success,
            quality_before=quality_before,
            quality_after=quality_after,
            repair_time=repair_time,
            error_message=error_message,
        )

        # Add to recent attempts
        self.recent_attempts.append(attempt)
        if len(self.recent_attempts) > self.max_recent_attempts:
            self.recent_attempts.pop(0)

        # Update aggregated metrics
        self.metrics.total_attempts += 1
        self.metrics.attempts_by_category[error_category] += 1
        self.metrics.attempts_by_strategy[strategy_used] += 1

        if success:
            self.metrics.total_successes += 1
            self.metrics.successes_by_category[error_category] += 1
            self.metrics.successes_by_strategy[strategy_used] += 1

            # Track quality improvement
            if quality_before is not None and quality_after is not None:
                improvement = quality_after - quality_before
                # Update running average
                current_avg = self.metrics.average_quality_improvement
                count = self.metrics.total_successes
                self.metrics.average_quality_improvement = (
                    (current_avg * (count - 1) + improvement) / count
                )
        else:
            self.metrics.total_failures += 1
            self.metrics.failures_by_category[error_category] += 1
            if strategy_used:
                self.metrics.failures_by_strategy[strategy_used] += 1

        # Update average repair time
        current_avg_time = self.metrics.average_repair_time
        count = self.metrics.total_attempts
        self.metrics.average_repair_time = (
            (current_avg_time * (count - 1) + repair_time) / count
        )

        # Store pattern for learning
        pattern = {
            "error_category": error_category,
            "error_type": error_type,
            "strategy": strategy_used,
            "success": success,
            "quality_improvement": (
                quality_after - quality_before
                if quality_before is not None and quality_after is not None
                else None
            ),
        }
        self.metrics.repair_patterns.append(pattern)
        if len(self.metrics.repair_patterns) > 1000:
            self.metrics.repair_patterns.pop(0)

        logger.debug(
            "repair_attempt_recorded",
            category=error_category,
            strategy=strategy_used,
            success=success,
        )

    def get_success_rate(self, category: str | None = None) -> float:
        """Get repair success rate.

        Args:
            category: Optional category to filter by

        Returns:
            Success rate (0.0-1.0)
        """
        if category:
            attempts = self.metrics.attempts_by_category.get(category, 0)
            successes = self.metrics.successes_by_category.get(category, 0)
        else:
            attempts = self.metrics.total_attempts
            successes = self.metrics.total_successes

        if attempts == 0:
            return 0.0

        return successes / attempts

    def get_best_strategy_for_category(self, category: str) -> str | None:
        """Get best performing strategy for a category.

        Args:
            category: Error category

        Returns:
            Strategy name with best success rate, or None
        """
        category_attempts = [
            a for a in self.recent_attempts if a.error_category == category
        ]

        if not category_attempts:
            return None

        # Count successes by strategy
        strategy_successes: dict[str, int] = defaultdict(int)
        strategy_attempts: dict[str, int] = defaultdict(int)

        for attempt in category_attempts:
            strategy_attempts[attempt.strategy_used] += 1
            if attempt.success:
                strategy_successes[attempt.strategy_used] += 1

        # Find strategy with best success rate
        best_strategy = None
        best_rate = 0.0

        for strategy, attempts_count in strategy_attempts.items():
            successes_count = strategy_successes.get(strategy, 0)
            rate = successes_count / attempts_count if attempts_count > 0 else 0.0

            if rate > best_rate:
                best_rate = rate
                best_strategy = strategy

        return best_strategy

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of repair metrics.

        Returns:
            Dictionary with key metrics
        """
        return {
            "total_attempts": self.metrics.total_attempts,
            "total_successes": self.metrics.total_successes,
            "total_failures": self.metrics.total_failures,
            "overall_success_rate": self.get_success_rate(),
            "success_rate_by_category": {
                cat: self.get_success_rate(cat)
                for cat in self.metrics.attempts_by_category.keys()
            },
            "average_repair_time": self.metrics.average_repair_time,
            "average_quality_improvement": self.metrics.average_quality_improvement,
            "most_common_category": max(
                self.metrics.attempts_by_category.items(),
                key=lambda x: x[1],
                default=("none", 0),
            )[0],
            "best_strategy_by_category": {
                cat: self.get_best_strategy_for_category(cat)
                for cat in self.metrics.attempts_by_category.keys()
            },
        }

    def get_recent_patterns(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent repair patterns for learning.

        Args:
            limit: Maximum number of patterns to return

        Returns:
            List of recent repair patterns
        """
        return self.metrics.repair_patterns[-limit:]


# Global metrics collector instance
_global_collector: RepairMetricsCollector | None = None


def get_repair_metrics_collector() -> RepairMetricsCollector:
    """Get global repair metrics collector instance.

    Returns:
        RepairMetricsCollector instance
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = RepairMetricsCollector()
    return _global_collector
