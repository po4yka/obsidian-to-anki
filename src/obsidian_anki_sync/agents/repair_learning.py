"""Repair learning and pattern recognition system.

This module learns from successful repair patterns and suggests repairs
based on historical success rates.
"""

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from obsidian_anki_sync.utils.logging import get_logger

from .repair_metrics import get_repair_metrics_collector

logger = get_logger(__name__)


@dataclass
class RepairPattern:
    """A learned repair pattern."""

    error_signature: str  # Hash of error characteristics
    error_category: str
    error_type: str
    successful_strategy: str
    success_count: int
    total_attempts: int
    average_quality_improvement: float
    repair_steps: list[str]  # Steps that led to success


class RepairLearningSystem:
    """Learns from repair patterns and suggests optimal strategies."""

    def __init__(self) -> None:
        """Initialize learning system."""
        self.patterns: dict[str, RepairPattern] = {}
        self.metrics_collector = get_repair_metrics_collector()

    def _create_error_signature(
        self, error_category: str, error_type: str, error_message: str
    ) -> str:
        """Create a signature hash for an error pattern.

        Args:
            error_category: Error category
            error_type: Error type
            error_message: Error message

        Returns:
            Hash signature string
        """
        # Normalize error message (remove file paths, timestamps, etc.)
        normalized_msg = error_message.lower()
        # Remove common variable parts
        normalized_msg = re.sub(r"/[^\s]+", "/path", normalized_msg)
        normalized_msg = re.sub(r"\d{4}-\d{2}-\d{2}", "date", normalized_msg)

        signature_str = f"{error_category}:{error_type}:{normalized_msg[:100]}"
        return hashlib.md5(signature_str.encode()).hexdigest()

    def learn_from_success(
        self,
        error_category: str,
        error_type: str,
        error_message: str,
        strategy_used: str,
        quality_improvement: float | None = None,
        repair_steps: list[str] | None = None,
    ) -> None:
        """Learn from a successful repair.

        Args:
            error_category: Error category
            error_type: Error type
            error_message: Error message
            strategy_used: Strategy that succeeded
            quality_improvement: Quality improvement achieved
            repair_steps: Steps taken during repair
        """
        signature = self._create_error_signature(
            error_category, error_type, error_message
        )

        if signature in self.patterns:
            pattern = self.patterns[signature]
            pattern.total_attempts += 1
            pattern.success_count += 1

            # Update average quality improvement
            if quality_improvement is not None:
                current_avg = pattern.average_quality_improvement
                count = pattern.success_count
                pattern.average_quality_improvement = (
                    current_avg * (count - 1) + quality_improvement
                ) / count
        else:
            # Create new pattern
            pattern = RepairPattern(
                error_signature=signature,
                error_category=error_category,
                error_type=error_type,
                successful_strategy=strategy_used,
                success_count=1,
                total_attempts=1,
                average_quality_improvement=quality_improvement or 0.0,
                repair_steps=repair_steps or [],
            )
            self.patterns[signature] = pattern

        logger.debug(
            "repair_pattern_learned",
            signature=signature[:8],
            strategy=strategy_used,
            success_rate=pattern.success_count / pattern.total_attempts,
        )

    def suggest_strategy(
        self, error_category: str, error_type: str, error_message: str
    ) -> str | None:
        """Suggest repair strategy based on learned patterns.

        Args:
            error_category: Error category
            error_type: Error type
            error_message: Error message

        Returns:
            Suggested strategy name, or None if no pattern found
        """
        signature = self._create_error_signature(
            error_category, error_type, error_message
        )

        # Exact match
        if signature in self.patterns:
            pattern = self.patterns[signature]
            if (
                pattern.success_count / pattern.total_attempts >= 0.7
            ):  # 70% success rate
                return pattern.successful_strategy

        # Category-based suggestion
        category_patterns = [
            p for p in self.patterns.values() if p.error_category == error_category
        ]

        if category_patterns:
            # Find best strategy for this category
            strategy_success: dict[str, tuple[int, int]] = defaultdict(
                lambda: (0, 0)
            )  # (successes, total)

            for pattern in category_patterns:
                successes, total = strategy_success[pattern.successful_strategy]
                strategy_success[pattern.successful_strategy] = (
                    successes + pattern.success_count,
                    total + pattern.total_attempts,
                )

            # Find strategy with best success rate
            best_strategy = None
            best_rate = 0.0

            for strategy, (successes, total) in strategy_success.items():
                rate = successes / total if total > 0 else 0.0
                if rate > best_rate and rate >= 0.6:  # Minimum 60% success rate
                    best_rate = rate
                    best_strategy = strategy

            if best_strategy:
                logger.info(
                    "repair_strategy_suggested",
                    category=error_category,
                    strategy=best_strategy,
                    success_rate=best_rate,
                )
                return best_strategy

        return None

    def get_similar_patterns(
        self, error_category: str, error_type: str, limit: int = 5
    ) -> list[RepairPattern]:
        """Get similar repair patterns.

        Args:
            error_category: Error category
            error_type: Error type
            limit: Maximum number of patterns to return

        Returns:
            List of similar patterns
        """
        similar = [
            p
            for p in self.patterns.values()
            if p.error_category == error_category and p.error_type == error_type
        ]

        # Sort by success rate
        similar.sort(
            key=lambda p: (
                p.success_count / p.total_attempts if p.total_attempts > 0 else 0
            ),
            reverse=True,
        )

        return similar[:limit]

    def get_learning_stats(self) -> dict[str, Any]:
        """Get learning system statistics.

        Returns:
            Dictionary with learning stats
        """
        total_patterns = len(self.patterns)
        patterns_by_category: dict[str, int] = defaultdict(int)

        for pattern in self.patterns.values():
            patterns_by_category[pattern.error_category] += 1

        return {
            "total_patterns": total_patterns,
            "patterns_by_category": dict(patterns_by_category),
            "average_pattern_success_rate": (
                sum(
                    p.success_count / p.total_attempts
                    for p in self.patterns.values()
                    if p.total_attempts > 0
                )
                / total_patterns
                if total_patterns > 0
                else 0.0
            ),
        }


# Global learning system instance
_global_learning: RepairLearningSystem | None = None


def get_repair_learning_system() -> RepairLearningSystem:
    """Get global repair learning system instance.

    Returns:
        RepairLearningSystem instance
    """
    global _global_learning
    if _global_learning is None:
        _global_learning = RepairLearningSystem()
    return _global_learning
