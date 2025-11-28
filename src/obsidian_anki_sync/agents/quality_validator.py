"""Comprehensive quality validation system for Anki cards."""

import time

from obsidian_anki_sync.anki.client import AnkiClient
from obsidian_anki_sync.models import NoteMetadata
from obsidian_anki_sync.providers.base import BaseLLMProvider
from obsidian_anki_sync.utils.logging import get_logger

from .card_improver import CardImprover
from .card_quality_agent import CardQualityAgent
from .models import GeneratedCard, QualityReport
from .performance_tracker import PerformanceTracker

logger = get_logger(__name__)


class QualityValidationResult:
    """Result of comprehensive quality validation."""

    def __init__(
        self,
        original_card: GeneratedCard,
        quality_report: QualityReport,
        improved_card: GeneratedCard | None = None,
        performance_data: dict | None = None,
        validation_time: float = 0.0,
    ):
        """Initialize validation result.

        Args:
            original_card: The original card that was validated
            quality_report: Quality assessment results
            improved_card: Improved version of the card (if any)
            performance_data: Historical performance data (if available)
            validation_time: Time taken for validation
        """
        self.original_card = original_card
        self.quality_report = quality_report
        self.improved_card = improved_card
        self.performance_data = performance_data or {}
        self.validation_time = validation_time

    @property
    def final_card(self) -> GeneratedCard:
        """Get the final card (improved if available, otherwise original)."""
        return self.improved_card if self.improved_card else self.original_card

    @property
    def needs_improvement(self) -> bool:
        """Check if the card needs improvement."""
        return self.quality_report.overall_score < 0.8

    @property
    def is_critical_issue(self) -> bool:
        """Check if there are critical quality issues."""
        return self.quality_report.overall_score < 0.6


class QualityValidator:
    """Comprehensive quality validation and improvement system for Anki cards.

    Orchestrates quality assessment, auto-improvement, and performance tracking
    to ensure high-quality cards that follow evidence-based learning principles.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        anki_client: AnkiClient | None = None,
        enable_auto_improvement: bool = True,
        enable_performance_tracking: bool = True,
        quality_threshold: float = 0.8,
        critical_threshold: float = 0.6,
    ):
        """Initialize quality validator.

        Args:
            llm_provider: LLM provider for quality assessment and improvement
            anki_client: AnkiConnect client for performance tracking
            enable_auto_improvement: Whether to automatically improve low-quality cards
            enable_performance_tracking: Whether to track card performance
            quality_threshold: Minimum quality score for acceptable cards
            critical_threshold: Threshold for critical quality issues
        """
        self.llm_provider = llm_provider
        self.quality_agent = CardQualityAgent(llm_provider)
        self.improver = CardImprover(llm_provider) if enable_auto_improvement else None
        self.performance_tracker = (
            PerformanceTracker(anki_client)
            if (anki_client and enable_performance_tracking)
            else None
        )

        self.quality_threshold = quality_threshold
        self.critical_threshold = critical_threshold

        logger.info(
            "quality_validator_initialized",
            auto_improvement=enable_auto_improvement,
            performance_tracking=enable_performance_tracking,
            quality_threshold=quality_threshold,
            critical_threshold=critical_threshold,
        )

    def validate_card(
        self,
        card: GeneratedCard,
        metadata: NoteMetadata,
        context_cards: list[GeneratedCard | None] | None = None,
        note_id: int | None = None,
    ) -> QualityValidationResult:
        """Perform comprehensive quality validation on a card.

        Args:
            card: Card to validate
            metadata: Note metadata for context
            context_cards: Other cards from the same note
            note_id: Anki note ID for performance tracking

        Returns:
            Comprehensive validation result
        """
        start_time = time.time()

        logger.debug("validating_card_quality", slug=card.slug)

        # Step 1: Quality Assessment
        quality_report = self.quality_agent.assess_card_quality(
            card, metadata, context_cards
        )

        # Step 2: Get Performance Data (if available)
        performance_data = None
        if self.performance_tracker and note_id:
            try:
                performance_data = self.performance_tracker.get_card_performance(
                    [note_id]
                )
                performance_data = performance_data.get(note_id, {})
            except Exception as e:
                logger.debug("performance_data_unavailable", error=str(e))

        # Step 3: Auto-Improvement (if enabled and needed)
        improved_card = None
        if self.improver and quality_report.overall_score < self.quality_threshold:
            try:
                improved_card = self.improver.improve_card(
                    card, quality_report, metadata, auto_apply=True
                )

                # Re-assess the improved card
                if improved_card != card:
                    improved_quality = self.quality_agent.assess_card_quality(
                        improved_card, metadata, context_cards
                    )
                    quality_report = improved_quality

            except Exception as e:
                logger.warning("auto_improvement_failed", slug=card.slug, error=str(e))

        validation_time = time.time() - start_time

        result = QualityValidationResult(
            original_card=card,
            quality_report=quality_report,
            improved_card=improved_card,
            performance_data=performance_data,
            validation_time=validation_time,
        )

        # Log validation outcome
        self._log_validation_result(result)

        return result

    def validate_cards_batch(
        self,
        cards: list[GeneratedCard],
        metadata: NoteMetadata,
        note_ids: list[int | None] | None = None,
    ) -> list[QualityValidationResult]:
        """Validate multiple cards in batch for efficiency.

        Args:
            cards: List of cards to validate
            metadata: Shared note metadata
            note_ids: Corresponding Anki note IDs for performance tracking

        Returns:
            List of validation results
        """
        start_time = time.time()

        logger.info("batch_validating_cards", count=len(cards), note=metadata.title)

        results = []

        # Process cards individually but reuse context
        for i, card in enumerate(cards):
            note_id = note_ids[i] if note_ids and i < len(note_ids) else None
            # Other cards as context
            context_cards = [c for c in cards if c != card]

            result = self.validate_card(card, metadata, context_cards, note_id)
            results.append(result)

        batch_time = time.time() - start_time

        # Summarize batch results
        total_cards = len(results)
        improved_cards = sum(1 for r in results if r.improved_card is not None)
        critical_issues = sum(1 for r in results if r.is_critical_issue)
        avg_quality = sum(r.quality_report.overall_score for r in results) / total_cards

        logger.info(
            "batch_validation_complete",
            total_cards=total_cards,
            improved_cards=improved_cards,
            critical_issues=critical_issues,
            average_quality=avg_quality,
            batch_time=batch_time,
        )

        return results

    def get_quality_statistics(self, results: list[QualityValidationResult]) -> dict:
        """Generate quality statistics from validation results.

        Args:
            results: List of validation results

        Returns:
            Dictionary with quality statistics
        """
        if not results:
            return {}

        stats = {
            "total_cards": len(results),
            "average_quality": sum(r.quality_report.overall_score for r in results)
            / len(results),
            "cards_improved": sum(1 for r in results if r.improved_card is not None),
            "critical_issues": sum(1 for r in results if r.is_critical_issue),
            "quality_distribution": self._calculate_quality_distribution(results),
            "common_issues": self._identify_common_issues(results),
        }

        return stats

    def _calculate_quality_distribution(
        self, results: list[QualityValidationResult]
    ) -> dict[str, int]:
        """Calculate distribution of quality scores."""
        distribution = {
            "excellent": 0,  # 0.9-1.0
            "good": 0,  # 0.8-0.9
            "fair": 0,  # 0.6-0.8
            "poor": 0,  # 0.4-0.6
            "critical": 0,  # 0.0-0.4
        }

        for result in results:
            score = result.quality_report.overall_score
            if score >= 0.9:
                distribution["excellent"] += 1
            elif score >= 0.8:
                distribution["good"] += 1
            elif score >= 0.6:
                distribution["fair"] += 1
            elif score >= 0.4:
                distribution["poor"] += 1
            else:
                distribution["critical"] += 1

        return distribution

    def _identify_common_issues(
        self, results: list[QualityValidationResult]
    ) -> list[dict]:
        """Identify most common quality issues."""
        issue_counts = {}

        for result in results:
            for dimension_name, dimension in result.quality_report.dimensions.items():
                for issue in dimension.issues:
                    key = f"{dimension_name}: {issue}"
                    issue_counts[key] = issue_counts.get(key, 0) + 1

        # Return top 10 most common issues
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [
            {"issue": issue, "count": count, "percentage": count / len(results) * 100}
            for issue, count in sorted_issues[:10]
        ]

    def _log_validation_result(self, result: QualityValidationResult) -> None:
        """Log validation result with appropriate level."""
        report = result.quality_report
        card = result.original_card

        log_data = {
            "slug": card.slug,
            "quality_score": report.overall_score,
            "confidence": report.confidence,
            "dimensions": {k: v.score for k, v in report.dimensions.items()},
            "suggestions_count": len(report.suggestions),
            "improved": result.improved_card is not None,
            "validation_time": result.validation_time,
        }

        if result.is_critical_issue:
            logger.error("card_quality_critical", **log_data)
        elif result.needs_improvement:
            logger.warning("card_quality_needs_improvement", **log_data)
        else:
            logger.info("card_quality_acceptable", **log_data)

    def export_quality_report(self, results: list[QualityValidationResult]) -> dict:
        """Export comprehensive quality report.

        Args:
            results: Validation results to include in report

        Returns:
            Dictionary containing complete quality analysis
        """
        stats = self.get_quality_statistics(results)

        # Add performance insights if available
        performance_insights = {}
        if self.performance_tracker:
            try:
                performance_insights = (
                    self.performance_tracker.export_performance_report()
                )
            except Exception as e:
                logger.debug("performance_insights_unavailable", error=str(e))

        report = {
            "timestamp": time.time(),
            "quality_statistics": stats,
            "performance_insights": performance_insights,
            "validation_results": [
                {
                    "slug": r.original_card.slug,
                    "quality_score": r.quality_report.overall_score,
                    "critical_issues": r.is_critical_issue,
                    "improved": r.improved_card is not None,
                    "top_issues": [
                        issue
                        for dim in r.quality_report.dimensions.values()
                        for issue in dim.issues[:2]
                    ],
                }
                for r in results
            ],
            "recommendations": self._generate_recommendations(stats),
        }

        return report

    def _generate_recommendations(self, stats: dict) -> list[str]:
        """Generate recommendations based on quality statistics.

        Args:
            stats: Quality statistics

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        avg_quality = stats.get("average_quality", 0)
        critical_issues = stats.get("critical_issues", 0)
        total_cards = stats.get("total_cards", 0)

        if avg_quality < 0.7:
            recommendations.append(
                "Overall card quality is low. Consider reviewing generation prompts and templates."
            )

        if critical_issues > total_cards * 0.1:  # More than 10% critical issues
            recommendations.append(
                "High number of critical quality issues. Manual review recommended for problematic cards."
            )

        distribution = stats.get("quality_distribution", {})
        poor_count = distribution.get("poor", 0) + distribution.get("critical", 0)
        if poor_count > total_cards * 0.2:  # More than 20% poor/critical
            recommendations.append(
                "Significant portion of cards need improvement. Consider adjusting quality thresholds."
            )

        common_issues = stats.get("common_issues", [])
        if common_issues:
            top_issue = common_issues[0]["issue"]
            recommendations.append(
                f"Most common issue: {top_issue}. Focus improvement efforts here."
            )

        return recommendations
