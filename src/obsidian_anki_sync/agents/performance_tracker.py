"""Performance tracking agent for monitoring card effectiveness in Anki."""

import time
from collections import defaultdict
from typing import Any

from obsidian_anki_sync.anki.client import AnkiClient
from obsidian_anki_sync.models import Card
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceTracker:
    """Tracks and analyzes card performance metrics from Anki.

    Integrates with AnkiConnect to gather review statistics and identify
    patterns in card effectiveness.
    """

    def __init__(self, anki_client: AnkiClient):
        """Initialize performance tracker.

        Args:
            anki_client: AnkiConnect client for data retrieval
        """
        self.anki_client = anki_client
        self._last_sync = 0.0
        self._cache_duration = 3600.0  # Cache for 1 hour

        logger.info("performance_tracker_initialized")

    def get_card_performance(self, note_ids: list[int]) -> dict[int, dict[str, float]]:
        """Get performance metrics for specific cards.

        Args:
            note_ids: List of Anki note IDs to analyze

        Returns:
            Dictionary mapping note IDs to performance metrics
        """
        try:
            if not note_ids:
                return {}

            # Get card IDs for these notes (batch operation)
            # AnkiConnect notesToCards takes a list of note IDs
            card_ids_list = self.anki_client.invoke("notesToCards", {"notes": note_ids})
            
            all_card_ids = []
            if card_ids_list:
                for ids in card_ids_list:
                    if ids:
                        all_card_ids.extend(ids)
            
            if not all_card_ids:
                return {}

            # Get cards info for these cards
            cards_info = self.anki_client.cards_info(all_card_ids)

            performance_data = {}

            for card_info in cards_info:
                note_id = card_info.get("note")  # AnkiConnect uses 'note' for note ID
                if note_id:
                    metrics = self._calculate_card_metrics(card_info)
                    # For simplicity with 1:1 mapping, simple assignment works.
                    # For 1:N, this stores the last card's metrics.
                    performance_data[note_id] = metrics

            return performance_data

        except Exception as e:
            logger.warning("failed_to_get_card_performance", error=str(e))
            return {}

    def get_review_statistics(self) -> dict[str, float]:
        """Get overall review statistics from Anki.

        Returns:
            Dictionary with review statistics
        """
        try:
            # Get today's review count
            reviews_today = self.anki_client.get_num_cards_reviewed_today()

            # Get collection stats (HTML, parse what we can)
            collection_stats = self.anki_client.get_collection_stats()

            stats: dict[str, Any] = {
                "reviews_today": float(reviews_today),
                "collection_stats_html": str(collection_stats),
            }

            return stats

        except Exception as e:
            logger.warning("failed_to_get_review_statistics", error=str(e))
            return {}

    def analyze_card_patterns(self, cards: list[Card]) -> dict[str, list[str]]:
        """Analyze patterns in card performance to identify quality issues.

        Args:
            cards: List of cards to analyze

        Returns:
            Dictionary mapping issue types to lists of affected card slugs
        """
        issues = defaultdict(list)

        try:
            # Map note_id -> Card for analysis
            card_map: dict[int, Card] = {}
            note_ids: list[int] = []

            # Resolve note IDs from cards
            for card in cards:
                if not card.guid:
                    continue
                try:
                    # Find note ID by GUID
                    # This might be slow for many cards, but necessary if we don't have ID stored
                    found_ids = self.anki_client.find_notes(f"guid:{card.guid}")
                    if found_ids:
                        # Assuming 1:1 mapping for simple cards
                        note_id = found_ids[0]
                        card_map[note_id] = card
                        note_ids.append(note_id)
                except Exception:
                    # Ignore cards that can't be found
                    continue

            if note_ids:
                performance_data = self.get_card_performance(note_ids)

                for note_id, metrics in performance_data.items():
                    card = card_map.get(note_id)
                    if card:
                        card_issues = self._analyze_single_card_performance(card, metrics)
                        for issue_type, affected_cards in card_issues.items():
                            issues[issue_type].extend(affected_cards)

        except Exception as e:
            logger.warning("card_pattern_analysis_failed", error=str(e))

        return dict(issues)

    def _calculate_card_metrics(self, card_info: dict) -> dict[str, float]:
        """Calculate performance metrics for a single card.

        Args:
            card_info: Card information from AnkiConnect

        Returns:
            Dictionary with calculated metrics
        """
        metrics = {}

        try:
            # These are example metrics - actual implementation depends on API
            reviews = card_info.get("reviews", 0)
            lapses = card_info.get("lapses", 0)
            interval = card_info.get("interval", 0)
            
            # Anki returns factor in permille (e.g. 2500 for 2.5)
            raw_factor = card_info.get("factor", 2500)
            ease_factor = raw_factor / 1000.0

            # Calculate derived metrics
            if reviews > 0:
                lapse_rate = lapses / reviews
                metrics["lapse_rate"] = lapse_rate

                retention = 1.0 - lapse_rate
                metrics["estimated_retention"] = max(0.0, min(1.0, retention))

            metrics["ease_factor"] = ease_factor
            metrics["interval_days"] = interval
            metrics["total_reviews"] = reviews
            metrics["lapses"] = lapses

            # Quality indicators
            if ease_factor < 1.8:
                metrics["difficulty_indicator"] = "hard"
            elif ease_factor > 2.8:
                metrics["difficulty_indicator"] = "easy"
            else:
                metrics["difficulty_indicator"] = "medium"

        except (KeyError, TypeError) as e:
            logger.debug(
                "metric_calculation_error",
                error=str(e),
                card_info_keys=list(card_info.keys()),
            )

        return metrics

    def _analyze_single_card_performance(
        self, card: Card, performance_data: dict[str, float]
    ) -> dict[str, list[str]]:
        """Analyze performance of a single card.

        Args:
            card: Card to analyze
            performance_data: Performance metrics for the card

        Returns:
            Dictionary mapping issue types to affected card slugs
        """
        issues = defaultdict(list)

        # Check for leech indicators
        if performance_data.get("lapses", 0) > 5:
            issues["potential_leeches"].append(card.slug)

        # Check retention rates
        retention = performance_data.get("estimated_retention", 1.0)
        if retention < 0.7:  # Below 70% retention
            issues["low_retention"].append(card.slug)

        # Check ease factors
        ease_factor = performance_data.get("ease_factor", 2.5)
        if ease_factor < 1.5:  # Very difficult cards
            issues["very_difficult"].append(card.slug)

        # Check for cards that are reviewed very frequently but still fail
        reviews = performance_data.get("total_reviews", 0)
        lapses = performance_data.get("lapses", 0)

        if reviews > 10 and lapses / reviews > 0.3:  # >30% failure rate
            issues["persistent_failures"].append(card.slug)

        return dict(issues)

    def get_quality_insights(
        self, performance_data: dict[int, dict[str, float]]
    ) -> dict[str, float]:
        """Generate quality insights from performance data.

        Args:
            performance_data: Performance metrics for multiple cards

        Returns:
            Dictionary with quality insights and averages
        """
        insights: dict[str, Any] = {}

        if not performance_data:
            return insights

        # Calculate averages
        total_cards = len(performance_data)
        total_retention = 0.0
        total_ease = 0.0
        retention_count = 0
        ease_count = 0

        for metrics in performance_data.values():
            if "estimated_retention" in metrics:
                total_retention += metrics["estimated_retention"]
                retention_count += 1

            if "ease_factor" in metrics:
                total_ease += metrics["ease_factor"]
                ease_count += 1

        if retention_count > 0:
            insights["average_retention"] = total_retention / retention_count

        if ease_count > 0:
            insights["average_ease_factor"] = total_ease / ease_count

        # Calculate quality score based on performance
        if retention_count > 0 and ease_count > 0:
            avg_retention = insights["average_retention"]
            avg_ease = insights["average_ease_factor"]

            quality_score = (avg_retention * 0.7) + ((avg_ease - 1.3) / 1.7 * 0.3)
            insights["overall_quality_score"] = max(0.0, min(1.0, quality_score))

        insights["total_cards_analyzed"] = total_cards

        return insights

    def export_performance_report(self) -> dict:
        """Export a comprehensive performance report.

        Returns:
            Dictionary containing performance analysis and recommendations
        """
        try:
            # Get current statistics
            review_stats = self.get_review_statistics()

            report = {
                "timestamp": time.time(),
                "review_statistics": review_stats,
                "recommendations": self._generate_recommendations(review_stats),
            }

            return report

        except Exception as e:
            logger.error("performance_report_export_failed", error=str(e))
            return {"error": str(e), "timestamp": time.time()}

    def _generate_recommendations(self, stats: dict[str, float]) -> list[str]:
        """Generate recommendations based on performance statistics.

        Args:
            stats: Review statistics

        Returns:
            List of recommendations
        """
        recommendations = []

        reviews_today = stats.get("reviews_today", 0)

        if reviews_today > 200:
            recommendations.append(
                "High review count today - consider adjusting daily limits"
            )
        elif reviews_today < 20:
            recommendations.append(
                "Low review count - you may need more cards or different scheduling"
            )

        # Add more sophisticated recommendations based on available metrics
        # This would be expanded based on what statistics are available

        return recommendations
