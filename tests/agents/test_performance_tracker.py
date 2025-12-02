"""Tests for performance tracker agent."""

import unittest
from unittest.mock import MagicMock, patch

from obsidian_anki_sync.agents.performance_tracker import PerformanceTracker
from obsidian_anki_sync.models import Card, Manifest


class TestPerformanceTracker(unittest.TestCase):
    def setUp(self):
        self.mock_anki_client = MagicMock()
        self.tracker = PerformanceTracker(self.mock_anki_client)

    def test_get_card_performance_success(self):
        """Test fetching card performance metrics."""

        # Mock Anki client response for notesToCards
        def invoke_side_effect(action, params=None):
            if action == "notesToCards":
                return [[1001]]  # Card ID for note 123
            return None

        self.mock_anki_client.invoke.side_effect = invoke_side_effect

        # Mock cards_info
        self.mock_anki_client.cards_info.return_value = [
            {
                "cardId": 1001,
                "note": 123,  # Note ID
                "reviews": 10,
                "lapses": 1,
                "interval": 5,
                "factor": 2500,  # 2.5 ease
            }
        ]

        metrics = self.tracker.get_card_performance([123])

        self.assertIn(123, metrics)
        card_metrics = metrics[123]
        self.assertEqual(card_metrics["total_reviews"], 10)
        self.assertEqual(card_metrics["lapses"], 1)
        self.assertEqual(card_metrics["lapse_rate"], 0.1)
        self.assertEqual(card_metrics["difficulty_indicator"], "medium")

    def test_get_review_statistics_success(self):
        """Test fetching overall review statistics."""
        self.mock_anki_client.get_num_cards_reviewed_today.return_value = 50
        self.mock_anki_client.get_collection_stats.return_value = "<html>Stats</html>"

        stats = self.tracker.get_review_statistics()

        self.assertEqual(stats["reviews_today"], 50.0)
        self.assertEqual(stats["collection_stats_html"], "<html>Stats</html>")

    def test_analyze_card_patterns_with_guids(self):
        """Test card pattern analysis using GUIDs."""
        # Create mock cards with GUIDs
        card1 = MagicMock(spec=Card)
        card1.guid = "guid1"
        card1.slug = "slug1"

        card2 = MagicMock(spec=Card)
        card2.guid = "guid2"
        card2.slug = "slug2"

        cards = [card1, card2]

        # Mock find_notes
        def find_notes_side_effect(query):
            if "guid1" in query:
                return [101]
            if "guid2" in query:
                return [102]
            return []

        self.mock_anki_client.find_notes.side_effect = find_notes_side_effect

        # Mock invoke for notesToCards
        def invoke_side_effect(action, params=None):
            if action == "notesToCards":
                # params["notes"] will be [101, 102] (order depends on implementation but likely preserved)
                # Return list of lists of card IDs
                # We need to be careful if the input list order matters or if we just return results
                # The code does: card_ids_list = self.anki_client.invoke("notesToCards", {"notes": note_ids})
                # Let's assume it returns [[1001], [1002]]
                return [[1001], [1002]]
            return None

        self.mock_anki_client.invoke.side_effect = invoke_side_effect

        # Mock cards_info
        self.mock_anki_client.cards_info.return_value = [
            {
                "cardId": 1001,
                "note": 101,
                "reviews": 20,
                "lapses": 10,  # High lapse rate
                "factor": 1300,  # 1.3 ease -> hard
            },
            {
                "cardId": 1002,
                "note": 102,
                "reviews": 5,
                "lapses": 0,
                "factor": 2900,  # 2.9 ease -> easy
            },
        ]

        issues = self.tracker.analyze_card_patterns(cards)

        # Check that issues were identified correctly
        self.assertIn("slug1", issues.get("persistent_failures", []))
        self.assertIn("slug1", issues.get("very_difficult", []))
        self.assertIn("slug1", issues.get("potential_leeches", []))

        self.assertNotIn("slug2", issues.get("persistent_failures", []))
        self.assertNotIn("slug2", issues.get("very_difficult", []))

    def test_analyze_card_patterns_no_guids(self):
        """Test pattern analysis with cards missing GUIDs."""
        card1 = MagicMock(spec=Card)
        card1.guid = ""  # No GUID
        card1.slug = "slug1"

        issues = self.tracker.analyze_card_patterns([card1])

        # Should find no issues and call no Anki methods
        self.assertEqual(len(issues), 0)
        self.mock_anki_client.find_notes.assert_not_called()


if __name__ == "__main__":
    unittest.main()
