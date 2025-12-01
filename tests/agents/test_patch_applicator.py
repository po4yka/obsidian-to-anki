"""Tests for patch_applicator module."""

import pytest

from obsidian_anki_sync.agents.models import CardCorrection, GeneratedCard
from obsidian_anki_sync.agents.patch_applicator import (
    PATCHABLE_FIELDS,
    apply_corrections,
)


@pytest.fixture
def sample_cards() -> list[GeneratedCard]:
    """Create sample cards for testing.

    Note: GeneratedCard uses 1-based card_index (ge=1).
    """
    return [
        GeneratedCard(
            card_index=1,
            slug="test-card-1",
            lang="en",
            apf_html="<p>Original content 1</p>",
            confidence=0.8,
        ),
        GeneratedCard(
            card_index=2,
            slug="test-card-2",
            lang="ru",
            apf_html="<p>Original content 2</p>",
            confidence=0.9,
        ),
        GeneratedCard(
            card_index=3,
            slug="test-card-3",
            lang="en",
            apf_html="<p>Original content 3</p>",
            confidence=0.7,
        ),
    ]


class TestApplyCorrections:
    """Tests for apply_corrections function."""

    def test_empty_corrections_returns_original(
        self, sample_cards: list[GeneratedCard]
    ) -> None:
        """Empty corrections list returns original cards unchanged."""
        corrected, changes = apply_corrections(sample_cards, [])

        assert corrected == sample_cards
        assert changes == []

    def test_apply_single_apf_html_correction(
        self, sample_cards: list[GeneratedCard]
    ) -> None:
        """Apply a single apf_html correction."""
        corrections = [
            CardCorrection(
                card_index=1,
                field_name="apf_html",
                current_value="<p>Original content 1</p>",
                suggested_value="<p>Fixed content 1</p>",
                rationale="Fixed HTML formatting",
            )
        ]

        corrected, changes = apply_corrections(sample_cards, corrections)

        assert len(corrected) == 3
        assert corrected[0].apf_html == "<p>Fixed content 1</p>"
        assert corrected[1].apf_html == "<p>Original content 2</p>"  # Unchanged
        assert corrected[2].apf_html == "<p>Original content 3</p>"  # Unchanged

        assert len(changes) == 1
        assert "Card 1" in changes[0]
        assert "apf_html" in changes[0]

    def test_apply_multiple_corrections(
        self, sample_cards: list[GeneratedCard]
    ) -> None:
        """Apply multiple corrections to different cards."""
        corrections = [
            CardCorrection(
                card_index=1,
                field_name="slug",
                suggested_value="fixed-slug-1",
                rationale="Fixed slug",
            ),
            CardCorrection(
                card_index=3,
                field_name="confidence",
                suggested_value="0.95",  # String - will be converted by Pydantic
                rationale="Increased confidence",
            ),
        ]

        corrected, changes = apply_corrections(sample_cards, corrections)

        assert corrected[0].slug == "fixed-slug-1"
        assert corrected[2].confidence == 0.95

        assert len(changes) == 2

    def test_apply_lang_correction(self, sample_cards: list[GeneratedCard]) -> None:
        """Apply a lang field correction."""
        corrections = [
            CardCorrection(
                card_index=2,
                field_name="lang",
                current_value="ru",
                suggested_value="en",
                rationale="Wrong language detected",
            )
        ]

        corrected, changes = apply_corrections(sample_cards, corrections)

        assert corrected[1].lang == "en"  # Card with index 2 is at position 1
        assert len(changes) == 1
        assert "'ru' -> 'en'" in changes[0]

    def test_nonexistent_card_index_skipped(
        self, sample_cards: list[GeneratedCard]
    ) -> None:
        """Correction for nonexistent card index is skipped."""
        corrections = [
            CardCorrection(
                card_index=99,  # Does not exist
                field_name="apf_html",
                suggested_value="<p>New content</p>",
                rationale="Fix HTML",
            )
        ]

        corrected, changes = apply_corrections(sample_cards, corrections)

        # All cards unchanged
        assert len(corrected) == 3
        for orig, corr in zip(sample_cards, corrected, strict=True):
            assert orig.slug == corr.slug
            assert orig.apf_html == corr.apf_html

        # No changes applied
        assert changes == []

    def test_non_patchable_field_skipped(
        self, sample_cards: list[GeneratedCard]
    ) -> None:
        """Correction for non-patchable field is skipped."""
        corrections = [
            CardCorrection(
                card_index=1,
                field_name="card_index",  # Not patchable
                suggested_value="5",
                rationale="Try to change index",
            )
        ]

        corrected, changes = apply_corrections(sample_cards, corrections)

        # Card index unchanged
        assert corrected[0].card_index == 1

        # No changes applied
        assert changes == []

    def test_original_cards_not_mutated(
        self, sample_cards: list[GeneratedCard]
    ) -> None:
        """Original cards list is not mutated."""
        original_apf = sample_cards[0].apf_html

        corrections = [
            CardCorrection(
                card_index=1,
                field_name="apf_html",
                suggested_value="<p>New content</p>",
                rationale="Fix HTML",
            )
        ]

        corrected, _ = apply_corrections(sample_cards, corrections)

        # Original unchanged
        assert sample_cards[0].apf_html == original_apf
        # Corrected is different
        assert corrected[0].apf_html == "<p>New content</p>"

    def test_cards_returned_in_order(self, sample_cards: list[GeneratedCard]) -> None:
        """Corrected cards are returned in card_index order."""
        # Apply correction to middle card only
        corrections = [
            CardCorrection(
                card_index=2,
                field_name="apf_html",
                suggested_value="<p>Fixed</p>",
                rationale="Fix",
            )
        ]

        corrected, _ = apply_corrections(sample_cards, corrections)

        # Verify order - cards should be sorted by card_index
        assert [c.card_index for c in corrected] == [1, 2, 3]

    def test_change_description_truncated(
        self, sample_cards: list[GeneratedCard]
    ) -> None:
        """Long values are truncated in change descriptions."""
        long_html = "<p>" + "x" * 100 + "</p>"
        corrections = [
            CardCorrection(
                card_index=1,
                field_name="apf_html",
                suggested_value=long_html,
                rationale="Long content",
            )
        ]

        corrected, changes = apply_corrections(sample_cards, corrections)

        # Value applied fully
        assert corrected[0].apf_html == long_html

        # Description truncated
        assert len(changes) == 1
        assert "..." in changes[0]  # Truncation indicator


class TestPatchableFields:
    """Tests for PATCHABLE_FIELDS constant."""

    def test_expected_fields_are_patchable(self) -> None:
        """Expected fields are in PATCHABLE_FIELDS."""
        assert "slug" in PATCHABLE_FIELDS
        assert "lang" in PATCHABLE_FIELDS
        assert "apf_html" in PATCHABLE_FIELDS
        assert "confidence" in PATCHABLE_FIELDS

    def test_card_index_not_patchable(self) -> None:
        """card_index is not patchable (structural field)."""
        assert "card_index" not in PATCHABLE_FIELDS

    def test_patchable_fields_count(self) -> None:
        """Exactly 4 fields are patchable."""
        assert len(PATCHABLE_FIELDS) == 4
