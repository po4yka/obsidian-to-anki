"""Tests for bilingual consistency validation."""

import pytest

from obsidian_anki_sync.agents.models import GeneratedCard
from obsidian_anki_sync.agents.post_validation.semantic_validator import (
    _check_bilingual_consistency,
    _compare_card_structures,
    _parse_card_for_comparison,
)


class TestBilingualConsistency:
    """Test bilingual card consistency validation."""

    def test_parse_card_for_comparison(self):
        """Test parsing card HTML for consistency comparison."""
        apf_html = """<!-- Card 1 | slug: test-1-en | CardType: Simple | Tags: kotlin classes -->

<!-- Title -->
How to prohibit object creation?

<!-- Key point (code block / image) -->
<pre><code class="language-kotlin">object Singleton { }</code></pre>

<!-- Key point notes -->
<ul>
  <li>Use private constructor</li>
  <li>Object declaration preferred</li>
  <li>Factory methods work</li>
</ul>

<!-- Other notes -->
<ul>
  <li>Ref: [[test.md#qa-1]]</li>
  <li>Note: Important detail</li>
</ul>

<!-- manifest: {"slug":"test-1-en","lang":"en","type":"Simple","tags":["kotlin","classes"]} -->"""

        result = _parse_card_for_comparison(apf_html)

        assert result["title"] == "How to prohibit object creation?"
        assert result["key_point_notes_count"] == 3
        assert len(result["key_point_notes"]) == 3
        assert result["other_notes_count"] == 2
        assert result["has_key_point_code"] is True
        assert result["card_type"] == "Simple"

    def test_compare_identical_structures(self):
        """Test comparison of identical EN/RU structures."""
        en_parsed = {
            "title": "Test Title",
            "key_point_notes_count": 2,
            "other_notes_count": 1,
            "has_key_point_code": True,
            "card_type": "Simple",
            "key_point_notes": ["Note 1", "Note 2"],
            "other_notes": ["Ref: test"]
        }
        ru_parsed = en_parsed.copy()  # Identical

        errors = _compare_card_structures(1, en_parsed, ru_parsed)

        assert len(errors) == 0  # No errors for identical structures

    def test_compare_different_note_counts(self):
        """Test detection of different note counts."""
        en_parsed = {
            "title": "Test Title",
            "key_point_notes_count": 3,  # EN has 3 notes
            "other_notes_count": 1,
            "has_key_point_code": True,
            "card_type": "Simple",
            "key_point_notes": ["Note 1", "Note 2", "Note 3"],
            "other_notes": ["Ref: test"]
        }
        ru_parsed = en_parsed.copy()
        ru_parsed["key_point_notes_count"] = 2  # RU has only 2 notes
        ru_parsed["key_point_notes"] = ["Note 1", "Note 2"]

        errors = _compare_card_structures(1, en_parsed, ru_parsed)

        assert len(errors) == 1
        assert "Key point notes count mismatch" in errors[0]
        assert "EN has 3, RU has 2" in errors[0]

    def test_compare_missing_code_block(self):
        """Test detection of missing code block in one language."""
        en_parsed = {
            "title": "Test Title",
            "key_point_notes_count": 2,
            "other_notes_count": 0,
            "has_key_point_code": True,  # EN has code
            "card_type": "Simple",
            "key_point_notes": ["Note 1", "Note 2"],
            "other_notes": []
        }
        ru_parsed = en_parsed.copy()
        ru_parsed["has_key_point_code"] = False  # RU lacks code

        errors = _compare_card_structures(1, en_parsed, ru_parsed)

        assert len(errors) == 1
        assert "Code block presence mismatch" in errors[0]

    def test_compare_preference_statement_mismatch(self):
        """Test detection of contradictory preference statements."""
        en_parsed = {
            "title": "Test Title",
            "key_point_notes_count": 2,
            "other_notes_count": 0,
            "has_key_point_code": False,
            "card_type": "Simple",
            "key_point_notes": ["Use private constructor", "Object declaration preferred"],
            "other_notes": []
        }
        ru_parsed = en_parsed.copy()
        ru_parsed["key_point_notes"] = [
            "Используйте приватный конструктор", "Приватный конструктор обязателен"]

        errors = _compare_card_structures(1, en_parsed, ru_parsed)

        assert len(errors) == 1
        assert "Preference statement mismatch" in errors[0]
        assert "EN has 'prefer' preference" in errors[0]
        assert "RU lacks 'предпочтительнее' preference" in errors[0]

    def test_bilingual_consistency_check(self):
        """Test full bilingual consistency checking with multiple cards."""
        # Create mock cards
        en_card = GeneratedCard(
            card_index=1,
            slug="test-1-en",
            lang="en",
            apf_html="""<!-- Card 1 | slug: test-1-en | CardType: Simple | Tags: kotlin -->

<!-- Title -->
English Title

<!-- Key point notes -->
<ul>
  <li>Note 1</li>
  <li>Note 2</li>
</ul>

<!-- manifest: {"slug":"test-1-en","lang":"en","type":"Simple","tags":["kotlin"]} -->""",
            confidence=0.9,
            content_hash="hash1"
        )

        ru_card = GeneratedCard(
            card_index=1,
            slug="test-1-ru",
            lang="ru",
            apf_html="""<!-- Card 1 | slug: test-1-ru | CardType: Simple | Tags: kotlin -->

<!-- Title -->
Russian Title

<!-- Key point notes -->
<ul>
  <li>Note 1</li>
</ul>

<!-- manifest: {"slug":"test-1-ru","lang":"ru","type":"Simple","tags":["kotlin"]} -->""",
            confidence=0.9,
            content_hash="hash2"
        )

        cards = [en_card, ru_card]

        errors = _check_bilingual_consistency(cards)

        assert len(errors) == 1
        assert "Key point notes count mismatch" in errors[0]

    def test_bilingual_consistency_no_errors(self):
        """Test bilingual consistency with no issues."""
        # Create consistent mock cards
        en_card = GeneratedCard(
            card_index=1,
            slug="test-1-en",
            lang="en",
            apf_html="""<!-- Card 1 | slug: test-1-en | CardType: Simple | Tags: kotlin -->

<!-- Title -->
English Title

<!-- Key point notes -->
<ul>
  <li>Note 1</li>
  <li>Note 2</li>
</ul>

<!-- manifest: {"slug":"test-1-en","lang":"en","type":"Simple","tags":["kotlin"]} -->""",
            confidence=0.9,
            content_hash="hash1"
        )

        ru_card = GeneratedCard(
            card_index=1,
            slug="test-1-ru",
            lang="ru",
            apf_html="""<!-- Card 1 | slug: test-1-ru | CardType: Simple | Tags: kotlin -->

<!-- Title -->
Russian Title

<!-- Key point notes -->
<ul>
  <li>Note 1</li>
  <li>Note 2</li>
</ul>

<!-- manifest: {"slug":"test-1-ru","lang":"ru","type":"Simple","tags":["kotlin"]} -->""",
            confidence=0.9,
            content_hash="hash2"
        )

        cards = [en_card, ru_card]

        errors = _check_bilingual_consistency(cards)

        assert len(errors) == 0

    def test_bilingual_consistency_single_language(self):
        """Test that single language cards don't trigger consistency checks."""
        en_card = GeneratedCard(
            card_index=1,
            slug="test-1-en",
            lang="en",
            apf_html="<!-- Title -->\nTest\n<!-- manifest: {} -->",
            confidence=0.9,
            content_hash="hash1"
        )

        cards = [en_card]  # Only English, no Russian pair

        errors = _check_bilingual_consistency(cards)

        assert len(errors) == 0  # No bilingual pairs to check
