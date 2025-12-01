from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_anki_sync.agents.models import HighlightedQA, HighlightResult
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.sync.card_generator import CardGenerator


@pytest.fixture()
def card_generator() -> CardGenerator:
    config = MagicMock(spec=Config)
    config.vault_path = Path()
    apf_gen = MagicMock()
    return CardGenerator(config, apf_gen)


def test_format_highlight_hint_includes_candidates(
    card_generator: CardGenerator,
) -> None:
    highlight = HighlightResult(
        qa_candidates=[
            HighlightedQA(
                question="What is binary search?",
                answer="A divide-and-conquer algorithm for sorted arrays",
                confidence=0.92,
                source_excerpt="Question (EN) ...",
                anchor="question-en",
            )
        ],
        summaries=["Discusses binary search variants"],
        suggestions=["Add Russian answer for Question 1"],
        detected_sections=["Question (EN)", "Answer (EN)"],
        confidence=0.88,
        note_status="draft",
        analysis_time=1.5,
        raw_excerpt="Binary search overview",
    )

    formatted = card_generator._format_highlight_hint(highlight)

    assert "Candidate Q&A pairs" in formatted
    assert "Add Russian answer" in formatted
    assert "draft" in formatted


def test_format_highlight_hint_returns_blank_for_empty(
    card_generator: CardGenerator,
) -> None:
    highlight = HighlightResult(
        qa_candidates=[],
        summaries=[],
        suggestions=[],
        detected_sections=[],
        confidence=0.0,
        note_status="unknown",
        analysis_time=0.5,
        raw_excerpt=None,
    )

    formatted = card_generator._format_highlight_hint(highlight)

    assert formatted == ""


def test_format_highlight_hint_none(card_generator: CardGenerator) -> None:
    assert card_generator._format_highlight_hint(None) == ""
