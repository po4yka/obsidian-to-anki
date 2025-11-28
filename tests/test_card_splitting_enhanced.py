"""Tests for enhanced Card Splitting Agent with advanced strategies and confidence scoring."""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from obsidian_anki_sync.agents.models import CardSplittingResult, CardSplitPlan
from obsidian_anki_sync.agents.pydantic import CardSplittingAgentAI
from obsidian_anki_sync.models import NoteMetadata, QAPair


@pytest.fixture
def sample_metadata():
    """Create sample note metadata."""
    return NoteMetadata(
        id="test-001",
        title="Test Note",
        topic="testing",
        language_tags=["en", "ru"],
        tags=[],
        created="2024-01-01",
        updated="2024-01-02",
    )


@pytest.fixture
def sample_qa_pairs():
    """Create sample Q&A pairs."""
    return [
        QAPair(
            card_index=1,
            question_en="What is testing?",
            question_ru="Что такое тестирование?",
            answer_en="Testing is verification.",
            answer_ru="Тестирование - это проверка.",
            context="",
            followups="",
            references="",
            related="",
        )
    ]


@pytest.fixture
def card_splitting_agent():
    """Create a CardSplittingAgentAI instance with mocked PydanticAI Agent."""
    mock_model = MagicMock()
    with patch("obsidian_anki_sync.agents.pydantic.card_splitting.Agent") as mock_agent_class:
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance
        agent = CardSplittingAgentAI(model=mock_model, temperature=0.0)
        agent.agent = mock_agent_instance
        return agent


class TestCardSplittingEnhanced:
    """Tests for enhanced card splitting features."""

    def test_confidence_scoring_high_confidence(
        self, card_splitting_agent, sample_metadata, sample_qa_pairs
    ):
        """Test high confidence scoring for clear split decision."""
        # Mock agent response with high confidence
        mock_result = MagicMock()
        mock_result.data = MagicMock()
        mock_result.data.should_split = True
        mock_result.data.card_count = 3
        mock_result.data.splitting_strategy = "list"
        mock_result.data.split_plan = [
            MagicMock(
                card_number=1,
                concept="Overview",
                question="What are the 3 items?",
                answer_summary="Item 1, Item 2, Item 3",
                rationale="Overview card",
            ),
            MagicMock(
                card_number=2,
                concept="Item 1",
                question="What is Item 1?",
                answer_summary="Description of Item 1",
                rationale="Individual item",
            ),
            MagicMock(
                card_number=3,
                concept="Item 2",
                question="What is Item 2?",
                answer_summary="Description of Item 2",
                rationale="Individual item",
            ),
        ]
        mock_result.data.reasoning = "Clear list pattern with 3 items"
        mock_result.data.confidence = 0.95
        mock_result.data.fallback_strategy = None

        card_splitting_agent.agent.run = AsyncMock(return_value=mock_result)

        result = asyncio.run(card_splitting_agent.analyze(
            note_content="Test content with 3 items",
            metadata=sample_metadata,
            qa_pairs=sample_qa_pairs,
        ))

        assert result.confidence == 0.95
        assert result.should_split is True
        assert result.card_count == 3
        assert result.splitting_strategy == "list"

    def test_confidence_scoring_low_confidence(
        self, card_splitting_agent, sample_metadata, sample_qa_pairs
    ):
        """Test low confidence scoring triggers fallback."""
        # Mock agent response with low confidence
        mock_result = MagicMock()
        mock_result.data = MagicMock()
        mock_result.data.should_split = True
        mock_result.data.card_count = 2
        mock_result.data.splitting_strategy = "concept"
        mock_result.data.split_plan = [
            MagicMock(
                card_number=1,
                concept="Concept 1",
                question="What is Concept 1?",
                answer_summary="Answer 1",
                rationale="",
            ),
            MagicMock(
                card_number=2,
                concept="Concept 2",
                question="What is Concept 2?",
                answer_summary="Answer 2",
                rationale="",
            ),
        ]
        mock_result.data.reasoning = "Uncertain - could be split or kept together"
        mock_result.data.confidence = 0.5
        mock_result.data.fallback_strategy = "none"

        card_splitting_agent.agent.run = AsyncMock(return_value=mock_result)

        result = asyncio.run(card_splitting_agent.analyze(
            note_content="Ambiguous content",
            metadata=sample_metadata,
            qa_pairs=sample_qa_pairs,
        ))

        assert result.confidence == 0.5
        assert result.fallback_strategy == "none"

    def test_advanced_strategy_difficulty(
        self, card_splitting_agent, sample_metadata, sample_qa_pairs
    ):
        """Test difficulty-based splitting strategy."""
        mock_result = MagicMock()
        mock_result.data = MagicMock()
        mock_result.data.should_split = True
        mock_result.data.card_count = 3
        mock_result.data.splitting_strategy = "difficulty"
        mock_result.data.split_plan = [
            MagicMock(
                card_number=1,
                concept="Easy concept",
                question="What is the basic concept?",
                answer_summary="Basic answer",
                rationale="Easy difficulty",
            ),
            MagicMock(
                card_number=2,
                concept="Medium concept",
                question="What is the intermediate concept?",
                answer_summary="Intermediate answer",
                rationale="Medium difficulty",
            ),
            MagicMock(
                card_number=3,
                concept="Hard concept",
                question="What is the advanced concept?",
                answer_summary="Advanced answer",
                rationale="Hard difficulty",
            ),
        ]
        mock_result.data.reasoning = "Concepts ordered by difficulty: easy -> medium -> hard"
        mock_result.data.confidence = 0.88
        mock_result.data.fallback_strategy = None

        card_splitting_agent.agent.run = AsyncMock(return_value=mock_result)

        result = asyncio.run(card_splitting_agent.analyze(
            note_content="Content with varying difficulty levels",
            metadata=sample_metadata,
            qa_pairs=sample_qa_pairs,
        ))

        assert result.splitting_strategy == "difficulty"
        assert result.card_count == 3
        assert len(result.split_plan) == 3

    def test_advanced_strategy_prerequisite(
        self, card_splitting_agent, sample_metadata, sample_qa_pairs
    ):
        """Test prerequisite-aware splitting strategy."""
        mock_result = MagicMock()
        mock_result.data = MagicMock()
        mock_result.data.should_split = True
        mock_result.data.card_count = 3
        mock_result.data.splitting_strategy = "prerequisite"
        mock_result.data.split_plan = [
            MagicMock(
                card_number=1,
                concept="Foundation",
                question="What is the foundational concept?",
                answer_summary="Base concept",
                rationale="Foundational - no prerequisites",
            ),
            MagicMock(
                card_number=2,
                concept="Intermediate",
                question="What is the intermediate concept?",
                answer_summary="Builds on foundation",
                rationale="Requires foundation",
            ),
            MagicMock(
                card_number=3,
                concept="Advanced",
                question="What is the advanced concept?",
                answer_summary="Builds on intermediate",
                rationale="Requires intermediate",
            ),
        ]
        mock_result.data.reasoning = "Ordered by prerequisites: foundation -> intermediate -> advanced"
        mock_result.data.confidence = 0.9
        mock_result.data.fallback_strategy = None

        card_splitting_agent.agent.run = AsyncMock(return_value=mock_result)

        result = asyncio.run(card_splitting_agent.analyze(
            note_content="Content with prerequisite relationships",
            metadata=sample_metadata,
            qa_pairs=sample_qa_pairs,
        ))

        assert result.splitting_strategy == "prerequisite"
        # Verify ordering (foundation first)
        assert result.split_plan[0].concept == "Foundation"

    def test_fallback_on_error(
        self, card_splitting_agent, sample_metadata, sample_qa_pairs
    ):
        """Test fallback behavior on agent error."""
        # Mock agent error
        card_splitting_agent.agent.run = AsyncMock(
            side_effect=Exception("Agent error"))

        result = asyncio.run(card_splitting_agent.analyze(
            note_content="Test content",
            metadata=sample_metadata,
            qa_pairs=sample_qa_pairs,
        ))

        # Should return conservative fallback
        assert result.should_split is False
        assert result.card_count == 1
        assert result.confidence == 0.3  # Low confidence for fallback
        assert result.fallback_strategy == "none"
