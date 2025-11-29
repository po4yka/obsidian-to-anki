"""Unit tests for SplitValidatorAgentAI."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from obsidian_anki_sync.agents.models import (
    CardSplitPlan,
    CardSplittingResult,
    NoteMetadata,
)
from obsidian_anki_sync.agents.pydantic.split_validator import (
    SplitValidationResult,
    SplitValidatorAgentAI,
)


@pytest.fixture
def mock_model():
    """Create a mock PydanticAI model."""
    return MagicMock()


@pytest.fixture
def split_validator(mock_model):
    """Create a split validator agent with mock model."""
    with patch(
        "obsidian_anki_sync.agents.pydantic.split_validator.Agent"
    ) as mock_agent_class:
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance
        agent = SplitValidatorAgentAI(model=mock_model)
        agent.agent = mock_agent_instance
        yield agent


@pytest.mark.asyncio
async def test_validate_valid_split(split_validator):
    """Test validation of a valid split plan."""
    # Mock agent run result
    mock_result = MagicMock()
    mock_result.output = SplitValidationResult(
        is_valid=True,
        validation_score=0.9,
        feedback="Good split",
        suggested_modifications=[],
    )
    split_validator.agent.run = AsyncMock(return_value=mock_result)

    # Input data
    note_content = "List of 5 items..."
    metadata = NoteMetadata(
        title="Test Note",
        topic="Test Topic",
        tags=["test"],
        file_path="/path/to/note.md",
    )
    splitting_result = CardSplittingResult(
        should_split=True,
        card_count=5,
        splitting_strategy="list",
        reasoning="List of items",
        split_plan=[
            CardSplitPlan(
                card_number=1,
                concept="Item 1",
                question="Q1",
                answer_summary="A1",
                rationale="R1",
            )
        ],
        decision_time=0.1,
    )

    # Run validation
    result = await split_validator.validate(
        note_content=note_content,
        metadata=metadata,
        splitting_result=splitting_result,
    )

    # Verify
    assert result.is_valid
    assert result.validation_score == 0.9
    assert result.feedback == "Good split"
    split_validator.agent.run.assert_called_once()


@pytest.mark.asyncio
async def test_validate_invalid_split(split_validator):
    """Test validation of an invalid split plan."""
    # Mock agent run result
    mock_result = MagicMock()
    mock_result.output = SplitValidationResult(
        is_valid=False,
        validation_score=0.2,
        feedback="Over-fragmentation",
        suggested_modifications=["Merge into single card"],
    )
    split_validator.agent.run = AsyncMock(return_value=mock_result)

    # Input data
    note_content = "Simple concept"
    metadata = NoteMetadata(
        title="Simple Note",
        topic="Test Topic",
        tags=["test"],
        file_path="/path/to/note.md",
    )
    splitting_result = CardSplittingResult(
        should_split=True,
        card_count=2,
        splitting_strategy="concept",
        reasoning="Split simple concept",
        split_plan=[],
        decision_time=0.1,
    )

    # Run validation
    result = await split_validator.validate(
        note_content=note_content,
        metadata=metadata,
        splitting_result=splitting_result,
    )

    # Verify
    assert not result.is_valid
    assert result.validation_score == 0.2
    assert "Over-fragmentation" in result.feedback


@pytest.mark.asyncio
async def test_validate_fallback_on_error(split_validator):
    """Test fallback behavior when validation fails."""
    # Mock agent run to raise exception
    split_validator.agent.run = AsyncMock(side_effect=Exception("Model error"))

    # Input data
    note_content = "Content"
    metadata = NoteMetadata(
        title="Test Note",
        topic="Test Topic",
        tags=["test"],
        file_path="/path/to/note.md",
    )
    splitting_result = CardSplittingResult(
        should_split=True,
        card_count=2,
        splitting_strategy="concept",
        reasoning="Reason",
        split_plan=[],
        decision_time=0.1,
    )

    # Run validation
    result = await split_validator.validate(
        note_content=note_content,
        metadata=metadata,
        splitting_result=splitting_result,
    )

    # Verify fallback (defaults to valid to avoid blocking)
    assert result.is_valid
    assert result.validation_score == 0.5
    assert "Validation failed" in result.feedback
