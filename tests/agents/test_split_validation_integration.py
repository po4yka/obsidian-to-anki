"""Integration tests for split validation workflow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from obsidian_anki_sync.agents.langgraph.nodes import split_validation_node
from obsidian_anki_sync.agents.langgraph.state import (
    PipelineState,
    register_runtime_resources,
)
from obsidian_anki_sync.agents.models import (
    CardSplitPlan,
    CardSplittingResult,
    NoteMetadata,
)
from obsidian_anki_sync.agents.pydantic.split_validator import SplitValidationResult
from obsidian_anki_sync.config import Config


@pytest.fixture()
def mock_config():
    """Create a mock configuration."""
    config = MagicMock(spec=Config)
    config.get_model_for_agent.return_value = "mock-model"
    return config


@pytest.fixture()
def pipeline_state(mock_config):
    """Create a sample pipeline state."""
    # Create a mock model factory
    mock_model_factory = MagicMock()
    mock_model_factory.get_model.side_effect = (
        lambda agent_type: MagicMock() if agent_type == "split_validator" else None
    )

    # Register runtime resources and get the key
    runtime_key = register_runtime_resources(
        config=mock_config,
        model_factory=mock_model_factory,
    )

    return PipelineState(
        note_content="Test content",
        metadata_dict=NoteMetadata(
            id="test-note-123",
            title="Test Note",
            topic="Test Topic",
            tags=["test"],
            file_path="/path/to/note.md",
            created="2024-01-01T00:00:00Z",
            updated="2024-01-01T00:00:00Z",
        ).model_dump(),
        qa_pairs_dicts=[],
        config=mock_config,
        messages=[],
        stage_times={},
        step_counts={},
        runtime_key=runtime_key,
        config_snapshot=mock_config.__dict__
        if hasattr(mock_config, "__dict__")
        else {},
    )


@pytest.mark.asyncio()
async def test_split_validation_node_skipped_no_split(pipeline_state):
    """Test that validation is skipped if no split is proposed."""
    pipeline_state["card_splitting"] = None

    new_state = await split_validation_node(pipeline_state)

    assert new_state["current_stage"] == "generation"
    assert "split_validation" not in new_state or new_state["split_validation"] is None


@pytest.mark.asyncio()
async def test_split_validation_node_approved(pipeline_state):
    """Test that validation approves a valid split."""
    # Setup splitting result
    splitting_result = CardSplittingResult(
        should_split=True,
        card_count=2,
        splitting_strategy="concept",
        reasoning="Reason",
        split_plan=[
            CardSplitPlan(
                card_number=1,
                concept="C1",
                question="Q1",
                answer_summary="A1",
                rationale="R1",
            )
        ],
        decision_time=0.1,
    )
    pipeline_state["card_splitting"] = splitting_result.model_dump()

    # Mock agent
    mock_agent = MagicMock()
    mock_agent.validate = AsyncMock(
        return_value=SplitValidationResult(
            is_valid=True,
            validation_score=0.9,
            feedback="Approved",
            suggested_modifications=[],
        )
    )

    with (
        patch(
            "obsidian_anki_sync.agents.langgraph.nodes.SplitValidatorAgentAI",
            return_value=mock_agent,
        ),
        patch(
            "obsidian_anki_sync.agents.langgraph.nodes.create_openrouter_model_from_env"
        ),
    ):
        new_state = await split_validation_node(pipeline_state)

    assert new_state["current_stage"] == "generation"
    assert new_state["split_validation"]["is_valid"]
    assert "Approved split" in new_state["messages"][-1]

    # Verify split plan remains unchanged
    assert new_state["card_splitting"]["should_split"]


@pytest.mark.asyncio()
async def test_split_validation_node_rejected(pipeline_state):
    """Test that validation rejects an invalid split and reverts to single card."""
    # Setup splitting result
    splitting_result = CardSplittingResult(
        should_split=True,
        card_count=2,
        splitting_strategy="concept",
        reasoning="Reason",
        split_plan=[
            CardSplitPlan(
                card_number=1,
                concept="C1",
                question="Q1",
                answer_summary="A1",
                rationale="R1",
            )
        ],
        decision_time=0.1,
    )
    pipeline_state["card_splitting"] = splitting_result.model_dump()

    # Mock agent
    mock_agent = MagicMock()
    mock_agent.validate = AsyncMock(
        return_value=SplitValidationResult(
            is_valid=False,
            validation_score=0.2,
            feedback="Rejected",
            suggested_modifications=[],
        )
    )

    with (
        patch(
            "obsidian_anki_sync.agents.langgraph.nodes.SplitValidatorAgentAI",
            return_value=mock_agent,
        ),
        patch(
            "obsidian_anki_sync.agents.langgraph.nodes.create_openrouter_model_from_env"
        ),
    ):
        new_state = await split_validation_node(pipeline_state)

    assert new_state["current_stage"] == "generation"
    assert not new_state["split_validation"]["is_valid"]
    assert "Rejected split" in new_state["messages"][-1]

    # Verify split plan was reverted
    assert not new_state["card_splitting"]["should_split"]
    assert new_state["card_splitting"]["card_count"] == 1
    assert new_state["card_splitting"]["splitting_strategy"] == "none"
