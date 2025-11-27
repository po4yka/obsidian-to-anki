"""Tests for Self-Reflection feature in LangGraph pipeline.

This module tests:
1. Reflection output models (Pydantic models)
2. Reflection trace storage
3. Reflection node functionality
4. Revision node functionality
5. Workflow routing with self-reflection
6. Config integration
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from obsidian_anki_sync.agents.langgraph.reflection_models import (
    EnrichmentReflectionOutput,
    GenerationReflectionOutput,
    ReflectionOutput,
    ReflectionTrace,
    RevisionInput,
    RevisionOutput,
    RevisionSuggestion,
)
from obsidian_anki_sync.agents.langgraph.reflection_nodes import (
    _can_revise,
    _increment_revision_count,
    _should_skip_reflection,
    _store_reflection_trace,
    reflect_after_enrichment_node,
    reflect_after_generation_node,
    revise_enrichment_node,
    revise_generation_node,
    should_revise_enrichment,
    should_revise_generation,
)


# Configure pytest-anyio mode - only use asyncio backend
pytestmark = [
    pytest.mark.anyio,
]


@pytest.fixture(scope="session")
def anyio_backend():
    """Only run async tests with asyncio backend."""
    return "asyncio"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_pipeline_state():
    """Create a mock pipeline state for self-reflection tests."""
    return {
        "note_content": "# Test Note\n\nQ: What is Python?\nA: A programming language.",
        "metadata_dict": {
            "id": "test-note-id",
            "title": "Test Note",
            "topic": "Programming",
            "tags": ["python", "basics"],
            "language_tags": ["en"],
            "source": "test.md",
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
        },
        "qa_pairs_dicts": [
            {"question": "What is Python?", "answer": "A programming language."}
        ],
        # Self-reflection configuration
        "enable_self_reflection": True,
        "store_reflection_traces": True,
        "log_reflection_traces": False,
        "reflection_enabled_stages": ["generation", "context_enrichment"],
        "reflection_model": MagicMock(),
        "reflection_traces": {},
        "current_reflection": None,
        "revision_count": 0,
        "max_revisions": 2,
        "stage_revision_counts": {},
        # Pipeline results
        "generation": {
            "success": True,
            "cards": [
                {
                    "card_type": "Simple",
                    "question": "What is Python?",
                    "answer": "A programming language.",
                    "slug": "test-python-001",
                }
            ],
        },
        "post_validation": {
            "is_valid": True,
            "issues": [],
        },
        "context_enrichment": {
            "enrichments": [
                {
                    "examples": ["print('Hello, World!')"],
                    "mnemonics": ["Python is named after Monty Python"],
                    "context": "Python was created by Guido van Rossum",
                }
            ],
        },
        "linter_results": [],
        # CoT context (for reflection to check)
        "reasoning_traces": {
            "generation": {
                "stage": "generation",
                "planned_approach": "Create simple Q&A card",
                "recommendations": ["Keep answer concise"],
                "confidence": 0.9,
            }
        },
        "current_reasoning": None,
        # Standard state fields
        "step_counts": {},
        "stage_times": {},
        "messages": [],
        "max_steps": 100,
        "enable_context_enrichment": True,
        "enable_memorization_quality": True,
    }


# =============================================================================
# Test ReflectionOutput Models
# =============================================================================


class TestReflectionOutputModels:
    """Test Pydantic models for reflection outputs."""

    def test_base_reflection_output_default_values(self):
        """Test ReflectionOutput has correct default values."""
        output = ReflectionOutput(
            reflection="Test reflection",
            quality_assessment="Good quality",
        )
        assert output.reflection == "Test reflection"
        assert output.quality_assessment == "Good quality"
        assert output.issues_found == []
        assert output.strengths == []
        assert output.revision_suggestions == []
        assert output.revision_needed is False
        assert output.revision_priority == "none"
        assert output.confidence == 0.5
        assert output.cot_plan_followed is True
        assert output.cot_deviations == []

    def test_reflection_output_with_issues(self):
        """Test ReflectionOutput with issues found."""
        output = ReflectionOutput(
            reflection="Found some issues",
            quality_assessment="Needs improvement",
            issues_found=["Question unclear", "Answer incomplete"],
            revision_needed=True,
            revision_priority="high",
            confidence=0.8,
        )
        assert output.revision_needed is True
        assert output.revision_priority == "high"
        assert len(output.issues_found) == 2

    def test_generation_reflection_output(self):
        """Test GenerationReflectionOutput specific fields."""
        output = GenerationReflectionOutput(
            reflection="Generation reflection",
            quality_assessment="Good",
            card_quality_scores=[0.9, 0.85],
            format_compliance="APF format correct",
            content_accuracy="Accurate",
            question_clarity="Clear",
            answer_completeness="Complete",
            memorization_potential="High",
            recommended_card_changes=[{"card_index": 0, "changes": ["Improve wording"]}],
        )
        assert output.card_quality_scores == [0.9, 0.85]
        assert output.format_compliance == "APF format correct"
        assert len(output.recommended_card_changes) == 1

    def test_enrichment_reflection_output(self):
        """Test EnrichmentReflectionOutput specific fields."""
        output = EnrichmentReflectionOutput(
            reflection="Enrichment reflection",
            quality_assessment="Good",
            example_quality="Helpful examples",
            mnemonic_effectiveness="Memorable",
            context_relevance="Relevant",
            enrichment_impact="Positive",
            over_enrichment_risk=False,
            recommended_enrichment_changes=["Add more examples"],
        )
        assert output.over_enrichment_risk is False
        assert len(output.recommended_enrichment_changes) == 1


class TestRevisionSuggestion:
    """Test RevisionSuggestion model."""

    def test_revision_suggestion_creation(self):
        """Test creating a revision suggestion."""
        suggestion = RevisionSuggestion(
            issue="Question is ambiguous",
            severity="high",
            suggestion="Rephrase to be more specific",
            affected_field="question",
        )
        assert suggestion.issue == "Question is ambiguous"
        assert suggestion.severity == "high"
        assert suggestion.suggestion == "Rephrase to be more specific"
        assert suggestion.affected_field == "question"

    def test_revision_suggestion_default_severity(self):
        """Test revision suggestion default severity."""
        suggestion = RevisionSuggestion(
            issue="Minor formatting issue",
            suggestion="Add period at end",
        )
        assert suggestion.severity == "medium"
        assert suggestion.affected_field == ""


class TestReflectionTrace:
    """Test ReflectionTrace model."""

    def test_reflection_trace_from_output(self):
        """Test creating ReflectionTrace from output."""
        output = GenerationReflectionOutput(
            reflection="Test reflection content",
            quality_assessment="Good quality",
            issues_found=["Issue 1"],
            strengths=["Strength 1"],
            revision_suggestions=[
                RevisionSuggestion(
                    issue="Test issue",
                    suggestion="Fix it",
                )
            ],
            revision_needed=True,
            confidence=0.85,
            cot_plan_followed=True,
            card_quality_scores=[0.9],
        )

        trace = ReflectionTrace.from_output(
            stage="generation",
            output=output,
            reflection_time=1.5,
            timestamp=1234567890.0,
            stage_specific_data={"card_quality_scores": output.card_quality_scores},
        )

        assert trace.stage == "generation"
        assert trace.reflection == "Test reflection content"
        assert trace.quality_assessment == "Good quality"
        assert len(trace.issues_found) == 1
        assert len(trace.strengths) == 1
        assert len(trace.revision_suggestions) == 1
        assert trace.revision_needed is True
        assert trace.confidence == 0.85
        assert trace.reflection_time == 1.5
        assert trace.timestamp == 1234567890.0
        assert trace.stage_specific_data["card_quality_scores"] == [0.9]


class TestRevisionInputOutput:
    """Test RevisionInput and RevisionOutput models."""

    def test_revision_input_creation(self):
        """Test RevisionInput model."""
        revision_input = RevisionInput(
            original_output={"cards": [{"question": "Test?"}]},
            reflection_trace={"issues_found": ["Issue 1"]},
            cot_reasoning={"planned_approach": "Simple approach"},
            revision_focus=["question", "answer"],
            max_changes=5,
        )
        assert revision_input.original_output is not None
        assert len(revision_input.revision_focus) == 2
        assert revision_input.max_changes == 5

    def test_revision_output_creation(self):
        """Test RevisionOutput model."""
        revision_output = RevisionOutput(
            revised_output={"cards": [{"question": "Improved question?"}]},
            changes_made=["Improved question clarity"],
            issues_addressed=["Issue 1"],
            issues_remaining=[],
            revision_confidence=0.9,
            further_revision_recommended=False,
        )
        assert revision_output.revised_output is not None
        assert len(revision_output.changes_made) == 1
        assert revision_output.revision_confidence == 0.9
        assert revision_output.further_revision_recommended is False


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestShouldSkipReflection:
    """Test _should_skip_reflection helper function."""

    def test_skip_when_disabled(self, mock_pipeline_state):
        """Test reflection is skipped when disabled."""
        mock_pipeline_state["enable_self_reflection"] = False
        assert _should_skip_reflection(mock_pipeline_state, "generation") is True

    def test_skip_when_stage_not_enabled(self, mock_pipeline_state):
        """Test reflection is skipped for non-enabled stage."""
        mock_pipeline_state["reflection_enabled_stages"] = ["generation"]
        assert _should_skip_reflection(mock_pipeline_state, "context_enrichment") is True

    def test_not_skip_when_enabled(self, mock_pipeline_state):
        """Test reflection is not skipped when properly enabled."""
        mock_pipeline_state["enable_self_reflection"] = True
        mock_pipeline_state["reflection_enabled_stages"] = ["generation"]
        assert _should_skip_reflection(mock_pipeline_state, "generation") is False


class TestStoreReflectionTrace:
    """Test _store_reflection_trace helper function."""

    def test_stores_trace_in_state(self, mock_pipeline_state):
        """Test reflection trace is stored in state."""
        output = GenerationReflectionOutput(
            reflection="Test reflection",
            quality_assessment="Good",
            confidence=0.9,
        )

        _store_reflection_trace(
            mock_pipeline_state,
            "generation",
            output,
            reflection_time=1.0,
        )

        assert "generation" in mock_pipeline_state["reflection_traces"]
        assert mock_pipeline_state["current_reflection"] is not None
        assert mock_pipeline_state["current_reflection"]["confidence"] == 0.9

    def test_skips_when_storage_disabled(self, mock_pipeline_state):
        """Test trace is not stored when storage is disabled."""
        mock_pipeline_state["store_reflection_traces"] = False

        output = GenerationReflectionOutput(
            reflection="Test",
            quality_assessment="Good",
        )

        _store_reflection_trace(
            mock_pipeline_state,
            "generation",
            output,
            reflection_time=1.0,
        )

        assert mock_pipeline_state["reflection_traces"] == {}


class TestCanRevise:
    """Test _can_revise helper function."""

    def test_can_revise_when_under_limit(self, mock_pipeline_state):
        """Test revision allowed when under limit."""
        mock_pipeline_state["max_revisions"] = 2
        mock_pipeline_state["stage_revision_counts"] = {"generation": 0}
        assert _can_revise(mock_pipeline_state, "generation") is True

    def test_cannot_revise_when_at_limit(self, mock_pipeline_state):
        """Test revision not allowed when at limit."""
        mock_pipeline_state["max_revisions"] = 2
        mock_pipeline_state["stage_revision_counts"] = {"generation": 2}
        assert _can_revise(mock_pipeline_state, "generation") is False

    def test_cannot_revise_when_disabled(self, mock_pipeline_state):
        """Test revision not allowed when max_revisions is 0."""
        mock_pipeline_state["max_revisions"] = 0
        assert _can_revise(mock_pipeline_state, "generation") is False


class TestIncrementRevisionCount:
    """Test _increment_revision_count helper function."""

    def test_increments_stage_count(self, mock_pipeline_state):
        """Test stage revision count is incremented."""
        mock_pipeline_state["stage_revision_counts"] = {}
        mock_pipeline_state["revision_count"] = 0

        _increment_revision_count(mock_pipeline_state, "generation")

        assert mock_pipeline_state["stage_revision_counts"]["generation"] == 1
        assert mock_pipeline_state["revision_count"] == 1

    def test_increments_existing_count(self, mock_pipeline_state):
        """Test incrementing existing count."""
        mock_pipeline_state["stage_revision_counts"] = {"generation": 1}
        mock_pipeline_state["revision_count"] = 1

        _increment_revision_count(mock_pipeline_state, "generation")

        assert mock_pipeline_state["stage_revision_counts"]["generation"] == 2
        assert mock_pipeline_state["revision_count"] == 2


# =============================================================================
# Test Routing Functions
# =============================================================================


class TestShouldReviseGeneration:
    """Test should_revise_generation routing function."""

    def test_returns_true_when_revision_needed(self, mock_pipeline_state):
        """Test returns True when revision is needed and allowed."""
        mock_pipeline_state["current_reflection"] = {
            "revision_needed": True,
        }
        mock_pipeline_state["max_revisions"] = 2
        mock_pipeline_state["stage_revision_counts"] = {"generation": 0}

        assert should_revise_generation(mock_pipeline_state) is True

    def test_returns_false_when_no_revision_needed(self, mock_pipeline_state):
        """Test returns False when revision not needed."""
        mock_pipeline_state["current_reflection"] = {
            "revision_needed": False,
        }

        assert should_revise_generation(mock_pipeline_state) is False

    def test_returns_false_when_no_reflection(self, mock_pipeline_state):
        """Test returns False when no reflection exists."""
        mock_pipeline_state["current_reflection"] = None

        assert should_revise_generation(mock_pipeline_state) is False


class TestShouldReviseEnrichment:
    """Test should_revise_enrichment routing function."""

    def test_returns_true_when_revision_needed(self, mock_pipeline_state):
        """Test returns True when revision is needed and allowed."""
        mock_pipeline_state["current_reflection"] = {
            "revision_needed": True,
        }
        mock_pipeline_state["max_revisions"] = 2
        mock_pipeline_state["stage_revision_counts"] = {"context_enrichment": 0}

        assert should_revise_enrichment(mock_pipeline_state) is True

    def test_returns_false_at_revision_limit(self, mock_pipeline_state):
        """Test returns False when at revision limit."""
        mock_pipeline_state["current_reflection"] = {
            "revision_needed": True,
        }
        mock_pipeline_state["max_revisions"] = 2
        mock_pipeline_state["stage_revision_counts"] = {"context_enrichment": 2}

        assert should_revise_enrichment(mock_pipeline_state) is False


# =============================================================================
# Test Reflection Nodes
# =============================================================================


class TestReflectAfterGenerationNode:
    """Test reflect_after_generation_node."""

    async def test_skips_when_disabled(self, mock_pipeline_state):
        """Test reflection is skipped when disabled."""
        mock_pipeline_state["enable_self_reflection"] = False

        result = await reflect_after_generation_node(mock_pipeline_state)

        assert result["current_reflection"] is None

    async def test_skips_when_stage_not_enabled(self, mock_pipeline_state):
        """Test reflection is skipped for non-enabled stage."""
        mock_pipeline_state["reflection_enabled_stages"] = ["context_enrichment"]

        result = await reflect_after_generation_node(mock_pipeline_state)

        assert "generation" not in result.get("reflection_traces", {})

    async def test_stores_reflection_on_success(self, mock_pipeline_state):
        """Test reflection trace is stored on success."""
        mock_output = GenerationReflectionOutput(
            reflection="Good quality cards",
            quality_assessment="Excellent",
            confidence=0.9,
            revision_needed=False,
            card_quality_scores=[0.95],
        )

        mock_result = MagicMock()
        mock_result.data = mock_output
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch(
            "obsidian_anki_sync.agents.langgraph.reflection_nodes.Agent",
            return_value=mock_agent,
        ):
            result = await reflect_after_generation_node(mock_pipeline_state)

            assert result["current_reflection"] is not None
            assert result["current_reflection"]["confidence"] == 0.9

    async def test_handles_missing_generation_output(self, mock_pipeline_state):
        """Test handles missing generation output gracefully."""
        mock_pipeline_state["generation"] = None

        result = await reflect_after_generation_node(mock_pipeline_state)

        # Should not crash, just skip
        assert result is not None


class TestReflectAfterEnrichmentNode:
    """Test reflect_after_enrichment_node."""

    async def test_stores_reflection_on_success(self, mock_pipeline_state):
        """Test reflection trace is stored on success."""
        mock_output = EnrichmentReflectionOutput(
            reflection="Good enrichment",
            quality_assessment="Excellent",
            confidence=0.85,
            revision_needed=False,
            example_quality="Good examples",
            mnemonic_effectiveness="Effective",
        )

        mock_result = MagicMock()
        mock_result.data = mock_output
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch(
            "obsidian_anki_sync.agents.langgraph.reflection_nodes.Agent",
            return_value=mock_agent,
        ):
            result = await reflect_after_enrichment_node(mock_pipeline_state)

            assert result["current_reflection"] is not None
            assert result["current_reflection"]["confidence"] == 0.85


# =============================================================================
# Test Revision Nodes
# =============================================================================


class TestReviseGenerationNode:
    """Test revise_generation_node."""

    async def test_skips_when_no_revision_needed(self, mock_pipeline_state):
        """Test revision is skipped when not needed."""
        mock_pipeline_state["current_reflection"] = {
            "revision_needed": False,
        }

        original_generation = mock_pipeline_state["generation"]
        result = await revise_generation_node(mock_pipeline_state)

        # Generation should remain unchanged
        assert result["generation"] == original_generation

    async def test_skips_when_at_revision_limit(self, mock_pipeline_state):
        """Test revision is skipped when at limit."""
        mock_pipeline_state["current_reflection"] = {
            "revision_needed": True,
            "issues_found": ["Issue 1"],
            "revision_suggestions": [],
        }
        mock_pipeline_state["stage_revision_counts"] = {"generation": 2}
        mock_pipeline_state["max_revisions"] = 2

        original_generation = mock_pipeline_state["generation"]
        result = await revise_generation_node(mock_pipeline_state)

        # Generation should remain unchanged
        assert result["generation"] == original_generation

    async def test_performs_revision_when_needed(self, mock_pipeline_state):
        """Test revision is performed when needed."""
        mock_pipeline_state["current_reflection"] = {
            "revision_needed": True,
            "issues_found": ["Question unclear"],
            "revision_suggestions": [
                {"suggestion": "Clarify question", "severity": "medium"}
            ],
            "quality_assessment": "Needs improvement",
            "revision_priority": "medium",
        }
        mock_pipeline_state["stage_revision_counts"] = {"generation": 0}

        mock_output = RevisionOutput(
            revised_output={
                "success": True,
                "cards": [{"question": "What is Python programming language?"}],
            },
            changes_made=["Clarified question"],
            issues_addressed=["Question unclear"],
            issues_remaining=[],
            revision_confidence=0.85,
            further_revision_recommended=False,
        )

        mock_result = MagicMock()
        mock_result.data = mock_output
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch(
            "obsidian_anki_sync.agents.langgraph.reflection_nodes.Agent",
            return_value=mock_agent,
        ):
            result = await revise_generation_node(mock_pipeline_state)

            assert result["stage_revision_counts"]["generation"] == 1
            assert result["revision_count"] == 1


# =============================================================================
# Test WorkflowBuilder Integration
# =============================================================================


class TestWorkflowBuilderSelfReflection:
    """Test WorkflowBuilder integration with self-reflection."""

    def test_reflection_nodes_added_when_enabled(self):
        """Test reflection nodes are added when enabled."""
        from obsidian_anki_sync.agents.langgraph.workflow_builder import WorkflowBuilder

        mock_config = MagicMock()
        mock_config.enable_cot_reasoning = False
        mock_config.enable_self_reflection = True
        mock_config.enable_note_correction = False

        builder = WorkflowBuilder(mock_config)
        workflow = builder.build_workflow()

        # Check that reflection nodes are in the graph
        node_names = list(workflow.nodes.keys())
        assert "reflect_after_generation" in node_names
        assert "reflect_after_enrichment" in node_names
        assert "revise_generation" in node_names
        assert "revise_enrichment" in node_names

    def test_reflection_nodes_not_added_when_disabled(self):
        """Test reflection nodes are not added when disabled."""
        from obsidian_anki_sync.agents.langgraph.workflow_builder import WorkflowBuilder

        mock_config = MagicMock()
        mock_config.enable_cot_reasoning = False
        mock_config.enable_self_reflection = False
        mock_config.enable_note_correction = False

        builder = WorkflowBuilder(mock_config)
        workflow = builder.build_workflow()

        # Check that reflection nodes are not in the graph
        node_names = list(workflow.nodes.keys())
        assert "reflect_after_generation" not in node_names
        assert "reflect_after_enrichment" not in node_names


# =============================================================================
# Test Config Integration
# =============================================================================


class TestConfigSelfReflection:
    """Test Config integration with self-reflection."""

    def test_default_reflection_config_fields_exist(self):
        """Test that self-reflection config fields exist on Config class."""
        from obsidian_anki_sync.config import Config

        # Check that the Config class has the expected fields defined
        field_names = list(Config.model_fields.keys())

        assert "enable_self_reflection" in field_names
        assert "store_reflection_traces" in field_names
        assert "log_reflection_traces" in field_names
        assert "max_revisions" in field_names
        assert "reflection_enabled_stages" in field_names
        assert "reflection_model" in field_names
        assert "reflection_temperature" in field_names

    def test_default_reflection_config_values(self):
        """Test default self-reflection config default values."""
        from obsidian_anki_sync.config import Config

        # Check default values from Field definitions
        enable_self_reflection_field = Config.model_fields["enable_self_reflection"]
        assert enable_self_reflection_field.default is False

        store_traces_field = Config.model_fields["store_reflection_traces"]
        assert store_traces_field.default is True

        max_revisions_field = Config.model_fields["max_revisions"]
        assert max_revisions_field.default == 2

    def test_reflection_model_in_agent_mappings(self):
        """Test reflection model is included in agent type mappings."""
        # The get_model_for_agent should handle "reflection" type
        # Test by checking the implementation includes "reflection"
        import inspect
        from obsidian_anki_sync.config import Config

        source = inspect.getsource(Config.get_model_for_agent)

        # Check that reflection is mapped to a task
        assert '"reflection"' in source or "'reflection'" in source
        # Check that reflection model is in the agent_model_map
        assert "reflection" in source
