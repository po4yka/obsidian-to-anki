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
    _is_simple_content,
    _should_skip_reflection,
    _store_reflection_trace,
    detect_domain,
    determine_revision_strategy,
    prioritize_issues,
    reflect_after_enrichment_node,
    reflect_after_generation_node,
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
        "current_reflection": None,
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
            recommended_card_changes=[
                {"card_index": 0, "changes": ["Improve wording"]}
            ],
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
        assert (
            _should_skip_reflection(mock_pipeline_state, "context_enrichment") is True
        )

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


# =============================================================================
# Test Domain Detection and Specialized Reflection
# =============================================================================


class TestDomainDetection:
    """Test domain detection functionality."""

    def test_domain_detection_from_topic_programming(self, mock_pipeline_state):
        """Test domain detection from topic metadata."""
        mock_pipeline_state["metadata_dict"]["topic"] = "Python Programming"
        domain = detect_domain(mock_pipeline_state)
        assert domain == "programming"

    def test_domain_detection_from_topic_medical(self):
        """Test domain detection from topic metadata for medical content."""
        # Use a completely fresh state dict
        state = {"metadata_dict": {"topic": "Clinical Pharmacology", "tags": []}}
        domain = detect_domain(state)
        assert domain == "medical"

    def test_domain_detection_from_topic_interview(self):
        """Test domain detection from topic metadata for interview content."""
        # Use a completely fresh state dict
        state = {"metadata_dict": {"topic": "System Design Interview", "tags": []}}
        domain = detect_domain(state)
        assert domain == "interview"

    def test_domain_detection_from_tags(self, mock_pipeline_state):
        """Test domain detection from tags."""
        # Create a fresh copy to avoid state pollution
        state = mock_pipeline_state.copy()
        state["metadata_dict"] = mock_pipeline_state["metadata_dict"].copy()
        state["metadata_dict"]["tags"] = ["javascript", "coding"]
        domain = detect_domain(state)
        assert domain == "programming"

    def test_domain_detection_fallback_to_general(self, mock_pipeline_state):
        """Test fallback to general domain for unrecognized content."""
        # Create a fresh copy to avoid state pollution
        state = mock_pipeline_state.copy()
        state["metadata_dict"] = mock_pipeline_state["metadata_dict"].copy()
        state["metadata_dict"]["topic"] = "Random Topic"
        state["metadata_dict"]["tags"] = ["misc"]
        domain = detect_domain(state)
        assert domain == "general"

    def test_domain_detection_cached_result(self, mock_pipeline_state):
        """Test that domain detection caches results."""
        mock_pipeline_state["detected_domain"] = "programming"
        # Different topic
        mock_pipeline_state["metadata_dict"]["topic"] = "Medical Content"
        domain = detect_domain(mock_pipeline_state)
        assert domain == "programming"  # Should return cached value


class TestDomainSpecificReflection:
    """Test domain-specific reflection criteria."""

    def test_domain_registry_loaded(self):
        """Test that domain registry is properly loaded."""
        from obsidian_anki_sync.agents.langgraph.reflection_domains import (
            DOMAIN_REGISTRY,
        )

        assert "programming" in DOMAIN_REGISTRY
        assert "medical" in DOMAIN_REGISTRY
        assert "interview" in DOMAIN_REGISTRY
        assert "general" in DOMAIN_REGISTRY

    def test_programming_domain_criteria(self):
        """Test programming domain has correct criteria."""
        from obsidian_anki_sync.agents.langgraph.reflection_domains import (
            get_domain_criteria,
        )

        criteria = get_domain_criteria("programming")
        assert criteria is not None
        assert "syntax" in criteria.quality_checks[0].lower()
        assert "python" in criteria.keywords
        assert criteria.revision_thresholds["critical"] == 0.9

    def test_medical_domain_criteria(self):
        """Test medical domain has correct criteria."""
        from obsidian_anki_sync.agents.langgraph.reflection_domains import (
            get_domain_criteria,
        )

        criteria = get_domain_criteria("medical")
        assert criteria is not None
        assert "facts" in criteria.quality_checks[0].lower()
        assert "medical" in criteria.keywords
        assert criteria.revision_thresholds["critical"] == 0.8


# =============================================================================
# Test Revision Strategy Selection
# =============================================================================


class TestRevisionStrategy:
    """Test revision strategy selection logic."""

    def test_light_edit_strategy_low_severity(self, mock_pipeline_state):
        """Test light edit strategy for low severity issues."""
        reflection = {
            "revision_suggestions": [{"severity_score": 0.2}, {"severity_score": 0.1}]
        }
        strategy = determine_revision_strategy(reflection, "general")
        assert strategy == "light_edit"

    def test_moderate_revision_strategy_medium_severity(self, mock_pipeline_state):
        """Test moderate revision strategy for medium severity issues."""
        reflection = {
            "revision_suggestions": [{"severity_score": 0.6}, {"severity_score": 0.7}]
        }
        strategy = determine_revision_strategy(reflection, "general")
        assert strategy == "moderate_revision"

    def test_major_rewrite_strategy_high_severity(self, mock_pipeline_state):
        """Test major rewrite strategy for high severity issues."""
        reflection = {
            "revision_suggestions": [{"severity_score": 0.8}, {"severity_score": 0.9}]
        }
        strategy = determine_revision_strategy(reflection, "general")
        assert strategy == "major_rewrite"

    def test_domain_specific_thresholds(self, mock_pipeline_state):
        """Test that domain-specific thresholds affect strategy selection."""
        reflection = {
            "revision_suggestions": [
                # Above medical medium threshold (0.3) but below high (0.6)
                {"severity_score": 0.5}
            ]
        }
        # Medical domain has lower thresholds, so 0.5 triggers major_rewrite
        strategy = determine_revision_strategy(reflection, "medical")
        assert strategy == "major_rewrite"


class TestIssuePrioritization:
    """Test issue prioritization based on domain weights."""

    def test_prioritize_issues_by_severity(self):
        """Test basic prioritization by severity score."""
        issues = [
            {"type": "minor", "severity_score": 0.3},
            {"type": "major", "severity_score": 0.8},
        ]
        prioritized = prioritize_issues(issues, "general", max_issues=2)
        assert len(prioritized) == 2
        # Should be sorted by severity (highest first)
        assert prioritized[0]["severity_score"] == 0.8

    def test_domain_weighted_prioritization(self):
        """Test prioritization with domain-specific weights."""
        issues = [
            # High weight in medical
            {"type": "factual_error", "severity_score": 0.5},
            # Not in medical weights (gets default weight 1.0)
            {"type": "poor_naming", "severity_score": 0.7},
        ]
        prioritized = prioritize_issues(issues, "medical", max_issues=2)
        # poor_naming gets higher weighted score due to default weight 1.0 vs 0.9
        assert prioritized[0]["type"] == "poor_naming"

    def test_max_issues_limit(self):
        """Test that prioritization respects max_issues limit."""
        issues = [{"type": "issue1", "severity_score": 0.5} for _ in range(10)]
        prioritized = prioritize_issues(issues, "general", max_issues=3)
        assert len(prioritized) == 3


# =============================================================================
# Test Smart Reflection Skipping
# =============================================================================


class TestSmartReflectionSkipping:
    """Test smart reflection skipping based on content complexity."""

    def test_skip_low_qa_count(self, mock_pipeline_state, mock_config):
        """Test skipping reflection for low Q/A count."""
        config = mock_config
        config.reflection_skip_qa_threshold = 2

        mock_pipeline_state["qa_pairs_dicts"] = [
            {"question": "Q1", "answer": "A1"}
        ]  # Only 1 pair
        mock_pipeline_state["note_content"] = (
            "Long content that exceeds length threshold"
        )
        mock_pipeline_state["pre_validation"] = {"confidence": 0.5}
        mock_pipeline_state["post_validation"] = {"confidence": 0.5}

        should_skip = _is_simple_content(mock_pipeline_state, config)
        assert should_skip is True

    def test_skip_short_content(self, mock_pipeline_state, mock_config):
        """Test skipping reflection for short content."""
        config = mock_config
        config.reflection_skip_content_length = 500

        mock_pipeline_state["qa_pairs_dicts"] = [
            {"question": "Q1"},
            {"question": "Q2"},
            {"question": "Q3"},
        ]  # 3 pairs
        mock_pipeline_state["note_content"] = "Short content"  # < 500 chars
        mock_pipeline_state["pre_validation"] = {"confidence": 0.5}
        mock_pipeline_state["post_validation"] = {"confidence": 0.5}

        should_skip = _is_simple_content(mock_pipeline_state, config)
        assert should_skip is True

    def test_skip_high_confidence_validation(self, mock_pipeline_state, mock_config):
        """Test skipping reflection for high validation confidence."""
        config = mock_config
        config.reflection_skip_confidence_threshold = 0.9

        mock_pipeline_state["qa_pairs_dicts"] = [
            {"question": "Q1"},
            {"question": "Q2"},
            {"question": "Q3"},
        ]  # 3 pairs
        # > 500 chars
        mock_pipeline_state["note_content"] = "Long content that exceeds threshold" * 20
        mock_pipeline_state["pre_validation"] = {"confidence": 0.95}  # High confidence
        mock_pipeline_state["post_validation"] = {"confidence": 0.92}  # High confidence

        should_skip = _is_simple_content(mock_pipeline_state, config)
        assert should_skip is True

    def test_no_skip_complex_content(self, mock_pipeline_state, mock_config):
        """Test not skipping reflection for complex content."""
        config = mock_config
        config.reflection_skip_qa_threshold = 2
        config.reflection_skip_content_length = 500
        config.reflection_skip_confidence_threshold = 0.9

        mock_pipeline_state["qa_pairs_dicts"] = [
            {"question": "Q1"},
            {"question": "Q2"},
            {"question": "Q3"},
        ]  # 3 pairs
        # > 500 chars
        mock_pipeline_state["note_content"] = "Long content that exceeds threshold" * 20
        mock_pipeline_state["pre_validation"] = {"confidence": 0.7}  # Medium confidence
        mock_pipeline_state["post_validation"] = {
            "confidence": 0.8
        }  # Medium confidence

        should_skip = _is_simple_content(mock_pipeline_state, config)
        assert should_skip is False

    def test_skip_reflection_marks_state(self, mock_pipeline_state, mock_config):
        """Test that skipping reflection updates state appropriately."""
        config = mock_config
        config.reflection_skip_qa_threshold = 2
        config.enable_self_reflection = True
        config.reflection_enabled_stages = ["generation"]

        mock_pipeline_state["config"] = config
        mock_pipeline_state["qa_pairs_dicts"] = [{"question": "Q1"}]  # Low count
        mock_pipeline_state["note_content"] = "Short"

        should_skip = _should_skip_reflection(mock_pipeline_state, "generation")
        assert should_skip is True
        assert mock_pipeline_state["reflection_skipped"] is True
        assert mock_pipeline_state["reflection_skip_reason"] == "simple_content"
