"""Tests for Chain of Thought (CoT) reasoning integration.

This module tests:
- Reasoning models (Pydantic validation)
- Reasoning nodes (skip logic, error handling, trace storage)
- Workflow builder CoT routing
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from obsidian_anki_sync.agents.langgraph.reasoning_models import (
    CardSplittingReasoningOutput,
    DuplicateReasoningOutput,
    EnrichmentReasoningOutput,
    GenerationReasoningOutput,
    MemorizationReasoningOutput,
    PostValidationReasoningOutput,
    PreValidationReasoningOutput,
    ReasoningTrace,
    ReasoningTraceOutput,
)
from obsidian_anki_sync.agents.langgraph.reasoning_nodes import (
    _should_skip_reasoning,
    _store_reasoning_trace,
    think_before_generation_node,
    think_before_post_validation_node,
    think_before_pre_validation_node,
)
from obsidian_anki_sync.agents.langgraph.workflow_builder import (
    WorkflowBuilder,
    should_continue_after_enrichment,
    should_continue_after_memorization_quality,
    should_continue_after_post_validation,
    should_continue_after_pre_validation,
)
from obsidian_anki_sync.config import Config

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
def base_reasoning_output():
    """Create a base reasoning output for testing."""
    return ReasoningTraceOutput(
        reasoning="This is the full chain of thought reasoning.",
        key_observations=["Observation 1", "Observation 2"],
        planned_approach="The recommended approach is X.",
        potential_issues=["Issue 1", "Issue 2"],
        recommendations=["Recommendation 1", "Recommendation 2"],
        confidence=0.85,
    )


@pytest.fixture
def mock_pipeline_state():
    """Create a minimal pipeline state for testing."""
    from datetime import datetime

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
        "enable_cot_reasoning": True,
        "store_reasoning_traces": True,
        "log_reasoning_traces": False,
        "cot_enabled_stages": ["pre_validation", "generation", "post_validation"],
        "reasoning_model": "openai:gpt-4",
        "reasoning_traces": {},
        "current_reasoning": None,
        "step_counts": {},
        "stage_times": {},
        "messages": [],
        "max_steps": 100,
    }


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = MagicMock(spec=Config)
    config.get_model_for_agent.return_value = "gpt-4o"
    config.get_model_config_for_task.return_value = {"max_tokens": 100}
    config.langgraph_max_retries = 3
    config.langgraph_auto_fix = True
    config.langgraph_strict_mode = True
    config.enable_card_splitting = True
    config.enable_context_enrichment = True
    config.enable_memorization_quality = True
    config.enable_duplicate_detection = False
    config.enable_cot_reasoning = False  # Default disabled
    config.enable_note_correction = False
    return config


# =============================================================================
# Test Reasoning Models
# =============================================================================


class TestReasoningTraceOutput:
    """Tests for base ReasoningTraceOutput model."""

    def test_create_with_required_fields(self):
        """Test creating output with only required fields."""
        output = ReasoningTraceOutput(
            reasoning="Test reasoning",
            planned_approach="Test approach",
        )
        assert output.reasoning == "Test reasoning"
        assert output.planned_approach == "Test approach"
        assert output.key_observations == []
        assert output.potential_issues == []
        assert output.recommendations == []
        assert output.confidence == 0.5  # Default

    def test_create_with_all_fields(self, base_reasoning_output):
        """Test creating output with all fields."""
        assert (
            base_reasoning_output.reasoning
            == "This is the full chain of thought reasoning."
        )
        assert len(base_reasoning_output.key_observations) == 2
        assert base_reasoning_output.confidence == 0.85

    def test_confidence_bounds(self):
        """Test confidence field validation."""
        # Valid bounds
        ReasoningTraceOutput(reasoning="test", planned_approach="test", confidence=0.0)
        ReasoningTraceOutput(reasoning="test", planned_approach="test", confidence=1.0)

        # Invalid bounds should raise
        with pytest.raises(ValueError):
            ReasoningTraceOutput(
                reasoning="test", planned_approach="test", confidence=-0.1
            )

        with pytest.raises(ValueError):
            ReasoningTraceOutput(
                reasoning="test", planned_approach="test", confidence=1.1
            )


class TestPreValidationReasoningOutput:
    """Tests for PreValidationReasoningOutput model."""

    def test_create_with_stage_specific_fields(self):
        """Test creating with pre-validation specific fields."""
        output = PreValidationReasoningOutput(
            reasoning="Test reasoning",
            planned_approach="Test approach",
            structure_assessment="Good structure",
            frontmatter_assessment="Complete frontmatter",
            content_quality_assessment="High quality",
            validation_focus=["tags", "formatting"],
        )
        assert output.structure_assessment == "Good structure"
        assert output.frontmatter_assessment == "Complete frontmatter"
        assert len(output.validation_focus) == 2


class TestGenerationReasoningOutput:
    """Tests for GenerationReasoningOutput model."""

    def test_create_with_stage_specific_fields(self):
        """Test creating with generation specific fields."""
        output = GenerationReasoningOutput(
            reasoning="Test reasoning",
            planned_approach="Test approach",
            card_type_recommendation="Cloze",
            complexity_assessment="Medium complexity",
            qa_pair_analysis=[{"pair": 1, "type": "factual"}],
            formatting_recommendations=["Use bullet points"],
        )
        assert output.card_type_recommendation == "Cloze"
        assert output.complexity_assessment == "Medium complexity"
        assert len(output.qa_pair_analysis) == 1


class TestPostValidationReasoningOutput:
    """Tests for PostValidationReasoningOutput model."""

    def test_create_with_stage_specific_fields(self):
        """Test creating with post-validation specific fields."""
        output = PostValidationReasoningOutput(
            reasoning="Test reasoning",
            planned_approach="Test approach",
            quality_concerns=["Missing examples"],
            validation_strategy="Strict validation",
            expected_issues=["HTML formatting"],
        )
        assert len(output.quality_concerns) == 1
        assert output.validation_strategy == "Strict validation"


class TestReasoningTrace:
    """Tests for ReasoningTrace model and factory method."""

    def test_from_output_factory(self, base_reasoning_output):
        """Test creating ReasoningTrace from output."""
        trace = ReasoningTrace.from_output(
            stage="pre_validation",
            output=base_reasoning_output,
            reasoning_time=1.5,
            timestamp=time.time(),
            stage_specific_data={"extra": "data"},
        )
        assert trace.stage == "pre_validation"
        assert trace.reasoning == base_reasoning_output.reasoning
        assert trace.confidence == 0.85
        assert trace.reasoning_time == 1.5
        assert trace.stage_specific_data == {"extra": "data"}

    def test_model_dump(self, base_reasoning_output):
        """Test that trace can be serialized."""
        trace = ReasoningTrace.from_output(
            stage="generation",
            output=base_reasoning_output,
            reasoning_time=0.5,
            timestamp=1234567890.0,
        )
        dumped = trace.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["stage"] == "generation"
        assert dumped["timestamp"] == 1234567890.0


# =============================================================================
# Test Reasoning Node Helpers
# =============================================================================


class TestShouldSkipReasoning:
    """Tests for _should_skip_reasoning helper."""

    def test_skip_when_cot_disabled(self, mock_pipeline_state):
        """Test skipping when CoT is disabled globally."""
        mock_pipeline_state["enable_cot_reasoning"] = False
        assert _should_skip_reasoning(mock_pipeline_state, "pre_validation") is True

    def test_skip_when_stage_not_enabled(self, mock_pipeline_state):
        """Test skipping when stage is not in enabled list."""
        mock_pipeline_state["cot_enabled_stages"] = [
            "generation"
        ]  # pre_validation not included
        assert _should_skip_reasoning(mock_pipeline_state, "pre_validation") is True

    def test_no_skip_when_enabled(self, mock_pipeline_state):
        """Test not skipping when CoT and stage are enabled."""
        assert _should_skip_reasoning(mock_pipeline_state, "pre_validation") is False
        assert _should_skip_reasoning(mock_pipeline_state, "generation") is False

    def test_no_skip_when_empty_stages_list(self, mock_pipeline_state):
        """Test that empty stages list means all stages enabled."""
        mock_pipeline_state["cot_enabled_stages"] = []
        # With empty list, stage check passes (all enabled)
        assert _should_skip_reasoning(mock_pipeline_state, "any_stage") is False


class TestStoreReasoningTrace:
    """Tests for _store_reasoning_trace helper."""

    def test_store_trace_when_enabled(self, mock_pipeline_state, base_reasoning_output):
        """Test trace is stored when store_reasoning_traces is True."""
        _store_reasoning_trace(
            mock_pipeline_state,
            "pre_validation",
            base_reasoning_output,
            1.5,
            {"extra": "data"},
        )

        assert "pre_validation" in mock_pipeline_state["reasoning_traces"]
        assert mock_pipeline_state["current_reasoning"] is not None
        trace = mock_pipeline_state["reasoning_traces"]["pre_validation"]
        assert trace["reasoning"] == base_reasoning_output.reasoning
        assert trace["stage_specific_data"] == {"extra": "data"}

    def test_no_store_when_disabled(self, mock_pipeline_state, base_reasoning_output):
        """Test trace is not stored when store_reasoning_traces is False."""
        mock_pipeline_state["store_reasoning_traces"] = False
        _store_reasoning_trace(
            mock_pipeline_state,
            "pre_validation",
            base_reasoning_output,
            1.5,
        )
        assert mock_pipeline_state["reasoning_traces"] == {}
        assert mock_pipeline_state["current_reasoning"] is None

    def test_initializes_reasoning_traces_if_missing(self, base_reasoning_output):
        """Test that reasoning_traces dict is initialized if missing."""
        state = {
            "store_reasoning_traces": True,
            "log_reasoning_traces": False,
        }
        _store_reasoning_trace(state, "test", base_reasoning_output, 1.0)
        assert "reasoning_traces" in state
        assert "test" in state["reasoning_traces"]


# =============================================================================
# Test Reasoning Nodes
# =============================================================================


class TestThinkBeforePreValidationNode:
    """Tests for think_before_pre_validation_node."""

    async def test_skips_when_cot_disabled(self, mock_pipeline_state):
        """Test node returns unchanged state when CoT disabled."""
        mock_pipeline_state["enable_cot_reasoning"] = False
        result = await think_before_pre_validation_node(mock_pipeline_state)
        assert result["current_reasoning"] is None

    async def test_skips_when_model_not_available(self, mock_pipeline_state):
        """Test node returns unchanged state when model is None."""
        mock_pipeline_state["reasoning_model"] = None
        result = await think_before_pre_validation_node(mock_pipeline_state)
        assert result["current_reasoning"] is None

    async def test_handles_exception_gracefully(self, mock_pipeline_state):
        """Test node logs warning and continues on exception."""
        # Mock agent to raise exception
        with patch(
            "obsidian_anki_sync.agents.langgraph.reasoning_nodes.Agent"
        ) as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=Exception("Test error"))
            mock_agent_class.return_value = mock_agent

            result = await think_before_pre_validation_node(mock_pipeline_state)

            # Should continue without raising
            assert result["current_reasoning"] is None

    async def test_stores_reasoning_on_success(self, mock_pipeline_state):
        """Test node stores reasoning trace on success."""
        mock_output = PreValidationReasoningOutput(
            reasoning="Test reasoning",
            planned_approach="Test approach",
            structure_assessment="Good",
            confidence=0.9,
        )

        mock_result = MagicMock()
        mock_result.data = mock_output
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "obsidian_anki_sync.agents.langgraph.reasoning_nodes.Agent",
                return_value=mock_agent,
            ),
            patch(
                "obsidian_anki_sync.agents.langgraph.reasoning_nodes.NoteMetadata",
            ) as mock_metadata_cls,
            patch(
                "obsidian_anki_sync.agents.langgraph.reasoning_nodes.QAPair",
            ) as mock_qa_cls,
        ):
            # Setup mock metadata
            mock_metadata = MagicMock()
            mock_metadata.title = "Test Note"
            mock_metadata.topic = "Programming"
            mock_metadata.tags = ["python"]
            mock_metadata.language_tags = ["en"]
            mock_metadata_cls.return_value = mock_metadata

            # Setup mock QAPair
            mock_qa = MagicMock()
            mock_qa.question = "What is Python?"
            mock_qa.answer = "A programming language"
            mock_qa_cls.return_value = mock_qa

            result = await think_before_pre_validation_node(mock_pipeline_state)

            assert result["current_reasoning"] is not None
            assert "pre_validation" in result["reasoning_traces"]
            assert "think_before_pre_validation" in result["stage_times"]


class TestThinkBeforeGenerationNode:
    """Tests for think_before_generation_node."""

    async def test_skips_when_stage_not_enabled(self, mock_pipeline_state):
        """Test node skips when generation not in enabled stages."""
        mock_pipeline_state["cot_enabled_stages"] = [
            "pre_validation"
        ]  # generation not included
        result = await think_before_generation_node(mock_pipeline_state)
        assert result["current_reasoning"] is None


class TestThinkBeforePostValidationNode:
    """Tests for think_before_post_validation_node."""

    async def test_skips_when_no_generation_data(self, mock_pipeline_state):
        """Test node returns early when generation data is missing."""
        with patch(
            "obsidian_anki_sync.agents.langgraph.reasoning_nodes.Agent"
        ) as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            result = await think_before_post_validation_node(mock_pipeline_state)
            # Should skip since no generation data
            assert result.get("generation") is None


# =============================================================================
# Test Workflow Builder CoT Integration
# =============================================================================


class TestWorkflowBuilderCoTRouting:
    """Tests for WorkflowBuilder CoT-specific routing."""

    def test_build_workflow_without_cot(self, mock_config):
        """Test workflow builds without CoT nodes when disabled."""
        mock_config.enable_cot_reasoning = False
        builder = WorkflowBuilder(mock_config)
        workflow = builder.build_workflow()

        # Workflow should compile without CoT nodes
        app = workflow.compile()
        assert app is not None

    def test_build_workflow_with_cot(self, mock_config):
        """Test workflow builds with CoT nodes when enabled."""
        mock_config.enable_cot_reasoning = True
        builder = WorkflowBuilder(mock_config)
        workflow = builder.build_workflow()

        # Workflow should compile with CoT nodes
        app = workflow.compile()
        assert app is not None

    def test_route_after_pre_validation_with_cot_valid(self, mock_config):
        """Test CoT routing after valid pre-validation."""
        mock_config.enable_cot_reasoning = True
        builder = WorkflowBuilder(mock_config)

        state = {
            "pre_validation": {"is_valid": True},
            "enable_card_splitting": True,
        }
        result = builder._route_after_pre_validation_with_cot(state)
        assert result == "think_card_splitting"

    def test_route_after_pre_validation_with_cot_no_splitting(self, mock_config):
        """Test CoT routing skips card splitting when disabled."""
        mock_config.enable_cot_reasoning = True
        builder = WorkflowBuilder(mock_config)

        state = {
            "pre_validation": {"is_valid": True},
            "enable_card_splitting": False,
        }
        result = builder._route_after_pre_validation_with_cot(state)
        assert result == "think_generation"

    def test_route_after_pre_validation_with_cot_invalid(self, mock_config):
        """Test CoT routing to failed on invalid pre-validation."""
        mock_config.enable_cot_reasoning = True
        builder = WorkflowBuilder(mock_config)

        state = {"pre_validation": {"is_valid": False}}
        result = builder._route_after_pre_validation_with_cot(state)
        assert result == "failed"

    def test_route_after_post_validation_with_cot_enrichment(self, mock_config):
        """Test CoT routing to enrichment after post-validation."""
        mock_config.enable_cot_reasoning = True
        builder = WorkflowBuilder(mock_config)

        state = {"current_stage": "context_enrichment"}
        result = builder._route_after_post_validation_with_cot(state)
        assert result == "think_enrichment"

    def test_route_after_post_validation_with_cot_retry(self, mock_config):
        """Test CoT routing to generation retry."""
        mock_config.enable_cot_reasoning = True
        builder = WorkflowBuilder(mock_config)

        state = {"current_stage": "generation"}
        result = builder._route_after_post_validation_with_cot(state)
        assert result == "think_generation"

    def test_route_after_enrichment_with_cot_memorization(self, mock_config):
        """Test CoT routing to memorization after enrichment."""
        mock_config.enable_cot_reasoning = True
        builder = WorkflowBuilder(mock_config)

        state = {"current_stage": "memorization_quality"}
        result = builder._route_after_enrichment_with_cot(state)
        assert result == "think_memorization"

    def test_route_after_memorization_with_cot_duplicate(self, mock_config):
        """Test CoT routing to duplicate detection after memorization."""
        mock_config.enable_cot_reasoning = True
        builder = WorkflowBuilder(mock_config)

        state = {"current_stage": "duplicate_detection"}
        result = builder._route_after_memorization_with_cot(state)
        assert result == "think_duplicate"


# =============================================================================
# Test Original Routing Functions (non-CoT)
# =============================================================================


class TestOriginalRoutingFunctions:
    """Tests for original (non-CoT) routing functions."""

    def test_should_continue_after_pre_validation_valid_with_splitting(self):
        """Test routing after valid pre-validation with splitting."""
        state = {
            "pre_validation": {"is_valid": True},
            "enable_card_splitting": True,
        }
        result = should_continue_after_pre_validation(state)
        assert result == "card_splitting"

    def test_should_continue_after_pre_validation_valid_no_splitting(self):
        """Test routing after valid pre-validation without splitting."""
        state = {
            "pre_validation": {"is_valid": True},
            "enable_card_splitting": False,
        }
        result = should_continue_after_pre_validation(state)
        assert result == "generation"

    def test_should_continue_after_pre_validation_invalid(self):
        """Test routing after invalid pre-validation."""
        state = {"pre_validation": {"is_valid": False}}
        result = should_continue_after_pre_validation(state)
        assert result == "failed"

    def test_should_continue_after_post_validation_enrichment(self):
        """Test routing to enrichment after post-validation."""
        state = {"current_stage": "context_enrichment"}
        result = should_continue_after_post_validation(state)
        assert result == "context_enrichment"

    def test_should_continue_after_post_validation_retry(self):
        """Test routing to generation retry."""
        state = {"current_stage": "generation"}
        result = should_continue_after_post_validation(state)
        assert result == "generation"

    def test_should_continue_after_enrichment_memorization(self):
        """Test routing to memorization after enrichment."""
        state = {"current_stage": "memorization_quality"}
        result = should_continue_after_enrichment(state)
        assert result == "memorization_quality"

    def test_should_continue_after_enrichment_complete(self):
        """Test routing to complete after enrichment."""
        state = {"current_stage": "complete"}
        result = should_continue_after_enrichment(state)
        assert result == "complete"

    def test_should_continue_after_memorization_quality_duplicate(self):
        """Test routing to duplicate after memorization."""
        state = {"current_stage": "duplicate_detection"}
        result = should_continue_after_memorization_quality(state)
        assert result == "duplicate_detection"

    def test_should_continue_after_memorization_quality_complete(self):
        """Test routing to complete after memorization."""
        state = {"current_stage": "complete"}
        result = should_continue_after_memorization_quality(state)
        assert result == "complete"


# =============================================================================
# Test All Reasoning Output Subclasses
# =============================================================================


class TestAllReasoningOutputSubclasses:
    """Test all reasoning output subclass models."""

    def test_card_splitting_reasoning_output(self):
        """Test CardSplittingReasoningOutput creation."""
        output = CardSplittingReasoningOutput(
            reasoning="Analyzing for splitting",
            planned_approach="Check complexity",
            complexity_indicators=["Long answer"],
            split_recommendation="Split into 2 cards",
            concept_boundaries=["Part 1", "Part 2"],
        )
        assert len(output.complexity_indicators) == 1
        assert output.split_recommendation == "Split into 2 cards"

    def test_enrichment_reasoning_output(self):
        """Test EnrichmentReasoningOutput creation."""
        output = EnrichmentReasoningOutput(
            reasoning="Analyzing for enrichment",
            planned_approach="Add examples",
            enrichment_opportunities=["Add code example"],
            mnemonic_suggestions=["Acronym"],
            example_types=["Code", "Diagram"],
        )
        assert len(output.enrichment_opportunities) == 1
        assert len(output.example_types) == 2

    def test_memorization_reasoning_output(self):
        """Test MemorizationReasoningOutput creation."""
        output = MemorizationReasoningOutput(
            reasoning="Analyzing memorization",
            planned_approach="Check retention",
            retention_factors=["Spaced repetition"],
            cognitive_load_assessment="Medium load",
        )
        assert output.cognitive_load_assessment == "Medium load"

    def test_duplicate_reasoning_output(self):
        """Test DuplicateReasoningOutput creation."""
        output = DuplicateReasoningOutput(
            reasoning="Checking duplicates",
            planned_approach="Compare with existing",
            similarity_indicators=["Similar topic"],
            comparison_strategy="Semantic similarity",
        )
        assert output.comparison_strategy == "Semantic similarity"
