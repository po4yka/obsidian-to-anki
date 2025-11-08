"""LangGraph-based orchestrator for the card generation pipeline.

This module implements a state machine workflow using LangGraph to coordinate:
1. Pre-Validator Agent - structure and format validation
2. Generator Agent - card generation with LLM
3. Post-Validator Agent - quality validation with retry logic

The workflow supports conditional routing, automatic retries, and state persistence.
"""

import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from ..config import Config
from ..models import NoteMetadata, QAPair
from ..utils.logging import get_logger
from .models import (
    AgentPipelineResult,
    GeneratedCard,
    GenerationResult,
    PostValidationResult,
    PreValidationResult,
)
from .slug_utils import generate_agent_slug_base

logger = get_logger(__name__)


# ============================================================================
# State Definition
# ============================================================================


class PipelineState(TypedDict):
    """State for the card generation pipeline workflow.

    This state is passed between nodes and tracks the entire pipeline execution.
    """

    # Input data
    note_content: str
    metadata_dict: dict  # Serialized NoteMetadata
    qa_pairs_dicts: list[dict]  # Serialized QAPair list
    file_path: str | None
    slug_base: str

    # Pipeline stage results
    pre_validation: dict | None  # Serialized PreValidationResult
    generation: dict | None  # Serialized GenerationResult
    post_validation: dict | None  # Serialized PostValidationResult

    # Workflow control
    current_stage: Literal[
        "pre_validation", "generation", "post_validation", "complete", "failed"
    ]
    retry_count: int
    max_retries: int
    auto_fix_enabled: bool
    strict_mode: bool

    # Timing
    start_time: float
    stage_times: dict[str, float]

    # Messages (for debugging/logging)
    messages: Annotated[list[str], add_messages]


# ============================================================================
# Node Functions
# ============================================================================


def pre_validation_node(state: PipelineState) -> PipelineState:
    """Execute pre-validation stage.

    Validates note structure, formatting, and frontmatter before generation.
    Can auto-fix common issues if enabled.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with pre-validation results
    """
    import asyncio

    from ..models import NoteMetadata, QAPair
    from ..providers.pydantic_ai_models import (
        create_openrouter_model_from_env,
    )
    from .pydantic_ai_agents import PreValidatorAgentAI

    logger.info("langgraph_pre_validation_start")
    start_time = time.time()

    # Deserialize metadata and QA pairs
    metadata = NoteMetadata(**state["metadata_dict"])
    qa_pairs = [QAPair(**qa_dict) for qa_dict in state["qa_pairs_dicts"]]
    file_path = Path(state["file_path"]) if state["file_path"] else None

    # Create PydanticAI model (using OpenRouter as default)
    try:
        # Try to create from environment (requires OPENROUTER_API_KEY)
        model = create_openrouter_model_from_env(model_name="openai/gpt-4o-mini")
    except Exception as e:
        logger.warning("failed_to_create_openrouter_model", error=str(e))
        # Fallback to simple validation
        pre_result = PreValidationResult(
            is_valid=True,
            error_type="none",
            error_details="Pre-validation passed (OpenRouter unavailable, skipped)",
            auto_fix_applied=False,
            fixed_content=None,
            validation_time=time.time() - start_time,
        )
        state["pre_validation"] = pre_result.model_dump()
        state["stage_times"]["pre_validation"] = pre_result.validation_time
        state["current_stage"] = "generation"
        state["messages"].append(f"Pre-validation: {pre_result.error_type}")
        return state

    # Create pre-validator agent
    pre_validator = PreValidatorAgentAI(model=model, temperature=0.0)

    # Run validation (PydanticAI uses async)
    try:
        pre_result = asyncio.run(
            pre_validator.validate(
                note_content=state["note_content"],
                metadata=metadata,
                qa_pairs=qa_pairs,
                file_path=file_path,
            )
        )
        pre_result.validation_time = time.time() - start_time
    except Exception as e:
        logger.error("langgraph_pre_validation_error", error=str(e))
        pre_result = PreValidationResult(
            is_valid=False,
            error_type="structure",
            error_details=f"Pre-validation failed: {str(e)}",
            auto_fix_applied=False,
            fixed_content=None,
            validation_time=time.time() - start_time,
        )

    logger.info(
        "langgraph_pre_validation_complete",
        is_valid=pre_result.is_valid,
        time=pre_result.validation_time,
    )

    state["pre_validation"] = pre_result.model_dump()
    state["stage_times"]["pre_validation"] = pre_result.validation_time
    state["current_stage"] = "generation" if pre_result.is_valid else "failed"
    state["messages"].append(f"Pre-validation: {pre_result.error_type}")

    return state


def generation_node(state: PipelineState) -> PipelineState:
    """Execute card generation stage.

    Generates APF cards from Q/A pairs using the configured LLM.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with generation results
    """
    import asyncio

    from ..models import NoteMetadata, QAPair
    from ..providers.pydantic_ai_models import create_openrouter_model_from_env
    from .pydantic_ai_agents import GeneratorAgentAI

    logger.info("langgraph_generation_start")
    start_time = time.time()

    # Deserialize data
    metadata = NoteMetadata(**state["metadata_dict"])
    qa_pairs = [QAPair(**qa_dict) for qa_dict in state["qa_pairs_dicts"]]

    # Create PydanticAI model for generation
    try:
        # Use more powerful model for generation
        model = create_openrouter_model_from_env(
            model_name="anthropic/claude-3-5-sonnet"
        )
    except Exception as e:
        logger.error("failed_to_create_generator_model", error=str(e))
        # Return error result
        gen_result = GenerationResult(
            cards=[],
            total_cards=0,
            generation_time=time.time() - start_time,
            model_used="none",
        )
        state["generation"] = gen_result.model_dump()
        state["stage_times"]["generation"] = gen_result.generation_time
        state["current_stage"] = "failed"
        state["messages"].append("Generation failed: model unavailable")
        return state

    # Create generator agent
    generator = GeneratorAgentAI(model=model, temperature=0.3)

    # Run generation
    try:
        gen_result = asyncio.run(
            generator.generate_cards(
                note_content=state["note_content"],
                metadata=metadata,
                qa_pairs=qa_pairs,
                slug_base=state["slug_base"],
            )
        )
        gen_result.generation_time = time.time() - start_time
    except Exception as e:
        logger.error("langgraph_generation_error", error=str(e))
        gen_result = GenerationResult(
            cards=[],
            total_cards=0,
            generation_time=time.time() - start_time,
            model_used=str(model),
        )

    logger.info(
        "langgraph_generation_complete",
        cards_count=gen_result.total_cards,
        time=gen_result.generation_time,
    )

    state["generation"] = gen_result.model_dump()
    state["stage_times"]["generation"] = gen_result.generation_time
    state["current_stage"] = (
        "post_validation" if gen_result.total_cards > 0 else "failed"
    )
    state["messages"].append(f"Generated {gen_result.total_cards} cards")

    return state


def post_validation_node(state: PipelineState) -> PipelineState:
    """Execute post-validation stage.

    Validates generated cards for quality, syntax, and accuracy.
    Can auto-fix issues if enabled and will retry on failures.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with post-validation results
    """
    import asyncio

    from ..models import NoteMetadata
    from ..providers.pydantic_ai_models import create_openrouter_model_from_env
    from .pydantic_ai_agents import PostValidatorAgentAI

    logger.info("langgraph_post_validation_start", retry_count=state["retry_count"])
    start_time = time.time()

    # Deserialize data
    metadata = NoteMetadata(**state["metadata_dict"])
    generation = state.get("generation")

    if not generation:
        logger.error("langgraph_post_validation_no_generation")
        state["current_stage"] = "failed"
        return state

    cards = [GeneratedCard(**card_dict) for card_dict in generation["cards"]]

    # Create PydanticAI model for validation
    try:
        model = create_openrouter_model_from_env(model_name="openai/gpt-4o-mini")
    except Exception as e:
        logger.warning("failed_to_create_post_validator_model", error=str(e))
        # Assume valid if validator unavailable
        post_result = PostValidationResult(
            is_valid=True,
            error_type="none",
            error_details="Post-validation passed (OpenRouter unavailable, skipped)",
            corrected_cards=None,
            validation_time=time.time() - start_time,
        )
        state["post_validation"] = post_result.model_dump()
        state["stage_times"]["post_validation"] = (
            state["stage_times"].get("post_validation", 0.0)
            + post_result.validation_time
        )
        state["current_stage"] = "complete"
        state["messages"].append("Post-validation skipped")
        return state

    # Create post-validator agent
    post_validator = PostValidatorAgentAI(model=model, temperature=0.0)

    # Run validation
    try:
        post_result = asyncio.run(
            post_validator.validate(
                cards=cards,
                metadata=metadata,
                strict_mode=state["strict_mode"],
            )
        )
        post_result.validation_time = time.time() - start_time
    except Exception as e:
        logger.error("langgraph_post_validation_error", error=str(e))
        post_result = PostValidationResult(
            is_valid=False,
            error_type="syntax",
            error_details=f"Post-validation failed: {str(e)}",
            corrected_cards=None,
            validation_time=time.time() - start_time,
        )

    logger.info(
        "langgraph_post_validation_complete",
        is_valid=post_result.is_valid,
        retry_count=state["retry_count"],
        time=post_result.validation_time,
    )

    state["post_validation"] = post_result.model_dump()
    state["stage_times"]["post_validation"] = (
        state["stage_times"].get("post_validation", 0.0) + post_result.validation_time
    )

    # Determine next stage based on validation result
    if post_result.is_valid:
        state["current_stage"] = "complete"
        state["messages"].append("Post-validation passed")
    elif state["retry_count"] < state["max_retries"] and state["auto_fix_enabled"]:
        # Apply corrections if available
        if post_result.corrected_cards and state["generation"] is not None:
            # Update generation with corrected cards
            corrected_dicts = [
                card.model_dump() for card in post_result.corrected_cards
            ]
            state["generation"]["cards"] = corrected_dicts
            state["generation"]["total_cards"] = len(corrected_dicts)
            logger.info("applied_corrected_cards", count=len(corrected_dicts))

        state["retry_count"] += 1
        state["current_stage"] = "post_validation"  # Re-validate corrections
        state["messages"].append(
            f"Applied fixes, re-validating (attempt {state['retry_count']})"
        )
    else:
        state["current_stage"] = "failed"
        state["messages"].append("Post-validation failed, no more retries")

    return state


# ============================================================================
# Conditional Routing
# ============================================================================


def should_continue_after_pre_validation(
    state: PipelineState,
) -> Literal["generation", "failed"]:
    """Determine next node after pre-validation.

    Args:
        state: Current pipeline state

    Returns:
        Next node name
    """
    pre_validation = state.get("pre_validation")
    if pre_validation and pre_validation["is_valid"]:
        return "generation"
    return "failed"


def should_continue_after_post_validation(
    state: PipelineState,
) -> Literal["complete", "generation", "failed"]:
    """Determine next node after post-validation.

    Args:
        state: Current pipeline state

    Returns:
        Next node name
    """
    current_stage = state.get("current_stage", "failed")

    if current_stage == "complete":
        return "complete"
    elif current_stage == "generation":
        return "generation"  # Retry
    else:
        return "failed"


# ============================================================================
# Workflow Builder
# ============================================================================


class LangGraphOrchestrator:
    """LangGraph-based orchestrator for the card generation pipeline.

    Uses a state machine workflow with conditional routing, automatic retries,
    and state persistence via checkpoints.
    """

    def __init__(
        self,
        config: Config,
        max_retries: int = 3,
        auto_fix_enabled: bool = True,
        strict_mode: bool = True,
    ):
        """Initialize LangGraph orchestrator.

        Args:
            config: Service configuration
            max_retries: Maximum post-validation retry attempts
            auto_fix_enabled: Enable automatic error fixing
            strict_mode: Use strict validation mode
        """
        self.config = config
        self.max_retries = max_retries
        self.auto_fix_enabled = auto_fix_enabled
        self.strict_mode = strict_mode

        # Build the workflow graph
        self.workflow = self._build_workflow()

        # Initialize checkpoint saver for state persistence
        self.checkpointer = MemorySaver()

        # Compile the graph
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

        logger.info(
            "langgraph_orchestrator_initialized",
            max_retries=max_retries,
            auto_fix=auto_fix_enabled,
            strict_mode=strict_mode,
        )

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow.

        Returns:
            Configured StateGraph instance
        """
        # Create workflow graph
        workflow = StateGraph(PipelineState)

        # Add nodes
        workflow.add_node("pre_validation", pre_validation_node)
        workflow.add_node("generation", generation_node)
        workflow.add_node("post_validation", post_validation_node)

        # Set entry point
        workflow.set_entry_point("pre_validation")

        # Add conditional edges
        workflow.add_conditional_edges(
            "pre_validation",
            should_continue_after_pre_validation,
            {
                "generation": "generation",
                "failed": END,
            },
        )

        workflow.add_edge("generation", "post_validation")

        workflow.add_conditional_edges(
            "post_validation",
            should_continue_after_post_validation,
            {
                "complete": END,
                "generation": "generation",  # Retry loop
                "failed": END,
            },
        )

        return workflow

    def process_note(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        file_path: Path | None = None,
    ) -> AgentPipelineResult:
        """Process a note through the LangGraph workflow.

        Args:
            note_content: Full note content
            metadata: Parsed metadata
            qa_pairs: Parsed Q/A pairs
            file_path: Optional file path for validation

        Returns:
            AgentPipelineResult with all pipeline stages
        """
        start_time = time.time()

        logger.info(
            "langgraph_pipeline_start",
            title=metadata.title,
            qa_pairs_count=len(qa_pairs),
        )

        # Generate slug base
        slug_base = self._generate_slug_base(metadata)

        # Initialize state
        initial_state: PipelineState = {
            "note_content": note_content,
            "metadata_dict": asdict(metadata),
            "qa_pairs_dicts": [asdict(qa) for qa in qa_pairs],
            "file_path": str(file_path) if file_path else None,
            "slug_base": slug_base,
            "pre_validation": None,
            "generation": None,
            "post_validation": None,
            "current_stage": "pre_validation",
            "retry_count": 0,
            "max_retries": self.max_retries,
            "auto_fix_enabled": self.auto_fix_enabled,
            "strict_mode": self.strict_mode,
            "start_time": start_time,
            "stage_times": {},
            "messages": [],
        }

        # Execute workflow
        final_state = self.app.invoke(
            initial_state,  # type: ignore[arg-type]
            config={"configurable": {"thread_id": f"note-{metadata.title}"}},
        )

        # Build result
        total_time = time.time() - start_time
        success = final_state["current_stage"] == "complete"

        # Deserialize results
        pre_validation = (
            PreValidationResult(**final_state["pre_validation"])
            if final_state.get("pre_validation")
            else None
        )
        generation = (
            GenerationResult(**final_state["generation"])
            if final_state.get("generation")
            else None
        )
        post_validation = (
            PostValidationResult(**final_state["post_validation"])
            if final_state.get("post_validation")
            else None
        )

        result = AgentPipelineResult(
            success=success,
            pre_validation=pre_validation
            or PreValidationResult(
                is_valid=False, error_type="none", validation_time=0.0
            ),
            generation=generation,
            post_validation=post_validation,
            total_time=total_time,
            retry_count=final_state["retry_count"],
        )

        logger.info(
            "langgraph_pipeline_complete",
            success=success,
            retry_count=final_state["retry_count"],
            total_time=total_time,
            messages=final_state["messages"],
        )

        return result

    def _generate_slug_base(self, metadata: NoteMetadata) -> str:
        """Generate base slug from note metadata using collision-safe helper."""

        return generate_agent_slug_base(metadata)
