"""Main LangGraph orchestrator for card generation pipeline.

This module implements the orchestrator class that builds and executes
the LangGraph state machine workflow.
"""

import time
import uuid
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ...config import Config
from ...models import NoteMetadata, QAPair
from ...utils.logging import get_logger
from ..models import (
    AgentPipelineResult,
    CardSplittingResult,
    ContextEnrichmentResult,
    GeneratedCard,
    GenerationResult,
    MemorizationQualityResult,
    PostValidationResult,
    PreValidationResult,
)
from ..slug_utils import generate_agent_slug_base
from .nodes import (
    card_splitting_node,
    context_enrichment_node,
    duplicate_detection_node,
    generation_node,
    memorization_quality_node,
    note_correction_node,
    post_validation_node,
    pre_validation_node,
)
from .retry_policies import TRANSIENT_RETRY_POLICY, VALIDATION_RETRY_POLICY
from .state import PipelineState

# Optional agent memory and observability (requires chromadb, motor, langsmith)
try:
    from ..agent_memory import AgentMemoryStore
    from ..advanced_memory import AdvancedMemoryStore  # NEW: Advanced memory system
    # NEW: Enhanced observability
    from ..enhanced_observability import EnhancedObservabilitySystem
except ImportError:
    AgentMemoryStore = None
    AdvancedMemoryStore = None
    EnhancedObservabilitySystem = None

logger = get_logger(__name__)


# ============================================================================
# Conditional Routing
# ============================================================================


def should_continue_after_pre_validation(
    state: PipelineState,
) -> Literal["card_splitting", "generation", "failed"]:
    """Determine next node after pre-validation.

    Args:
        state: Current pipeline state

    Returns:
        Next node name
    """
    pre_validation = state.get("pre_validation")
    if pre_validation and pre_validation["is_valid"]:
        # Route to card splitting if enabled, otherwise directly to generation
        if state.get("enable_card_splitting", True):
            return "card_splitting"
        return "generation"
    return "failed"


def should_continue_after_post_validation(
    state: PipelineState,
) -> Literal["context_enrichment", "generation", "failed"]:
    """Determine next node after post-validation.

    Args:
        state: Current pipeline state

    Returns:
        Next node name
    """
    current_stage = state.get("current_stage", "failed")

    if current_stage == "context_enrichment":
        return "context_enrichment"
    elif current_stage == "generation":
        return "generation"  # Retry
    else:
        return "failed"


def should_continue_after_enrichment(
    state: PipelineState,
) -> Literal["memorization_quality", "complete"]:
    """Determine next node after context enrichment.

    Args:
        state: Current pipeline state

    Returns:
        Next node name
    """
    current_stage = state.get("current_stage", "complete")

    if current_stage == "memorization_quality":
        return "memorization_quality"
    else:
        return "complete"


def should_continue_after_memorization_quality(
    state: PipelineState,
) -> Literal["duplicate_detection", "complete"]:
    """Determine next node after memorization quality.

    Args:
        state: Current pipeline state

    Returns:
        Next node name
    """
    current_stage = state.get("current_stage", "complete")

    if current_stage == "duplicate_detection":
        return "duplicate_detection"
    else:
        return "complete"


# ============================================================================
# Workflow Builder
# ============================================================================


class LangGraphOrchestrator:
    """LangGraph-based orchestrator for the card generation pipeline.

    Uses a state machine workflow with conditional routing, automatic retries,
    and state persistence via checkpoints.
    """

    async def __init__(  # CHANGED: Made async for MongoDB connection
        self,
        config: Config,
        max_retries: int | None = None,
        auto_fix_enabled: bool | None = None,
        strict_mode: bool | None = None,
        enable_card_splitting: bool | None = None,
        enable_context_enrichment: bool | None = None,
        enable_memorization_quality: bool | None = None,
        enable_duplicate_detection: bool | None = None,
    ):
        """Initialize LangGraph orchestrator.

        Args:
            config: Service configuration
            max_retries: Maximum post-validation retry attempts (uses config if None)
            auto_fix_enabled: Enable automatic error fixing (uses config if None)
            strict_mode: Use strict validation mode (uses config if None)
            enable_card_splitting: Enable card splitting agent (uses config if None)
            enable_context_enrichment: Enable context enrichment agent (uses config if None)
            enable_memorization_quality: Enable memorization quality agent (uses config if None)
            enable_duplicate_detection: Enable duplicate detection agent (uses config if None)
        """
        self.config = config
        # Use config values as defaults if not explicitly provided
        self.max_retries = (
            max_retries if max_retries is not None else config.langgraph_max_retries
        )
        self.auto_fix_enabled = (
            auto_fix_enabled
            if auto_fix_enabled is not None
            else config.langgraph_auto_fix
        )
        self.strict_mode = (
            strict_mode if strict_mode is not None else config.langgraph_strict_mode
        )
        self.enable_card_splitting = (
            enable_card_splitting
            if enable_card_splitting is not None
            # Default to True
            else getattr(config, "enable_card_splitting", True)
        )
        self.enable_context_enrichment = (
            enable_context_enrichment
            if enable_context_enrichment is not None
            else config.enable_context_enrichment
        )
        self.enable_memorization_quality = (
            enable_memorization_quality
            if enable_memorization_quality is not None
            else config.enable_memorization_quality
        )
        self.enable_duplicate_detection = (
            enable_duplicate_detection
            if enable_duplicate_detection is not None
            else getattr(
                config, "enable_duplicate_detection", False
            )  # Default to False
        )

        # Create and cache PydanticAI models once during initialization
        # This avoids recreating models (and HTTP clients) on every node execution
        from ...providers.pydantic_ai_models import PydanticAIModelFactory

        try:
            # Create models with full configuration including reasoning
            self.pre_validator_model = self._create_model_with_config(
                config, "pre_validator"
            )
            self.card_splitting_model = self._create_model_with_config(
                config, "card_splitting"
            )
            self.generator_model = self._create_model_with_config(
                config, "generator"
            )
            self.post_validator_model = self._create_model_with_config(
                config, "post_validator"
            )
            self.context_enrichment_model = self._create_model_with_config(
                config, "context_enrichment"
            )
            self.memorization_quality_model = self._create_model_with_config(
                config, "memorization_quality"
            )
            self.duplicate_detection_model = self._create_model_with_config(
                config, "duplicate_detection"
            )
            logger.info("pydantic_ai_models_cached", models_created=7)
        except Exception as e:
            logger.warning(
                "failed_to_cache_models_will_create_on_demand", error=str(e))
            # Set to None - nodes will create models on demand as fallback
            self.pre_validator_model = None
            self.card_splitting_model = None
            self.generator_model = None
            self.post_validator_model = None
            self.context_enrichment_model = None
            self.memorization_quality_model = None
            self.duplicate_detection_model = None

        # Initialize memory stores if enabled
        self.memory_store = None
        self.advanced_memory_store = None

    def _create_model_with_config(self, config: Config, agent_type: str) -> Any:
        """Create a PydanticAI model with full configuration including reasoning.

        Args:
            config: Configuration object
            agent_type: Agent type (e.g., "pre_validator", "generator")

        Returns:
            Configured PydanticAI model
        """
        model_name = config.get_model_for_agent(agent_type)
        model_config = config.get_model_config_for_task(agent_type)

        return PydanticAIModelFactory.create_from_config(
            config,
            model_name=model_name,
            reasoning_enabled=model_config.get("reasoning_enabled", False),
            max_tokens=model_config.get("max_tokens"),
        )

        # Legacy ChromaDB memory store
        if getattr(config, "enable_agent_memory", True) and AgentMemoryStore:
            try:
                memory_storage_path = getattr(
                    config, "memory_storage_path", Path(".agent_memory")
                )
                enable_semantic_search = getattr(
                    config, "enable_semantic_search", True)
                embedding_model = getattr(
                    config, "embedding_model", "text-embedding-3-small"
                )

                self.memory_store = AgentMemoryStore(
                    storage_path=memory_storage_path,
                    embedding_model=embedding_model,
                    enable_semantic_search=enable_semantic_search,
                )
                logger.info(
                    "langgraph_memory_store_initialized",
                    path=str(memory_storage_path),
                )
            except Exception as e:
                logger.warning(
                    "langgraph_memory_store_init_failed", error=str(e))

        # NEW: Advanced MongoDB memory store
        if getattr(config, "use_advanced_memory", False) and AdvancedMemoryStore:
            try:
                mongodb_url = getattr(
                    config, "mongodb_url", "mongodb://localhost:27017")
                memory_db_name = getattr(
                    config, "memory_db_name", "obsidian_anki_memory")

                self.advanced_memory_store = AdvancedMemoryStore(
                    config=config,
                    mongodb_url=mongodb_url,
                    db_name=memory_db_name,
                    embedding_store=self.memory_store,  # Link to ChromaDB for embeddings
                )

                # Connect to MongoDB (synchronous wrapper for async connect)
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If already in async context, schedule the coroutine
                        # Connection will be established on first use
                        logger.info(
                            "advanced_memory_store_deferred_connection",
                            mongodb_url=mongodb_url,
                            db_name=memory_db_name,
                        )
                    else:
                        connected = loop.run_until_complete(
                            self.advanced_memory_store.connect()
                        )
                        if connected:
                            logger.info(
                                "advanced_memory_store_initialized",
                                mongodb_url=mongodb_url,
                                db_name=memory_db_name,
                            )
                        else:
                            logger.warning(
                                "advanced_memory_store_connection_failed")
                            self.advanced_memory_store = None
                except RuntimeError:
                    # No event loop available, will connect on first use
                    logger.info("advanced_memory_store_lazy_connection")

            except Exception as e:
                logger.warning(
                    "advanced_memory_store_init_failed", error=str(e))

        # NEW: Enhanced observability system
        self.observability = None
        if getattr(config, "enable_enhanced_observability", False) and EnhancedObservabilitySystem:
            try:
                self.observability = EnhancedObservabilitySystem(config)
                logger.info("enhanced_observability_system_initialized")
            except Exception as e:
                logger.warning(
                    "enhanced_observability_init_failed", error=str(e))

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
            card_splitting=enable_card_splitting,
            context_enrichment=enable_context_enrichment,
            memorization_quality=enable_memorization_quality,
            duplicate_detection=enable_duplicate_detection,
        )

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow.

        Best practices applied:
        - RetryPolicy with exponential backoff for transient failures
        - Different retry policies for different node types
        - Conditional routing for error handling

        Returns:
            Configured StateGraph instance
        """
        # Create workflow graph
        workflow = StateGraph(PipelineState)

        # Add optional note correction node (if enabled)
        # Best practice: Use RetryPolicy for nodes that call external APIs
        enable_note_correction = getattr(
            self.config, "enable_note_correction", False)
        if enable_note_correction:
            workflow.add_node(
                "note_correction",
                note_correction_node,
                retry=TRANSIENT_RETRY_POLICY,
            )

        # Add core nodes with appropriate retry policies
        # Validation nodes: lighter retry (faster, less critical)
        workflow.add_node(
            "pre_validation",
            pre_validation_node,
            retry=VALIDATION_RETRY_POLICY,
        )

        # Card splitting: lighter retry (optional enhancement)
        workflow.add_node(
            "card_splitting",
            card_splitting_node,
            retry=VALIDATION_RETRY_POLICY,
        )

        # Generation: full retry policy (critical, expensive operation)
        workflow.add_node(
            "generation",
            generation_node,
            retry=TRANSIENT_RETRY_POLICY,
        )

        # Post-validation: lighter retry (validation step)
        workflow.add_node(
            "post_validation",
            post_validation_node,
            retry=VALIDATION_RETRY_POLICY,
        )

        # Add enhancement nodes with appropriate retry policies
        workflow.add_node(
            "context_enrichment",
            context_enrichment_node,
            retry=VALIDATION_RETRY_POLICY,
        )
        workflow.add_node(
            "memorization_quality",
            memorization_quality_node,
            retry=VALIDATION_RETRY_POLICY,
        )
        workflow.add_node(
            "duplicate_detection",
            duplicate_detection_node,
            retry=VALIDATION_RETRY_POLICY,
        )

        # Set entry point (note_correction if enabled, otherwise pre_validation)
        if enable_note_correction:
            workflow.set_entry_point("note_correction")
            # Note correction always goes to pre-validation
            workflow.add_edge("note_correction", "pre_validation")
        else:
            workflow.set_entry_point("pre_validation")

        # Add conditional edges
        workflow.add_conditional_edges(
            "pre_validation",
            should_continue_after_pre_validation,
            {
                "card_splitting": "card_splitting",
                "generation": "generation",
                "failed": END,
            },
        )

        # Card splitting always goes to generation
        workflow.add_edge("card_splitting", "generation")

        workflow.add_edge("generation", "post_validation")

        workflow.add_conditional_edges(
            "post_validation",
            should_continue_after_post_validation,
            {
                "context_enrichment": "context_enrichment",
                "generation": "generation",  # Retry loop
                "failed": END,
            },
        )

        # Add enrichment to quality routing
        workflow.add_conditional_edges(
            "context_enrichment",
            should_continue_after_enrichment,
            {
                "memorization_quality": "memorization_quality",
                "complete": END,
            },
        )

        # Memorization quality to duplicate detection routing
        workflow.add_conditional_edges(
            "memorization_quality",
            should_continue_after_memorization_quality,
            {
                "duplicate_detection": "duplicate_detection",
                "complete": END,
            },
        )

        # Duplicate detection always goes to END
        workflow.add_edge("duplicate_detection", END)

        return workflow

    async def process_note(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        file_path: Path | None = None,
        existing_cards: list[GeneratedCard] | None = None,
    ) -> AgentPipelineResult:
        """Process a note through the LangGraph workflow.

        Args:
            note_content: Full note content
            metadata: Parsed metadata
            qa_pairs: Parsed Q/A pairs
            file_path: Optional file path for validation
            existing_cards: Optional list of existing cards for duplicate detection

        Returns:
            AgentPipelineResult with all pipeline stages
        """
        start_time = time.time()

        logger.info(
            "langgraph_pipeline_start",
            title=metadata.title,
            qa_pairs_count=len(qa_pairs),
        )

        # NEW: Start observability tracking
        observability_context = None
        if self.observability:
            observability_context = self.observability.trace_agent_execution(
                agent_name="langgraph_orchestrator",
                title=metadata.title,
                qa_pairs_count=len(qa_pairs),
                has_existing_cards=existing_cards is not None,
            )

        # Generate slug base
        slug_base = self._generate_slug_base(metadata)

        # Initialize state
        # Best practice: Initialize all state fields explicitly
        initial_state: PipelineState = {
            # Input data
            "note_content": note_content,
            "metadata_dict": metadata.model_dump(),
            "qa_pairs_dicts": [qa.model_dump() for qa in qa_pairs],
            "file_path": str(file_path) if file_path else None,
            "slug_base": slug_base,
            "config": self.config,  # Pass config for model selection
            "existing_cards_dicts": (
                [card.model_dump() for card in existing_cards]
                if existing_cards
                else None
            ),
            # Pass cached models through state for reuse
            "pre_validator_model": self.pre_validator_model,
            "card_splitting_model": self.card_splitting_model,
            "generator_model": self.generator_model,
            "post_validator_model": self.post_validator_model,
            "context_enrichment_model": self.context_enrichment_model,
            "memorization_quality_model": self.memorization_quality_model,
            "duplicate_detection_model": self.duplicate_detection_model,
            # Pipeline stage results
            "pre_validation": None,
            "card_splitting": None,
            "generation": None,
            "post_validation": None,
            "context_enrichment": None,
            "memorization_quality": None,
            "duplicate_detection": None,
            # Workflow control
            "current_stage": (
                "note_correction"
                if getattr(self.config, "enable_note_correction", False)
                else "pre_validation"
            ),
            "enable_card_splitting": self.enable_card_splitting,
            "enable_context_enrichment": self.enable_context_enrichment,
            "enable_memorization_quality": self.enable_memorization_quality,
            "enable_duplicate_detection": self.enable_duplicate_detection,
            "retry_count": 0,
            "max_retries": self.max_retries,
            "auto_fix_enabled": self.auto_fix_enabled,
            "strict_mode": self.strict_mode,
            # Cycle protection (best practice: add hard stops)
            "step_count": 0,
            "max_steps": getattr(self.config, "langgraph_max_steps", 20),
            # Error tracking (best practice: track errors for debugging)
            "last_error": None,
            "last_error_severity": None,
            "errors": [],
            # Timing
            "start_time": start_time,
            "stage_times": {},
            "messages": [],
        }

        # Execute workflow with async invocation
        # Use unique thread ID with UUID to avoid collisions
        thread_id = f"note-{metadata.title}-{uuid.uuid4().hex[:8]}"
        # LangGraph's ainvoke type checking is imperfect with TypedDict states
        final_state = await self.app.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
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
        # Card splitting result is available in state but not used in AgentPipelineResult
        # (kept for future use or logging)
        _card_splitting = (
            CardSplittingResult(**final_state["card_splitting"])
            if final_state.get("card_splitting")
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
        # Context enrichment result is available in state but not used in AgentPipelineResult
        # (kept for future use or logging)
        context_enrichment = (
            ContextEnrichmentResult(**final_state["context_enrichment"])
            if final_state.get("context_enrichment")
            else None
        )
        memorization_quality = (
            MemorizationQualityResult(**final_state["memorization_quality"])
            if final_state.get("memorization_quality")
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
            memorization_quality=memorization_quality,
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

        # NEW: Record observability metrics
        if self.observability:
            try:
                from ..enhanced_observability import AgentMetrics
                metrics = AgentMetrics(
                    agent_name="langgraph_orchestrator",
                    execution_time=total_time,
                    success=success,
                    retry_count=final_state["retry_count"],
                    step_count=final_state.get("step_count", 0),
                    quality_score=memorization_quality.memorization_score if memorization_quality else None,
                    api_cost=None,  # Could be calculated from API usage
                    token_count=None,  # Could be tracked from model usage
                    error_type=final_state.get("last_error_severity"),
                    handoffs=0,  # Could be tracked in swarm mode
                    timestamp=start_time,
                )
                self.observability.record_metrics(metrics)
                logger.info("observability_metrics_recorded")
            except Exception as e:
                logger.warning(
                    "observability_metrics_recording_failed", error=str(e))

        # NEW: Learn from execution if advanced memory is enabled
        if self.advanced_memory_store and self.advanced_memory_store.connected:
            try:
                await self.advanced_memory_store.learn_from_pipeline_result(
                    agent_name="langgraph_orchestrator",
                    task_type="note_processing",
                    input_data={
                        # Truncate for hashing
                        "note_content": note_content[:500],
                        "qa_pairs_count": len(qa_pairs),
                        "metadata_title": metadata.title,
                    },
                    pipeline_result={
                        "success": success,
                        "total_time": total_time,
                        "retry_count": final_state["retry_count"],
                        "step_count": final_state.get("step_count", 0),
                        "quality_score": memorization_quality.memorization_score if memorization_quality else None,
                        "context_enrichment": context_enrichment is not None,
                        "error_type": last_error.get("type") if (last_error := final_state.get("last_error")) else None,
                    },
                    execution_time=total_time,
                )
                logger.info("advanced_memory_learning_completed")
            except Exception as e:
                logger.warning("advanced_memory_learning_failed", error=str(e))

        return result

    def _generate_slug_base(self, metadata: NoteMetadata) -> str:
        """Generate base slug from note metadata using collision-safe helper."""

        return generate_agent_slug_base(metadata)
