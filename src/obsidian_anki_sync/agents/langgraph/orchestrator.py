"""Main LangGraph orchestrator for card generation pipeline.

This module implements the orchestrator class that builds and executes
the LangGraph state machine workflow.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from obsidian_anki_sync.config import Config
    from obsidian_anki_sync.models import Card, NoteMetadata, QAPair

    from .state import PipelineState

from obsidian_anki_sync.agents.agent_monitoring import (
    PerformanceTracker,
    get_pipeline_performance_tracker,
)
from obsidian_anki_sync.agents.models import (
    AgentPipelineResult,
    CardSplittingResult,
    ContextEnrichmentResult,
    GeneratedCard,
    GenerationResult,
    HighlightResult,
    MemorizationQualityResult,
    PostValidationResult,
    PreValidationResult,
)
from obsidian_anki_sync.agents.slug_utils import generate_agent_slug_base
from obsidian_anki_sync.agents.unified_agent import UnifiedAgentSelector
from obsidian_anki_sync.error_codes import ErrorCode
from obsidian_anki_sync.utils.logging import get_logger

from .model_factory import ModelFactory
from .state import PipelineState, cleanup_runtime_resources, register_runtime_resources
from .workflow_builder import WorkflowBuilder

# Optional agent memory and observability (requires chromadb, motor, langsmith)
try:
    from obsidian_anki_sync.agents.advanced_memory import (
        AdvancedMemoryStore,  # NEW: Advanced memory system
    )
    from obsidian_anki_sync.agents.agent_memory import AgentMemoryStore

    # NEW: Enhanced observability
    from obsidian_anki_sync.agents.enhanced_observability import (
        EnhancedObservabilitySystem,
    )
except ImportError:
    AgentMemoryStore = None
    AdvancedMemoryStore = None
    EnhancedObservabilitySystem = None

# Optional RAG integration
try:
    from obsidian_anki_sync.rag.integration import RAGIntegration, get_rag_integration
except ImportError:
    RAGIntegration = None
    get_rag_integration = None

logger = get_logger(__name__)


class LangGraphOrchestrator:
    """LangGraph-based orchestrator for the card generation pipeline.

    Uses a state machine workflow with conditional routing, automatic retries,
    and state persistence via checkpoints.
    """

    def __init__(
        self,
        config: Config,
        max_retries: int | None = None,
        auto_fix_enabled: bool | None = None,
        strict_mode: bool | None = None,
        enable_card_splitting: bool | None = None,
        enable_context_enrichment: bool | None = None,
        enable_memorization_quality: bool | None = None,
        enable_duplicate_detection: bool | None = None,
        agent_framework: str | None = None,  # NEW: Agent framework selection
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
            agent_framework: Agent framework to use ("pydantic_ai" or "langchain", uses config if None)
        """
        self.config = config
        self.performance_tracker: PerformanceTracker | None = (
            get_pipeline_performance_tracker(config)
        )
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
        self.enable_highlight_agent = getattr(config, "enable_highlight_agent", True)
        self.highlight_max_candidates = getattr(config, "highlight_max_candidates", 3)

        # NEW: Agent framework selection (can be overridden by memory)
        self.agent_framework = (
            agent_framework
            if agent_framework is not None
            else getattr(config, "agent_framework", "pydantic_ai")
        )

        # NEW: Initialize unified agent selector for framework switching
        self.agent_selector = UnifiedAgentSelector(config)

        # Initialize ModelFactory (still needed for some legacy functionality)
        self.model_factory = ModelFactory(config)

        # Initialize WorkflowBuilder
        self.workflow_builder = WorkflowBuilder(config)

        # Initialize memory stores if enabled
        self.memory_store = None
        self.advanced_memory_store = None

        # Legacy ChromaDB memory store
        if getattr(config, "enable_agent_memory", True) and AgentMemoryStore:
            try:
                memory_storage_path = getattr(
                    config, "memory_storage_path", Path(".agent_memory")
                )
                enable_semantic_search = getattr(config, "enable_semantic_search", True)

                self.memory_store = AgentMemoryStore(
                    storage_path=memory_storage_path,
                    config=config,
                    enable_semantic_search=enable_semantic_search,
                )
                logger.info(
                    "langgraph_memory_store_initialized",
                    path=str(memory_storage_path),
                )
            except Exception as e:
                logger.warning("langgraph_memory_store_init_failed", error=str(e))

        # NEW: Advanced MongoDB memory store (deferred connection)
        self.advanced_memory_store = None
        if getattr(config, "use_advanced_memory", False) and AdvancedMemoryStore:
            try:
                mongodb_url = getattr(
                    config, "mongodb_url", "mongodb://localhost:27017"
                )
                memory_db_name = getattr(
                    config, "memory_db_name", "obsidian_anki_memory"
                )

                self.advanced_memory_store = AdvancedMemoryStore(
                    config=config,
                    mongodb_url=mongodb_url,
                    db_name=memory_db_name,
                    embedding_store=self.memory_store,  # Link to ChromaDB for embeddings
                )
                logger.info("advanced_memory_store_deferred_connection")
            except Exception as e:
                logger.warning("advanced_memory_store_init_failed", error=str(e))

        # NEW: Enhanced observability system
        self.observability = None
        if (
            getattr(config, "enable_enhanced_observability", False)
            and EnhancedObservabilitySystem
        ):
            try:
                self.observability = EnhancedObservabilitySystem(config)
                logger.info("enhanced_observability_system_initialized")
            except Exception as e:
                logger.warning("enhanced_observability_init_failed", error=str(e))

        # RAG integration for context enrichment and duplicate detection
        self.rag_integration = None
        self.enable_rag = getattr(config, "rag_enabled", False)
        if self.enable_rag and get_rag_integration is not None:
            try:
                self.rag_integration = get_rag_integration(config)
                if self.rag_integration.is_enabled:
                    logger.info(
                        "rag_integration_initialized",
                        context_enrichment=getattr(
                            config, "rag_context_enrichment", True
                        ),
                        duplicate_detection=getattr(
                            config, "rag_duplicate_detection", True
                        ),
                        few_shot_examples=getattr(
                            config, "rag_few_shot_examples", True
                        ),
                    )
                else:
                    logger.info(
                        "rag_integration_not_indexed",
                        hint="Run 'obsidian-anki-sync rag index' to build the index for enhanced features",
                    )
            except Exception as e:
                logger.warning("rag_integration_init_failed", error=str(e))

        # Build the workflow graph
        self.workflow = self.workflow_builder.build_workflow()

        # Compile the graph
        # Note: We do not use a checkpointer (MemorySaver) here because:
        # 1. We don't need to resume interrupted workflows
        # 2. MemorySaver accumulates state history indefinitely, causing memory leaks in long-running processes
        # 3. We use ainvoke for single-shot execution
        self.app = self.workflow.compile()

        # Provider compatibility - create a dummy provider for compatibility
        # The LangGraph orchestrator uses PydanticAI models internally
        self._provider = None  # Will be set up async if needed

        logger.info(
            "langgraph_orchestrator_initialized",
            max_retries=self.max_retries,
            auto_fix=self.auto_fix_enabled,
            strict_mode=self.strict_mode,
            card_splitting=self.enable_card_splitting,
            context_enrichment=self.enable_context_enrichment,
            memorization_quality=self.enable_memorization_quality,
            duplicate_detection=self.enable_duplicate_detection,
        )

    async def setup_async(self):
        """Async setup for components that need async initialization (e.g., MongoDB)."""
        # Connect to MongoDB if advanced memory is enabled
        if self.advanced_memory_store and hasattr(
            self.advanced_memory_store, "connect"
        ):
            try:
                connected = await self.advanced_memory_store.connect()
                if connected:
                    logger.info("advanced_memory_store_connected")
                else:
                    logger.warning("advanced_memory_store_connection_failed")
                    self.advanced_memory_store = None
            except Exception as e:
                logger.warning("advanced_memory_store_async_setup_failed", error=str(e))
                self.advanced_memory_store = None

    def convert_to_cards(
        self,
        generated_cards: list[GeneratedCard],
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        file_path: Path | None = None,
    ) -> list[Card]:
        """Convert GeneratedCard instances to Card instances.

        This method replicates the legacy orchestrator's card conversion logic.
        """
        import hashlib

        from obsidian_anki_sync.anki.field_mapper import map_apf_to_anki_fields
        from obsidian_anki_sync.models import Card, Manifest
        from obsidian_anki_sync.utils.content_hash import compute_content_hash

        cards: list[Card] = []
        qa_lookup = {qa.card_index: qa for qa in qa_pairs}

        for gen_card in generated_cards:
            # Validate that we can extract fields from the generated HTML
            try:
                # Use default note type for validation check
                fields = map_apf_to_anki_fields(gen_card.apf_html, "APF::Simple")

                # Log all extracted fields for debugging purposes
                logger.debug(
                    "extracted_fields_from_generated_card",
                    slug=gen_card.slug,
                    apf_html_length=len(gen_card.apf_html),
                    extracted_fields=fields,
                )

                if not fields.get("Primary Title"):
                    logger.warning(
                        "empty_primary_title_detected_in_generated_card",
                        slug=gen_card.slug,
                        apf_html_preview=gen_card.apf_html[:500],
                        hint="Card fields extraction failed (Primary Title is empty). Check APF format.",
                    )
                    continue  # Skip this card

            except Exception as e:
                logger.warning(
                    "field_extraction_failed_during_conversion",
                    slug=gen_card.slug,
                    error=str(e),
                    apf_html_preview=gen_card.apf_html[:500],
                )
                continue  # Skip this card if parsing fails entirely

            # Create manifest
            # Safely extract slug_base by removing -index-lang suffix
            parts = gen_card.slug.rsplit("-", 2)
            slug_base = parts[0] if len(parts) >= 3 else gen_card.slug

            manifest = Manifest(
                slug=gen_card.slug,
                slug_base=slug_base,
                lang=gen_card.lang,
                source_path=str(file_path) if file_path else "unknown",
                source_anchor=f"qa-{gen_card.card_index}",
                note_id=metadata.id,
                note_title=metadata.title,
                card_index=gen_card.card_index,
                guid=gen_card.slug,  # Use slug as GUID for now
                hash6=None,
            )

            qa_pair = qa_lookup.get(gen_card.card_index)
            content_hash = gen_card.content_hash
            if not content_hash and qa_pair:
                content_hash = compute_content_hash(qa_pair, metadata, gen_card.lang)
            elif not content_hash:
                content_hash = hashlib.sha256(
                    gen_card.apf_html.encode("utf-8")
                ).hexdigest()

            cards.append(
                Card(
                    slug=gen_card.slug,
                    lang=gen_card.lang,
                    apf_html=gen_card.apf_html,
                    manifest=manifest,
                    content_hash=content_hash,
                    note_type="APF::Simple",  # Default, can be detected from HTML
                    tags=[],  # Extract from manifest in HTML
                    guid=gen_card.slug,
                )
            )

        return cards

    @property
    def provider(self):
        """Provider compatibility property for sync engine access.

        Returns a dummy provider for compatibility with sync engine expectations.
        """
        if self._provider is None:
            # Create a minimal provider for compatibility
            # This is only used by the sync engine to access provider name/check_connection
            from obsidian_anki_sync.providers.base import BaseLLMProvider

            class LangGraphCompatibilityProvider(BaseLLMProvider):
                def get_provider_name(self) -> str:
                    return "langgraph_pydantic_ai"

                def check_connection(self) -> bool:
                    return True  # LangGraph handles its own connection checks

                def list_models(self) -> list[str]:
                    return []  # LangGraph uses PydanticAI models directly

                async def generate(self, model: str, prompt: str, **kwargs) -> str:
                    msg = "LangGraph orchestrator handles generation internally"
                    raise NotImplementedError(msg)

                def generate_sync(self, prompt: str, **kwargs) -> str:
                    msg = "LangGraph orchestrator handles generation internally"
                    raise NotImplementedError(msg)

            self._provider = LangGraphCompatibilityProvider()

        return self._provider

    @provider.setter
    def provider(self, value):
        """Allow setting provider for compatibility."""
        self._provider = value

    async def _determine_optimal_agent_framework(
        self,
        note_content: str,
        metadata: NoteMetadata,
    ) -> str:
        """Determine the optimal agent framework based on memory and content analysis.

        Args:
            note_content: Full note content
            metadata: Note metadata

        Returns:
            Optimal agent framework: "pydantic_ai", "langchain", or "memory_enhanced"
        """
        # Start with configured default
        optimal_framework = self.agent_framework

        # If memory is available, use it for routing decisions
        if self.advanced_memory_store and self.advanced_memory_store.connected:
            try:
                topic = getattr(metadata, "topic", "general")
                user_id = getattr(metadata, "user_id", "default")

                # Check user preferences for framework
                user_prefs = await self.advanced_memory_store.get_user_card_preferences(
                    user_id, topic
                )
                if user_prefs and user_prefs.confidence > 0.8:
                    # If user strongly prefers memory-enhanced and it's available
                    if self.agent_selector.memory_enhanced_available:
                        optimal_framework = "memory_enhanced"
                        logger.info(
                            "memory_based_routing_selected_memory_enhanced",
                            topic=topic,
                            user_id=user_id,
                            confidence=user_prefs.confidence,
                        )

                # Check topic performance patterns
                topic_stats = await self.advanced_memory_store.get_topic_feedback_stats(
                    topic
                )
                if topic_stats and topic_stats.get("total_feedback", 0) > 10:
                    avg_quality = topic_stats.get("avg_quality_score", 0.5)

                    # If topic has high quality scores and memory-enhanced is available,
                    # prefer it for consistency
                    if (
                        avg_quality > 0.8
                        and self.agent_selector.memory_enhanced_available
                    ):
                        optimal_framework = "memory_enhanced"
                        logger.info(
                            "topic_performance_routing_selected_memory_enhanced",
                            topic=topic,
                            avg_quality=avg_quality,
                            total_feedback=topic_stats.get("total_feedback"),
                        )
                    # If topic has poor quality scores, try langchain for different approach
                    elif avg_quality < 0.6:
                        optimal_framework = "langchain"
                        logger.info(
                            "topic_performance_routing_selected_langchain",
                            topic=topic,
                            avg_quality=avg_quality,
                            reason="poor_historical_performance",
                        )

                # Content complexity routing
                content_length = len(note_content)
                if content_length > 5000:  # Very complex content
                    # Use langchain for complex multi-step reasoning
                    optimal_framework = "langchain"
                    logger.info(
                        "content_complexity_routing_selected_langchain",
                        content_length=content_length,
                        reason="high_complexity_content",
                    )
                elif content_length < 500:  # Simple content
                    # Use memory-enhanced for simple, consistent generation
                    if self.agent_selector.memory_enhanced_available:
                        optimal_framework = "memory_enhanced"
                        logger.info(
                            "content_complexity_routing_selected_memory_enhanced",
                            content_length=content_length,
                            reason="simple_content",
                        )

            except Exception as e:
                logger.warning(
                    "memory_based_routing_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    fallback=self.agent_framework,
                )
                optimal_framework = self.agent_framework

        return optimal_framework

    def _average_generation_confidence(
        self, generation: GenerationResult | None
    ) -> float:
        """Compute the average confidence emitted by the generator."""
        if not generation or not generation.cards:
            return 0.0

        total_confidence = sum(card.confidence for card in generation.cards)
        return total_confidence / max(len(generation.cards), 1)

    def _stage_success(
        self,
        stage_name: str,
        pre_validation: PreValidationResult | None,
        generation: GenerationResult | None,
        post_validation: PostValidationResult | None,
        final_state: PipelineState,
    ) -> bool:
        """Heuristically determine whether a stage succeeded."""
        match stage_name:
            case "pre_validation":
                return bool(pre_validation and pre_validation.is_valid)
            case "generation":
                return bool(generation and generation.total_cards > 0)
            case "post_validation":
                return bool(post_validation and post_validation.is_valid)
            case "linter_validation":
                return bool(final_state.get("linter_valid", False))
            case _:
                # Consider other stages successful if they recorded time
                return True

    def _record_pipeline_metrics(
        self,
        *,
        success: bool,
        total_time: float,
        stage_times: dict[str, float],
        pre_validation: PreValidationResult | None,
        generation: GenerationResult | None,
        post_validation: PostValidationResult | None,
        final_state: PipelineState,
    ) -> None:
        """Record per-stage metrics using the shared performance tracker."""
        if not self.performance_tracker:
            return

        pipeline_confidence = self._average_generation_confidence(generation)
        error_type = final_state.get("last_error_severity")
        if not error_type and post_validation and not post_validation.is_valid:
            error_type = f"post_validation.{post_validation.error_type}"

        self.performance_tracker.record_call(
            agent_name="langgraph_pipeline",
            success=success,
            confidence=pipeline_confidence,
            response_time=total_time,
            error_type=error_type,
        )

        for stage_name, duration in stage_times.items():
            stage_success = self._stage_success(
                stage_name,
                pre_validation,
                generation,
                post_validation,
                final_state,
            )
            self.performance_tracker.record_call(
                agent_name=f"langgraph_stage_{stage_name}",
                success=stage_success,
                confidence=1.0 if stage_success else 0.0,
                response_time=duration,
                error_type=None if stage_success else stage_name,
            )

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
        import uuid

        start_time = time.time()
        pipeline_id = str(uuid.uuid4())[:8]

        logger.info(
            "pipeline_started",
            pipeline_id=pipeline_id,
            note_id=metadata.id,
            title=metadata.title,
            qa_pairs_count=len(qa_pairs),
            agent_framework="langgraph",
            file_path=str(file_path) if file_path else None,
            initial_state_keys=[
                "note_content",
                "metadata_dict",
                "qa_pairs_dicts",
                "file_path",
            ],
        )

        # NEW: Start observability tracking
        if self.observability:
            self.observability.trace_agent_execution(
                agent_name="langgraph_orchestrator",
                title=metadata.title,
                qa_pairs_count=len(qa_pairs),
                has_existing_cards=existing_cards is not None,
            )

        # Generate slug base
        slug_base = self._generate_slug_base(metadata)

        # Determine optimal agent framework based on memory and content
        optimal_framework = await self._determine_optimal_agent_framework(
            note_content, metadata
        )

        runtime_key = register_runtime_resources(
            config=self.config,
            model_factory=self.model_factory,
            agent_selector=self.agent_selector,
            rag_integration=self.rag_integration,
        )

        try:
            # Initialize state
            # Best practice: Initialize all state fields explicitly
            initial_state: PipelineState = {
                # Pipeline tracking
                "pipeline_id": pipeline_id,
                # Input data
                "note_content": note_content,
                "metadata_dict": metadata.model_dump(),
                "qa_pairs_dicts": [qa.model_dump() for qa in qa_pairs],
                "file_path": str(file_path) if file_path else None,
                "slug_base": slug_base,
                "runtime_key": runtime_key,
                "config_snapshot": self.config.model_dump(mode="json"),
                "existing_cards_dicts": (
                    [card.model_dump() for card in existing_cards]
                    if existing_cards
                    else None
                ),
                # NEW: Agent framework configuration (dynamically determined)
                "agent_framework": optimal_framework,
                "agent_selector": runtime_key,  # Present for backward compatibility
                # Cached model names (models fetched via runtime registry)
                "pre_validator_model": self.config.get_model_for_agent("pre_validator"),
                "card_splitting_model": self.config.get_model_for_agent(
                    "card_splitting"
                ),
                "generator_model": self.config.get_model_for_agent("generator"),
                "post_validator_model": self.config.get_model_for_agent(
                    "post_validator"
                ),
                "context_enrichment_model": self.config.get_model_for_agent(
                    "context_enrichment"
                ),
                "memorization_quality_model": self.config.get_model_for_agent(
                    "memorization_quality"
                ),
                "duplicate_detection_model": self.config.get_model_for_agent(
                    "duplicate_detection"
                ),
                "split_validator_model": self.config.get_model_for_agent(
                    "split_validator"
                ),
                "highlight_model": self.config.get_model_for_agent("highlight"),
                # Chain of Thought (CoT) configuration
                "enable_cot_reasoning": getattr(
                    self.config, "enable_cot_reasoning", False
                ),
                "store_reasoning_traces": getattr(
                    self.config, "store_reasoning_traces", True
                ),
                "log_reasoning_traces": getattr(
                    self.config, "log_reasoning_traces", False
                ),
                "cot_enabled_stages": getattr(
                    self.config,
                    "cot_enabled_stages",
                    ["pre_validation", "generation", "post_validation"],
                ),
                "reasoning_model": self.config.get_model_for_agent("reasoning"),
                "reasoning_traces": {},
                "current_reasoning": None,
                # Self-Reflection configuration
                "enable_self_reflection": getattr(
                    self.config, "enable_self_reflection", False
                ),
                "store_reflection_traces": getattr(
                    self.config, "store_reflection_traces", True
                ),
                "log_reflection_traces": getattr(
                    self.config, "log_reflection_traces", False
                ),
                "reflection_enabled_stages": getattr(
                    self.config,
                    "reflection_enabled_stages",
                    ["generation", "context_enrichment"],
                ),
                "reflection_model": self.config.get_model_for_agent("reflection"),
                "reflection_traces": {},
                "current_reflection": None,
                "revision_count": 0,
                "max_revisions": getattr(self.config, "max_revisions", 2),
                "stage_revision_counts": {},
                # Domain Detection and Smart Skipping
                "detected_domain": None,  # Will be set by first reflection node
                "reflection_skipped": False,
                "reflection_skip_reason": None,
                "revision_strategy": None,
                # RAG (Retrieval-Augmented Generation) configuration
                "enable_rag": self.enable_rag,
                "rag_context_enrichment": getattr(
                    self.config, "rag_context_enrichment", True
                ),
                "rag_duplicate_detection": getattr(
                    self.config, "rag_duplicate_detection", True
                ),
                "rag_few_shot_examples": getattr(
                    self.config, "rag_few_shot_examples", True
                ),
                "rag_integration": runtime_key,
                "rag_enrichment": None,
                "rag_examples": None,
                "rag_duplicate_results": None,
                # Auto-Fix configuration (autofix always runs as first step)
                "autofix_write_back": getattr(self.config, "autofix_write_back", False),
                "autofix_handlers": getattr(self.config, "autofix_handlers", None),
                # Pipeline stage results
                "autofix": None,  # Will be populated by autofix_node
                "pre_validation": None,
                "note_correction": None,
                "card_splitting": None,
                "generation": None,
                "linter_valid": False,  # Will be set by linter_validation_node
                "linter_results": [],  # Will be populated by linter_validation_node
                "post_validation": None,
                "context_enrichment": None,
                "memorization_quality": None,
                "duplicate_detection": None,
                "highlight_result": None,
                # Workflow control
                "current_stage": self._determine_entry_stage(),
                "enable_card_splitting": self.enable_card_splitting,
                "enable_context_enrichment": self.enable_context_enrichment,
                "enable_memorization_quality": self.enable_memorization_quality,
                "enable_duplicate_detection": self.enable_duplicate_detection,
                "enable_highlight_agent": self.enable_highlight_agent,
                "retry_count": 0,
                "max_retries": self.max_retries,
                "post_validator_timeout_seconds": getattr(
                    self.config,
                    "post_validator_timeout_seconds",
                    900.0,  # Increased for complex notes
                ),
                "post_validator_retry_backoff_seconds": getattr(
                    self.config, "post_validator_retry_backoff_seconds", 3.0
                ),
                "post_validator_retry_jitter_seconds": getattr(
                    self.config, "post_validator_retry_jitter_seconds", 1.5
                ),
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
            # Use unique thread ID with full UUID to avoid collisions
            thread_id = f"note-{metadata.title}-{uuid.uuid4().hex}"
            # LangGraph's ainvoke type checking is imperfect with TypedDict states
            final_state = await self.app.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}},
            )

            # Build result
            total_time = time.time() - start_time
            success = final_state["current_stage"] == "complete"
        finally:
            # Clean up runtime resources to prevent memory leak
            cleanup_runtime_resources(runtime_key)

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
        highlight_result = (
            HighlightResult(**final_state["highlight_result"])
            if final_state.get("highlight_result")
            else None
        )

        stage_times = final_state.get("stage_times", {})
        result = AgentPipelineResult(
            success=success,
            pre_validation=pre_validation
            or PreValidationResult(
                is_valid=False, error_type="none", validation_time=0.0
            ),
            generation=generation,
            post_validation=post_validation,
            memorization_quality=memorization_quality,
            highlight_result=highlight_result,
            total_time=total_time,
            retry_count=final_state["retry_count"],
            stage_times=stage_times,
            last_error=final_state.get("last_error"),
        )

        nodes_executed = final_state.get("step_count", 0)
        logger.info(
            "pipeline_completed",
            pipeline_id=pipeline_id,
            note_id=metadata.id,
            success=success,
            total_duration=round(total_time, 3),
            retry_count=final_state["retry_count"],
            nodes_executed=nodes_executed,
            stage_times=stage_times,
            current_stage=final_state.get("current_stage", "unknown"),
            cards_generated=len(generation.cards)
            if generation and generation.cards
            else 0,
        )

        self._record_pipeline_metrics(
            success=success,
            total_time=total_time,
            stage_times=stage_times,
            pre_validation=pre_validation,
            generation=generation,
            post_validation=post_validation,
            final_state=final_state,
        )

        # Post-pipeline tasks with timeout to prevent blocking result return
        # These are non-critical tasks (observability, memory learning) that
        # should not prevent the pipeline result from being returned to the worker
        post_pipeline_timeout = getattr(
            self.config, "post_pipeline_timeout_seconds", 30.0
        )
        logger.info(
            "post_pipeline_tasks_starting",
            pipeline_id=pipeline_id,
            note_id=metadata.id,
            timeout=post_pipeline_timeout,
            has_observability=self.observability is not None,
            has_advanced_memory=self.advanced_memory_store is not None,
            advanced_memory_connected=self.advanced_memory_store.connected
            if self.advanced_memory_store
            else False,
        )

        try:
            await asyncio.wait_for(
                self._run_post_pipeline_tasks(
                    pipeline_id=pipeline_id,
                    metadata=metadata,
                    note_content=note_content,
                    qa_pairs=qa_pairs,
                    success=success,
                    total_time=total_time,
                    final_state=final_state,
                    generation=generation,
                    memorization_quality=memorization_quality,
                    context_enrichment=context_enrichment,
                    start_time=start_time,
                ),
                timeout=post_pipeline_timeout,
            )
            logger.info(
                "post_pipeline_tasks_completed",
                pipeline_id=pipeline_id,
                note_id=metadata.id,
            )
        except TimeoutError:
            # Upgrade to error - timeouts indicate resource issues
            logger.error(
                "post_pipeline_tasks_timeout",
                pipeline_id=pipeline_id,
                note_id=metadata.id,
                timeout=post_pipeline_timeout,
                error_code=ErrorCode.PRV_MEMORY_FAILED.value,
                message="Post-pipeline tasks timed out, returning result anyway",
            )
        except Exception as e:
            # Upgrade to error - failures affect learning and observability
            logger.error(
                "post_pipeline_tasks_failed",
                pipeline_id=pipeline_id,
                note_id=metadata.id,
                error=str(e),
                error_code=ErrorCode.PRV_MEMORY_FAILED.value,
                message="Post-pipeline tasks failed, returning result anyway",
            )

        logger.info(
            "pipeline_returning_result",
            pipeline_id=pipeline_id,
            note_id=metadata.id,
            success=success,
            cards_count=len(generation.cards) if generation and generation.cards else 0,
        )
        return result

    async def _run_post_pipeline_tasks(
        self,
        pipeline_id: str,
        metadata: NoteMetadata,
        note_content: str,
        qa_pairs: list[QAPair],
        success: bool,
        total_time: float,
        final_state: dict,
        generation: GenerationResult | None,
        memorization_quality: MemorizationQualityResult | None,
        context_enrichment: ContextEnrichmentResult | None,
        start_time: float,
    ) -> None:
        """Run non-critical post-pipeline tasks (observability, memory learning).

        These tasks are wrapped in a timeout to prevent blocking result return.
        """
        logger.info(
            "post_pipeline_tasks_executing",
            pipeline_id=pipeline_id,
            note_id=metadata.id,
        )

        # Record observability metrics
        if self.observability:
            try:
                from obsidian_anki_sync.agents.enhanced_observability import (
                    AgentMetrics,
                )

                metrics = AgentMetrics(
                    agent_name="langgraph_orchestrator",
                    execution_time=total_time,
                    success=success,
                    retry_count=final_state["retry_count"],
                    step_count=final_state.get("step_count", 0),
                    quality_score=(
                        memorization_quality.memorization_score
                        if memorization_quality
                        else None
                    ),
                    api_cost=None,  # Could be calculated from API usage
                    token_count=None,  # Could be tracked from model usage
                    error_type=final_state.get("last_error_severity"),
                    handoffs=0,  # Could be tracked in swarm mode
                    timestamp=start_time,
                )
                self.observability.record_metrics(metrics)
                logger.info("observability_metrics_recorded", pipeline_id=pipeline_id)
            except Exception as e:
                logger.warning(
                    "observability_metrics_recording_failed",
                    pipeline_id=pipeline_id,
                    error=str(e),
                )

        # Learn from execution if advanced memory is enabled
        logger.info(
            "advanced_memory_check",
            pipeline_id=pipeline_id,
            has_store=self.advanced_memory_store is not None,
            connected=self.advanced_memory_store.connected
            if self.advanced_memory_store
            else False,
        )
        if self.advanced_memory_store and self.advanced_memory_store.connected:
            try:
                logger.info(
                    "advanced_memory_learning_starting", pipeline_id=pipeline_id
                )
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
                        "quality_score": (
                            memorization_quality.memorization_score
                            if memorization_quality
                            else None
                        ),
                        "context_enrichment": context_enrichment is not None,
                        "error_type": (
                            last_error.get("type")
                            if (last_error := final_state.get("last_error"))
                            else None
                        ),
                    },
                    execution_time=total_time,
                )

                # Learn user preferences from successful generation
                if success and generation and generation.success:
                    await self._learn_user_preferences(
                        metadata, generation, memorization_quality
                    )

                # Store memorization feedback for learning
                if memorization_quality:
                    await self._store_memorization_feedback(
                        metadata, generation, memorization_quality
                    )

                logger.info(
                    "advanced_memory_learning_completed", pipeline_id=pipeline_id
                )
            except Exception as e:
                logger.warning(
                    "advanced_memory_learning_failed",
                    pipeline_id=pipeline_id,
                    error=str(e),
                )

    async def _learn_user_preferences(
        self,
        metadata: NoteMetadata,
        generation: GenerationResult,
        memorization_quality: MemorizationQualityResult | None,
    ):
        """Learn user preferences from successful card generation.

        Args:
            metadata: Note metadata
            generation: Generation results
            memorization_quality: Memorization quality assessment
        """
        if not self.advanced_memory_store:
            return

        try:
            import time

            from obsidian_anki_sync.agents.advanced_memory import UserCardPreferences

            user_id = getattr(metadata, "user_id", "default")
            topic = getattr(metadata, "topic", "general")

            # Analyze generated cards for preferences
            if generation.cards:
                card_types = [card.card_type for card in generation.cards]
                dominant_card_type = max(set(card_types), key=card_types.count)

                # Determine difficulty based on content and quality
                difficulty = "medium"  # default
                if memorization_quality and memorization_quality.memorization_score:
                    score = memorization_quality.memorization_score
                    if score > 0.8:
                        difficulty = "easy"
                    elif score < 0.6:
                        difficulty = "hard"

                # Get existing preferences to update
                existing_prefs = (
                    await self.advanced_memory_store.get_user_card_preferences(
                        user_id, topic
                    )
                )

                # Update or create preferences
                if existing_prefs:
                    # Update existing preferences with new observations
                    confidence_boost = (
                        0.1
                        if memorization_quality
                        and memorization_quality.memorization_score > 0.7
                        else 0.05
                    )
                    new_confidence = min(
                        1.0, existing_prefs.confidence + confidence_boost
                    )

                    # Update dominant preferences
                    if card_types.count(dominant_card_type) > card_types.count(
                        existing_prefs.preferred_card_type
                    ):
                        preferred_type = dominant_card_type
                    else:
                        preferred_type = existing_prefs.preferred_card_type

                    updated_prefs = UserCardPreferences(
                        user_id=user_id,
                        topic=topic,
                        preferred_card_type=preferred_type,
                        preferred_difficulty=difficulty,
                        formatting_preferences=existing_prefs.formatting_preferences,
                        rejection_patterns=existing_prefs.rejection_patterns,
                        confidence=new_confidence,
                        last_updated=time.time(),
                        observation_count=existing_prefs.observation_count + 1,
                    )
                else:
                    # Create new preferences
                    updated_prefs = UserCardPreferences(
                        user_id=user_id,
                        topic=topic,
                        preferred_card_type=dominant_card_type,
                        preferred_difficulty=difficulty,
                        formatting_preferences={},
                        rejection_patterns=[],
                        confidence=0.6,  # Initial confidence
                        last_updated=time.time(),
                        observation_count=1,
                    )

                await self.advanced_memory_store.store_user_card_preferences(
                    updated_prefs
                )

                logger.info(
                    "user_preferences_learned",
                    user_id=user_id,
                    topic=topic,
                    preferred_type=updated_prefs.preferred_card_type,
                    confidence=updated_prefs.confidence,
                )

        except Exception as e:
            logger.warning(
                "user_preferences_learning_failed",
                error=str(e),
                error_type=type(e).__name__,
                user_id=getattr(metadata, "user_id", "default"),
                topic=getattr(metadata, "topic", "general"),
            )

    async def _store_memorization_feedback(
        self,
        metadata: NoteMetadata,
        generation: GenerationResult,
        memorization_quality: MemorizationQualityResult,
    ):
        """Store memorization quality feedback for learning.

        Args:
            metadata: Note metadata
            generation: Generation results
            memorization_quality: Memorization quality assessment
        """
        if not self.advanced_memory_store:
            return

        try:
            import time

            from obsidian_anki_sync.agents.advanced_memory import MemorizationFeedback

            # Store feedback for each card
            if generation.cards:
                for card in generation.cards:
                    feedback = MemorizationFeedback(
                        card_id=card.slug,  # Use slug as unique identifier
                        quality_score=memorization_quality.memorization_score,
                        issues_found=memorization_quality.issues,
                        strengths_identified=memorization_quality.strengths,
                        improvement_suggestions=memorization_quality.suggested_improvements,
                        topic=getattr(metadata, "topic", "general"),
                        card_type=card.card_type,
                        timestamp=time.time(),
                        metadata={
                            "title": getattr(metadata, "title", ""),
                            "language": getattr(card, "lang", "en"),
                            "card_index": getattr(card, "card_index", 0),
                            "tags": getattr(card, "tags", []),
                            "is_memorizable": memorization_quality.is_memorizable,
                        },
                    )

                    await self.advanced_memory_store.store_memorization_feedback(
                        feedback
                    )

                logger.info(
                    "memorization_feedback_stored",
                    cards_count=len(generation.cards),
                    quality_score=memorization_quality.quality_score,
                    topic=getattr(metadata, "topic", "general"),
                    issues_count=len(memorization_quality.issues),
                )

        except Exception as e:
            logger.warning(
                "memorization_feedback_storage_failed",
                error=str(e),
                error_type=type(e).__name__,
                cards_count=len(generation.cards) if generation.cards else 0,
                topic=getattr(metadata, "topic", "general"),
            )

    def _generate_slug_base(self, metadata: NoteMetadata) -> str:
        """Generate base slug from note metadata using collision-safe helper."""

        return generate_agent_slug_base(metadata)

    def _determine_entry_stage(self) -> str:
        """Determine the entry stage based on enabled features.

        Returns:
            The name of the first stage to execute in the pipeline.
        """
        # Autofix is always the first stage (permanent step)
        return "autofix"
