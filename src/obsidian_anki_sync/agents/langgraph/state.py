"""State definition and helpers for the LangGraph pipeline.

This module defines the TypedDict state structure used throughout the
card generation pipeline workflow and provides helpers for accessing
runtime-only resources (Config, model factory, selectors) without putting
non-serializable objects into the persisted state.
"""

from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph.message import add_messages

from obsidian_anki_sync.agents.langgraph.model_factory import ModelFactory
from obsidian_anki_sync.config import Config


class PipelineState(TypedDict):
    """State for the card generation pipeline workflow.

    This state is passed between nodes and tracks the entire pipeline execution.

    Best practices applied:
    - Keep state minimal and explicit
    - Use TypedDict for consistency
    - Track step count for cycle protection
    - Include error context for debugging
    """

    # Input data
    note_content: str
    metadata_dict: dict  # Serialized NoteMetadata
    qa_pairs_dicts: list[dict]  # Serialized QAPair list
    file_path: str | None
    slug_base: str
    # Lookup key for runtime resources (config, models, selectors)
    runtime_key: str
    config_snapshot: dict  # JSON-serializable config snapshot for reference
    existing_cards_dicts: (
        list[dict] | None
    )  # Serialized existing cards for duplicate check

    # Cached model names (models retrieved from runtime cache for execution)
    pre_validator_model: str | None  # Model name for pre-validation
    card_splitting_model: str | None
    generator_model: str | None
    post_validator_model: str | None
    context_enrichment_model: str | None
    memorization_quality_model: str | None
    duplicate_detection_model: str | None
    highlight_model: str | None

    # Pipeline stage results
    autofix: dict | None  # Serialized AutoFixResult
    note_correction: dict | None  # Serialized NoteCorrectionResult
    pre_validation: dict | None  # Serialized PreValidationResult
    card_splitting: dict | None  # Serialized CardSplittingResult
    generation: dict | None  # Serialized GenerationResult
    linter_valid: bool  # Did cards pass deterministic APF linting?
    # Per-card linter results: [{slug, is_valid, errors, warnings}]
    linter_results: list[dict]
    post_validation: dict | None  # Serialized PostValidationResult
    context_enrichment: dict | None  # Serialized ContextEnrichmentResult
    memorization_quality: dict | None  # Serialized MemorizationQualityResult
    # Serialized DuplicateDetectionResult (per card)
    duplicate_detection: dict | None
    highlight_result: dict | None  # Serialized HighlightResult

    # Workflow control
    current_stage: Literal[
        "autofix",
        "note_correction",
        "pre_validation",
        "card_splitting",
        "generation",
        "linter_validation",
        "post_validation",
        "context_enrichment",
        "memorization_quality",
        "duplicate_detection",
        "highlight",
        "complete",
        "failed",
    ]
    # Auto-fix always runs (permanent step) - configurable options only
    autofix_write_back: bool  # Write fixes back to source files
    # List of enabled handler types (None = all)
    autofix_handlers: list[str] | None
    enable_card_splitting: bool
    enable_context_enrichment: bool
    enable_memorization_quality: bool
    enable_duplicate_detection: bool
    enable_highlight_agent: bool
    retry_count: int
    max_retries: int
    auto_fix_enabled: bool
    strict_mode: bool

    # Cycle protection (best practice: add hard stops for bounded cycles)
    step_count: int  # Current number of steps executed
    max_steps: int  # Maximum allowed steps (prevents infinite loops)

    # Error tracking (best practice: track errors for debugging and routing)
    last_error: str | None  # Last error message
    # Error severity (critical/recoverable/warning)
    last_error_severity: str | None
    # History of errors: [{stage, error, severity, timestamp}]
    errors: list[dict]

    # Timing
    start_time: float
    stage_times: dict[str, float]

    # Messages (for debugging/logging)
    messages: Annotated[list[str], add_messages]

    # ============================================================================
    # Chain of Thought (CoT) Configuration
    # ============================================================================
    enable_cot_reasoning: bool  # Master toggle for CoT reasoning
    store_reasoning_traces: bool  # Store traces in state for inspection
    log_reasoning_traces: bool  # Log traces to logger (can be verbose)
    cot_enabled_stages: list[str]  # Stages where CoT is applied

    # Reasoning Model (cached for performance)
    reasoning_model: str | None  # Model name for reasoning

    # Reasoning Traces (per stage)
    # Structure: {stage_name: ReasoningTrace.model_dump()}
    reasoning_traces: dict[str, dict]

    # Current reasoning context (passed to action nodes)
    # Contains the latest reasoning output for the current stage
    current_reasoning: dict | None

    # ============================================================================
    # Self-Reflection Configuration
    # ============================================================================
    enable_self_reflection: bool  # Master toggle for self-reflection
    store_reflection_traces: bool  # Store traces in state for inspection
    log_reflection_traces: bool  # Log traces to logger (can be verbose)
    reflection_enabled_stages: list[str]  # Stages where reflection is applied

    # Reflection Model (cached for performance)
    reflection_model: str | None  # Model name for reflection

    # Reflection Traces (per stage)
    # Structure: {stage_name: ReflectionTrace.model_dump()}
    reflection_traces: dict[str, dict]

    # Current reflection context (passed to revision nodes)
    # Contains the latest reflection output for the current stage
    current_reflection: dict | None

    # Revision Control
    revision_count: int  # Current number of revisions in this pass
    max_revisions: int  # Maximum revisions allowed per stage
    # Tracks revisions per stage: {stage_name: count}
    stage_revision_counts: dict[str, int]

    # Domain Detection and Smart Skipping
    detected_domain: str | None  # Detected content domain for specialized reflection
    reflection_skipped: bool  # Whether reflection was skipped for this content
    reflection_skip_reason: str | None  # Reason for skipping reflection
    # Selected revision strategy (light_edit, moderate_revision, major_rewrite)
    revision_strategy: str | None

    # ============================================================================
    # RAG (Retrieval-Augmented Generation) Configuration
    # ============================================================================
    enable_rag: bool  # Master toggle for RAG features
    rag_context_enrichment: bool  # Use RAG for context enrichment
    rag_duplicate_detection: bool  # Use RAG for duplicate detection
    rag_few_shot_examples: bool  # Use RAG for few-shot examples

    # RAG Integration (runtime cache key)
    rag_integration: str | None  # RAGIntegration runtime key

    # RAG Results (per-stage outputs)
    rag_enrichment: dict | None  # RAG context enrichment data
    rag_examples: list[dict] | None  # Few-shot examples from RAG
    # RAG-based duplicate check results
    rag_duplicate_results: list[dict] | None

    # Unified agent framework configuration
    # Agent framework to use ("pydantic_ai", "langchain", etc.)
    agent_framework: str
    agent_selector: str | None  # Runtime cache key for agent selector
    split_validator_model: str | None  # Model name for split validation


# =============================================================================
# Runtime resource cache
# =============================================================================

_RUNTIME_RESOURCES: dict[str, dict[str, Any]] = {}


def register_runtime_resources(
    config: Config,
    model_factory: ModelFactory,
    agent_selector: Any | None = None,
    rag_integration: Any | None = None,
) -> str:
    """Register runtime-only resources and return the lookup key.

    These objects are intentionally kept out of the serialized PipelineState to
    keep LangGraph checkpoints JSON-serializable. Nodes can retrieve them using
    the runtime key stored in the state.
    """

    runtime_key = str(uuid4())
    _RUNTIME_RESOURCES[runtime_key] = {
        "config": config,
        "model_factory": model_factory,
        "agent_selector": agent_selector,
        "rag_integration": rag_integration,
    }
    return runtime_key


def _get_resource(state: PipelineState, name: str) -> Any:
    runtime_key = state.get("runtime_key")
    if not runtime_key:
        return None

    resources = _RUNTIME_RESOURCES.get(runtime_key)
    if not resources:
        return None

    return resources.get(name)


def get_config(state: PipelineState) -> Config:
    """Return the Config object associated with the current state."""

    config = _get_resource(state, "config")
    if config is None:
        error_message = "Config unavailable for runtime key"
        raise KeyError(error_message)
    return config


def get_model(state: PipelineState, agent_type: str) -> Any:
    """Fetch a model by agent type using the runtime model factory."""

    factory: ModelFactory | None = _get_resource(state, "model_factory")
    if factory is None:
        return None
    return factory.get_model(agent_type)


def get_agent_selector(state: PipelineState) -> Any:
    """Retrieve the agent selector from the runtime cache."""

    return _get_resource(state, "agent_selector")


def get_rag_integration(state: PipelineState) -> Any:
    """Retrieve the RAG integration instance from the runtime cache."""

    return _get_resource(state, "rag_integration")
