"""State definition for LangGraph pipeline.

This module defines the TypedDict state structure used throughout the
card generation pipeline workflow.
"""

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages


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
    existing_cards_dicts: (
        list[dict] | None
    )  # Serialized existing cards for duplicate check

    # Pipeline stage results
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

    # Workflow control
    current_stage: Literal[
        "note_correction",
        "pre_validation",
        "card_splitting",
        "generation",
        "linter_validation",
        "post_validation",
        "context_enrichment",
        "memorization_quality",
        "duplicate_detection",
        "complete",
        "failed",
    ]
    enable_card_splitting: bool
    enable_context_enrichment: bool
    enable_memorization_quality: bool
    enable_duplicate_detection: bool
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
