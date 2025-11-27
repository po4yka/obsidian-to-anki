"""State definition for LangGraph pipeline.

This module defines the TypedDict state structure used throughout the
card generation pipeline workflow.
"""

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

from ...config import Config


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
    config: Config  # Service configuration for model selection
    existing_cards_dicts: (
        list[dict] | None
    )  # Serialized existing cards for duplicate check

    # Cached models (for performance - created once during orchestrator init)
    pre_validator_model: Any | None  # PydanticAI OpenAIModel instance
    card_splitting_model: Any | None
    generator_model: Any | None
    post_validator_model: Any | None
    context_enrichment_model: Any | None
    memorization_quality_model: Any | None
    duplicate_detection_model: Any | None

    # Pipeline stage results
    note_correction: dict | None  # Serialized NoteCorrectionResult
    pre_validation: dict | None  # Serialized PreValidationResult
    card_splitting: dict | None  # Serialized CardSplittingResult
    generation: dict | None  # Serialized GenerationResult
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
