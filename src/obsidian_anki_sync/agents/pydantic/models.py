"""Shared models for PydanticAI agents.

Contains all BaseModel output types and dependency models used across agents.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from obsidian_anki_sync.models import NoteMetadata, QAPair


class PreValidationOutput(BaseModel):
    """Structured output from pre-validation agent."""

    is_valid: bool = Field(description="Whether the note passes validation")
    error_type: str = Field(
        description="Type of error: format, structure, frontmatter, content, or none"
    )
    error_details: str = Field(default="", description="Detailed error description")
    suggested_fixes: list[str] = Field(
        default_factory=list, description="Suggested fixes for validation errors"
    )
    confidence: float = Field(
        default=0.5, description="Confidence in validation result"
    )


class CardGenerationOutput(BaseModel):
    """Structured output from card generation agent."""

    cards: list[dict[str, Any]] = Field(
        description="Generated cards with all APF fields"
    )
    total_generated: int = Field(ge=0, description="Total number of cards generated")
    generation_notes: str = Field(
        default="", description="Notes about the generation process"
    )
    confidence: float = Field(default=0.5, description="Overall generation confidence")


class PostValidationOutput(BaseModel):
    """Structured output from post-validation agent."""

    is_valid: bool = Field(description="Whether all cards pass validation")
    error_type: str = Field(
        description="Type of error: syntax, factual, semantic, template, or none"
    )
    error_details: str = Field(default="", description="Detailed validation errors")
    card_issues: list[dict[str, str]] = Field(
        default_factory=list,
        description="Per-card issues with card_index and issue description",
    )
    suggested_corrections: list[dict[str, Any]] = Field(
        default_factory=list, description="Suggested card corrections"
    )
    confidence: float = Field(
        default=0.5, description="Confidence in validation result"
    )


class MemorizationIssue(BaseModel):
    """A single memorization quality issue."""

    type: str = Field(
        description="Issue type (atomic_violation, information_leakage, etc.)"
    )
    severity: str = Field(description="Severity: low, medium, high")
    message: str = Field(description="Detailed description of the issue")


class MemorizationQualityOutput(BaseModel):
    """Structured output from memorization quality agent."""

    is_memorizable: bool = Field(
        description="Whether card is suitable for spaced repetition"
    )
    memorization_score: float = Field(
        default=0.5, description="Quality score for long-term retention"
    )
    issues: list[MemorizationIssue] = Field(
        default_factory=list, description="List of memorization issues found"
    )
    strengths: list[str] = Field(
        default_factory=list, description="What the card does well"
    )
    suggested_improvements: list[str] = Field(
        default_factory=list, description="Actionable improvements"
    )
    confidence: float = Field(default=0.5, description="Confidence in assessment")


class CardSplitPlanOutput(BaseModel):
    """Single card plan in split output."""

    card_number: int = Field(default=1, ge=1)
    concept: str
    question: str
    answer_summary: str
    rationale: str


class CardSplittingOutput(BaseModel):
    """Structured output from card splitting agent."""

    should_split: bool = Field(description="Whether to split into multiple cards")
    card_count: int = Field(default=1, ge=1, description="Number of cards to generate")
    splitting_strategy: str = Field(
        description="Strategy: none/concept/list/example/hierarchical/step/difficulty/prerequisite/context_aware"
    )
    split_plan: list[CardSplitPlanOutput] = Field(
        default_factory=list, description="Plan for each card"
    )
    reasoning: str = Field(description="Explanation of the decision")
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Decision confidence (0.0-1.0)"
    )
    fallback_strategy: str | None = Field(
        default=None, description="Fallback strategy if primary fails"
    )


class DuplicateMatchOutput(BaseModel):
    """Output model for a duplicate match."""

    card_slug: str = Field(min_length=1)
    similarity_score: float = Field(default=0.0)
    duplicate_type: str = Field(description="exact/semantic/partial_overlap/unique")
    reasoning: str = Field(default="")


class DuplicateDetectionOutput(BaseModel):
    """Structured output from duplicate detection agent."""

    is_duplicate: bool = Field(description="True if exact or semantic duplicate")
    similarity_score: float = Field(default=0.0)
    duplicate_type: str = Field(description="exact/semantic/partial_overlap/unique")
    reasoning: str = Field(description="Explanation of similarity assessment")
    recommendation: str = Field(description="delete/merge/keep_both/review_manually")
    better_card: str | None = Field(
        default=None, description="'new' or existing card slug if duplicate"
    )
    merge_suggestion: str | None = Field(default=None)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EnrichmentAdditionOutput(BaseModel):
    """Output model for an enrichment addition."""

    enrichment_type: str = Field(
        description="example/mnemonic/visual/related/practical"
    )
    content: str = Field(min_length=1)
    rationale: str = Field(default="")


class ContextEnrichmentOutput(BaseModel):
    """Structured output from context enrichment agent."""

    should_enrich: bool = Field(description="Whether card needs enrichment")
    enrichment_type: list[str] = Field(
        default_factory=list, description="Types of enrichment to add"
    )
    enriched_answer: str = Field(default="", description="Enhanced answer text")
    enriched_extra: str = Field(default="", description="Enhanced Extra section")
    additions_summary: str = Field(default="", description="Summary of additions")
    rationale: str = Field(default="", description="Why enrichment helps")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ============================================================================
# Agent Dependencies
# ============================================================================


class PreValidationDeps(BaseModel):
    """Dependencies for pre-validation agent."""

    note_content: str
    metadata: NoteMetadata
    qa_pairs: list[QAPair]
    file_path: Path | None = None


class GenerationDeps(BaseModel):
    """Dependencies for card generation agent."""

    note_content: str
    metadata: NoteMetadata
    qa_pairs: list[QAPair]
    slug_base: str


class PostValidationDeps(BaseModel):
    """Dependencies for post-validation agent."""

    cards: list  # list[GeneratedCard] - avoid circular import
    metadata: NoteMetadata
    strict_mode: bool = True


class CardSplittingDeps(BaseModel):
    """Dependencies for card splitting agent."""

    note_content: str
    metadata: NoteMetadata
    qa_pairs: list[QAPair]


class DuplicateDetectionDeps(BaseModel):
    """Dependencies for duplicate detection agent."""

    new_card_question: str
    new_card_answer: str
    existing_card_question: str
    existing_card_answer: str
    existing_card_slug: str


class ContextEnrichmentDeps(BaseModel):
    """Dependencies for context enrichment agent."""

    question: str
    answer: str
    extra: str
    card_slug: str
    note_title: str
