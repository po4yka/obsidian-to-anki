"""Structured output models for PydanticAI agents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from obsidian_anki_sync.agents.models import CardCorrection
from obsidian_anki_sync.apf.linter import validate_apf

from .streaming import _decode_html_encoded_apf


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
    confidence: float = Field(default=0.5, description="Confidence in validation result")


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

    @field_validator("cards")
    @classmethod
    def validate_apf_format(cls, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate that generated cards comply with APF format."""
        errors = []
        normalized_cards: list[dict[str, Any]] = []
        for i, card in enumerate(cards):
            apf_html_raw = card.get("apf_html", "")
            decoded_apf_html = _decode_html_encoded_apf(str(apf_html_raw))
            normalized_card = dict(card)
            normalized_card["apf_html"] = decoded_apf_html
            slug = card.get("slug")

            result = validate_apf(decoded_apf_html, slug)

            if result.errors:
                errors.append(f"Card {i + 1} ({slug}): {'; '.join(result.errors)}")
            normalized_cards.append(normalized_card)

        if errors:
            msg = f"APF Validation Failed: {'; '.join(errors)}"
            raise ValueError(msg)

        return normalized_cards


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
    suggested_corrections: list[CardCorrection] = Field(
        default_factory=list, description="Field-level card corrections"
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
        description=(
            "Strategy: none/concept/list/example/hierarchical/step/difficulty/"
            "prerequisite/context_aware/prerequisite_aware/cloze"
        )
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

