"""Pydantic models for Chain of Thought (CoT) reasoning outputs.

This module defines structured output models for reasoning nodes that
analyze state and provide recommendations before action nodes execute.
"""

from pydantic import BaseModel, Field


class ReasoningTraceOutput(BaseModel):
    """Base structured output from reasoning agent.

    All stage-specific reasoning outputs inherit from this base class.
    """

    reasoning: str = Field(
        description="Full chain of thought reasoning process"
    )
    key_observations: list[str] = Field(
        default_factory=list,
        description="Key insights from analyzing the current state",
    )
    planned_approach: str = Field(
        description="Description of the recommended approach for the next action"
    )
    potential_issues: list[str] = Field(
        default_factory=list,
        description="Anticipated problems or edge cases to handle",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Specific recommendations for the action node",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the reasoning (0.0-1.0)",
    )


class PreValidationReasoningOutput(ReasoningTraceOutput):
    """Reasoning specific to pre-validation stage."""

    structure_assessment: str = Field(
        default="",
        description="Assessment of note structure quality",
    )
    frontmatter_assessment: str = Field(
        default="",
        description="Assessment of YAML frontmatter completeness",
    )
    content_quality_assessment: str = Field(
        default="",
        description="Assessment of content quality and clarity",
    )
    validation_focus: list[str] = Field(
        default_factory=list,
        description="Areas to focus validation on",
    )


class GenerationReasoningOutput(ReasoningTraceOutput):
    """Reasoning specific to card generation stage."""

    card_type_recommendation: str = Field(
        default="",
        description="Recommended card type (Simple, Cloze, etc.)",
    )
    complexity_assessment: str = Field(
        default="",
        description="Assessment of content complexity",
    )
    qa_pair_analysis: list[dict] = Field(
        default_factory=list,
        description="Analysis of each Q&A pair",
    )
    formatting_recommendations: list[str] = Field(
        default_factory=list,
        description="Formatting recommendations for generated cards",
    )


class PostValidationReasoningOutput(ReasoningTraceOutput):
    """Reasoning specific to post-validation stage."""

    quality_concerns: list[str] = Field(
        default_factory=list,
        description="Quality concerns to check during validation",
    )
    validation_strategy: str = Field(
        default="",
        description="Recommended validation strategy",
    )
    expected_issues: list[str] = Field(
        default_factory=list,
        description="Expected validation issues based on generation output",
    )


class CardSplittingReasoningOutput(ReasoningTraceOutput):
    """Reasoning specific to card splitting stage."""

    complexity_indicators: list[str] = Field(
        default_factory=list,
        description="Indicators of content complexity",
    )
    split_recommendation: str = Field(
        default="",
        description="Recommendation on whether to split",
    )
    concept_boundaries: list[str] = Field(
        default_factory=list,
        description="Identified concept boundaries for splitting",
    )


class EnrichmentReasoningOutput(ReasoningTraceOutput):
    """Reasoning specific to context enrichment stage."""

    enrichment_opportunities: list[str] = Field(
        default_factory=list,
        description="Identified opportunities for enrichment",
    )
    mnemonic_suggestions: list[str] = Field(
        default_factory=list,
        description="Potential mnemonic devices",
    )
    example_types: list[str] = Field(
        default_factory=list,
        description="Types of examples that could help",
    )


class MemorizationReasoningOutput(ReasoningTraceOutput):
    """Reasoning specific to memorization quality stage."""

    retention_factors: list[str] = Field(
        default_factory=list,
        description="Factors affecting retention",
    )
    cognitive_load_assessment: str = Field(
        default="",
        description="Assessment of cognitive load",
    )


class DuplicateReasoningOutput(ReasoningTraceOutput):
    """Reasoning specific to duplicate detection stage."""

    similarity_indicators: list[str] = Field(
        default_factory=list,
        description="Indicators of potential similarity",
    )
    comparison_strategy: str = Field(
        default="",
        description="Strategy for comparing cards",
    )


class ReasoningTrace(BaseModel):
    """Serialized reasoning trace for state storage.

    This model represents a complete reasoning trace that can be
    stored in pipeline state and logged for observability.
    """

    stage: str = Field(description="Pipeline stage this reasoning is for")
    reasoning: str = Field(description="Full chain of thought reasoning")
    key_observations: list[str] = Field(default_factory=list)
    planned_approach: str = Field(default="")
    potential_issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_time: float = Field(
        default=0.0,
        description="Time taken for reasoning in seconds",
    )
    timestamp: float = Field(
        default=0.0,
        description="Unix timestamp when reasoning completed",
    )
    stage_specific_data: dict = Field(
        default_factory=dict,
        description="Additional stage-specific data from reasoning output",
    )

    @classmethod
    def from_output(
        cls,
        stage: str,
        output: ReasoningTraceOutput,
        reasoning_time: float,
        timestamp: float,
        stage_specific_data: dict | None = None,
    ) -> "ReasoningTrace":
        """Create a ReasoningTrace from a reasoning output.

        Args:
            stage: Pipeline stage name
            output: Reasoning output from the agent
            reasoning_time: Time taken for reasoning
            timestamp: Unix timestamp
            stage_specific_data: Additional stage-specific data

        Returns:
            ReasoningTrace instance
        """
        return cls(
            stage=stage,
            reasoning=output.reasoning,
            key_observations=output.key_observations,
            planned_approach=output.planned_approach,
            potential_issues=output.potential_issues,
            recommendations=output.recommendations,
            confidence=output.confidence,
            reasoning_time=reasoning_time,
            timestamp=timestamp,
            stage_specific_data=stage_specific_data or {},
        )
