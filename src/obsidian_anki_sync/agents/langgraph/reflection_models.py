"""Pydantic models for Self-Reflection outputs.

This module defines structured output models for reflection nodes that
evaluate outputs AFTER action nodes execute and determine if revision is needed.

Self-Reflection pattern:
1. Action node produces output
2. Reflection node evaluates the output
3. If issues found and revision needed, trigger revision node
4. Revision node uses reflection feedback to improve output
5. Loop back to action node with revised input
"""

from pydantic import BaseModel, Field


class RevisionSuggestion(BaseModel):
    """A specific suggestion for revising the output."""

    issue: str = Field(
        description="The specific issue identified in the output"
    )
    severity: str = Field(
        default="medium",
        description="Severity of the issue: 'low', 'medium', 'high', 'critical'",
    )
    suggestion: str = Field(
        description="Specific suggestion for how to fix this issue"
    )
    affected_field: str = Field(
        default="",
        description="The field or section of the output affected by this issue",
    )


class ReflectionOutput(BaseModel):
    """Base structured output from self-reflection agent.

    All stage-specific reflection outputs inherit from this base class.
    """

    reflection: str = Field(
        description="Full self-reflection analysis of the output"
    )
    quality_assessment: str = Field(
        description="Overall quality assessment of the output"
    )
    issues_found: list[str] = Field(
        default_factory=list,
        description="Issues identified in the output",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Strengths identified in the output",
    )
    revision_suggestions: list[RevisionSuggestion] = Field(
        default_factory=list,
        description="Specific suggestions for revision",
    )
    revision_needed: bool = Field(
        default=False,
        description="Whether the output needs revision",
    )
    revision_priority: str = Field(
        default="none",
        description="Priority for revision: 'none', 'low', 'medium', 'high'",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the reflection assessment (0.0-1.0)",
    )
    cot_plan_followed: bool = Field(
        default=True,
        description="Whether the original CoT plan was followed (if available)",
    )
    cot_deviations: list[str] = Field(
        default_factory=list,
        description="Deviations from the original CoT plan",
    )


class GenerationReflectionOutput(ReflectionOutput):
    """Reflection specific to card generation output."""

    card_quality_scores: list[float] = Field(
        default_factory=list,
        description="Quality scores for each generated card (0.0-1.0)",
    )
    format_compliance: str = Field(
        default="",
        description="Assessment of APF format compliance",
    )
    content_accuracy: str = Field(
        default="",
        description="Assessment of content accuracy and completeness",
    )
    question_clarity: str = Field(
        default="",
        description="Assessment of question clarity and specificity",
    )
    answer_completeness: str = Field(
        default="",
        description="Assessment of answer completeness and correctness",
    )
    memorization_potential: str = Field(
        default="",
        description="Assessment of how well cards support memorization",
    )
    recommended_card_changes: list[dict] = Field(
        default_factory=list,
        description="Specific recommended changes per card [{card_index, changes}]",
    )


class EnrichmentReflectionOutput(ReflectionOutput):
    """Reflection specific to context enrichment output."""

    example_quality: str = Field(
        default="",
        description="Assessment of added examples quality",
    )
    mnemonic_effectiveness: str = Field(
        default="",
        description="Assessment of mnemonic device effectiveness",
    )
    context_relevance: str = Field(
        default="",
        description="Assessment of added context relevance",
    )
    enrichment_impact: str = Field(
        default="",
        description="Assessment of enrichment impact on learning",
    )
    over_enrichment_risk: bool = Field(
        default=False,
        description="Whether cards may be over-enriched (cognitive overload)",
    )
    recommended_enrichment_changes: list[str] = Field(
        default_factory=list,
        description="Specific recommended changes to enrichment",
    )


class ReflectionTrace(BaseModel):
    """Serialized reflection trace for state storage.

    This model represents a complete reflection trace that can be
    stored in pipeline state and logged for observability.
    """

    stage: str = Field(description="Pipeline stage this reflection is for")
    reflection: str = Field(description="Full reflection analysis")
    quality_assessment: str = Field(default="")
    issues_found: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    revision_suggestions: list[dict] = Field(default_factory=list)
    revision_needed: bool = Field(default=False)
    revision_priority: str = Field(default="none")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    cot_plan_followed: bool = Field(default=True)
    cot_deviations: list[str] = Field(default_factory=list)
    reflection_time: float = Field(
        default=0.0,
        description="Time taken for reflection in seconds",
    )
    timestamp: float = Field(
        default=0.0,
        description="Unix timestamp when reflection completed",
    )
    stage_specific_data: dict = Field(
        default_factory=dict,
        description="Additional stage-specific data from reflection output",
    )

    @classmethod
    def from_output(
        cls,
        stage: str,
        output: ReflectionOutput,
        reflection_time: float,
        timestamp: float,
        stage_specific_data: dict | None = None,
    ) -> "ReflectionTrace":
        """Create a ReflectionTrace from a reflection output.

        Args:
            stage: Pipeline stage name
            output: Reflection output from the agent
            reflection_time: Time taken for reflection
            timestamp: Unix timestamp
            stage_specific_data: Additional stage-specific data

        Returns:
            ReflectionTrace instance
        """
        return cls(
            stage=stage,
            reflection=output.reflection,
            quality_assessment=output.quality_assessment,
            issues_found=output.issues_found,
            strengths=output.strengths,
            revision_suggestions=[
                s.model_dump() for s in output.revision_suggestions
            ],
            revision_needed=output.revision_needed,
            revision_priority=output.revision_priority,
            confidence=output.confidence,
            cot_plan_followed=output.cot_plan_followed,
            cot_deviations=output.cot_deviations,
            reflection_time=reflection_time,
            timestamp=timestamp,
            stage_specific_data=stage_specific_data or {},
        )


class RevisionInput(BaseModel):
    """Input for revision node containing reflection feedback."""

    original_output: dict = Field(
        description="The original output that needs revision"
    )
    reflection_trace: dict = Field(
        description="The reflection trace with issues and suggestions"
    )
    cot_reasoning: dict | None = Field(
        default=None,
        description="Original CoT reasoning for context (if available)",
    )
    revision_focus: list[str] = Field(
        default_factory=list,
        description="Specific areas to focus revision on",
    )
    max_changes: int = Field(
        default=5,
        description="Maximum number of changes to make in this revision",
    )


class RevisionOutput(BaseModel):
    """Output from a revision node."""

    revised_output: dict = Field(
        description="The revised output after applying changes"
    )
    changes_made: list[str] = Field(
        default_factory=list,
        description="Description of changes made during revision",
    )
    issues_addressed: list[str] = Field(
        default_factory=list,
        description="Issues from reflection that were addressed",
    )
    issues_remaining: list[str] = Field(
        default_factory=list,
        description="Issues that could not be addressed in this revision",
    )
    revision_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence that revision improved the output",
    )
    further_revision_recommended: bool = Field(
        default=False,
        description="Whether another revision pass is recommended",
    )
