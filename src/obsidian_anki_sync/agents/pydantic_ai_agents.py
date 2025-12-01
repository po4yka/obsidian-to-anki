"""PydanticAI-based agent implementations for card generation pipeline.

This module provides type-safe agents using PydanticAI for:
1. Pre-validation - note structure and format checking
2. Card generation - converting Q/A pairs to APF cards
3. Post-validation - quality and accuracy checking

All agents use structured outputs and proper type validation.
"""

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from obsidian_anki_sync.apf.linter import validate_apf
from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.utils.content_hash import compute_content_hash
from obsidian_anki_sync.utils.logging import get_logger

from .card_splitting_prompts import CARD_SPLITTING_DECISION_PROMPT
from .context_enrichment_prompts import CONTEXT_ENRICHMENT_PROMPT
from .duplicate_detection_prompts import DUPLICATE_DETECTION_PROMPT
from .exceptions import (
    GenerationError,
    ModelError,
    PostValidationError,
    PreValidationError,
    StructuredOutputError,
)
from .improved_prompts import (
    CARD_GENERATION_SYSTEM_PROMPT,
    POST_VALIDATION_SYSTEM_PROMPT,
    PRE_VALIDATION_SYSTEM_PROMPT,
)
from .memorization_prompts import MEMORIZATION_QUALITY_PROMPT
from .models import (
    CardCorrection,
    CardSplitPlan,
    CardSplittingResult,
    ContextEnrichmentResult,
    DuplicateDetectionResult,
    DuplicateMatch,
    EnrichmentAddition,
    GeneratedCard,
    GenerationResult,
    MemorizationQualityResult,
    PostValidationResult,
    PreValidationResult,
)
from .patch_applicator import apply_corrections

logger = get_logger(__name__)


# ============================================================================
# Result Types for PydanticAI Agents
# ============================================================================


class PreValidationOutput(BaseModel):
    """Structured output from pre-validation agent."""

    is_valid: bool = Field(description="Whether the note passes validation")
    error_type: str = Field(
        description="Type of error: format, structure, frontmatter, content, or none"
    )
    error_details: str = Field(
        default="", description="Detailed error description")
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
    total_generated: int = Field(
        ge=0, description="Total number of cards generated")
    generation_notes: str = Field(
        default="", description="Notes about the generation process"
    )
    confidence: float = Field(
        default=0.5, description="Overall generation confidence")

    @field_validator("cards")
    @classmethod
    def validate_apf_format(cls, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate that generated cards comply with APF format."""
        errors = []
        for i, card in enumerate(cards):
            apf_html = card.get("apf_html", "")
            slug = card.get("slug")

            # Run deterministic linter
            result = validate_apf(apf_html, slug)

            if result.errors:
                errors.append(
                    f"Card {i + 1} ({slug}): {'; '.join(result.errors)}")

        if errors:
            msg = f"APF Validation Failed: {'; '.join(errors)}"
            raise ValueError(msg)

        return cards


class PostValidationOutput(BaseModel):
    """Structured output from post-validation agent."""

    is_valid: bool = Field(description="Whether all cards pass validation")
    error_type: str = Field(
        description="Type of error: syntax, factual, semantic, template, or none"
    )
    error_details: str = Field(
        default="", description="Detailed validation errors")
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
    confidence: float = Field(
        default=0.5, description="Confidence in assessment")


class CardSplitPlanOutput(BaseModel):
    """Single card plan in split output."""

    card_number: int = Field(default=1, ge=1)
    concept: str
    question: str
    answer_summary: str
    rationale: str


class CardSplittingOutput(BaseModel):
    """Structured output from card splitting agent."""

    should_split: bool = Field(
        description="Whether to split into multiple cards")
    card_count: int = Field(
        default=1, ge=1, description="Number of cards to generate")
    splitting_strategy: str = Field(
        description="Strategy: none/concept/list/example/hierarchical/step/difficulty/prerequisite/context_aware/prerequisite_aware/cloze"
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

    cards: list[GeneratedCard]
    metadata: NoteMetadata
    strict_mode: bool = True


class CardSplittingDeps(BaseModel):
    """Dependencies for card splitting agent."""

    note_content: str
    metadata: NoteMetadata
    qa_pairs: list[QAPair]


# ============================================================================
# PydanticAI Agent Implementations
# ============================================================================


class PreValidatorAgentAI:
    """PydanticAI-based pre-validation agent.

    Validates note structure, formatting, and frontmatter before generation.
    Uses structured outputs for type-safe validation results.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.0):
        """Initialize pre-validator agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model = model
        self.temperature = temperature

        # Create PydanticAI agent with structured output
        self.agent: Agent[PreValidationDeps, PreValidationOutput] = Agent(
            model=self.model,
            output_type=PreValidationOutput,
            system_prompt=self._get_system_prompt(),
        )

        logger.info("pydantic_ai_pre_validator_initialized", model=str(model))

    def _get_system_prompt(self) -> str:
        """Get system prompt for pre-validation with few-shot examples."""
        return PRE_VALIDATION_SYSTEM_PROMPT

    async def validate(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        file_path: Path | None = None,
    ) -> PreValidationResult:
        """Validate note before card generation.

        Args:
            note_content: Full note content
            metadata: Parsed metadata
            qa_pairs: Parsed Q/A pairs
            file_path: Optional file path

        Returns:
            PreValidationResult with validation outcome
        """
        logger.info("pydantic_ai_pre_validation_start", title=metadata.title)

        # Create dependencies
        deps = PreValidationDeps(
            note_content=note_content,
            metadata=metadata,
            qa_pairs=qa_pairs,
            file_path=file_path,
        )

        # Build validation prompt
        prompt = f"""Validate this note for card generation:

Title: {metadata.title}
Topic: {metadata.topic}
Tags: {", ".join(metadata.tags)}
Language Tags: {", ".join(metadata.language_tags)}
Q&A Pairs: {len(qa_pairs)}

Note Content Preview:
{note_content[:500]}...

Validate the structure, frontmatter, and content quality."""

        try:
            # Run agent
            result = await self.agent.run(prompt, deps=deps)

            # Convert to PreValidationResult
            # result.data is typed as PreValidationOutput by pydantic-ai
            output: PreValidationOutput = result.data
            validation_result = PreValidationResult(
                is_valid=output.is_valid,
                error_type=output.error_type,
                error_details=output.error_details,
                auto_fix_applied=False,
                fixed_content=None,
                validation_time=0.0,  # Would track timing in production
            )

            logger.info(
                "pydantic_ai_pre_validation_complete",
                is_valid=output.is_valid,
                confidence=output.confidence,
            )

            return validation_result

        except ValueError as e:
            # Structured output parsing error
            logger.error(
                "pydantic_ai_pre_validation_parse_error", error=str(e))
            msg = "Failed to parse pre-validation output"
            raise StructuredOutputError(
                msg,
                details={"error": str(e), "title": metadata.title},
            ) from e
        except TimeoutError as e:
            logger.error("pydantic_ai_pre_validation_timeout", error=str(e))
            msg = "Pre-validation timed out"
            raise ModelError(msg, details={"title": metadata.title}) from e
        except Exception as e:
            logger.error("pydantic_ai_pre_validation_failed", error=str(e))
            msg = f"Pre-validation failed: {e!s}"
            raise PreValidationError(
                msg, details={"title": metadata.title}) from e


class GeneratorAgentAI:
    """PydanticAI-based card generation agent.

    Generates APF cards from Q/A pairs using structured outputs.
    Ensures type-safe card generation with validation.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.3):
        """Initialize generator agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature for creativity
        """
        self.model = model
        self.temperature = temperature

        # Use improved system prompt with few-shot examples
        self.system_prompt = CARD_GENERATION_SYSTEM_PROMPT

        # Create PydanticAI agent
        self.agent: Agent[GenerationDeps, CardGenerationOutput] = Agent(
            model=self.model,
            output_type=CardGenerationOutput,
            system_prompt=self.system_prompt,
        )

        logger.info("pydantic_ai_generator_initialized", model=str(model))

    async def generate_cards(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        slug_base: str,
    ) -> GenerationResult:
        """Generate APF cards from Q/A pairs.

        Args:
            note_content: Full note content for context
            metadata: Note metadata
            qa_pairs: Q/A pairs to convert
            slug_base: Base slug for card identifiers

        Returns:
            GenerationResult with all generated cards
        """
        logger.info(
            "pydantic_ai_generation_start",
            title=metadata.title,
            qa_count=len(qa_pairs),
        )

        # Create dependencies
        deps = GenerationDeps(
            note_content=note_content,
            metadata=metadata,
            qa_pairs=qa_pairs,
            slug_base=slug_base,
        )

        # Build generation prompt
        prompt = f"""Generate APF cards for these Q&A pairs:

Title: {metadata.title}
Topic: {metadata.topic}
Languages: {", ".join(metadata.language_tags)}
Slug Base: {slug_base}

Q&A Pairs ({len(qa_pairs)}):
"""
        for idx, qa in enumerate(qa_pairs, 1):
            prompt += f"\n{idx}. Q: {qa.question_en[:100]}...\n   A: {qa.answer_en[:100]}...\n"

        prompt += (
            "\nGenerate complete APF HTML cards for all Q&A pairs in all languages."
        )

        try:
            # Run agent
            result = await self.agent.run(prompt, deps=deps)
            output: CardGenerationOutput = result.data

            # Convert cards to GeneratedCard instances with content hashes
            generated_cards: list[GeneratedCard] = []
            qa_lookup = {qa.card_index: qa for qa in qa_pairs}
            for card_dict in output.cards:
                try:
                    card_index = card_dict["card_index"]
                    lang = card_dict["lang"]
                    qa_pair = qa_lookup.get(card_index)
                    content_hash = ""
                    if qa_pair is not None:
                        content_hash = compute_content_hash(
                            qa_pair, metadata, lang)

                    generated_card = GeneratedCard(
                        card_index=card_index,
                        slug=card_dict["slug"],
                        lang=lang,
                        apf_html=card_dict["apf_html"],
                        confidence=card_dict.get("confidence", output.confidence),
                        content_hash=content_hash,
                    )
                    generated_cards.append(generated_card)
                except (KeyError, ValueError) as e:
                    logger.warning(
                        "invalid_generated_card", error=str(e), card=card_dict
                    )
                    continue

            if not generated_cards:
                msg = "No valid cards generated"
                raise GenerationError(
                    msg,
                    details={
                        "title": metadata.title,
                        "qa_pairs_count": len(qa_pairs),
                        "raw_cards_count": len(output.cards),
                    },
                )

            generation_result = GenerationResult(
                cards=generated_cards,
                total_cards=len(generated_cards),
                generation_time=0.0,  # Would track timing
                model_used=str(self.model),
            )

            logger.info(
                "pydantic_ai_generation_complete",
                cards_generated=len(generated_cards),
                confidence=output.confidence,
            )

            return generation_result

        except GenerationError:
            raise  # Re-raise our custom error
        except ValueError as e:
            logger.error("pydantic_ai_generation_parse_error", error=str(e))
            msg = "Failed to parse generation output"
            raise StructuredOutputError(
                msg,
                details={"error": str(e), "title": metadata.title},
            ) from e
        except TimeoutError as e:
            logger.error("pydantic_ai_generation_timeout", error=str(e))
            msg = "Card generation timed out"
            raise ModelError(msg, details={"title": metadata.title}) from e
        except Exception as e:
            logger.error("pydantic_ai_generation_failed", error=str(e))
            msg = f"Card generation failed: {e!s}"
            raise GenerationError(
                msg, details={"title": metadata.title}) from e


class PostValidatorAgentAI:
    """PydanticAI-based post-validation agent.

    Validates generated cards for quality, syntax, and accuracy.
    Uses structured outputs to identify and suggest corrections.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.0):
        """Initialize post-validator agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model = model
        self.temperature = temperature

        # Create PydanticAI agent
        self.agent: Agent[PostValidationDeps, PostValidationOutput] = Agent(
            model=self.model,
            output_type=PostValidationOutput,
            system_prompt=self._get_system_prompt(),
        )

        logger.info("pydantic_ai_post_validator_initialized", model=str(model))

    def _get_system_prompt(self) -> str:
        """Get system prompt for post-validation with few-shot examples."""
        return POST_VALIDATION_SYSTEM_PROMPT

    async def validate(
        self,
        cards: list[GeneratedCard],
        metadata: NoteMetadata,
        strict_mode: bool = True,
    ) -> PostValidationResult:
        """Validate generated cards.

        Args:
            cards: Generated cards to validate
            metadata: Note metadata for context
            strict_mode: Enable strict validation

        Returns:
            PostValidationResult with validation outcome
        """
        logger.info("pydantic_ai_post_validation_start",
                    cards_count=len(cards))

        # Create dependencies
        deps = PostValidationDeps(
            cards=cards, metadata=metadata, strict_mode=strict_mode
        )

        # Build validation prompt with FULL card content
        prompt = f"""Validate these {len(cards)} generated APF cards:

Metadata:
- Title: {metadata.title}
- Topic: {metadata.topic}
- Languages: {", ".join(metadata.language_tags)}

Cards to validate:
"""
        # Include full HTML content for proper validation (up to 3 cards to avoid token limits)
        for card in cards[:3]:
            prompt += f"\n--- Card {card.card_index} ({card.lang}): {card.slug} ---\n"
            prompt += f"```html\n{card.apf_html}\n```\n"

        if len(cards) > 3:
            prompt += f"\n(Plus {len(cards) - 3} more cards with similar structure)\n"

        prompt += (
            f"\nValidate all {len(cards)} cards for APF v2.1 compliance, correctness, and quality."
        )

        try:
            # Run agent
            result = await self.agent.run(prompt, deps=deps)
            output: PostValidationOutput = result.data

            # Apply suggested corrections if any
            corrected_cards = None
            applied_changes: list[str] = []

            if output.suggested_corrections:
                corrected_cards, applied_changes = apply_corrections(
                    cards, output.suggested_corrections
                )
                logger.info(
                    "post_validation_corrections_applied",
                    suggested_count=len(output.suggested_corrections),
                    applied_count=len(applied_changes),
                )

            validation_result = PostValidationResult(
                is_valid=output.is_valid,
                error_type=output.error_type,
                error_details=output.error_details,
                suggested_corrections=output.suggested_corrections,
                corrected_cards=corrected_cards,
                applied_changes=applied_changes,
                validation_time=0.0,
            )

            logger.info(
                "pydantic_ai_post_validation_complete",
                is_valid=output.is_valid,
                confidence=output.confidence,
                issues_found=len(output.card_issues),
                corrections_applied=len(applied_changes),
            )

            return validation_result

        except ValueError as e:
            logger.error(
                "pydantic_ai_post_validation_parse_error", error=str(e))
            msg = "Failed to parse post-validation output"
            raise StructuredOutputError(
                msg,
                details={"error": str(e), "cards_count": len(cards)},
            ) from e
        except TimeoutError as e:
            logger.error("pydantic_ai_post_validation_timeout", error=str(e))
            msg = "Post-validation timed out"
            raise ModelError(msg, details={"cards_count": len(cards)}) from e
        except Exception as e:
            logger.error("pydantic_ai_post_validation_failed", error=str(e))
            msg = f"Post-validation failed: {e!s}"
            raise PostValidationError(
                msg, details={"cards_count": len(cards)}) from e


class MemorizationQualityAgentAI:
    """PydanticAI-based memorization quality agent.

    Evaluates whether generated cards are effective for spaced repetition
    and long-term memory retention.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.0):
        """Initialize memorization quality agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model = model
        self.temperature = temperature

        # Create PydanticAI agent
        self.agent: Agent[PostValidationDeps, MemorizationQualityOutput] = Agent(
            model=self.model,
            output_type=MemorizationQualityOutput,
            system_prompt=MEMORIZATION_QUALITY_PROMPT,
        )

        logger.info("pydantic_ai_memorization_agent_initialized",
                    model=str(model))

    async def assess(
        self,
        cards: list[GeneratedCard],
        metadata: NoteMetadata,
    ) -> MemorizationQualityResult:
        """Assess memorization quality of generated cards.

        Args:
            cards: Generated cards to assess
            metadata: Note metadata for context

        Returns:
            MemorizationQualityResult with assessment
        """
        logger.info("pydantic_ai_memorization_assessment_start",
                    cards_count=len(cards))

        # Create dependencies (reuse PostValidationDeps)
        deps = PostValidationDeps(
            cards=cards, metadata=metadata, strict_mode=True)

        # Build assessment prompt
        prompt = f"""Assess memorization quality of these {len(cards)} Anki cards:

Metadata:
- Title: {metadata.title}
- Topic: {metadata.topic}
- Languages: {", ".join(metadata.language_tags)}

Cards to assess:
"""
        for card in cards[:5]:  # Show first 5 for context
            prompt += f"\nCard {card.card_index} ({card.lang}): {card.slug}\n"
            front_match = card.apf_html.split("<!-- Front -->")
            if len(front_match) > 1:
                front_text = front_match[1].split(
                    "<!-- Back -->")[0].strip()[:150]
                prompt += f"Front: {front_text}...\n"
            back_match = card.apf_html.split("<!-- Back -->")
            if len(back_match) > 1:
                back_text = back_match[1].split("<!--")[0].strip()[:150]
                prompt += f"Back: {back_text}...\n"

        prompt += f"\nAssess all {len(cards)} cards for memorization effectiveness."

        try:
            # Run agent
            result = await self.agent.run(prompt, deps=deps)
            output: MemorizationQualityOutput = result.data

            # Convert to MemorizationQualityResult
            issues_list = [issue.model_dump() for issue in output.issues]

            quality_result = MemorizationQualityResult(
                is_memorizable=output.is_memorizable,
                memorization_score=output.memorization_score,
                issues=issues_list,
                strengths=output.strengths,
                suggested_improvements=output.suggested_improvements,
                assessment_time=0.0,
            )

            logger.info(
                "pydantic_ai_memorization_assessment_complete",
                is_memorizable=output.is_memorizable,
                score=output.memorization_score,
                issues_found=len(output.issues),
            )

            return quality_result

        except ValueError as e:
            logger.error("pydantic_ai_memorization_parse_error", error=str(e))
            msg = "Failed to parse memorization assessment output"
            raise StructuredOutputError(
                msg,
                details={"error": str(e), "cards_count": len(cards)},
            ) from e
        except TimeoutError as e:
            logger.error("pydantic_ai_memorization_timeout", error=str(e))
            msg = "Memorization assessment timed out"
            raise ModelError(msg, details={"cards_count": len(cards)}) from e
        except Exception as e:
            logger.error("pydantic_ai_memorization_failed", error=str(e))
            # Return permissive result instead of failing
            logger.warning("memorization_agent_fallback", error=str(e))
            return MemorizationQualityResult(
                is_memorizable=True,  # Assume cards are okay if agent fails
                memorization_score=0.7,  # Default acceptable score
                issues=[],
                strengths=[],
                suggested_improvements=[f"Memorization agent failed: {e!s}"],
                assessment_time=0.0,
            )


class CardSplittingAgentAI:
    """PydanticAI-based card splitting agent.

    Determines whether a note should generate one card or multiple cards,
    and provides a splitting plan if needed.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.0):
        """Initialize card splitting agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model = model
        self.temperature = temperature

        # Create PydanticAI agent
        self.agent: Agent[CardSplittingDeps, CardSplittingOutput] = Agent(
            model=self.model,
            output_type=CardSplittingOutput,
            system_prompt=CARD_SPLITTING_DECISION_PROMPT,
        )

        logger.info("pydantic_ai_card_splitting_agent_initialized",
                    model=str(model))

    async def analyze(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
    ) -> CardSplittingResult:
        """Analyze note and determine if splitting is needed.

        Args:
            note_content: Full note content
            metadata: Note metadata
            qa_pairs: Parsed Q/A pairs

        Returns:
            CardSplittingResult with splitting decision and plan
        """
        logger.info(
            "pydantic_ai_card_splitting_start",
            title=metadata.title,
            qa_count=len(qa_pairs),
        )

        # Create dependencies
        deps = CardSplittingDeps(
            note_content=note_content, metadata=metadata, qa_pairs=qa_pairs
        )

        # Build analysis prompt
        prompt = f"""Analyze this note and determine if it should generate one or multiple cards:

Title: {metadata.title}
Topic: {metadata.topic}
Q&A Pairs: {len(qa_pairs)}

Content Preview:
{note_content[:800]}...

Questions:
"""
        for idx, qa in enumerate(qa_pairs[:3], 1):
            prompt += f"\n{idx}. {qa.question_en[:150]}"
            if len(qa.question_en) > 150:
                prompt += "..."

        prompt += "\n\nShould this note generate one card or split into multiple cards? Provide a detailed splitting plan."

        try:
            import time

            start_time = time.time()

            # Run agent
            result = await self.agent.run(prompt, deps=deps)
            output: CardSplittingOutput = result.data

            # Convert split plan
            split_plans = []
            for plan_output in output.split_plan:
                split_plan = CardSplitPlan(
                    card_number=plan_output.card_number,
                    concept=plan_output.concept,
                    question=plan_output.question,
                    answer_summary=plan_output.answer_summary,
                    rationale=plan_output.rationale,
                )
                split_plans.append(split_plan)

            # Create result with confidence and fallback
            splitting_result = CardSplittingResult(
                should_split=output.should_split,
                card_count=output.card_count,
                # type: ignore[arg-type]
                splitting_strategy=output.splitting_strategy,
                split_plan=split_plans,
                reasoning=output.reasoning,
                decision_time=time.time() - start_time,
                confidence=output.confidence,
                fallback_strategy=output.fallback_strategy,
            )

            logger.info(
                "pydantic_ai_card_splitting_complete",
                should_split=output.should_split,
                card_count=output.card_count,
                strategy=output.splitting_strategy,
                confidence=output.confidence,
                fallback_strategy=output.fallback_strategy,
            )

            # Log warning if confidence is low
            if output.confidence < 0.6:
                logger.warning(
                    "card_splitting_low_confidence",
                    confidence=output.confidence,
                    strategy=output.splitting_strategy,
                    fallback=output.fallback_strategy,
                )

            return splitting_result

        except ValueError as e:
            logger.error(
                "pydantic_ai_card_splitting_parse_error", error=str(e))
            msg = "Failed to parse card splitting output"
            raise StructuredOutputError(
                msg,
                details={"error": str(e), "title": metadata.title},
            ) from e
        except TimeoutError as e:
            logger.error("pydantic_ai_card_splitting_timeout", error=str(e))
            msg = "Card splitting analysis timed out"
            raise ModelError(msg, details={"title": metadata.title}) from e
        except Exception as e:
            logger.error("pydantic_ai_card_splitting_failed", error=str(e))
            # Return conservative fallback (no split)
            logger.warning("card_splitting_agent_fallback", error=str(e))
            return CardSplittingResult(
                should_split=False,
                card_count=1,
                splitting_strategy="none",
                split_plan=[
                    CardSplitPlan(
                        card_number=1,
                        concept=metadata.title,
                        question=qa_pairs[0].question_en if qa_pairs else "Question",
                        answer_summary=(
                            qa_pairs[0].answer_en[:100] if qa_pairs else "Answer"
                        ),
                        rationale=f"Fallback: Agent failed ({e!s})",
                    )
                ],
                reasoning=f"Card splitting agent failed: {e!s}. Defaulting to single card.",
                decision_time=0.0,
                confidence=0.3,  # Low confidence for fallback
                fallback_strategy="none",
            )


# ============================================================================
# Duplicate Detection Agent
# ============================================================================


class DuplicateMatchOutput(BaseModel):
    """Output model for a duplicate match."""

    card_slug: str = Field(min_length=1)
    similarity_score: float = Field(default=0.0)
    duplicate_type: str = Field(
        description="exact/semantic/partial_overlap/unique")
    reasoning: str = Field(default="")


class DuplicateDetectionOutput(BaseModel):
    """Structured output from duplicate detection agent."""

    is_duplicate: bool = Field(
        description="True if exact or semantic duplicate")
    similarity_score: float = Field(default=0.0)
    duplicate_type: str = Field(
        description="exact/semantic/partial_overlap/unique")
    reasoning: str = Field(description="Explanation of similarity assessment")
    recommendation: str = Field(
        description="delete/merge/keep_both/review_manually")
    better_card: str | None = Field(
        default=None, description="'new' or existing card slug if duplicate"
    )
    merge_suggestion: str | None = Field(default=None)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class DuplicateDetectionDeps(BaseModel):
    """Dependencies for duplicate detection agent."""

    new_card_question: str
    new_card_answer: str
    existing_card_question: str
    existing_card_answer: str
    existing_card_slug: str


class DuplicateDetectionAgentAI:
    """PydanticAI-based duplicate detection agent.

    Analyzes cards to identify duplicates and overlapping content.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.0):
        """Initialize duplicate detection agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model = model
        self.temperature = temperature

        # Create PydanticAI agent
        self.agent: Agent[DuplicateDetectionDeps, DuplicateDetectionOutput] = Agent(
            model=self.model,
            output_type=DuplicateDetectionOutput,
            system_prompt=DUPLICATE_DETECTION_PROMPT,
        )

        logger.info(
            "pydantic_ai_duplicate_detection_agent_initialized", model=str(model)
        )

    async def check_duplicate(
        self, new_card: GeneratedCard, existing_card: GeneratedCard
    ) -> tuple[bool, float, str]:
        """Check if new card is a duplicate of existing card.

        Args:
            new_card: Newly generated card
            existing_card: Existing card to compare against

        Returns:
            Tuple of (is_duplicate, similarity_score, reasoning)
        """
        import time

        start_time = time.time()

        try:
            new_q, new_a = self._extract_qa_from_apf(new_card.apf_html)
            existing_q, existing_a = self._extract_qa_from_apf(
                existing_card.apf_html)

            # Create dependencies
            deps = DuplicateDetectionDeps(
                new_card_question=new_q,
                new_card_answer=new_a,
                existing_card_question=existing_q,
                existing_card_answer=existing_a,
                existing_card_slug=existing_card.slug,
            )

            # Build prompt
            prompt = f"""Compare these two cards and determine if they are duplicates:

**New Card**:
Q: {new_q}
A: {new_a}

**Existing Card (slug: {existing_card.slug})**:
Q: {existing_q}
A: {existing_a}

Analyze similarity and provide your assessment."""

            logger.info(
                "pydantic_ai_duplicate_check_start",
                new_slug=new_card.slug,
                existing_slug=existing_card.slug,
            )

            # Run agent
            result = await self.agent.run(prompt, deps=deps)
            output: DuplicateDetectionOutput = result.data

            elapsed = time.time() - start_time

            logger.info(
                "pydantic_ai_duplicate_check_complete",
                is_duplicate=output.is_duplicate,
                similarity=output.similarity_score,
                type=output.duplicate_type,
                elapsed=elapsed,
            )

            return (output.is_duplicate, output.similarity_score, output.reasoning)

        except Exception as e:
            logger.error("pydantic_ai_duplicate_check_failed", error=str(e))
            # Conservative fallback: assume not duplicate
            return (False, 0.0, f"Detection failed: {e!s}")

    async def find_duplicates(
        self, new_card: GeneratedCard, existing_cards: list[GeneratedCard]
    ) -> DuplicateDetectionResult:
        """Find all potential duplicates of a new card.

        Args:
            new_card: Newly generated card
            existing_cards: List of existing cards to check against

        Returns:
            DuplicateDetectionResult with all matches
        """
        import time

        start_time = time.time()

        try:
            matches = []

            # Check against each existing card
            for existing_card in existing_cards:
                _is_dup, score, reasoning = await self.check_duplicate(
                    new_card, existing_card
                )

                # Determine duplicate type based on score
                if score >= 0.95:
                    dup_type = "exact"
                elif score >= 0.80:
                    dup_type = "semantic"
                elif score >= 0.50:
                    dup_type = "partial_overlap"
                else:
                    dup_type = "unique"

                if score >= 0.50:  # Only include significant matches
                    match = DuplicateMatch(
                        card_slug=existing_card.slug,
                        similarity_score=score,
                        duplicate_type=dup_type,  # type: ignore[arg-type]
                        reasoning=reasoning,
                    )
                    matches.append(match)

            # Sort by similarity (highest first)
            matches.sort(key=lambda m: m.similarity_score, reverse=True)

            # Determine overall result
            best_match = matches[0] if matches else None
            is_duplicate = (
                best_match is not None and best_match.similarity_score >= 0.80
            )

            # Recommendation based on best match
            if best_match:
                if best_match.similarity_score >= 0.95:
                    recommendation = "delete"
                elif best_match.similarity_score >= 0.80:
                    recommendation = "merge"
                elif best_match.similarity_score >= 0.65:
                    recommendation = "review_manually"
                else:
                    recommendation = "keep_both"
            else:
                recommendation = "keep_both"

            result = DuplicateDetectionResult(
                is_duplicate=is_duplicate,
                best_match=best_match,
                all_matches=matches,
                recommendation=recommendation,  # type: ignore[arg-type]
                better_card=None,  # Would need more logic to determine
                merge_suggestion=None,
                detection_time=time.time() - start_time,
            )

            logger.info(
                "pydantic_ai_find_duplicates_complete",
                new_slug=new_card.slug,
                matches_found=len(matches),
                is_duplicate=is_duplicate,
                recommendation=recommendation,
            )

            return result

        except Exception as e:
            logger.error("pydantic_ai_find_duplicates_failed", error=str(e))
            # Return safe fallback
            return DuplicateDetectionResult(
                is_duplicate=False,
                best_match=None,
                all_matches=[],
                recommendation="keep_both",  # type: ignore[arg-type]
                better_card=None,
                merge_suggestion=None,
                detection_time=0.0,
            )

    def _extract_qa_from_apf(self, apf_html: str) -> tuple[str, str]:
        """Extract question and answer from APF HTML.

        Args:
            apf_html: APF format HTML

        Returns:
            Tuple of (question, answer)
        """
        question = ""
        answer = ""

        # Extract Front (question)
        front_match = re.search(
            r'<div class="front">(.*?)</div>', apf_html, re.DOTALL)
        if front_match:
            # Remove HTML tags
            question = re.sub(r"<[^>]+>", "", front_match.group(1)).strip()

        # Extract Back (answer)
        back_match = re.search(
            r'<div class="back">(.*?)</div>', apf_html, re.DOTALL)
        if back_match:
            answer = re.sub(r"<[^>]+>", "", back_match.group(1)).strip()

        return question or "Unknown question", answer or "Unknown answer"


# ============================================================================
# Context Enrichment Agent
# ============================================================================


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
    enriched_answer: str = Field(
        default="", description="Enhanced answer text")
    enriched_extra: str = Field(
        default="", description="Enhanced Extra section")
    additions_summary: str = Field(
        default="", description="Summary of additions")
    rationale: str = Field(default="", description="Why enrichment helps")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ContextEnrichmentDeps(BaseModel):
    """Dependencies for context enrichment agent."""

    question: str
    answer: str
    extra: str
    card_slug: str
    note_title: str


class ContextEnrichmentAgentAI:
    """PydanticAI-based context enrichment agent.

    Enhances cards with examples, mnemonics, and helpful context.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.3):
        """Initialize context enrichment agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature (0.3 for some creativity)
        """
        self.model = model
        self.temperature = temperature

        # Create PydanticAI agent
        self.agent: Agent[ContextEnrichmentDeps, ContextEnrichmentOutput] = Agent(
            model=self.model,
            output_type=ContextEnrichmentOutput,
            system_prompt=CONTEXT_ENRICHMENT_PROMPT,
        )

        logger.info(
            "pydantic_ai_context_enrichment_agent_initialized", model=str(model)
        )

    async def enrich(
        self, card: GeneratedCard, metadata: NoteMetadata
    ) -> ContextEnrichmentResult:
        """Enrich a card with additional context and examples.

        Args:
            card: Card to enrich
            metadata: Note metadata for context

        Returns:
            ContextEnrichmentResult with enriched card
        """
        import time

        start_time = time.time()

        try:
            # Extract Q/A from APF HTML
            question, answer = self._extract_qa_from_apf(card.apf_html)
            extra = self._extract_extra_from_apf(card.apf_html)

            # Create dependencies
            deps = ContextEnrichmentDeps(
                question=question,
                answer=answer,
                extra=extra or "",
                card_slug=card.slug,
                note_title=metadata.title,
            )

            # Build prompt
            prompt = f"""Analyze this flashcard and determine if it needs enrichment:

**Card (slug: {card.slug})**
Note: {metadata.title}

Q: {question}
A: {answer}
Extra: {extra or "(none)"}

Consider adding:
- Concrete examples (code, real-world scenarios)
- Mnemonics or memory aids
- Visual structure (formatting, bullets)
- Related concepts or comparisons
- Practical tips or common pitfalls

Provide your enrichment assessment."""

            logger.info("pydantic_ai_enrichment_start", slug=card.slug)

            # Run agent
            result = await self.agent.run(prompt, deps=deps)
            output: ContextEnrichmentOutput = result.data

            # If enrichment recommended, create enriched card
            enriched_card = None
            additions = []

            if output.should_enrich:
                # Reconstruct APF HTML with enrichments
                enriched_apf = self._rebuild_apf_html(
                    card.apf_html,
                    output.enriched_answer or answer,
                    output.enriched_extra or extra,
                )

                enriched_card = GeneratedCard(
                    card_index=card.card_index,
                    slug=card.slug,
                    lang=card.lang,
                    apf_html=enriched_apf,
                    confidence=card.confidence,
                    content_hash=card.content_hash,
                )

                # Create additions list
                for enrich_type in output.enrichment_type:
                    addition = EnrichmentAddition(
                        enrichment_type=enrich_type,
                        content=(
                            output.enriched_extra[:200] + "..."
                            if len(output.enriched_extra) > 200
                            else output.enriched_extra
                        ),
                        rationale=output.rationale,
                    )
                    additions.append(addition)

            enrichment_result = ContextEnrichmentResult(
                should_enrich=output.should_enrich,
                enriched_card=enriched_card,
                additions=additions,
                additions_summary=output.additions_summary,
                enrichment_rationale=output.rationale,
                enrichment_time=time.time() - start_time,
            )

            logger.info(
                "pydantic_ai_enrichment_complete",
                slug=card.slug,
                should_enrich=output.should_enrich,
                types=output.enrichment_type,
                confidence=output.confidence,
            )

            return enrichment_result

        except Exception as e:
            logger.error("pydantic_ai_enrichment_failed",
                         error=str(e), slug=card.slug)
            # Return safe fallback (no enrichment)
            return ContextEnrichmentResult(
                should_enrich=False,
                enriched_card=None,
                additions=[],
                additions_summary=f"Enrichment failed: {e!s}",
                enrichment_rationale="Agent encountered error",
                enrichment_time=0.0,
            )

    def _extract_qa_from_apf(self, apf_html: str) -> tuple[str, str]:
        """Extract question and answer from APF HTML."""
        question = ""
        answer = ""

        # Extract Front (question)
        front_match = re.search(
            r'<div class="front">(.*?)</div>', apf_html, re.DOTALL)
        if front_match:
            question = re.sub(r"<[^>]+>", "", front_match.group(1)).strip()

        # Extract Back (answer)
        back_match = re.search(
            r'<div class="back">(.*?)</div>', apf_html, re.DOTALL)
        if back_match:
            answer = re.sub(r"<[^>]+>", "", back_match.group(1)).strip()

        return question or "Unknown", answer or "Unknown"

    def _extract_extra_from_apf(self, apf_html: str) -> str:
        """Extract Extra section from APF HTML."""
        extra_match = re.search(
            r'<div class="extra">(.*?)</div>', apf_html, re.DOTALL)
        if extra_match:
            return re.sub(r"<[^>]+>", "", extra_match.group(1)).strip()
        return ""

    def _rebuild_apf_html(
        self, original_apf: str, new_answer: str, new_extra: str
    ) -> str:
        """Rebuild APF HTML with enriched content."""
        # Replace answer
        apf_html = re.sub(
            r'(<div class="back">)(.*?)(</div>)',
            rf"\1{new_answer}\3",
            original_apf,
            flags=re.DOTALL,
        )

        # Replace or add extra
        if '<div class="extra">' in apf_html:
            apf_html = re.sub(
                r'(<div class="extra">)(.*?)(</div>)',
                rf"\1{new_extra}\3",
                apf_html,
                flags=re.DOTALL,
            )
        else:
            # Add extra before closing tag
            apf_html = apf_html.replace(
                "</div>\n</div>",
                f'</div>\n<div class="extra">{new_extra}</div>\n</div>',
            )

        return apf_html
