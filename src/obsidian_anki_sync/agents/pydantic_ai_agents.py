"""PydanticAI-based agent implementations for card generation pipeline.

This module provides type-safe agents using PydanticAI for:
1. Pre-validation - note structure and format checking
2. Card generation - converting Q/A pairs to APF cards
3. Post-validation - quality and accuracy checking

All agents use structured outputs and proper type validation.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

from ..models import NoteMetadata, QAPair
from ..utils.logging import get_logger
from .models import GeneratedCard, GenerationResult, PostValidationResult, PreValidationResult

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
    error_details: str = Field(default="", description="Detailed error description")
    suggested_fixes: list[str] = Field(
        default_factory=list, description="Suggested fixes for validation errors"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in validation result"
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
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall generation confidence"
    )


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
        ge=0.0, le=1.0, description="Confidence in validation result"
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


# ============================================================================
# PydanticAI Agent Implementations
# ============================================================================


class PreValidatorAgentAI:
    """PydanticAI-based pre-validation agent.

    Validates note structure, formatting, and frontmatter before generation.
    Uses structured outputs for type-safe validation results.
    """

    def __init__(self, model: OpenAIModel, temperature: float = 0.0):
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
            result_type=PreValidationOutput,
            system_prompt=self._get_system_prompt(),
        )

        logger.info("pydantic_ai_pre_validator_initialized", model=str(model))

    def _get_system_prompt(self) -> str:
        """Get system prompt for pre-validation."""
        return """You are a pre-validation agent for Obsidian notes converted to Anki cards.

Your task is to validate note structure, formatting, and frontmatter before card generation.

Check for:
1. Valid YAML frontmatter with required fields (title, topic, tags, language_tags)
2. Proper Q&A pair formatting
3. Correct markdown structure
4. Content completeness
5. Language tag consistency

Provide detailed error information and suggested fixes if validation fails.
Be strict but helpful in your validation."""

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
Tags: {', '.join(metadata.tags)}
Language Tags: {', '.join(metadata.language_tags)}
Q&A Pairs: {len(qa_pairs)}

Note Content Preview:
{note_content[:500]}...

Validate the structure, frontmatter, and content quality."""

        try:
            # Run agent
            result = await self.agent.run(prompt, deps=deps)

            # Convert to PreValidationResult
            output = result.data
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

        except Exception as e:
            logger.error("pydantic_ai_pre_validation_failed", error=str(e))
            return PreValidationResult(
                is_valid=False,
                error_type="structure",
                error_details=f"Validation failed: {str(e)}",
                auto_fix_applied=False,
                fixed_content=None,
                validation_time=0.0,
            )


class GeneratorAgentAI:
    """PydanticAI-based card generation agent.

    Generates APF cards from Q/A pairs using structured outputs.
    Ensures type-safe card generation with validation.
    """

    def __init__(self, model: OpenAIModel, temperature: float = 0.3):
        """Initialize generator agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature for creativity
        """
        self.model = model
        self.temperature = temperature

        # Load APF prompt
        prompt_path = Path(__file__).parents[3] / ".docs" / "CARDS_PROMPT.md"
        if prompt_path.exists():
            self.system_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            logger.warning("cards_prompt_not_found", path=str(prompt_path))
            self.system_prompt = (
                "Generate APF cards following strict APF v2.1 format."
            )

        # Create PydanticAI agent
        self.agent: Agent[GenerationDeps, CardGenerationOutput] = Agent(
            model=self.model,
            result_type=CardGenerationOutput,
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
Languages: {', '.join(metadata.language_tags)}
Slug Base: {slug_base}

Q&A Pairs ({len(qa_pairs)}):
"""
        for idx, qa in enumerate(qa_pairs, 1):
            prompt += f"\n{idx}. Q: {qa.question[:100]}...\n   A: {qa.answer[:100]}...\n"

        prompt += "\nGenerate complete APF HTML cards for all Q&A pairs in all languages."

        try:
            # Run agent
            result = await self.agent.run(prompt, deps=deps)
            output = result.data

            # Convert cards to GeneratedCard instances
            generated_cards: list[GeneratedCard] = []
            for card_dict in output.cards:
                try:
                    generated_card = GeneratedCard(
                        card_index=card_dict["card_index"],
                        slug=card_dict["slug"],
                        lang=card_dict["lang"],
                        apf_html=card_dict["apf_html"],
                        confidence=card_dict.get("confidence", output.confidence),
                    )
                    generated_cards.append(generated_card)
                except Exception as e:
                    logger.warning("invalid_generated_card", error=str(e), card=card_dict)

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

        except Exception as e:
            logger.error("pydantic_ai_generation_failed", error=str(e))
            return GenerationResult(
                cards=[],
                total_cards=0,
                generation_time=0.0,
                model_used=str(self.model),
            )


class PostValidatorAgentAI:
    """PydanticAI-based post-validation agent.

    Validates generated cards for quality, syntax, and accuracy.
    Uses structured outputs to identify and suggest corrections.
    """

    def __init__(self, model: OpenAIModel, temperature: float = 0.0):
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
            result_type=PostValidationOutput,
            system_prompt=self._get_system_prompt(),
        )

        logger.info("pydantic_ai_post_validator_initialized", model=str(model))

    def _get_system_prompt(self) -> str:
        """Get system prompt for post-validation."""
        return """You are a post-validation agent for APF (Active Prompt Framework) cards.

Your task is to validate generated cards for:
1. APF syntax correctness (HTML structure, required fields)
2. Factual accuracy (content matches source Q&A)
3. Semantic coherence (cards make sense)
4. Template compliance (proper APF v2.1 format)
5. Language consistency

For each issue found:
- Identify the card_index
- Describe the specific problem
- Suggest a correction if possible

Be thorough and strict in your validation."""

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
        logger.info("pydantic_ai_post_validation_start", cards_count=len(cards))

        # Create dependencies
        deps = PostValidationDeps(
            cards=cards, metadata=metadata, strict_mode=strict_mode
        )

        # Build validation prompt
        prompt = f"""Validate these {len(cards)} generated APF cards:

Metadata:
- Title: {metadata.title}
- Topic: {metadata.topic}
- Languages: {', '.join(metadata.language_tags)}

Cards to validate:
"""
        for card in cards[:3]:  # Show first 3 for context
            prompt += f"\nCard {card.card_index} ({card.lang}): {card.slug}\n"
            prompt += f"HTML Preview: {card.apf_html[:200]}...\n"

        prompt += f"\nValidate all {len(cards)} cards for correctness, accuracy, and quality."

        try:
            # Run agent
            result = await self.agent.run(prompt, deps=deps)
            output = result.data

            # Convert suggested corrections to GeneratedCard list
            corrected_cards: list[GeneratedCard] | None = None
            if output.suggested_corrections:
                corrected_cards = []
                for correction in output.suggested_corrections:
                    try:
                        corrected_card = GeneratedCard(**correction)
                        corrected_cards.append(corrected_card)
                    except Exception as e:
                        logger.warning("invalid_correction", error=str(e))

            validation_result = PostValidationResult(
                is_valid=output.is_valid,
                error_type=output.error_type,
                error_details=output.error_details,
                corrected_cards=corrected_cards,
                validation_time=0.0,
            )

            logger.info(
                "pydantic_ai_post_validation_complete",
                is_valid=output.is_valid,
                confidence=output.confidence,
                issues_found=len(output.card_issues),
            )

            return validation_result

        except Exception as e:
            logger.error("pydantic_ai_post_validation_failed", error=str(e))
            return PostValidationResult(
                is_valid=False,
                error_type="syntax",
                error_details=f"Validation failed: {str(e)}",
                corrected_cards=None,
                validation_time=0.0,
            )
