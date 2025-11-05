"""Post-validator agent for card quality validation.

This agent validates generated APF cards for:
- APF format syntax compliance
- Factual accuracy vs source content
- Semantic coherence
- Template compliance
"""

import json
import time

from ..apf.html_validator import validate_card_html
from ..apf.linter import validate_apf
from ..models import NoteMetadata
from ..utils.logging import get_logger
from .models import GeneratedCard, PostValidationResult
from .ollama_client import OllamaClient

logger = get_logger(__name__)


class PostValidatorAgent:
    """Agent for post-validation of generated cards.

    Uses medium model (qwen3:14b) for quality validation with thinking mode.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: str = "qwen3:14b",
        temperature: float = 0.0,
    ):
        """Initialize post-validator agent.

        Args:
            ollama_client: Ollama client instance
            model: Model to use for validation
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature
        logger.info("post_validator_agent_initialized", model=model)

    def validate(
        self,
        cards: list[GeneratedCard],
        metadata: NoteMetadata,
        strict_mode: bool = True,
    ) -> PostValidationResult:
        """Validate generated cards.

        Args:
            cards: List of generated cards to validate
            metadata: Note metadata for context
            strict_mode: Enable strict validation

        Returns:
            PostValidationResult with validation outcome
        """
        start_time = time.time()

        logger.info(
            "post_validation_start",
            cards_count=len(cards),
            strict_mode=strict_mode,
        )

        # Step 1: Syntax validation (fast, deterministic)
        syntax_errors = self._syntax_validation(cards)

        if syntax_errors:
            validation_time = time.time() - start_time
            logger.warning(
                "post_validation_syntax_failed",
                errors_count=len(syntax_errors),
                errors=syntax_errors[:3],  # Log first 3 errors
            )

            return PostValidationResult(
                is_valid=False,
                error_type="syntax",
                error_details="; ".join(syntax_errors),
                corrected_cards=None,
                validation_time=validation_time,
            )

        # Step 2: Semantic validation (AI-powered)
        try:
            semantic_result = self._semantic_validation(cards, metadata, strict_mode)

            validation_time = time.time() - start_time

            logger.info(
                "post_validation_complete",
                is_valid=semantic_result.is_valid,
                error_type=semantic_result.error_type,
                time=validation_time,
            )

            return PostValidationResult(
                is_valid=semantic_result.is_valid,
                error_type=semantic_result.error_type,
                error_details=semantic_result.error_details,
                corrected_cards=semantic_result.corrected_cards,
                validation_time=validation_time,
            )

        except Exception as e:
            validation_time = time.time() - start_time
            logger.error(
                "post_validation_llm_error", error=str(e), time=validation_time
            )

            return PostValidationResult(
                is_valid=False,
                error_type="semantic",
                error_details=f"Semantic validation failed: {str(e)}",
                corrected_cards=None,
                validation_time=validation_time,
            )

    def _syntax_validation(self, cards: list[GeneratedCard]) -> list[str]:
        """Perform syntax validation using existing linters.

        Args:
            cards: Generated cards to validate

        Returns:
            List of validation errors (empty if valid)
        """
        all_errors = []

        for card in cards:
            # Validate APF format
            apf_result = validate_apf(card.apf_html, slug=card.slug)

            if not apf_result.is_valid:
                for error in apf_result.errors:
                    all_errors.append(f"[{card.slug}] APF format: {error}")

            # Validate HTML structure
            html_errors = validate_card_html(card.apf_html)

            for error in html_errors:
                all_errors.append(f"[{card.slug}] HTML: {error}")

        return all_errors

    def _semantic_validation(
        self, cards: list[GeneratedCard], metadata: NoteMetadata, strict_mode: bool
    ) -> PostValidationResult:
        """Perform semantic validation using LLM.

        Args:
            cards: Generated cards
            metadata: Note metadata for context
            strict_mode: Enable strict validation

        Returns:
            PostValidationResult
        """
        # Build validation prompt
        prompt = self._build_semantic_prompt(cards, metadata, strict_mode)

        system_prompt = """You are a quality validation agent for Anki flashcards.
Your job is to check:
1. Factual accuracy - no information loss or hallucinations
2. Semantic coherence - questions and answers are well-matched
3. Template compliance - follows APF v2.1 format
4. Card quality - atomic, clear, answerable

Always respond in valid JSON format.
Be thorough but constructive - suggest fixes when possible."""

        # Call LLM
        result = self.ollama_client.generate_json(
            model=self.model,
            prompt=prompt,
            system=system_prompt,
            temperature=self.temperature,
        )

        # Parse LLM response
        is_valid = result.get("is_valid", False)
        error_type = result.get("error_type", "none")
        error_details = result.get("error_details", "")
        corrected_cards_data = result.get("corrected_cards")

        # Convert corrected cards if provided
        corrected_cards = None
        if corrected_cards_data:
            try:
                corrected_cards = [
                    GeneratedCard(**card_data) for card_data in corrected_cards_data
                ]
            except Exception as e:
                logger.warning("failed_to_parse_corrected_cards", error=str(e))

        return PostValidationResult(
            is_valid=is_valid,
            error_type=error_type,
            error_details=error_details,
            corrected_cards=corrected_cards,
            validation_time=0.0,  # Will be set by caller
        )

    def _build_semantic_prompt(
        self, cards: list[GeneratedCard], metadata: NoteMetadata, strict_mode: bool
    ) -> str:
        """Build semantic validation prompt.

        Args:
            cards: Generated cards
            metadata: Note metadata
            strict_mode: Enable strict validation

        Returns:
            Formatted prompt string
        """
        # Build card summaries
        card_summaries = []
        for card in cards[:5]:  # Limit to first 5 cards to avoid token limits
            # Extract title from HTML
            title = "Unknown"
            if "<!-- Title -->" in card.apf_html:
                lines = card.apf_html.split("\n")
                for i, line in enumerate(lines):
                    if "<!-- Title -->" in line and i + 1 < len(lines):
                        title = lines[i + 1].strip()[:100]  # First 100 chars
                        break

            card_summaries.append(
                f"Card {card.card_index} (slug: {card.slug}, lang: {card.lang}):\n"
                f"  Title: {title}\n"
                f"  Confidence: {card.confidence}\n"
                f"  HTML length: {len(card.apf_html)} chars"
            )

        cards_summary = "\n".join(card_summaries)

        return f"""Validate these APF flashcards for quality and correctness.

NOTE METADATA:
- Title: {metadata.title}
- Topic: {metadata.topic}
- Subtopics: {', '.join(metadata.subtopics)}
- Total Cards: {len(cards)}

GENERATED CARDS SUMMARY:
{cards_summary}

VALIDATION REQUIREMENTS:
1. Factual Accuracy: Cards must accurately reflect the source material
2. Semantic Coherence: Questions and answers should be well-matched
3. Template Compliance: Must follow APF v2.1 format strictly
4. Card Quality: Each card should be atomic (one concept), clear, and answerable
5. No Information Loss: All important details preserved
6. No Hallucinations: No invented or incorrect information

STRICT MODE: {"ENABLED" if strict_mode else "DISABLED"}
{"In strict mode, reject cards with any quality issues." if strict_mode else "In lenient mode, only reject cards with critical errors."}

ANALYZE AND RESPOND IN JSON FORMAT:
{{
    "is_valid": true/false,
    "error_type": "syntax" | "factual" | "semantic" | "template" | "none",
    "error_details": "Detailed description of any errors",
    "corrected_cards": null or [
        {{
            "card_index": 1,
            "slug": "slug-here",
            "lang": "en",
            "apf_html": "corrected HTML if auto-fix possible",
            "confidence": 0.9
        }}
    ]
}}

If you can auto-fix issues, provide corrected_cards with the fixes applied.
Be specific about errors. If everything is valid, set error_type to "none"."""

    def attempt_auto_fix(
        self, cards: list[GeneratedCard], error_details: str
    ) -> list[GeneratedCard] | None:
        """Attempt to auto-fix validation errors.

        Args:
            cards: Original cards with errors
            error_details: Description of errors

        Returns:
            Corrected cards if successful, None otherwise
        """
        try:
            prompt = f"""Fix these APF cards based on the validation errors.

VALIDATION ERRORS:
{error_details}

CARDS TO FIX:
{json.dumps([card.model_dump() for card in cards[:3]], indent=2)}

Provide corrected cards in JSON format:
{{
    "corrected_cards": [
        {{
            "card_index": 1,
            "slug": "slug",
            "lang": "en",
            "apf_html": "corrected HTML",
            "confidence": 0.9
        }}
    ]
}}"""

            system_prompt = "You are a card correction agent. Fix validation errors while preserving card content and intent."

            result = self.ollama_client.generate_json(
                model=self.model,
                prompt=prompt,
                system=system_prompt,
                temperature=self.temperature,
            )

            corrected_data = result.get("corrected_cards", [])
            if corrected_data:
                return [GeneratedCard(**card) for card in corrected_data]

            return None

        except Exception as e:
            logger.error("auto_fix_failed", error=str(e))
            return None
