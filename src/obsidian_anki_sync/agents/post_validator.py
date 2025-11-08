"""Post-validator agent for card quality validation.

This agent validates generated APF cards for:
- APF format syntax compliance
- Factual accuracy vs source content
- Semantic coherence
- Template compliance
"""

import time

from ..apf.html_validator import validate_card_html
from ..apf.linter import validate_apf
from ..models import NoteMetadata
from ..providers.base import BaseLLMProvider
from ..utils.logging import get_logger
from .llm_errors import (
    categorize_llm_error,
    format_llm_error_for_user,
    log_llm_error,
)
from .models import GeneratedCard, PostValidationResult

logger = get_logger(__name__)


class PostValidatorAgent:
    """Agent for post-validation of generated cards.

    Uses medium model (qwen3:14b) for quality validation with thinking mode.
    """

    def __init__(
        self,
        ollama_client: BaseLLMProvider,
        model: str = "qwen3:14b",
        temperature: float = 0.0,
    ):
        """Initialize post-validator agent.

        Args:
            ollama_client: LLM provider instance (BaseLLMProvider)
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

            # Categorize errors by type for better diagnostics
            error_by_type: dict[str, int] = {}
            for error in syntax_errors:
                # Extract error type from error message
                if "APF format:" in error:
                    error_type = error.split("APF format:")[1].strip().split()[0:3]
                    error_key = " ".join(error_type) if error_type else "unknown"
                elif "HTML:" in error:
                    error_key = "HTML validation"
                else:
                    error_key = "other"
                error_by_type[error_key] = error_by_type.get(error_key, 0) + 1

            # Log error summary
            logger.warning(
                "post_validation_syntax_failed",
                errors_count=len(syntax_errors),
                error_breakdown=error_by_type,
            )

            # Log each error individually (up to 20 to avoid spam)
            for i, error in enumerate(syntax_errors[:20]):
                logger.warning("validation_error_detail", error_num=i + 1, error=error)

            if len(syntax_errors) > 20:
                logger.warning(
                    "validation_errors_truncated",
                    shown=20,
                    total=len(syntax_errors),
                    additional=len(syntax_errors) - 20,
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

            # Categorize and log the error properly
            llm_error = categorize_llm_error(
                error=e,
                model=self.model,
                operation="post-validation",
                duration=validation_time,
            )

            log_llm_error(
                llm_error,
                cards_count=len(cards),
                strict_mode=strict_mode,
            )

            logger.error(
                "post_validation_llm_error",
                error_type=llm_error.error_type.value,
                error=str(llm_error),
                user_message=format_llm_error_for_user(llm_error),
                time=validation_time,
            )

            return PostValidationResult(
                is_valid=False,
                error_type="semantic",
                error_details=f"Semantic validation failed: {format_llm_error_for_user(llm_error)}",
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
            # DEBUG: Log first 500 chars of the card for debugging
            logger.debug(
                "validating_card_syntax",
                slug=card.slug,
                apf_preview=card.apf_html[:500] if card.apf_html else "(empty)",
                apf_length=len(card.apf_html),
            )

            # Validate APF format
            apf_result = validate_apf(card.apf_html, slug=card.slug)

            if not apf_result.is_valid:
                # DEBUG: Log the full card when validation fails
                logger.warning(
                    "card_validation_failed",
                    slug=card.slug,
                    apf_html=card.apf_html[:1000],  # First 1000 chars
                    errors=apf_result.errors,
                )
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
        # Limit to first 5 cards to avoid token limits, but warn if truncating
        cards_to_validate = cards[:5]
        if len(cards) > 5:
            logger.warning(
                "semantic_validation_truncated",
                total_cards=len(cards),
                validated_cards=5,
                skipped_cards=len(cards) - 5,
            )

        card_summaries = []
        for card in cards_to_validate:
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
- Subtopics: {", ".join(metadata.subtopics)}
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

    def _rule_based_header_fix(
        self, cards: list[GeneratedCard]
    ) -> list[GeneratedCard] | None:
        """Attempt to fix card headers using rule-based transformations.

        Args:
            cards: Cards with potential header format issues

        Returns:
            Fixed cards if successful, None otherwise
        """
        import re

        fixed_cards = []
        any_fixes = False

        for card in cards:
            lines = card.apf_html.split("\n")
            if not lines:
                continue

            # Find the card header line (should be after BEGIN_CARDS)
            header_line_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith("<!-- Card "):
                    header_line_idx = i
                    break

            if header_line_idx is None:
                logger.warning(
                    "auto_fix_no_header_found",
                    slug=card.slug,
                )
                continue

            original_header = lines[header_line_idx].strip()
            fixed_header = original_header

            # Common fixes:
            # 1. Normalize spacing around pipes: "Card 1|slug:" -> "Card 1 | slug:"
            fixed_header = re.sub(r"\s*\|\s*", " | ", fixed_header)

            # 2. Fix CardType spacing and capitalization
            # Match variations like "CardType:Simple", "cardtype: Simple", "CardType :Simple"
            fixed_header = re.sub(
                r"[Cc]ard[Tt]ype\s*:\s*([Ss]imple|[Mm]issing|[Dd]raw)",
                lambda m: f"CardType: {m.group(1).capitalize()}",
                fixed_header,
            )

            # 3. Normalize Tags format: ensure it's "Tags: " not "tags:" or "Tags :"
            fixed_header = re.sub(r"[Tt]ags\s*:", "Tags:", fixed_header)

            # 4. Fix slug format: convert underscores to hyphens, lowercase
            slug_match = re.search(r"slug:\s*([^\s|]+)", fixed_header)
            if slug_match:
                original_slug = slug_match.group(1)
                fixed_slug = original_slug.lower().replace("_", "-")
                # Remove any invalid characters
                fixed_slug = re.sub(r"[^a-z0-9-]", "-", fixed_slug)
                # Remove multiple consecutive hyphens
                fixed_slug = re.sub(r"-+", "-", fixed_slug)
                # Remove leading/trailing hyphens
                fixed_slug = fixed_slug.strip("-")
                fixed_header = fixed_header.replace(
                    f"slug: {original_slug}", f"slug: {fixed_slug}"
                )

            # 5. Ensure tags are space-separated, not comma-separated
            tags_match = re.search(r"Tags:\s*([^>]+)", fixed_header)
            if tags_match:
                tags_str = tags_match.group(1).strip()
                # Replace commas with spaces
                fixed_tags = re.sub(r"\s*,\s*", " ", tags_str)
                # Normalize multiple spaces
                fixed_tags = re.sub(r"\s+", " ", fixed_tags)
                fixed_header = fixed_header.replace(
                    f"Tags: {tags_str}", f"Tags: {fixed_tags}"
                )

            # 6. Ensure proper comment ending
            if not fixed_header.endswith("-->"):
                if fixed_header.endswith("->"):
                    fixed_header += ">"
                elif fixed_header.endswith("-"):
                    fixed_header += "->"
                else:
                    fixed_header += " -->"

            # 7. Ensure proper comment beginning
            if not fixed_header.startswith("<!--"):
                fixed_header = "<!-- " + fixed_header.lstrip("<!")

            if fixed_header != original_header:
                logger.info(
                    "auto_fix_header_corrected",
                    slug=card.slug,
                    original=original_header[:100],
                    fixed=fixed_header[:100],
                )
                lines[header_line_idx] = fixed_header
                fixed_html = "\n".join(lines)
                any_fixes = True

                fixed_card = GeneratedCard(
                    card_index=card.card_index,
                    slug=card.slug,
                    lang=card.lang,
                    apf_html=fixed_html,
                    confidence=card.confidence,
                )
                fixed_cards.append(fixed_card)
            else:
                # No fix needed or fix didn't help
                fixed_cards.append(card)

        if any_fixes:
            return fixed_cards
        return None

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
            # First, try rule-based fixes for common issues
            if "Invalid card header format" in error_details:
                fixed_cards = self._rule_based_header_fix(cards)
                if fixed_cards:
                    logger.info(
                        "auto_fix_rule_based_success", cards_fixed=len(fixed_cards)
                    )
                    return fixed_cards

            # Fallback to LLM-based fix for complex issues
            # Include all cards (not just first 3) to ensure comprehensive fix
            # Limit to reasonable size to avoid token limits
            cards_to_fix = cards[:10]
            if len(cards) > 10:
                logger.warning(
                    "auto_fix_truncated",
                    total_cards=len(cards),
                    fixing_cards=10,
                    skipped_cards=len(cards) - 10,
                )

            # Build detailed card information with FULL HTML
            card_details = []
            for i, card in enumerate(cards_to_fix, 1):
                card_details.append(
                    f"=== Card {i} ===\n"
                    f"Slug: {card.slug}\n"
                    f"Lang: {card.lang}\n"
                    f"Card Index: {card.card_index}\n"
                    f"Confidence: {card.confidence}\n"
                    f"Full HTML:\n{card.apf_html}\n"
                )

            cards_info = "\n\n".join(card_details)

            prompt = f"""Fix the APF card validation errors and return the corrected cards.

VALIDATION ERRORS:
{error_details}

CARDS TO FIX (with full HTML):
{cards_info}

COMMON FIXES:
1. Card header format: Ensure "<!-- Card N | slug: name | CardType: Simple/Missing/Draw | Tags: tag1 tag2 tag3 -->"
2. Ensure spaces before and after pipe characters |
3. CardType must be exactly: Simple, Missing, or Draw (case-sensitive)
4. Tags must be space-separated, not comma-separated
5. Slug must be lowercase with only letters, numbers, hyphens

EXAMPLE VALID HEADER:
<!-- Card 1 | slug: android-lifecycle-methods | CardType: Simple | Tags: android lifecycle architecture -->

OUTPUT FORMAT (return ONLY this JSON structure):
{{
    "corrected_cards": [
        {{
            "card_index": 1,
            "slug": "android-lifecycle-methods",
            "lang": "en",
            "apf_html": "<!-- PROMPT_VERSION: apf-v2.1 -->\\n<!-- BEGIN_CARDS -->\\n<!-- Card 1 | slug: android-lifecycle-methods | CardType: Simple | Tags: android lifecycle architecture -->\\n... rest of HTML ...",
            "confidence": 0.9
        }}
    ]
}}

CRITICAL INSTRUCTIONS:
- Return the COMPLETE corrected apf_html for each card
- Include all cards that need fixing in the corrected_cards array
- Do NOT return an empty object {{}}
- Do NOT omit the corrected_cards array
- Ensure all JSON is properly escaped"""

            system_prompt = """You are an expert APF card correction agent. Your task is to fix syntax and format errors in APF flashcards.

CRITICAL RULES:
1. Always return valid JSON with a "corrected_cards" array
2. Never return an empty object or incomplete response
3. Include the FULL corrected HTML for each card
4. Maintain all content, only fix format/syntax issues
5. Be precise with card header format requirements"""

            # Call LLM with error handling
            try:
                llm_start_time = time.time()

                # Use slightly higher temperature for auto-fix to allow creativity
                # but keep it low for consistency
                auto_fix_temperature = max(0.1, self.temperature)

                logger.info(
                    "attempting_manual_auto_fix",
                    model=self.model,
                    temperature=auto_fix_temperature,
                    cards_to_fix=len(cards_to_fix),
                )

                result = self.ollama_client.generate_json(
                    model=self.model,
                    prompt=prompt,
                    system=system_prompt,
                    temperature=auto_fix_temperature,
                )

                llm_duration = time.time() - llm_start_time

                logger.info(
                    "auto_fix_llm_complete",
                    duration=round(llm_duration, 2),
                    result_type=type(result).__name__,
                )

                # Handle case where model returns empty or invalid JSON
                if not result or not isinstance(result, dict):
                    logger.warning(
                        "auto_fix_invalid_response",
                        response_type=type(result).__name__,
                        response=str(result)[:200],
                    )
                    return None

                corrected_data = result.get("corrected_cards", [])
                if not corrected_data:
                    logger.warning(
                        "auto_fix_no_corrections",
                        result_keys=list(result.keys()),
                        result_preview=str(result)[:200],
                    )
                    return None

                logger.info(
                    "auto_fix_success", corrected_cards_count=len(corrected_data)
                )
                return [GeneratedCard(**card) for card in corrected_data]

            except Exception as llm_error:
                llm_duration = time.time() - llm_start_time

                # Categorize and log the error
                categorized_error = categorize_llm_error(
                    error=llm_error,
                    model=self.model,
                    operation="auto-fix",
                    duration=llm_duration,
                )

                log_llm_error(
                    categorized_error,
                    cards_to_fix=len(cards_to_fix),
                    error_details_length=len(error_details),
                )

                logger.error(
                    "auto_fix_llm_failed",
                    error_type=categorized_error.error_type.value,
                    error=str(categorized_error),
                    user_message=format_llm_error_for_user(categorized_error),
                )
                return None

        except Exception as e:
            logger.error("auto_fix_failed", error=str(e))
            return None
