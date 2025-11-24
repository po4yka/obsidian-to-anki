"""Post-validator agent for card quality validation.

This agent validates generated APF cards for:
- APF format syntax compliance
- Factual accuracy vs source content
- Semantic coherence
- Template compliance
"""

import time
from enum import Enum

from ..apf.html_validator import validate_card_html
from ..apf.linter import validate_apf
from ..models import NoteMetadata
from ..providers.base import BaseLLMProvider
from ..utils.logging import get_logger
from .json_schemas import get_post_validation_schema
from .llm_errors import (
    categorize_llm_error,
    format_llm_error_for_user,
    log_llm_error,
)
from .models import GeneratedCard, PostValidationResult

logger = get_logger(__name__)


class ErrorCategory(str, Enum):
    """Error categories for validation errors."""

    SYNTAX = "syntax"  # APF format or HTML syntax errors (most fixable)
    HTML = "html"  # HTML structure validation errors (fixable)
    APF_FORMAT = "template"  # APF v2.1 template compliance (fixable)
    MANIFEST = "manifest"  # Manifest slug/format mismatches (fixable)
    SEMANTIC = "semantic"  # Question-answer mismatch or coherence issues (harder to fix)
    FACTUAL = "factual"  # Information inaccuracies or hallucinations (hardest to fix)
    NONE = "none"  # No errors

    @classmethod
    def from_error_string(cls, error_str: str) -> "ErrorCategory":
        """Categorize error from error string.

        Args:
            error_str: Error description string

        Returns:
            ErrorCategory enum value
        """
        error_lower = error_str.lower()
        if "html" in error_lower or "inline" in error_lower or "<code>" in error_lower:
            return cls.HTML
        elif "apf format" in error_lower or "template" in error_lower or "format" in error_lower:
            return cls.APF_FORMAT
        elif "manifest" in error_lower or "slug" in error_lower:
            return cls.MANIFEST
        elif "factual" in error_lower or "hallucination" in error_lower or "inaccurate" in error_lower:
            return cls.FACTUAL
        elif "semantic" in error_lower or "mismatch" in error_lower or "coherence" in error_lower:
            return cls.SEMANTIC
        elif "syntax" in error_lower or "parse" in error_lower or "invalid" in error_lower:
            return cls.SYNTAX
        else:
            return cls.NONE


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
            error_by_category: dict[str, int] = {}
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
                # Categorize error
                category = ErrorCategory.from_error_string(error)
                error_by_category[category.value] = error_by_category.get(category.value, 0) + 1

            # Log error summary
            logger.warning(
                "post_validation_syntax_failed",
                errors_count=len(syntax_errors),
                error_breakdown=error_by_type,
                error_by_category=error_by_category,
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

        system_prompt = """<role>
You are an expert quality validation agent for Anki flashcards using the APF v2.1 format. Your expertise includes educational content design, factual verification, semantic analysis, and template compliance checking.
</role>

<approach>
Think step by step through validation:
1. First, assess factual accuracy against source material
2. Second, analyze semantic coherence of Q&A pairs
3. Third, verify APF v2.1 template compliance
4. Fourth, evaluate overall card quality for learning
5. Finally, provide specific, actionable feedback

Be thorough and systematic in your analysis.
</approach>

<validation_priorities>
Critical (must pass):
- Factual accuracy: No hallucinations or information loss
- Answerability: Questions can be answered from context
- Template compliance: Strict APF v2.1 format adherence
- Semantic coherence: Q&A pairs are logically matched

Important (context-dependent):
- Card atomicity: One concept per card
- Clarity: Clear, unambiguous language
- Completeness: All important details preserved
</validation_priorities>

<output_requirements>
- Always respond in valid JSON format matching the schema
- Be specific about errors: identify card numbers and exact issues
- Be constructive: suggest fixes when issues are correctable
- Be thorough: check all cards systematically
- Provide corrected_cards only if fixes are deterministic
</output_requirements>

<constraints>
NEVER approve cards with:
- Factual hallucinations or inaccuracies
- Question-answer semantic mismatches
- Missing or malformed APF structure
- Unanswerable questions

DO suggest auto-fixes for:
- Minor formatting issues
- Correctable template compliance problems
- Structural improvements
</constraints>"""

        # Get JSON schema for structured output
        json_schema = get_post_validation_schema()

        # Call LLM
        result = self.ollama_client.generate_json(
            model=self.model,
            prompt=prompt,
            system=system_prompt,
            temperature=self.temperature,
            json_schema=json_schema,
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

        return f"""<task>
Validate generated APF flashcards for quality, correctness, and template compliance. Think step by step to identify any issues with factual accuracy, semantic coherence, or format compliance.
</task>

<input>
<note_metadata>
Title: {metadata.title}
Topic: {metadata.topic}
Subtopics: {", ".join(metadata.subtopics)}
Total Cards Generated: {len(cards)}
</note_metadata>

<cards_summary>
{cards_summary}
</cards_summary>

<validation_mode>
Strict Mode: {"ENABLED" if strict_mode else "DISABLED"}
{"- Reject cards with ANY quality issues" if strict_mode else "- Only reject cards with CRITICAL errors"}
{"- Minor issues should result in validation failure" if strict_mode else "- Minor issues are acceptable"}
</validation_mode>
</input>

<validation_steps>
Step 1: Factual Accuracy Assessment
- Verify cards accurately reflect source material
- Check for information loss or distortion
- Identify any hallucinated or invented information
- Confirm all important details are preserved

Step 2: Semantic Coherence Analysis
- Verify questions are well-formed and clear
- Ensure answers directly address their questions
- Check that context is relevant and helpful
- Confirm language consistency (en/ru matching)

Step 3: Template Compliance Check
- Validate APF v2.1 format is followed strictly
- Verify all required sections are present
- Check HTML structure is valid
- Ensure card headers match required format

Step 4: Card Quality Evaluation
- Confirm each card is atomic (single concept)
- Verify questions are answerable from provided context
- Check clarity and specificity
- Assess overall card usefulness for learning

Step 5: Technical Validation
- Verify slugs are properly formatted
- Check language codes are correct
- Validate confidence scores are reasonable
- Ensure no duplicate cards
</validation_steps>

<validation_criteria>
FACTUAL ACCURACY:
- No information from source material is lost
- No hallucinated or invented details
- All facts are correctly represented
- Context is preserved accurately

SEMANTIC COHERENCE:
- Question-answer pairs are logically matched
- Language versions (en/ru) are semantically equivalent
- Context enhances understanding
- No ambiguous or unclear phrasing

TEMPLATE COMPLIANCE:
- Strict adherence to APF v2.1 format
- All required HTML sections present
- Card headers properly formatted
- Valid HTML structure

CARD QUALITY:
- Atomic: Each card tests one concept
- Clear: Question is unambiguous
- Answerable: Can be answered from context
- Useful: Aids learning and retention

WHAT CONSTITUTES AN ERROR:
Critical errors (always reject):
- Factual inaccuracies or hallucinations
- Missing or malformed APF structure
- Unanswerable questions
- Information loss from source

Minor issues ({"reject in strict mode" if strict_mode else "acceptable in lenient mode"}):
- Suboptimal wording choices
- Minor formatting inconsistencies
- Less-than-ideal context presentation
</validation_criteria>

<output_format>
Respond with valid JSON matching this structure:

{{
    "is_valid": true/false,
    "error_type": "syntax" | "factual" | "semantic" | "template" | "none",
    "error_details": "Specific description of errors found, or empty string if valid",
    "corrected_cards": null or [
        {{
            "card_index": 1,
            "slug": "example-slug",
            "lang": "en",
            "apf_html": "complete corrected HTML",
            "confidence": 0.9
        }}
    ]
}}

error_type values:
- "syntax": APF format or HTML syntax errors
- "factual": Information inaccuracies or hallucinations
- "semantic": Question-answer mismatch or coherence issues
- "template": APF v2.1 template compliance failures
- "none": No errors found (is_valid = true)

If you can auto-fix issues:
- Set corrected_cards to array of corrected card objects
- Include COMPLETE apf_html for each corrected card
- Maintain all original content, only fix format/structure
</output_format>

<examples>
<example_1>
Input: Card with factual hallucination

Analysis: Card claims Android was released in 2009, but source material states 2008.

Output:
{{
    "is_valid": false,
    "error_type": "factual",
    "error_details": "Card 1 contains factual error: states Android released in 2009, but source indicates 2008. This is a hallucination.",
    "corrected_cards": null
}}
</example_1>

<example_2>
Input: Card with question-answer mismatch

Analysis: Question asks "What is polymorphism?" but answer describes inheritance.

Output:
{{
    "is_valid": false,
    "error_type": "semantic",
    "error_details": "Card 2 semantic error: question asks about polymorphism, but answer describes inheritance. Question and answer are not semantically matched.",
    "corrected_cards": null
}}
</example_2>

<example_3>
Input: All cards valid and well-formed

Output:
{{
    "is_valid": true,
    "error_type": "none",
    "error_details": "",
    "corrected_cards": null
}}
</example_3>
</examples>"""

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
                lambda m: f"CardType: {str(m.group(1)).capitalize()}",
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
                    content_hash=card.content_hash,
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

            prompt = f"""<task>
Fix APF card validation errors and return the corrected cards. Diagnose the specific issues and apply targeted corrections while preserving all original content.
</task>

<input>
<validation_errors>
{error_details}
</validation_errors>

<cards_to_fix>
{cards_info}
</cards_to_fix>
</input>

<diagnostic_approach>
Think step by step:
1. Analyze the validation errors to understand what's wrong
2. Identify the specific cards and fields that need correction
3. Determine the appropriate fix for each error
4. Apply corrections while preserving all original content
5. Verify the fix resolves the validation error
</diagnostic_approach>

<common_fixes>
Card Header Format Issues:
- Required format: "<!-- Card N | slug: name | CardType: Type | Tags: tag1 tag2 tag3 -->"
- MUST have spaces before and after each pipe character |
- Card number N must be sequential (1, 2, 3, ...)
- Slug must be lowercase with only letters, numbers, and hyphens (no underscores)
- CardType must be exactly one of: Simple, Missing, Draw (case-sensitive, capital C and T)
- Tags must be space-separated, NOT comma-separated

Slug Format Issues:
- Convert to lowercase
- Replace underscores with hyphens
- Remove invalid characters (keep only a-z, 0-9, hyphen)
- Remove multiple consecutive hyphens
- Remove leading/trailing hyphens

Tag Format Issues:
- Change from comma-separated to space-separated
- Remove extra whitespace
- Ensure tags are lowercase with underscores for multi-word tags

CardType Issues:
- Ensure exact capitalization: Simple, Missing, or Draw
- No variations like "simple", "SIMPLE", or "SimplE"

HTML Validation Issues:
- Inline <code> elements MUST be wrapped in <pre><code>...</code></pre>
- Standalone <code> tags are invalid - always wrap in <pre>
- Example fix: <code>text</code> -> <pre><code>text</code></pre>

Missing END_CARDS:
- ALWAYS include <!-- END_CARDS --> before END_OF_CARDS
- If missing, add it right before END_OF_CARDS line
- Format: <!-- END_CARDS -->\nEND_OF_CARDS

Manifest Slug Mismatches:
- Ensure slug in card header matches manifest slug exactly
- Check both <!-- Card ... | slug: ... --> and <!-- manifest: ... -->
- They must match or card will fail validation
</common_fixes>

<examples>
<example_1_invalid_header>
Before: <!-- Card 1|slug:android_lifecycle|cardtype:simple|Tags:android, lifecycle -->
After: <!-- Card 1 | slug: android-lifecycle | CardType: Simple | Tags: android lifecycle -->

Fixes applied:
- Added spaces around pipe characters
- Converted slug underscores to hyphens
- Fixed CardType capitalization
- Changed tags from comma-separated to space-separated
</example_1_invalid_header>

<example_2_valid_header>
<!-- Card 1 | slug: android-lifecycle-methods | CardType: Simple | Tags: android lifecycle architecture -->
</example_2_valid_header>
</examples>

<output_format>
Return ONLY valid JSON matching this EXACT structure:

{{
    "corrected_cards": [
        {{
            "card_index": 1,
            "slug": "android-lifecycle-methods",
            "lang": "en",
            "apf_html": "<!-- PROMPT_VERSION: apf-v2.1 -->\\n<!-- BEGIN_CARDS -->\\n<!-- Card 1 | slug: android-lifecycle-methods | CardType: Simple | Tags: android lifecycle architecture -->\\n... complete HTML content ...",
            "confidence": 0.9
        }}
    ]
}}
</output_format>

<critical_requirements>
MUST include:
- The COMPLETE corrected apf_html for each card (not truncated)
- ALL cards that need fixing in the corrected_cards array
- The corrected_cards array (never omit it)
- Properly escaped JSON strings

MUST NOT:
- Return an empty object {{}}
- Omit the corrected_cards array
- Return incomplete or truncated apf_html
- Change the semantic content of cards (only fix format/syntax)
- Add or remove cards (only correct existing ones)

PRESERVE:
- All original content and meaning
- All HTML sections
- All metadata fields
- Card ordering
</critical_requirements>"""

            system_prompt = """<role>
You are an expert APF card correction agent specializing in fixing syntax and format errors in APF v2.1 flashcards. Your expertise includes HTML structure validation, APF format compliance, and diagnostic problem-solving.
</role>

<approach>
Think step by step to fix errors:
1. Diagnose the specific error from validation feedback
2. Locate the problematic section in the card HTML
3. Determine the correct format based on APF v2.1 spec
4. Apply minimal, targeted corrections
5. Verify the fix resolves the error
6. Preserve all original semantic content
</approach>

<critical_rules>
MUST always:
- Return valid JSON with a "corrected_cards" array
- Include the FULL corrected HTML for each card (no truncation)
- Maintain all original content and meaning
- Only fix format/syntax issues, never change semantics
- Be precise with APF v2.1 format requirements
- Include ALL cards that need fixing

MUST NEVER:
- Return an empty object {{}} or incomplete response
- Omit the corrected_cards array
- Truncate or abbreviate the apf_html field
- Change the semantic content of cards
- Add or remove cards
- Skip cards that have errors
</critical_rules>

<format_requirements>
APF v2.1 Card Header Format:
<!-- Card N | slug: lowercase-slug | CardType: Simple|Missing|Draw | Tags: tag1 tag2 tag3 -->

Requirements:
- Spaces before and after each | pipe character
- CardType with capital C and T, followed by: Simple, Missing, or Draw
- Slug: lowercase, letters/numbers/hyphens only
- Tags: space-separated (not comma-separated)
</format_requirements>"""

            # Call LLM with error handling
            try:
                llm_start_time = time.time()

                # Use slightly higher temperature for auto-fix to allow creativity
                # but keep it low for consistency
                base_temperature = self.temperature if self.temperature is not None else 0.0
                auto_fix_temperature = max(0.1, base_temperature)

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

                # Fallback: Try deterministic fixes for common issues
                logger.info("auto_fix_llm_failed_trying_deterministic", error=str(llm_error))
                fixed_cards = self._apply_deterministic_fixes(cards_to_fix, error_details)
                if fixed_cards:
                    logger.info("auto_fix_deterministic_success", cards_fixed=len(fixed_cards))
                    return fixed_cards

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

    def _apply_deterministic_fixes(
        self, cards: list[GeneratedCard], error_details: str
    ) -> list[GeneratedCard] | None:
        """Apply deterministic fixes without LLM for common issues.

        Args:
            cards: Cards to fix
            error_details: Error description

        Returns:
            Fixed cards if any fixes applied, None otherwise
        """
        import re

        fixed_cards = []
        any_fixes = False

        for card in cards:
            fixed_html = card.apf_html
            card_fixed = False

            # Fix 1: Missing END_CARDS
            if "Missing" in error_details and "END_CARDS" in error_details:
                if "<!-- END_CARDS -->" not in fixed_html:
                    # Add END_CARDS before END_OF_CARDS if present
                    if "END_OF_CARDS" in fixed_html:
                        fixed_html = fixed_html.replace("END_OF_CARDS", "<!-- END_CARDS -->\nEND_OF_CARDS")
                    else:
                        # Add both if neither present
                        fixed_html += "\n<!-- END_CARDS -->\nEND_OF_CARDS"
                    any_fixes = True
                    card_fixed = True
                    logger.debug("deterministic_fix_added_end_cards", slug=card.slug)

            # Fix 2: Inline <code> without <pre> wrapper
            if "HTML" in error_details and ("code" in error_details.lower() or "inline" in error_details.lower()):
                # Find standalone <code> tags not inside <pre>
                code_pattern = r"<code(?:\s[^>]*)?>.*?</code>"
                matches = list(re.finditer(code_pattern, fixed_html, re.DOTALL | re.IGNORECASE))

                # Process in reverse to avoid offset issues
                for match in reversed(matches):
                    code_tag = match.group(0)
                    start_pos = match.start()
                    end_pos = match.end()

                    # Check if already inside <pre>
                    context_before = fixed_html[max(0, start_pos - 500):start_pos]
                    pre_matches = list(re.finditer(r"<pre(?:\s[^>]*)?>", context_before, re.IGNORECASE))
                    if pre_matches:
                        last_pre = pre_matches[-1]
                        pre_start = last_pre.start() + (start_pos - 500)
                        between = fixed_html[pre_start:start_pos]
                        if "</pre>" not in between and "</PRE>" not in between:
                            continue  # Already wrapped

                    # Wrap standalone code tag
                    wrapped = f"<pre>{code_tag}</pre>"
                    fixed_html = fixed_html[:start_pos] + wrapped + fixed_html[end_pos:]
                    any_fixes = True
                    card_fixed = True
                    logger.debug("deterministic_fix_wrapped_code", slug=card.slug)

            # Fix 3: Manifest slug mismatch (if error mentions it)
            if "manifest" in error_details.lower() and "slug" in error_details.lower():
                # Extract slug from card header
                header_match = re.search(r"<!--\s*Card\s+\d+\s*\|\s*slug:\s*([^\s|]+)", fixed_html)
                if header_match:
                    header_slug = header_match.group(1)
                    # Check manifest slug
                    manifest_match = re.search(r'<!--\s*manifest:.*?"slug"\s*:\s*"([^"]+)"', fixed_html, re.DOTALL)
                    if manifest_match:
                        manifest_slug = manifest_match.group(1)
                        if header_slug != manifest_slug:
                            # Fix manifest to match header
                            fixed_html = re.sub(
                                r'(<!--\s*manifest:.*?"slug"\s*:\s*")[^"]+(")',
                                rf'\1{header_slug}\2',
                                fixed_html,
                                flags=re.DOTALL
                            )
                            any_fixes = True
                            card_fixed = True
                            logger.debug("deterministic_fix_manifest_slug", slug=card.slug, fixed_slug=header_slug)

            if card_fixed:
                fixed_card = GeneratedCard(
                    card_index=card.card_index,
                    slug=card.slug,
                    lang=card.lang,
                    apf_html=fixed_html,
                    confidence=card.confidence,
                    content_hash=card.content_hash,
                )
                fixed_cards.append(fixed_card)
            else:
                fixed_cards.append(card)

        return fixed_cards if any_fixes else None
