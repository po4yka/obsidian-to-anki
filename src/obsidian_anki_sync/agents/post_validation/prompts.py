"""Prompt templates for post-validation and auto-fix."""

from ...models import NoteMetadata
from ...utils.logging import get_logger
from ..models import GeneratedCard

logger = get_logger(__name__)


def build_semantic_prompt(
    cards: list[GeneratedCard], metadata: NoteMetadata, strict_mode: bool
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

        # Include truncated HTML content for validation (first 2000 chars should be enough for key content)
        html_preview = card.apf_html[:2000]
        if len(card.apf_html) > 2000:
            html_preview += "...[truncated]"

        card_summaries.append(
            f"Card {card.card_index} (slug: {card.slug}, lang: {card.lang}):\n"
            f"  Title: {title}\n"
            f"  Confidence: {card.confidence}\n"
            f"  HTML Content:\n{html_preview}\n"
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
- Strict adherence to APF v2.1 format with exact section names:
  * Required: <!-- Title -->, <!-- Key point -->, <!-- Key point notes -->
  * Optional: <!-- Subtitle -->, <!-- Syntax (inline) -->, <!-- Sample (caption) -->, <!-- Sample (code block or image) -->, <!-- Other notes -->, <!-- Markdown -->
- Card headers must follow: <!-- Card N | slug: name | CardType: Simple/Missing/Draw | Tags: tag1 tag2 tag3 -->
- All required HTML comment delimiters present
- Valid HTML structure with proper nesting

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

Template compliance errors (always reject):
- Wrong section names (APF v2.1 uses 'Sample (caption)' and 'Sample (code block)', NOT 'Code Example')
- Incorrect card header format (must use 'CardType:', not 'type:')
  IMPORTANT: Card headers MUST use 'CardType:' (with capital C and T). The manifest uses '"type"' but card headers use 'CardType:' - this is correct per APF v2.1 specification
- Missing required sections (Title, Key point, Key point notes)
- Invalid card structure or HTML syntax
- Extra 'END_OF_CARDS' text after proper <!-- END_CARDS --> marker

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


SEMANTIC_VALIDATION_SYSTEM_PROMPT = """<role>
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


def build_autofix_prompt(cards: list[GeneratedCard], error_details: str) -> str:
    """Build auto-fix prompt for LLM-based error correction.

    Args:
        cards: Cards to fix (limited to 10)
        error_details: Description of validation errors

    Returns:
        Formatted prompt string
    """
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

    return f"""<task>
Fix APF v2.1 card validation errors. Apply minimal, targeted corrections while preserving all original content.
</task>

<validation_errors>
{error_details}
</validation_errors>

<cards_to_fix>
{cards_info}
</cards_to_fix>

<critical_rules>
CRITICAL: END_OF_CARDS must be the absolute last line with no content after it.
CRITICAL: Card headers MUST use 'CardType:' (capital C and T) - NOT 'type:'.
CRITICAL: If one language version has optional section (like Subtitle), all versions must have it.
CRITICAL: Return FULL corrected HTML for each card - no truncation.
CRITICAL: Include ALL cards that need fixing in corrected_cards array.
</critical_rules>

<targeted_fixes>
For "Extra 'END_OF_CARDS' text after proper <!-- END_CARDS --> marker":
BEFORE:
<!-- END_CARDS -->
END_OF_CARDS
some extra text here

AFTER:
<!-- END_CARDS -->
END_OF_CARDS

For "Incorrect card header format: uses 'CardType: Simple' instead of 'type: Simple'":
WRONG: CardType should be "type" to match manifest
CORRECT: CardType MUST be "CardType:" (capital C and T) in headers
BEFORE: <!-- Card 1 | slug: test | type: Simple | Tags: test -->
AFTER:  <!-- Card 1 | slug: test | CardType: Simple | Tags: test -->

For "Missing <!-- Subtitle (optional) --> section":
BEFORE (RU version missing Subtitle that EN version has):
<!-- Title -->
<p>Question</p>
<!-- Key point -->

AFTER (add consistent Subtitle):
<!-- Title -->
<p>Question</p>
<!-- Subtitle (optional) -->

<!-- Key point -->
</targeted_fixes>

<validation_checklist>
Before returning, verify your corrections meet ALL these requirements:

[ ] No text appears after END_OF_CARDS line (it must be the absolute last line)
[ ] All card headers use 'CardType:' (capital C and T) - NOT 'type:'
[ ] If any language version has an optional section (like Subtitle), ALL language versions have it
[ ] All corrected_cards have FULL apf_html content (no truncation)
[ ] All cards that need fixing are included in the corrected_cards array
[ ] JSON structure is valid and complete
[ ] All original semantic content is preserved
</validation_checklist>

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


AUTOFIX_SYSTEM_PROMPT = """<role>
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
