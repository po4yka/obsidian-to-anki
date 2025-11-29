"""Prompt templates for post-validation and auto-fix."""

from obsidian_anki_sync.agents.models import GeneratedCard
from obsidian_anki_sync.models import NoteMetadata
from obsidian_anki_sync.utils.logging import get_logger

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
Validate generated APF flashcards for CONTENT QUALITY - factual accuracy and semantic coherence. Template compliance is handled by a separate linter.
</task>

<critical_notice>
IMPORTANT: Template/format compliance (APF structure, headers, HTML syntax, sentinels) is checked by a deterministic linter. DO NOT check or report template issues - focus ONLY on content quality.
</critical_notice>

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
{"- Reject cards with ANY content quality issues" if strict_mode else "- Only reject cards with CRITICAL content errors"}
{"- Minor content issues should result in validation failure" if strict_mode else "- Minor content issues are acceptable"}
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

Step 3: Card Quality Evaluation
- Confirm each card is atomic (single concept)
- Verify questions are answerable from provided context
- Check clarity and specificity
- Assess overall card usefulness for learning

DO NOT CHECK (handled by linter):
- APF v2.1 format compliance
- Card header format (CardType, Tags, etc.)
- HTML structure or syntax
- Sentinel markers (PROMPT_VERSION, BEGIN_CARDS, END_CARDS)
- Section headers
</validation_steps>

<validation_criteria>
FACTUAL ACCURACY (YOUR RESPONSIBILITY):
- No information from source material is lost
- No hallucinated or invented details
- All facts are correctly represented
- Context is preserved accurately

SEMANTIC COHERENCE (YOUR RESPONSIBILITY):
- Question-answer pairs are logically matched
- Language versions (en/ru) are semantically equivalent
- Context enhances understanding
- No ambiguous or unclear phrasing

CARD QUALITY (YOUR RESPONSIBILITY):
- Atomic: Each card tests one concept
- Clear: Question is unambiguous
- Answerable: Can be answered from context
- Useful: Aids learning and retention

NOT YOUR RESPONSIBILITY (linter handles these):
- Template compliance (APF structure)
- Card header format
- HTML syntax
- Section names and structure

WHAT CONSTITUTES AN ERROR:
Critical errors (always reject):
- Factual inaccuracies or hallucinations
- Question-answer semantic mismatches
- Unanswerable questions
- Information loss from source

Minor content issues ({"reject in strict mode" if strict_mode else "acceptable in lenient mode"}):
- Suboptimal wording choices
- Less-than-ideal context presentation
- Minor clarity issues
</validation_criteria>

<output_format>
Respond with valid JSON matching this structure:

{{
    "is_valid": true/false,
    "error_type": "factual" | "semantic" | "none",
    "error_details": "Specific description of CONTENT errors found, or empty string if valid",
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

error_type values (CONTENT ERRORS ONLY):
- "factual": Information inaccuracies or hallucinations in content
- "semantic": Question-answer mismatch or coherence issues
- "none": No content errors found (is_valid = true)

DO NOT use these error_type values (linter handles these):
- "syntax": Handled by linter - do not report
- "template": Handled by linter - do not report

If you can auto-fix content issues:
- Set corrected_cards to array of corrected card objects
- Include COMPLETE apf_html for each corrected card
- Only fix content issues (factual/semantic), not format/structure
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
You are a content quality validator for Anki flashcards. Your expertise is in educational content design, factual verification, and semantic analysis.
</role>

<critical_instruction>
IMPORTANT: Template compliance (APF format, sentinels, headers, tags, HTML structure) is checked by a deterministic linter BEFORE your validation runs. The linter is authoritative for template compliance.

Your role is ONLY to validate CONTENT QUALITY:
1. Factual Accuracy - Is the content truthful and accurate?
2. Semantic Coherence - Do questions and answers match?
3. Information Quality - Is the content useful for learning?

DO NOT report template or format issues. Any template errors you find will be IGNORED because the linter has already validated them. Focus on CONTENT QUALITY ONLY.
</critical_instruction>

<approach>
Think step by step through validation:
1. First, assess factual accuracy - verify information is correct
2. Second, analyze semantic coherence - ensure Q&A pairs match
3. Third, evaluate learning quality - is this useful for memorization?
4. Finally, provide specific, actionable feedback on CONTENT issues only

Do NOT check or report:
- APF format or structure issues
- HTML syntax or tag issues
- Card header format (CardType, Tags, etc.)
- Sentinel markers (PROMPT_VERSION, BEGIN_CARDS, END_CARDS)
- Section headers (Title, Key point, etc.)
These are handled by the deterministic linter.
</approach>

<validation_priorities>
Critical (must pass) - YOUR RESPONSIBILITY:
- Factual accuracy: No hallucinations or information loss
- Semantic coherence: Q&A pairs are logically matched
- Answerability: Questions can be answered from provided context
- Information preservation: Key details from source are captured

NOT your responsibility (handled by linter):
- Template/format compliance
- HTML structure
- APF v2.1 format adherence
</validation_priorities>

<output_requirements>
- Always respond in valid JSON format matching the schema
- Be specific about CONTENT errors: identify card numbers and exact issues
- Only report "factual" or "semantic" error_type, NOT "template" or "syntax"
- Provide corrected_cards only if fixes are for content issues
</output_requirements>

<constraints>
NEVER approve cards with:
- Factual hallucinations or inaccuracies
- Question-answer semantic mismatches
- Unanswerable questions
- Information loss from source material

NEVER report these (linter handles them):
- APF structure issues
- Card header format problems
- HTML syntax errors
- Missing sections or sentinels
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

For "HTML validation failed: Tag 'div' is not closed":
BEFORE: <div>Some content
AFTER: <div>Some content</div>

For "Markdown code block found":
BEFORE: ```python\nprint("hello")\n```
AFTER: <pre><code class="language-python">print("hello")</code></pre>

For "Missing '<!-- Title -->'" or malformed section header:
BEFORE: <!-- Title ->\n<p>Title</p>
AFTER: <!-- Title -->\n<p>Title</p>
</targeted_fixes>

<validation_checklist>
Before returning, verify your corrections meet ALL these requirements:

[ ] No text appears after END_OF_CARDS line (it must be the absolute last line)
[ ] All card headers use 'CardType:' (capital C and T) - NOT 'type:'
[ ] If any language version has an optional section (like Subtitle), ALL language versions have it
[ ] All HTML tags are properly closed and nested
[ ] No Markdown code blocks (```) remain - converted to <pre><code>
[ ] All APF sentinels (PROMPT_VERSION, BEGIN_CARDS, END_CARDS) are present and correct
[ ] All section headers (Title, Key point, etc.) are properly formatted
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
- Fix ALL identified errors in a single pass
- Convert any Markdown code blocks (```) to HTML <pre><code> tags
- Ensure all HTML tags are properly closed

MUST NEVER:
- Return an empty object {{}} or incomplete response
- Omit the corrected_cards array
- Truncate or abbreviate the apf_html field
- Change the semantic content of cards
- Add or remove cards
- Skip cards that have errors
- Leave unclosed HTML tags
</critical_rules>

<format_requirements>
APF v2.1 Card Header Format:
<!-- Card N | slug: lowercase-slug | CardType: Simple|Missing|Draw | Tags: tag1 tag2 tag3 -->

HTML Requirements:
- Valid HTML5 syntax
- No Markdown syntax (like ``` or **bold**) inside HTML
- All tags properly closed
- Classes used for styling (e.g., class="language-python")

Sentinel Requirements:
- <!-- PROMPT_VERSION: apf-v2.1 --> at start
- <!-- BEGIN_CARDS --> before first card
- <!-- END_CARDS --> after last card
- END_OF_CARDS as absolute last line
</format_requirements>"""
