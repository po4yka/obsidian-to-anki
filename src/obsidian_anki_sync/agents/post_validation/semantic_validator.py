"""Semantic validation for APF cards using LLM."""

import re

from ...models import NoteMetadata
from ...providers.base import BaseLLMProvider
from ...utils.logging import get_logger
from ..json_schemas import get_post_validation_schema
from ..models import GeneratedCard, PostValidationResult
from .prompts import SEMANTIC_VALIDATION_SYSTEM_PROMPT, build_semantic_prompt

logger = get_logger(__name__)


# Known false positive patterns that LLMs hallucinate
# These are patterns where the LLM inverts or misremembers the correct rules
# APF v2.1 uses 'CardType:' in headers, NOT 'type:' - LLMs frequently get this wrong
FALSE_POSITIVE_PATTERNS = [
    # LLM sometimes says CardType should be type, but CardType is correct per APF v2.1
    r"CardType.*should.*be.*'?type'?",
    r"'CardType'.*but.*requires.*'type'",
    r"use.*'CardType:.*but.*APF.*requires.*'type:",
    r"headers.*use.*'CardType.*but.*requires.*'type",
    # "uses X instead of Y" format - comprehensive patterns
    r"uses\s+'?CardType:?\s*\w*'?\s+instead\s+of\s+'?type:?",
    r"'CardType:\s*\w+'\s+instead\s+of\s+'type:\s*\w+'",
    r"uses\s+CardType:\s*\w+\s+instead\s+of\s+type:\s*\w+",
    r"CardType:\s*\w+\s+instead\s+of\s+type:\s*\w+\s+to\s+match",
    r"'CardType:\s+Simple'\s+instead\s+of\s+'type:\s+Simple'",
    r"uses\s+'CardType:\s+Simple'\s+instead\s+of\s+'type:\s+Simple'\s+to\s+match\s+manifest",
    # "to match manifest" variations - more comprehensive
    r"CardType.*instead.*type.*match.*manifest",
    r"type:\s*\w+.*to\s+match\s+manifest",
    r"to\s+match\s+manifest.*CardType.*instead.*type",
    r"match\s+manifest.*should\s+use\s+type.*instead.*CardType",
    # Generic patterns for the CardType/type confusion
    r"Incorrect.*card\s+header.*CardType.*type",
    r"card\s+header.*'CardType.*'type",
    r"card\s+header.*format.*CardType.*type",
    # Confusing CardType complaints that reference the correct format
    r"CardType:\s*\w+.*but.*APF.*requires.*exact\s+format",
    r"CardType:\s*\w+.*but.*requires.*exact\s+format",
    r"uses\s+'CardType:\s*\w+'.*but.*APF.*v2\.1\s+requires",
    # Inverse: LLM might say type should be CardType when type is used (also wrong direction)
    r"'type'.*should.*be.*'CardType'",
    r"uses\s+'?type:?\s*\w*'?\s+instead\s+of\s+'?CardType:?",
    # END_OF_CARDS hallucination - LLM thinks END_OF_CARDS after <!-- END_CARDS --> is wrong,
    # but APF v2.1 requires both: <!-- END_CARDS --> followed by END_OF_CARDS on last line
    r"Extra\s+'?END_OF_CARDS'?\s+text\s+after.*END_CARDS.*marker",
    r"extra\s+END_OF_CARDS",
    r"END_OF_CARDS.*after.*<!--\s*END_CARDS\s*-->",
    # Tag format hallucinations - underscores ARE correct for multi-word tags
    r"Tags.*use\s+underscores.*instead\s+of\s+spaces",
    r"Tags.*underscores.*instead.*hyphens",
    r"tags.*header.*use.*underscores",
    # Vague "proper spacing" complaints without specific issues
    r"proper\s+spacing\s+and\s+structure",
    r"with\s+proper\s+spacing",
]


def _filter_false_positives(error_details: str) -> tuple[str, bool]:
    """Filter out known LLM hallucination patterns from error details.

    Args:
        error_details: Error details string from LLM

    Returns:
        Tuple of (filtered error details, whether any false positives were removed)
    """
    if not error_details:
        return error_details, False

    original = error_details

    # Split into separate error items if numbered
    # Pattern matches "1) ...", "2) ...", etc.
    error_items = re.split(r"(?=\d+\)\s)", error_details)
    filtered_items = []
    removed_any = False

    for item in error_items:
        item = item.strip()
        if not item:
            continue

        is_false_positive = False
        for pattern in FALSE_POSITIVE_PATTERNS:
            if re.search(pattern, item, re.IGNORECASE):
                is_false_positive = True
                logger.debug(
                    "filtered_false_positive_error",
                    pattern=pattern,
                    error_text=item[:100],
                )
                removed_any = True
                break

        if not is_false_positive:
            filtered_items.append(item)

    if not filtered_items:
        # All errors were false positives
        return "", removed_any

    # Renumber remaining errors
    if len(filtered_items) == 1:
        # Single error, remove numbering
        result = re.sub(r"^\d+\)\s*", "", filtered_items[0])
    else:
        # Multiple errors, renumber
        result_parts = []
        for i, item in enumerate(filtered_items, 1):
            # Remove old numbering and add new
            item_text = re.sub(r"^\d+\)\s*", "", item)
            result_parts.append(f"{i}) {item_text}")
        result = " ".join(result_parts)

    if removed_any:
        logger.info(
            "false_positives_filtered",
            original_errors=original[:200],
            filtered_errors=(
                result[:200] if result else "(all errors were false positives)"
            ),
        )

    return result, removed_any


def _check_bilingual_consistency(cards: list[GeneratedCard]) -> list[str]:
    """Check for consistency between English and Russian cards with same card_index.

    Args:
        cards: All generated cards

    Returns:
        List of consistency error messages
    """
    errors = []

    # Group cards by card_index
    cards_by_index = {}
    for card in cards:
        if card.card_index not in cards_by_index:
            cards_by_index[card.card_index] = {}
        cards_by_index[card.card_index][card.lang] = card

    # Check each card_index that has both languages
    for card_index, lang_cards in cards_by_index.items():
        if "en" in lang_cards and "ru" in lang_cards:
            en_card = lang_cards["en"]
            ru_card = lang_cards["ru"]

            # Parse both cards to compare structure
            en_parsed = _parse_card_for_comparison(en_card.apf_html)
            ru_parsed = _parse_card_for_comparison(ru_card.apf_html)

            # Check for structural consistency
            consistency_errors = _compare_card_structures(
                card_index, en_parsed, ru_parsed
            )
            errors.extend(consistency_errors)

    return errors


def _parse_card_for_comparison(apf_html: str) -> dict:
    """Parse card HTML for bilingual consistency comparison.

    Args:
        apf_html: APF HTML content

    Returns:
        Dict with parsed components
    """
    result = {
        "title": "",
        "key_point_notes_count": 0,
        "key_point_notes": [],
        "other_notes_count": 0,
        "other_notes": [],
        "has_key_point_code": False,
        "card_type": "Simple",
    }

    # Extract title
    title_match = re.search(r"<!-- Title -->\s*\n(.*?)\n\s*\n", apf_html, re.DOTALL)
    if title_match:
        result["title"] = title_match.group(1).strip()

    # Extract key point notes
    notes_match = re.search(
        r"<!-- Key point notes -->\s*\n<ul>\s*\n(.*?)\n\s*</ul>", apf_html, re.DOTALL
    )
    if notes_match:
        ul_content = notes_match.group(1)
        li_matches = re.findall(r"<li>(.*?)</li>", ul_content, re.DOTALL)
        result["key_point_notes"] = [li.strip() for li in li_matches]
        result["key_point_notes_count"] = len(result["key_point_notes"])

    # Check for key point code
    if "<!-- Key point (code block / image) -->" in apf_html:
        code_match = re.search(
            r"<!-- Key point \(code block / image\) -->\s*\n(.*?)\n\s*\n<!-- Key point notes -->",
            apf_html,
            re.DOTALL,
        )
        if code_match:
            code_block = code_match.group(1).strip()
            result["has_key_point_code"] = code_block.startswith(
                "<pre><code"
            ) and code_block.endswith("</code></pre>")

    # Extract other notes
    other_match = re.search(
        r"<!-- Other notes.*? -->\s*\n<ul>\s*\n(.*?)\n\s*</ul>", apf_html, re.DOTALL
    )
    if other_match:
        ul_content = other_match.group(1)
        li_matches = re.findall(r"<li>(.*?)</li>", ul_content, re.DOTALL)
        result["other_notes"] = [li.strip() for li in li_matches]
        result["other_notes_count"] = len(result["other_notes"])

    # Extract card type from header
    header_match = re.search(
        r"<!-- Card \d+ \| slug: .*? \| CardType: (\w+) \| Tags:", apf_html
    )
    if header_match:
        result["card_type"] = header_match.group(1)

    return result


def _compare_card_structures(
    card_index: int, en_parsed: dict, ru_parsed: dict
) -> list[str]:
    """Compare parsed structures of EN and RU cards for consistency.

    Args:
        card_index: Card index number
        en_parsed: Parsed English card structure
        ru_parsed: Parsed Russian card structure

    Returns:
        List of consistency error messages
    """
    errors = []

    # Check key point notes count
    if en_parsed["key_point_notes_count"] != ru_parsed["key_point_notes_count"]:
        errors.append(
            f"Card {card_index}: Key point notes count mismatch - "
            f"EN has {en_parsed['key_point_notes_count']}, "
            f"RU has {ru_parsed['key_point_notes_count']}"
        )

    # Check other notes count (if present)
    if en_parsed["other_notes_count"] != ru_parsed["other_notes_count"]:
        errors.append(
            f"Card {card_index}: Other notes count mismatch - "
            f"EN has {en_parsed['other_notes_count']}, "
            f"RU has {ru_parsed['other_notes_count']}"
        )

    # Check card type consistency
    if en_parsed["card_type"] != ru_parsed["card_type"]:
        errors.append(
            f"Card {card_index}: Card type mismatch - "
            f"EN is {en_parsed['card_type']}, RU is {ru_parsed['card_type']}"
        )

    # Check code block presence
    if en_parsed["has_key_point_code"] != ru_parsed["has_key_point_code"]:
        errors.append(
            f"Card {card_index}: Code block presence mismatch - "
            f"EN {'has' if en_parsed['has_key_point_code'] else 'lacks'} code block, "
            f"RU {'has' if ru_parsed['has_key_point_code'] else 'lacks'} code block"
        )

    # Check for preference statements in key point notes (common source of inconsistency)
    en_notes_lower = " ".join(en_parsed["key_point_notes"]).lower()
    ru_notes_lower = " ".join(ru_parsed["key_point_notes"]).lower()

    # Check for contradictory preference statements
    preference_indicators = [
        ("prefer", "предпочтительнее"),
        ("better", "лучше"),
        ("recommended", "рекомендуется"),
        ("should use", "следует использовать"),
        ("instead of", "вместо"),
    ]

    for en_term, ru_term in preference_indicators:
        en_has_term = en_term in en_notes_lower
        ru_has_term = ru_term in ru_notes_lower

        if en_has_term != ru_has_term:
            errors.append(
                f"Card {card_index}: Preference statement mismatch - "
                f"EN {'has' if en_has_term else 'lacks'} '{en_term}' preference, "
                f"RU {'has' if ru_has_term else 'lacks'} '{ru_term}' preference"
            )

    return errors


def semantic_validation(
    cards: list[GeneratedCard],
    metadata: NoteMetadata,
    strict_mode: bool,
    ollama_client: BaseLLMProvider,
    model: str,
    temperature: float,
) -> PostValidationResult:
    """Perform semantic validation using LLM.

    Args:
        cards: Generated cards
        metadata: Note metadata for context
        strict_mode: Enable strict validation
        ollama_client: LLM provider instance
        model: Model to use for validation
        temperature: Sampling temperature

    Returns:
        PostValidationResult
    """
    # First, check bilingual consistency (fast, deterministic)
    bilingual_errors = _check_bilingual_consistency(cards)

    # If we have bilingual errors and are in strict mode, fail immediately
    if bilingual_errors and strict_mode:
        return PostValidationResult(
            is_valid=False,
            error_type="bilingual_consistency",
            error_details="; ".join(bilingual_errors),
            corrected_cards=None,
            validation_time=0.0,  # Will be set by caller
        )

    # Build validation prompt
    prompt = build_semantic_prompt(cards, metadata, strict_mode)

    # Get JSON schema for structured output
    json_schema = get_post_validation_schema()

    # Call LLM
    result = ollama_client.generate_json(
        model=model,
        prompt=prompt,
        system=SEMANTIC_VALIDATION_SYSTEM_PROMPT,
        temperature=temperature,
        json_schema=json_schema,
    )

    # Parse LLM response
    is_valid = result.get("is_valid", False)
    error_type = result.get("error_type", "none")
    error_details = result.get("error_details", "")
    corrected_cards_data = result.get("corrected_cards")

    # Include bilingual errors in the final result
    all_error_details = []
    if bilingual_errors:
        all_error_details.extend(bilingual_errors)
    if error_details:
        all_error_details.append(error_details)

    combined_error_details = "; ".join(all_error_details) if all_error_details else ""

    # If we have bilingual errors, validation should fail
    if bilingual_errors:
        is_valid = False
        error_type = "bilingual_consistency"

    # Filter out known false positive errors (LLM hallucinations) from LLM-generated errors only
    if error_details and not bilingual_errors:  # Only filter if no bilingual errors
        filtered_errors, had_false_positives = _filter_false_positives(error_details)

        if had_false_positives:
            if not filtered_errors:
                # All errors were false positives - mark as valid
                logger.info(
                    "validation_passed_after_false_positive_filter",
                    original_error_type=error_type,
                    original_errors=error_details[:200],
                )
                is_valid = True
                error_type = "none"
                combined_error_details = ""
            else:
                # Some real errors remain
                combined_error_details = filtered_errors

    error_details = combined_error_details

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
