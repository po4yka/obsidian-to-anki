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
    # Inverse: LLM might say type should be CardType when type is used (also wrong direction)
    r"'type'.*should.*be.*'CardType'",
    r"uses\s+'?type:?\s*\w*'?\s+instead\s+of\s+'?CardType:?",
    # END_OF_CARDS hallucination - LLM thinks END_OF_CARDS after <!-- END_CARDS --> is wrong,
    # but APF v2.1 requires both: <!-- END_CARDS --> followed by END_OF_CARDS on last line
    r"Extra\s+'?END_OF_CARDS'?\s+text\s+after.*END_CARDS.*marker",
    r"extra\s+END_OF_CARDS",
    r"END_OF_CARDS.*after.*<!--\s*END_CARDS\s*-->",
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
    error_items = re.split(r'(?=\d+\)\s)', error_details)
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
        result = re.sub(r'^\d+\)\s*', '', filtered_items[0])
    else:
        # Multiple errors, renumber
        result_parts = []
        for i, item in enumerate(filtered_items, 1):
            # Remove old numbering and add new
            item_text = re.sub(r'^\d+\)\s*', '', item)
            result_parts.append(f"{i}) {item_text}")
        result = " ".join(result_parts)

    if removed_any:
        logger.info(
            "false_positives_filtered",
            original_errors=original[:200],
            filtered_errors=result[:
                                   200] if result else "(all errors were false positives)",
        )

    return result, removed_any


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

    # Filter out known false positive errors (LLM hallucinations)
    if error_details and not is_valid:
        filtered_errors, had_false_positives = _filter_false_positives(
            error_details)

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
                error_details = ""
            else:
                # Some real errors remain
                error_details = filtered_errors

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
