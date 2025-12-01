"""Utility for applying CardCorrection patches to GeneratedCard objects."""

from __future__ import annotations

import html
from copy import deepcopy
from typing import TYPE_CHECKING

from obsidian_anki_sync.utils.logging import get_logger

if TYPE_CHECKING:
    from obsidian_anki_sync.agents.models import CardCorrection, GeneratedCard

logger = get_logger(__name__)


def _decode_html_entities(value: str) -> str:
    """Decode HTML entities only if APF structure markers are incorrectly escaped.

    APF cards have two types of content:
    1. APF structure (<!-- Title -->, <pre><code>) - should be raw HTML
    2. Code content inside <code> blocks - should use &lt; &gt; for literal display

    This function only decodes if APF STRUCTURE markers are HTML-escaped,
    indicating the LLM incorrectly escaped the entire APF structure.
    Entities inside code blocks (which are correct) are preserved.

    Args:
        value: The APF HTML value that might have escaped structure markers

    Returns:
        Value with APF structure decoded, code content entities preserved
    """
    # Only decode if APF structure markers are HTML-escaped
    # This indicates the LLM incorrectly escaped the entire APF structure
    apf_structure_escaped = (
        "&lt;!--" in value  # Escaped APF comment marker
        or "&lt;pre&gt;" in value  # Escaped <pre> tag
        or "&lt;ul&gt;" in value  # Escaped <ul> tag
        or "&lt;li&gt;" in value  # Escaped <li> tag
        or "&lt;p&gt;" in value  # Escaped <p> tag
    )

    if not apf_structure_escaped:
        # Entities are only in code content (correct) - don't decode
        return value

    # LLM escaped everything - decode the entire value
    # Note: &amp;lt; will decode to &lt; (correct for code content)
    decoded = html.unescape(value)
    if decoded != value:
        logger.warning(
            "apf_structure_decoded",
            original_preview=value[:100],
            decoded_preview=decoded[:100],
        )
    return decoded

# Fields that can be patched on GeneratedCard
PATCHABLE_FIELDS = {"slug", "lang", "apf_html", "confidence"}


def apply_corrections(
    cards: list[GeneratedCard],
    corrections: list[CardCorrection],
) -> tuple[list[GeneratedCard], list[str]]:
    """Apply field-level corrections to cards.

    Args:
        cards: Original generated cards
        corrections: Field-level patches to apply

    Returns:
        Tuple of (corrected_cards, applied_changes)
        - corrected_cards: New list with corrections applied (deep copies)
        - applied_changes: Human-readable descriptions of changes made
    """
    if not corrections:
        return cards, []

    # Index cards by card_index for O(1) lookup
    card_map = {card.card_index: deepcopy(card) for card in cards}
    applied_changes: list[str] = []

    for correction in corrections:
        card = card_map.get(correction.card_index)
        if not card:
            logger.warning(
                "correction_card_not_found",
                card_index=correction.card_index,
                field=correction.field_name,
            )
            continue

        if correction.field_name not in PATCHABLE_FIELDS:
            logger.warning(
                "correction_field_not_patchable",
                field=correction.field_name,
                allowed=list(PATCHABLE_FIELDS),
            )
            continue

        # Get current value for comparison and logging
        old_value = getattr(card, correction.field_name, None)

        try:
            # Decode HTML entities if the LLM incorrectly HTML-escaped the content
            suggested_value = correction.suggested_value
            if correction.field_name == "apf_html" and isinstance(suggested_value, str):
                suggested_value = _decode_html_entities(suggested_value)

            # Skip no-op corrections (after decoding, value is same as original)
            if old_value == suggested_value:
                logger.debug(
                    "correction_skipped_noop",
                    card_index=correction.card_index,
                    field=correction.field_name,
                    rationale=correction.rationale,
                )
                continue

            # Apply the correction with proper validation/coercion
            # Use model_validate to ensure type coercion (e.g., str -> float for confidence)
            current_data = card.model_dump()
            current_data[correction.field_name] = suggested_value

            # Import here to avoid circular imports at module level
            from obsidian_anki_sync.agents.models import GeneratedCard as CardModel

            updated_card = CardModel.model_validate(current_data)
            card_map[correction.card_index] = updated_card

            # Build human-readable change description (use decoded value, not raw LLM response)
            old_preview = _truncate_value(old_value)
            new_preview = _truncate_value(suggested_value)
            change_desc = (
                f"Card {correction.card_index}: "
                f"{correction.field_name} '{old_preview}' -> '{new_preview}'"
            )
            applied_changes.append(change_desc)

            logger.info(
                "correction_applied",
                card_index=correction.card_index,
                field=correction.field_name,
                rationale=correction.rationale,
            )

        except (TypeError, ValueError) as e:
            logger.error(
                "correction_apply_failed",
                card_index=correction.card_index,
                field=correction.field_name,
                error=str(e),
            )

    # Return cards in original order (sorted by card_index)
    corrected_cards = [
        card_map[idx] for idx in sorted(card_map.keys())
    ]

    return corrected_cards, applied_changes


def _truncate_value(value: str | None, max_len: int = 50) -> str:
    """Truncate a value for logging display."""
    if value is None:
        return "<none>"
    value_str = str(value)
    if len(value_str) <= max_len:
        return value_str
    return value_str[:max_len] + "..."
