"""Utility for applying CardCorrection patches to GeneratedCard objects."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from obsidian_anki_sync.utils.logging import get_logger

if TYPE_CHECKING:
    from obsidian_anki_sync.agents.models import CardCorrection, GeneratedCard

logger = get_logger(__name__)

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

        # Get current value for logging
        old_value = getattr(card, correction.field_name, None)

        try:
            # Apply the correction with proper validation/coercion
            # Use model_validate to ensure type coercion (e.g., str -> float for confidence)
            current_data = card.model_dump()
            current_data[correction.field_name] = correction.suggested_value

            # Import here to avoid circular imports at module level
            from obsidian_anki_sync.agents.models import GeneratedCard as CardModel

            updated_card = CardModel.model_validate(current_data)
            card_map[correction.card_index] = updated_card

            # Build human-readable change description
            old_preview = _truncate_value(old_value)
            new_preview = _truncate_value(correction.suggested_value)
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
