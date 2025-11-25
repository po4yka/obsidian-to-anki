"""Syntax validation for APF cards."""

from ...apf.html_validator import validate_card_html
from ...apf.linter import validate_apf
from ...utils.logging import get_logger
from ..models import GeneratedCard

logger = get_logger(__name__)


def syntax_validation(cards: list[GeneratedCard]) -> list[str]:
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
