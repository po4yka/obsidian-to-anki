"""Syntax validation for APF cards."""

from obsidian_anki_sync.agents.models import GeneratedCard
from obsidian_anki_sync.apf.html_validator import validate_card_html
from obsidian_anki_sync.apf.linter import validate_apf
from obsidian_anki_sync.utils.logging import get_logger

from .error_categories import ErrorCategory
from .validation_models import ValidationError

logger = get_logger(__name__)


def syntax_validation(cards: list[GeneratedCard]) -> list[ValidationError]:
    """Perform syntax validation using existing linters.

    Args:
        cards: Generated cards to validate

    Returns:
        List of validation errors (empty if valid)
    """
    all_errors: list[ValidationError] = []

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
                all_errors.append(
                    ValidationError(
                        category=ErrorCategory.APF_FORMAT,
                        message=f"[{card.slug}] APF format: {error}",
                        code="apf_format_error",
                        context={"slug": card.slug, "raw_error": error},
                    )
                )

        # Validate HTML structure
        html_errors = validate_card_html(card.apf_html)

        for error in html_errors:
            all_errors.append(
                ValidationError(
                    category=ErrorCategory.HTML,
                    message=f"[{card.slug}] HTML: {error}",
                    code="html_syntax_error",
                    context={"slug": card.slug, "raw_error": error},
                )
            )

    return all_errors
