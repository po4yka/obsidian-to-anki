"""Validation for Obsidian notes."""

from pathlib import Path

from ..models import NoteMetadata, QAPair
from ..utils.logging import get_logger

logger = get_logger(__name__)


def validate_note(metadata: NoteMetadata, qa_pairs: list[QAPair], file_path: Path) -> list[str]:
    """
    Validate note structure and content.

    Args:
        metadata: Parsed metadata
        qa_pairs: Parsed Q/A pairs
        file_path: Path to the file

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check language tags
    if not metadata.language_tags:
        errors.append("No language_tags specified")
    else:
        allowed_langs = {'en', 'ru'}  # Configurable
        invalid = set(metadata.language_tags) - allowed_langs
        if invalid:
            errors.append(f"Invalid language tags: {invalid}")

    # Check Q/A pairs
    if not qa_pairs:
        errors.append("No Q/A pairs found in note")

    for qa_pair in qa_pairs:
        # Check for each language in language_tags
        if 'en' in metadata.language_tags:
            if not qa_pair.question_en:
                errors.append(f"Missing English question in pair {qa_pair.card_index}")
            if not qa_pair.answer_en:
                errors.append(f"Missing English answer in pair {qa_pair.card_index}")

        if 'ru' in metadata.language_tags:
            if not qa_pair.question_ru:
                errors.append(f"Missing Russian question in pair {qa_pair.card_index}")
            if not qa_pair.answer_ru:
                errors.append(f"Missing Russian answer in pair {qa_pair.card_index}")

    # Check file naming convention
    if not file_path.name.startswith('q-'):
        errors.append(f"File should start with 'q-': {file_path.name}")

    if errors:
        logger.warning("validation_failed", file=str(file_path), errors=errors)

    return errors

