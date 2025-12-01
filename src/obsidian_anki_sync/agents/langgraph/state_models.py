"""Schema models for LangGraph pipeline state validation.

These models provide runtime validation for serialized state structures
used in the LangGraph workflow.
"""

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class NoteMetadataSchema(BaseModel):
    """Schema for serialized note metadata in pipeline state."""

    id: str = Field(min_length=1)
    title: str
    topic: str
    tags: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    language: str = "en"


class QAPairSchema(BaseModel):
    """Schema for serialized Q&A pair in pipeline state."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    index: int = Field(ge=0)


class AutoFixResultSchema(BaseModel):
    """Schema for serialized autofix result in pipeline state."""

    fixed: bool
    changes: list[str] = Field(default_factory=list)
    error: str | None = None


class NoteCorrectionResultSchema(BaseModel):
    """Schema for serialized note correction result."""

    corrected: bool
    corrections: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


def validate_metadata_dict(data: dict[str, Any] | None) -> bool:
    """Validate metadata_dict structure.

    Args:
        data: Serialized metadata dictionary

    Returns:
        True if valid, False otherwise
    """
    if data is None:
        return True
    try:
        NoteMetadataSchema.model_validate(data)
        return True
    except ValidationError as e:
        logger.warning("invalid_metadata_dict", errors=str(e))
        return False


def validate_qa_pairs_dicts(data: list[dict[str, Any]] | None) -> bool:
    """Validate qa_pairs_dicts structure.

    Args:
        data: List of serialized Q&A pair dictionaries

    Returns:
        True if valid, False otherwise
    """
    if data is None:
        return True
    try:
        for qa in data:
            QAPairSchema.model_validate(qa)
        return True
    except ValidationError as e:
        logger.warning("invalid_qa_pairs_dicts", errors=str(e))
        return False


def validate_autofix_dict(data: dict[str, Any] | None) -> bool:
    """Validate autofix result structure.

    Args:
        data: Serialized autofix result dictionary

    Returns:
        True if valid, False otherwise
    """
    if data is None:
        return True
    try:
        AutoFixResultSchema.model_validate(data)
        return True
    except ValidationError as e:
        logger.warning("invalid_autofix_dict", errors=str(e))
        return False


def validate_pipeline_state(state: dict[str, Any]) -> list[str]:
    """Validate all serialized structures in pipeline state.

    Args:
        state: Full pipeline state dictionary

    Returns:
        List of validation error messages (empty if all valid)
    """
    errors: list[str] = []

    if state.get("metadata_dict"):
        if not validate_metadata_dict(state["metadata_dict"]):
            errors.append("Invalid metadata_dict structure")

    if state.get("qa_pairs_dicts"):
        if not validate_qa_pairs_dicts(state["qa_pairs_dicts"]):
            errors.append("Invalid qa_pairs_dicts structure")

    if state.get("autofix"):
        if not validate_autofix_dict(state["autofix"]):
            errors.append("Invalid autofix result structure")

    if state.get("note_correction"):
        try:
            NoteCorrectionResultSchema.model_validate(state["note_correction"])
        except ValidationError:
            errors.append("Invalid note_correction result structure")

    return errors
