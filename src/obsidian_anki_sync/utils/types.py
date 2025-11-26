"""Shared type definitions for the obsidian-anki-sync project."""

from pydantic import BaseModel, ConfigDict, Field

from ..models import NoteMetadata, QAPair


class RecoveryResult(BaseModel):
    """Result of error recovery attempt."""

    model_config = ConfigDict(extra="allow")

    success: bool = Field(description="Whether recovery was successful")
    metadata: NoteMetadata | None = Field(
        default=None, description="Recovered note metadata")
    qa_pairs: list[QAPair] | None = Field(
        default=None, description="Recovered Q&A pairs")
    method_used: str = Field(
        default="", description="Recovery method that was used")
    warnings: list[str] = Field(
        default_factory=list, description="Recovery warnings")
    original_error: str | None = Field(
        default=None, description="Original error message")
