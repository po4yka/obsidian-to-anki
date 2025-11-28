"""Data models for the sync service."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ManifestData(BaseModel):
    """Validated manifest data structure for parsing from Anki cards."""

    model_config = ConfigDict(extra="allow")

    slug: str = Field(min_length=1, description="Card slug identifier")
    source_path: str = Field(min_length=1, description="Source note path")
    card_index: int = Field(ge=0, description="Card index in note")
    lang: str = Field(min_length=2, max_length=2, description="Language code")


class NoteMetadata(BaseModel):
    """Metadata extracted from Obsidian note YAML frontmatter."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1, description="Unique identifier for the note")
    title: str = Field(min_length=1, description="Note title")
    topic: str = Field(min_length=1, description="Note topic")
    language_tags: list[str] = Field(default_factory=list, description="Language tags")
    created: datetime = Field(description="Creation timestamp")
    updated: datetime = Field(description="Last update timestamp")
    aliases: list[str] = Field(default_factory=list, description="Note aliases")
    subtopics: list[str] = Field(default_factory=list, description="Subtopics")
    question_kind: str | None = Field(default=None, description="Type of questions")
    difficulty: str | None = Field(default=None, description="Difficulty level")
    original_language: str | None = Field(default=None, description="Original language")
    source: str | None = Field(default=None, description="Source information")
    source_note: str | None = Field(default=None, description="Source note reference")
    status: str | None = Field(default=None, description="Note status")
    moc: str | None = Field(default=None, description="Map of Content reference")
    related: list[str] = Field(default_factory=list, description="Related notes")
    tags: list[str] = Field(default_factory=list, description="Note tags")
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="Source metadata"
    )
    anki_note_type: str | None = Field(default=None, description="Anki note type")
    anki_slugs: list[str] = Field(
        default_factory=list, description="Generated Anki slugs"
    )


class QAPair(BaseModel):
    """A single Q&A pair extracted from Obsidian note."""

    model_config = ConfigDict(extra="allow")

    card_index: int = Field(ge=1, description="Card index (1-based)")
    question_en: str = Field(min_length=1, description="Question in English")
    question_ru: str = Field(min_length=1, description="Question in Russian")
    answer_en: str = Field(min_length=1, description="Answer in English")
    answer_ru: str = Field(min_length=1, description="Answer in Russian")
    followups: str = Field(default="", description="Follow-up questions")
    references: str = Field(default="", description="References and citations")
    related: str = Field(default="", description="Related information")
    context: str = Field(default="", description="Additional context")


class Manifest(BaseModel):
    """Card manifest for tracking and linking."""

    model_config = ConfigDict(extra="allow")

    slug: str = Field(min_length=1, description="Unique card slug")
    slug_base: str = Field(
        min_length=1, description="Base slug without language suffix"
    )
    lang: str = Field(min_length=2, max_length=2, description="Language code")
    source_path: str = Field(min_length=1, description="Source note file path")
    source_anchor: str = Field(min_length=1, description="Anchor in source file")
    note_id: str = Field(min_length=1, description="Note unique identifier")
    note_title: str = Field(min_length=1, description="Note title")
    card_index: int = Field(ge=0, description="Card index in note")
    guid: str = Field(min_length=1, description="Anki GUID")
    hash6: str | None = Field(
        default=None, min_length=6, max_length=6, description="6-character content hash"
    )


class Card(BaseModel):
    """An APF card ready for Anki."""

    model_config = ConfigDict(extra="allow")

    slug: str = Field(min_length=1, description="Unique card slug")
    lang: str = Field(min_length=2, max_length=2, description="Language code")
    apf_html: str = Field(min_length=1, description="APF formatted HTML content")
    manifest: Manifest = Field(description="Card manifest information")
    content_hash: str = Field(
        min_length=1, description="Content hash for change detection"
    )
    note_type: str = Field(default="APF::Simple", description="Anki note type")
    tags: list[str] = Field(default_factory=list, description="Card tags")
    guid: str = Field(default="", description="Anki GUID (populated after creation)")


class ValidationResult(BaseModel):
    """Result of APF validation."""

    model_config = ConfigDict(extra="allow")

    errors: list[str] = Field(default_factory=list, description="Validation errors")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")

    @computed_field
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return len(self.errors) == 0


class SyncAction(BaseModel):
    """An action to be performed during sync."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(
        description="Action type: 'create', 'update', 'delete', 'restore', 'skip'"
    )
    card: Card = Field(description="Card to be acted upon")
    anki_guid: int | None = Field(
        default=None, description="Anki GUID for existing cards"
    )
    reason: str | None = Field(default=None, description="Reason for this action")
