"""Data models for the sync service."""

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ManifestData(BaseModel):
    """Validated manifest data structure for parsing from Anki cards."""

    model_config = ConfigDict(extra="allow")

    slug: str = Field(min_length=1, description="Card slug identifier")
    source_path: str = Field(min_length=1, description="Source note path")
    card_index: int = Field(ge=0, description="Card index in note")
    lang: str = Field(min_length=2, max_length=2, description="Language code")


@dataclass
class NoteMetadata:
    """Metadata extracted from Obsidian note YAML frontmatter."""

    id: str
    title: str
    topic: str
    language_tags: list[str]
    created: datetime
    updated: datetime
    aliases: list[str] = field(default_factory=list)
    subtopics: list[str] = field(default_factory=list)
    question_kind: str | None = None
    difficulty: str | None = None
    original_language: str | None = None
    source: str | None = None
    source_note: str | None = None
    status: str | None = None
    moc: str | None = None
    related: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    anki_note_type: str | None = None
    anki_slugs: list[str] = field(default_factory=list)


@dataclass
class QAPair:
    """A single Q&A pair extracted from Obsidian note."""

    card_index: int  # 1-based
    question_en: str
    question_ru: str
    answer_en: str
    answer_ru: str
    followups: str = ""
    references: str = ""
    related: str = ""
    context: str = ""


@dataclass
class Manifest:
    """Card manifest for tracking and linking."""

    slug: str
    slug_base: str
    lang: str
    source_path: str
    source_anchor: str
    note_id: str
    note_title: str
    card_index: int
    guid: str
    hash6: str | None = None


@dataclass
class Card:
    """An APF card ready for Anki."""

    slug: str
    lang: str
    apf_html: str
    manifest: Manifest
    content_hash: str
    note_type: str = "APF::Simple"
    tags: list[str] = field(default_factory=list)
    guid: str = ""


@dataclass
class ValidationResult:
    """Result of APF validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return len(self.errors) == 0


@dataclass
class SyncAction:
    """An action to be performed during sync."""

    type: str  # 'create', 'update', 'delete', 'restore', 'skip'
    card: Card
    anki_guid: int | None = None
    reason: str | None = None
