"""Domain entity for Anki cards."""

from __future__ import annotations
from enum import Enum

from dataclasses import dataclass, field
from typing import Any

from ..services.slug_service import SlugService


@dataclass(frozen=True)
class Card:
    """Domain entity representing an Anki card.

    This encapsulates the business logic for cards,
    including validation, slug generation, and sync state.
    """

    slug: str
    language: str
    apf_html: str
    manifest: CardManifest
    note_type: str
    tags: list[str] = field(default_factory=list)
    anki_guid: str | None = None

    def __post_init__(self) -> None:
        """Validate entity invariants."""
        if not self.slug:
            raise ValueError("Card slug cannot be empty")
        if len(self.language) != 2:
            raise ValueError("Language must be 2-character code")
        if not self.apf_html.strip():
            raise ValueError("APF HTML content cannot be empty")
        if not self.note_type:
            raise ValueError("Note type cannot be empty")

    @property
    def is_new(self) -> bool:
        """Check if card is new (not yet in Anki)."""
        return self.anki_guid is None

    @property
    def is_valid(self) -> bool:
        """Check if card meets business requirements."""
        return (
            self.has_valid_apf_format() and
            self.has_valid_manifest() and
            self.has_required_fields()
        )

    def has_valid_apf_format(self) -> bool:
        """Check if APF format is valid."""
        # Basic APF validation - look for required markers
        html = self.apf_html
        return (
            "<!-- PROMPT_VERSION: apf-v" in html and
            "<!-- BEGIN_CARDS -->" in html and
            "<!-- END_CARDS -->" in html
        )

    def has_valid_manifest(self) -> bool:
        """Check if manifest is valid."""
        return (
            self.manifest.slug == self.slug and
            self.manifest.lang == self.language and
            self.manifest.note_id and
            self.manifest.card_index >= 0
        )

    def has_required_fields(self) -> bool:
        """Check if card has all required fields."""
        # APF cards must have at least question and answer
        html = self.apf_html.lower()
        return "question" in html and "answer" in html

    @property
    def content_hash(self) -> str:
        """Calculate content hash for change detection."""
        content = f"{self.apf_html}{self.note_type}{','.join(sorted(self.tags))}"
        return SlugService.compute_hash(content)

    def with_guid(self, guid: str) -> Card:
        """Create a new Card instance with Anki GUID."""
        return Card(
            slug=self.slug,
            language=self.language,
            apf_html=self.apf_html,
            manifest=self.manifest,
            note_type=self.note_type,
            tags=self.tags,
            anki_guid=guid,
        )

    def update_content(self, new_apf_html: str) -> Card:
        """Create a new Card with updated content."""
        return Card(
            slug=self.slug,
            language=self.language,
            apf_html=new_apf_html,
            manifest=self.manifest,
            note_type=self.note_type,
            tags=self.tags,
            anki_guid=self.anki_guid,
        )


@dataclass(frozen=True)
class CardManifest:
    """Value object for card manifest information."""

    slug: str
    slug_base: str
    lang: str
    source_path: str
    source_anchor: str
    note_id: str
    note_title: str
    card_index: int
    guid: str | None = None
    hash6: str | None = None

    def __post_init__(self) -> None:
        """Validate manifest invariants."""
        if not self.slug:
            raise ValueError("Slug cannot be empty")
        if not self.slug_base:
            raise ValueError("Slug base cannot be empty")
        if len(self.lang) != 2:
            raise ValueError("Language must be 2-character code")
        if not self.source_path:
            raise ValueError("Source path cannot be empty")
        if not self.source_anchor:
            raise ValueError("Source anchor cannot be empty")
        if not self.note_id:
            raise ValueError("Note ID cannot be empty")
        if not self.note_title:
            raise ValueError("Note title cannot be empty")
        if self.card_index < 0:
            raise ValueError("Card index cannot be negative")

    @property
    def is_linked_to_note(self) -> bool:
        """Check if manifest links to a valid note."""
        return bool(self.note_id and self.source_path)

    @property
    def anchor_url(self) -> str:
        """Generate anchor URL for Obsidian linking."""
        return f"[[{self.source_path}#{self.source_anchor}]]"


@dataclass(frozen=True)
class SyncAction:
    """Domain entity representing a sync action."""

    action_type: SyncActionType
    card: Card
    anki_guid: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate action invariants."""
        if not isinstance(self.action_type, SyncActionType):
            raise ValueError("Invalid action type")

    @property
    def is_create(self) -> bool:
        """Check if this is a create action."""
        return self.action_type == SyncActionType.CREATE

    @property
    def is_update(self) -> bool:
        """Check if this is an update action."""
        return self.action_type == SyncActionType.UPDATE

    @property
    def is_delete(self) -> bool:
        """Check if this is a delete action."""
        return self.action_type == SyncActionType.DELETE

    @property
    def is_skip(self) -> bool:
        """Check if this is a skip action."""
        return self.action_type == SyncActionType.SKIP


class SyncActionType(Enum):
    """Enumeration of possible sync action types."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"
    SKIP = "skip"
