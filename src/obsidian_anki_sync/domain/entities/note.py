"""Domain entity for Obsidian notes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..services.content_hash_service import ContentHashService


@dataclass(frozen=True)
class Note:
    """Domain entity representing an Obsidian note.

    This is a domain entity that encapsulates the business logic
    related to notes, following Domain-Driven Design principles.
    """

    id: str
    title: str
    content: str
    file_path: Path
    metadata: NoteMetadata = field(compare=False)
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        """Validate entity invariants."""
        if not self.id:
            raise ValueError("Note ID cannot be empty")
        if not self.title:
            raise ValueError("Note title cannot be empty")
        if not self.content:
            raise ValueError("Note content cannot be empty")
        if not self.file_path.exists():
            raise ValueError(f"Note file does not exist: {self.file_path}")

    @property
    def content_hash(self) -> str:
        """Calculate content hash for change detection."""
        return ContentHashService.compute_hash(self.content)

    @property
    def is_valid(self) -> bool:
        """Check if note meets business requirements."""
        return (
            self.has_minimum_content() and
            self.has_valid_metadata() and
            self.has_qa_pairs()
        )

    def has_minimum_content(self) -> bool:
        """Check if note has minimum required content."""
        return len(self.content.strip()) >= 100  # Business rule

    def has_valid_metadata(self) -> bool:
        """Check if metadata meets requirements."""
        return (
            self.metadata.topic and
            len(self.metadata.language_tags) > 0 and
            self.metadata.status != "draft"  # Business rule: no draft notes
        )

    def has_qa_pairs(self) -> bool:
        """Check if note contains Q&A pairs."""
        # Look for Q&A patterns in content
        content_lower = self.content.lower()
        return (
            "# question" in content_lower or
            "# вопрос" in content_lower or
            "## answer" in content_lower or
            "## ответ" in content_lower
        )

    def get_relative_path(self, vault_root: Path) -> str:
        """Get relative path from vault root."""
        try:
            return str(self.file_path.relative_to(vault_root))
        except ValueError:
            return str(self.file_path)

    def get_qa_pairs_count(self) -> int:
        """Count Q&A pairs in the note."""
        # Simple heuristic - count question headers
        return self.content.count("# Question") + self.content.count("# Вопрос")


@dataclass(frozen=True)
class NoteMetadata:
    """Value object for note metadata."""

    topic: str
    language_tags: list[str]
    subtopics: list[str] = field(default_factory=list)
    difficulty: str | None = None
    question_kind: str | None = None
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    status: str | None = None
    original_language: str | None = None
    source: str | None = None
    moc: str | None = None
    related: list[str] = field(default_factory=list)
    anki_note_type: str | None = None
    anki_slugs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate metadata invariants."""
        if not self.topic:
            raise ValueError("Topic cannot be empty")
        if not self.language_tags:
            raise ValueError("At least one language tag required")
        if len(self.language_tags) > 2:
            raise ValueError("Maximum 2 language tags allowed")

    @property
    def primary_language(self) -> str:
        """Get primary language tag."""
        return self.language_tags[0] if self.language_tags else "en"

    @property
    def is_bilingual(self) -> bool:
        """Check if note is bilingual."""
        return len(self.language_tags) == 2


@dataclass(frozen=True)
class QAPair:
    """Value object representing a Q&A pair."""

    card_index: int
    question_en: str
    question_ru: str
    answer_en: str
    answer_ru: str
    followups: str = ""
    references: str = ""
    related: str = ""
    context: str = ""

    def __post_init__(self) -> None:
        """Validate Q&A pair invariants."""
        if self.card_index < 1:
            raise ValueError("Card index must be positive")
        if not self.question_en.strip():
            raise ValueError("English question cannot be empty")
        if not self.question_ru.strip():
            raise ValueError("Russian question cannot be empty")
        if not self.answer_en.strip():
            raise ValueError("English answer cannot be empty")
        if not self.answer_ru.strip():
            raise ValueError("Russian answer cannot be empty")

    @property
    def content_hash(self) -> str:
        """Calculate content hash for this Q&A pair."""
        content = f"{self.question_en}{self.question_ru}{self.answer_en}{self.answer_ru}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:6]
