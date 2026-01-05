"""Dependency models for PydanticAI agents."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from obsidian_anki_sync.models import NoteMetadata, QAPair


class PreValidationDeps(BaseModel):
    """Dependencies for pre-validation agent."""

    note_content: str
    metadata: NoteMetadata
    qa_pairs: list[QAPair]
    file_path: Path | None = None


class GenerationDeps(BaseModel):
    """Dependencies for card generation agent."""

    note_content: str
    metadata: NoteMetadata
    qa_pairs: list[QAPair]
    slug_base: str


class PostValidationDeps(BaseModel):
    """Dependencies for post-validation and memorization agents."""

    cards: list
    metadata: NoteMetadata
    strict_mode: bool = True


class CardSplittingDeps(BaseModel):
    """Dependencies for card splitting agent."""

    note_content: str
    metadata: NoteMetadata
    qa_pairs: list[QAPair]
