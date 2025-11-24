"""Shared type definitions for the obsidian-anki-sync project."""

from dataclasses import dataclass
from typing import List, Optional

from ..models import NoteMetadata, QAPair


@dataclass
class RecoveryResult:
    """Result of error recovery attempt."""
    success: bool
    metadata: Optional[NoteMetadata] = None
    qa_pairs: Optional[List[QAPair]] = None
    method_used: str = ""
    warnings: List[str] = None
    original_error: Optional[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
