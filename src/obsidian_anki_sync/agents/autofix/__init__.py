"""Auto-fix agents for correcting note issues during sync."""

from obsidian_anki_sync.agents.autofix.handlers import (
    AutoFixHandler,
    BrokenRelatedEntryHandler,
    BrokenWikilinkHandler,
    EmptyReferencesHandler,
    MissingRelatedQuestionsHandler,
    MocMismatchHandler,
    SectionOrderHandler,
    TitleFormatHandler,
    TrailingWhitespaceHandler,
)
from obsidian_anki_sync.agents.autofix.registry import AutoFixRegistry

__all__ = [
    "AutoFixHandler",
    "AutoFixRegistry",
    "BrokenRelatedEntryHandler",
    "BrokenWikilinkHandler",
    "EmptyReferencesHandler",
    "MissingRelatedQuestionsHandler",
    "MocMismatchHandler",
    "SectionOrderHandler",
    "TitleFormatHandler",
    "TrailingWhitespaceHandler",
]
