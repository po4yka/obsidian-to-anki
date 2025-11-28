"""Domain layer for the Obsidian to Anki sync service.

This package contains the domain entities, services, and interfaces,
following Domain-Driven Design principles.
"""

from .entities.card import Card, CardManifest, SyncAction, SyncActionType
from .entities.note import Note, NoteMetadata, QAPair
from .interfaces.anki_client import IAnkiClient
from .interfaces.card_generator import ICardGenerator
from .interfaces.llm_provider import ILLMProvider
from .interfaces.note_parser import INoteParser
from .interfaces.state_repository import IStateRepository
from .services.content_hash_service import ContentHashService
from .services.slug_service import SlugService

__all__ = [
    # Entities
    "Card",
    "CardManifest",
    # Services
    "ContentHashService",
    # Interfaces
    "IAnkiClient",
    "ICardGenerator",
    "ILLMProvider",
    "INoteParser",
    "IStateRepository",
    "Note",
    "NoteMetadata",
    "QAPair",
    "SlugService",
    "SyncAction",
    "SyncActionType",
]
