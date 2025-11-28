"""Mock implementation of IStateRepository for testing."""

from typing import Any

from obsidian_anki_sync.domain.entities.card import Card
from obsidian_anki_sync.domain.entities.note import Note
from obsidian_anki_sync.domain.interfaces.state_repository import IStateRepository


class MockStateRepository(IStateRepository):
    """Mock implementation of state repository for testing.

    Provides in-memory storage for testing sync operations
    without requiring a real database.
    """

    def __init__(self):
        """Initialize mock repository."""
        self.notes: dict[str, Note] = {}
        self.cards: dict[str, Card] = {}
        self.content_hashes: dict[str, str] = {}
        self.sync_sessions: dict[str, dict[str, Any]] = {}
        self.session_counter = 0

    def get_note_by_id(self, note_id: str) -> Note | None:
        """Get note by ID."""
        return self.notes.get(note_id)

    def get_notes_by_path(self, file_path: str) -> list[Note]:
        """Get notes by file path."""
        return [
            note for note in self.notes.values() if str(note.file_path) == file_path
        ]

    def save_note(self, note: Note) -> None:
        """Save note."""
        self.notes[note.id] = note

    def delete_note(self, note_id: str) -> None:
        """Delete note."""
        self.notes.pop(note_id, None)

    def get_card_by_slug(self, slug: str) -> Card | None:
        """Get card by slug."""
        return self.cards.get(slug)

    def get_cards_by_note_id(self, note_id: str) -> list[Card]:
        """Get cards by note ID."""
        return [
            card for card in self.cards.values() if card.manifest.note_id == note_id
        ]

    def save_card(self, card: Card) -> None:
        """Save card."""
        self.cards[card.slug] = card

    def delete_card(self, slug: str) -> None:
        """Delete card."""
        self.cards.pop(slug, None)

    def get_all_notes(self) -> list[Note]:
        """Get all notes."""
        return list(self.notes.values())

    def get_all_cards(self) -> list[Card]:
        """Get all cards."""
        return list(self.cards.values())

    def get_sync_stats(self) -> dict[str, Any]:
        """Get sync statistics."""
        return {
            "total_notes": len(self.notes),
            "total_cards": len(self.cards),
            "last_sync": "2024-01-01T00:00:00Z",
            "sync_count": 42,
        }

    def save_sync_session(self, session_data: dict[str, Any]) -> str:
        """Save sync session."""
        session_id = f"session_{self.session_counter}"
        self.session_counter += 1
        self.sync_sessions[session_id] = session_data.copy()
        return session_id

    def get_sync_session(self, session_id: str) -> dict[str, Any] | None:
        """Get sync session."""
        return self.sync_sessions.get(session_id)

    def update_sync_progress(
        self, session_id: str, progress_data: dict[str, Any]
    ) -> None:
        """Update sync progress."""
        if session_id in self.sync_sessions:
            self.sync_sessions[session_id].update(progress_data)

    def get_content_hash(self, resource_id: str) -> str | None:
        """Get content hash."""
        return self.content_hashes.get(resource_id)

    def save_content_hash(self, resource_id: str, hash_value: str) -> None:
        """Save content hash."""
        self.content_hashes[resource_id] = hash_value

    def clear_expired_data(self, max_age_days: int) -> int:
        """Clear expired data (mock implementation)."""
        # Mock: just return a count
        return 5

    # Test helper methods

    def add_mock_note(self, note: Note) -> None:
        """Add a mock note for testing."""
        self.notes[note.id] = note

    def add_mock_card(self, card: Card) -> None:
        """Add a mock card for testing."""
        self.cards[card.slug] = card

    def get_notes_count(self) -> int:
        """Get number of stored notes."""
        return len(self.notes)

    def get_cards_count(self) -> int:
        """Get number of stored cards."""
        return len(self.cards)

    def has_note(self, note_id: str) -> bool:
        """Check if note exists."""
        return note_id in self.notes

    def has_card(self, slug: str) -> bool:
        """Check if card exists."""
        return slug in self.cards

    def reset(self) -> None:
        """Reset mock state."""
        self.notes.clear()
        self.cards.clear()
        self.content_hashes.clear()
        self.sync_sessions.clear()
        self.session_counter = 0
