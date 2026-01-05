"""SQLite database for tracking sync state using Clean Architecture patterns.

This module implements the StateDB as a composition of focused repositories,
following SOLID principles and Clean Architecture.

Security Note:
    All SQL queries in this module use parameterized statements (? placeholders)
    to prevent SQL injection vulnerabilities. Never use string formatting or
    concatenation to build SQL queries. Always use parameter binding.

    Example of correct usage:
        cursor.execute("SELECT * FROM cards WHERE slug = ?", (slug,))

    Example of INCORRECT usage (vulnerable to SQL injection):
        cursor.execute(f"SELECT * FROM cards WHERE slug = '{slug}'")
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from obsidian_anki_sync.models import Card as ModelCard

    from .progress import SyncProgress

from obsidian_anki_sync.domain.entities.card import Card as DomainCard
from obsidian_anki_sync.domain.entities.note import Note as DomainNote
from obsidian_anki_sync.domain.interfaces.state_repository import IStateRepository
from obsidian_anki_sync.infrastructure.card_index_repository import CardIndexRepository
from obsidian_anki_sync.infrastructure.card_repository import CardRepository
from obsidian_anki_sync.infrastructure.database_connection_manager import (
    DatabaseConnectionManager,
)
from obsidian_anki_sync.infrastructure.database_schema_manager import (
    DatabaseSchemaManager,
)
from obsidian_anki_sync.infrastructure.note_index_repository import NoteIndexRepository
from obsidian_anki_sync.infrastructure.progress_repository import ProgressRepository
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class StateDB(IStateRepository):
    """SQLite database for tracking sync state using Clean Architecture patterns.

    This class implements the Facade pattern, composing specialized repositories
    for different concerns while maintaining the IStateRepository interface.

    Thread Safety:
        This class is thread-safe. Each repository handles its own thread-local
        connections. The database uses WAL mode for concurrent access.

    Async Compatibility:
        This class uses synchronous sqlite3. If used in async contexts, it will
        block the event loop. For async usage, consider using aiosqlite or
        running database operations in a thread pool executor.

    Usage:
        # As context manager (recommended)
        with StateDB(db_path) as db:
            db.insert_card(card, anki_guid)

        # Direct usage
        db = StateDB(db_path)
        try:
            db.insert_card(card, anki_guid)
        finally:
            db.close()
    """

    def __init__(self, db_path: Path):
        """Initialize database with composed repositories.

        Args:
            db_path: Path to SQLite database file

        Note:
            Initializes all repositories and creates database schema if needed.
        """
        self._db_path = db_path

        # Initialize infrastructure layer
        self._connection_manager = DatabaseConnectionManager(db_path)
        self._schema_manager = DatabaseSchemaManager(self._connection_manager)

        # Initialize schema
        self._schema_manager.initialize_schema()

        # Initialize repositories
        self._card_repo = CardRepository(self._connection_manager)
        self._progress_repo = ProgressRepository(self._connection_manager)
        self._note_index_repo = NoteIndexRepository(self._connection_manager)
        self._card_index_repo = CardIndexRepository(self._connection_manager)

        logger.debug("StateDB_initialized", db_path=str(db_path))

    # Card Repository delegations
    def insert_card(self, card: "ModelCard", anki_guid: int) -> None:
        """Insert a new card record."""
        self._card_repo.insert_card(card, anki_guid)

    def update_card(self, card: "ModelCard") -> None:
        """Update existing card record."""
        self._card_repo.update_card(card)

    def get_by_slug(self, slug: str) -> dict | None:
        """Get card by slug."""
        return self._card_repo.get_by_slug(slug)

    def get_by_guid(self, anki_guid: int) -> dict | None:
        """Get card by Anki GUID."""
        return self._card_repo.get_by_guid(anki_guid)

    def get_by_source(self, source_path: str) -> list[dict]:
        """Get all cards from a source note."""
        return self._card_repo.get_by_source(source_path)

    def get_all_cards(self) -> list[DomainCard]:
        """Get all cards."""
        return self._card_repo.get_all_cards()

    def get_all_cards_raw(self) -> list[dict]:
        """Get all cards as raw dictionary records."""
        return self._card_repo.get_all_cards_raw()

    def get_processed_note_paths(self) -> set[str]:
        """Get set of all note paths that have been processed."""
        return self._card_repo.get_processed_note_paths()

    def delete_card(self, slug: str) -> None:
        """Delete a card record."""
        self._card_repo.delete_card(slug)

    def insert_card_extended(
        self,
        card: "ModelCard",
        anki_guid: int,
        fields: dict[str, str],
        tags: list[str],
        deck_name: str,
        apf_html: str,
    ) -> None:
        """Insert card with full content storage for atomicity support."""
        self._card_repo.insert_card_extended(
            card, anki_guid, fields, tags, deck_name, apf_html
        )

    def update_card_extended(
        self, card: "ModelCard", fields: dict[str, str], tags: list[str], apf_html: str
    ) -> None:
        """Update card with full content for atomicity support."""
        self._card_repo.update_card_extended(card, fields, tags, apf_html)

    def upsert_card_extended(
        self,
        card: "ModelCard",
        anki_guid: int | None,
        fields: dict[str, str],
        tags: list[str],
        deck_name: str,
        apf_html: str,
        creation_status: str = "success",
    ) -> None:
        """Insert or update card with full content."""
        self._card_repo.upsert_card_extended(
            card, anki_guid, fields, tags, deck_name, apf_html, creation_status
        )

    def upsert_batch_extended(self, cards_data: list[dict[str, Any]]) -> None:
        """Insert or update multiple cards in a single transaction."""
        self._card_repo.upsert_batch_extended(cards_data)

    def get_pending_cards(self) -> list[dict[str, Any]]:
        """Get all cards with 'pending' creation status."""
        return self._card_repo.get_pending_cards()

    def insert_cards_batch(
        self,
        cards_data: list[tuple["ModelCard", int, dict[str, str], list[str], str]],
        deck_name: str,
    ) -> None:
        """Insert multiple cards in a single batch operation."""
        self._card_repo.insert_cards_batch(cards_data, deck_name)

    def update_cards_batch(
        self,
        cards_data: list[tuple["ModelCard", dict[str, str], list[str]]],
    ) -> None:
        """Update multiple cards in a single batch operation."""
        self._card_repo.update_cards_batch(cards_data)

    def delete_cards_batch(self, slugs: list[str]) -> None:
        """Delete multiple cards in a single batch operation."""
        self._card_repo.delete_cards_batch(slugs)

    def update_card_status(
        self,
        slug: str,
        status: str,
        error_message: str | None = None,
        increment_retry: bool = False,
    ) -> None:
        """Update card creation/update status for error tracking."""
        self._card_repo.update_card_status(slug, status, error_message, increment_retry)

    # Progress Repository delegations
    def save_progress(self, progress: "SyncProgress") -> None:
        """Save sync progress state."""
        self._progress_repo.save_progress(progress)

    def get_progress(self, session_id: str) -> "SyncProgress | None":
        """Get sync progress by session ID."""
        return self._progress_repo.get_progress(session_id)

    def get_all_progress(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent sync progress records."""
        return self._progress_repo.get_all_progress(limit)

    def get_incomplete_progress(self) -> list[dict[str, Any]]:
        """Get all incomplete sync sessions."""
        return self._progress_repo.get_incomplete_progress()

    def delete_progress(self, session_id: str) -> None:
        """Delete a progress record."""
        self._progress_repo.delete_progress(session_id)

    # Note Index Repository delegations
    def upsert_note_index(
        self,
        source_path: str,
        note_id: str | None,
        note_title: str | None,
        topic: str | None,
        language_tags: list[str],
        qa_pair_count: int,
        file_modified_at: datetime | None,
        metadata_json: str | None = None,
    ) -> None:
        """Insert or update a note in the index."""
        self._note_index_repo.upsert_note_index(
            source_path,
            note_id,
            note_title,
            topic,
            language_tags,
            qa_pair_count,
            file_modified_at,
            metadata_json,
        )

    def update_note_sync_status(
        self, source_path: str, status: str, error_message: str | None = None
    ) -> None:
        """Update sync status for a note."""
        self._note_index_repo.update_note_sync_status(
            source_path, status, error_message
        )

    def get_note_index(self, source_path: str) -> dict | None:
        """Get note index entry."""
        return self._note_index_repo.get_note_index(source_path)

    def get_all_notes_index(self) -> list[dict]:
        """Get all notes from index."""
        return self._note_index_repo.get_all_notes_index()

    def get_notes_by_status(self, status: str) -> list[dict]:
        """Get notes by sync status."""
        return self._note_index_repo.get_notes_by_status(status)

    # Card Index Repository delegations
    def upsert_card_index(
        self,
        source_path: str,
        card_index: int,
        lang: str,
        slug: str | None = None,
        anki_guid: int | None = None,
        note_id: str | None = None,
        note_title: str | None = None,
        content_hash: str | None = None,
        status: str = "expected",
        in_obsidian: bool = True,
        in_anki: bool = False,
        in_database: bool = False,
    ) -> None:
        """Insert or update a card in the index."""
        self._card_index_repo.upsert_card_index(
            source_path,
            card_index,
            lang,
            slug,
            anki_guid,
            note_id,
            note_title,
            content_hash,
            status,
            in_obsidian,
            in_anki,
            in_database,
        )

    def get_card_index_by_source(self, source_path: str) -> list[dict]:
        """Get all cards for a note."""
        return self._card_index_repo.get_card_index_by_source(source_path)

    def get_card_index_by_slug(self, slug: str) -> dict | None:
        """Get card index entry by slug."""
        return self._card_index_repo.get_card_index_by_slug(slug)

    def get_index_statistics(self) -> dict:
        """Get statistics from the index."""
        return self._card_index_repo.get_index_statistics()

    def clear_index(self) -> None:
        """Clear all index data (for rebuilding)."""
        self._note_index_repo.clear_index()
        self._card_index_repo.clear_index()

    # IStateRepository interface implementation

    def get_note_by_id(self, note_id: str) -> DomainNote | None:
        """Retrieve a note by its ID."""
        # Not implemented - notes are stored in the file system
        return None

    def get_notes_by_path(self, file_path: str) -> list[DomainNote]:
        """Retrieve notes by file path."""
        # Not implemented - notes are stored in the file system
        return []

    def save_note(self, note: DomainNote) -> None:
        """Save a note to the repository."""
        # Not implemented - notes are stored in the file system

    def delete_note(self, note_id: str) -> None:
        """Delete a note from the repository."""
        # Not implemented - notes are stored in the file system

    def get_card_by_slug(self, slug: str) -> DomainCard | None:
        """Retrieve a card by its slug."""
        # Delegate to get_all_cards and find the matching card
        cards = self.get_all_cards()
        return next((card for card in cards if card.slug == slug), None)

    def get_cards_by_note_id(self, note_id: str) -> list[DomainCard]:
        """Retrieve all cards for a note."""
        # Find cards by note_id in their manifest
        cards = self.get_all_cards()
        return [card for card in cards if card.manifest.note_id == note_id]

    def save_card(self, card: DomainCard) -> None:
        """Save a card to the repository."""
        # This method expects domain entities, but our internal methods work with models
        # For now, raise NotImplementedError as this would require model conversion
        msg = "save_card requires model conversion - use insert_card instead"
        raise NotImplementedError(msg)

    def get_all_notes(self) -> list[DomainNote]:
        """Retrieve all notes."""
        # Not implemented - notes are stored in the file system
        return []

    def get_sync_stats(self) -> dict[str, Any]:
        """Get synchronization statistics."""
        return self.get_index_statistics()

    def save_sync_session(self, session_data: dict[str, Any]) -> str:
        """Save sync session data."""
        # Generate a simple session ID
        import uuid

        session_id = str(uuid.uuid4())
        # For now, just return the session ID - progress tracking is separate
        return session_id

    def get_sync_session(self, session_id: str) -> dict[str, Any | None]:
        """Retrieve sync session data."""
        progress = self.get_progress(session_id)
        if progress:
            # Convert progress to dict with Any | None values
            return {"progress": progress}  # type: ignore[return-value]
        # Return empty dict instead of None to match interface
        return {}

    def update_sync_progress(
        self, session_id: str, progress_data: dict[str, Any]
    ) -> None:
        """Update sync progress for a session."""
        # This is a simplified implementation - real progress tracking uses SyncProgress objects

    def get_content_hash(self, resource_id: str) -> str | None:
        """Get stored content hash for a resource."""
        # Content hashes are stored in card records
        card_data = self.get_by_slug(resource_id)
        return card_data.get("content_hash") if card_data else None

    def save_content_hash(self, resource_id: str, hash_value: str) -> None:
        """Save content hash for a resource."""
        # This would require updating card records - not implemented for simplicity

    def clear_expired_data(self, max_age_days: int) -> int:
        """Clear expired data from repository."""
        # Not implemented - would require date-based cleanup logic
        return 0

    def close(self) -> None:
        """Close all database connections."""
        self._connection_manager.close()

    def __enter__(self) -> "StateDB":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
