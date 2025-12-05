"""Safe and robust card exporter with validation, rollback, and progress tracking.

This module provides a safer alternative to the basic exporter with:
- Comprehensive output validation
- Transaction-like operations with cleanup
- Batch processing for large exports
- Progress tracking and detailed error reporting
- Resource cleanup and memory management
"""

import contextlib
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from obsidian_anki_sync.anki.validation import (
    ContentValidator,
    DataValidator,
    PathValidator,
    SafeFileOperations,
)
from obsidian_anki_sync.exceptions import DeckExportError
from obsidian_anki_sync.models import Card
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class ExportProgress:
    """Progress tracking for export operations."""

    def __init__(self, total_cards: int, progress_callback: Callable[[dict[str, Any]], None] | None = None):
        """Initialize progress tracker.

        Args:
            total_cards: Total number of cards to export
            progress_callback: Optional callback for progress updates
        """
        self.total_cards = total_cards
        self.processed = 0
        self.successful = 0
        self.errors = 0
        self.progress_callback = progress_callback

    def update(self, successful: int = 0, errors: int = 0) -> None:
        """Update progress counters.

        Args:
            successful: Number of successful exports
            errors: Number of errors
        """
        self.successful += successful
        self.errors += errors
        self.processed = self.successful + self.errors

        if self.progress_callback:
            progress_data = {
                'total': self.total_cards,
                'processed': self.processed,
                'successful': self.successful,
                'errors': self.errors,
                'percentage': (self.processed / self.total_cards * 100) if self.total_cards > 0 else 0,
            }
            self.progress_callback(progress_data)

    def get_summary(self) -> dict[str, Any]:
        """Get progress summary.

        Returns:
            Dictionary with final counts
        """
        return {
            'total': self.total_cards,
            'processed': self.processed,
            'successful': self.successful,
            'errors': self.errors,
        }


class ExportTransaction:
    """Transaction-like export operation with cleanup capability."""

    def __init__(self, output_path: Path):
        """Initialize export transaction.

        Args:
            output_path: Target output path
        """
        self.output_path = output_path
        self.temp_files: list[Path] = []
        self.backup_path: Path | None = None

    def add_temp_file(self, temp_path: Path) -> None:
        """Record a temporary file for cleanup.

        Args:
            temp_path: Temporary file path
        """
        self.temp_files.append(temp_path)

    def set_backup(self, backup_path: Path) -> None:
        """Set backup file for rollback.

        Args:
            backup_path: Backup file path
        """
        self.backup_path = backup_path

    def commit(self) -> None:
        """Commit the transaction by cleaning up temporary files."""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug("cleaned_up_temp_file", path=str(temp_file))
            except OSError as e:
                logger.warning("failed_to_cleanup_temp_file", path=str(temp_file), error=str(e))

        self.temp_files.clear()
        logger.debug("export_transaction_committed", output_path=str(self.output_path))

    def rollback(self) -> bool:
        """Rollback the transaction by restoring backup and cleaning up.

        Returns:
            True if rollback was successful, False otherwise
        """
        logger.warning("rolling_back_export_transaction", output_path=str(self.output_path))

        success = True

        # Restore backup if it exists
        if self.backup_path and self.backup_path.exists():
            try:
                if self.output_path.exists():
                    self.output_path.unlink()
                self.backup_path.rename(self.output_path)
                logger.debug("restored_backup_file", path=str(self.output_path))
            except OSError as e:
                logger.error("failed_to_restore_backup", error=str(e))
                success = False

        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug("cleaned_up_temp_file_on_rollback", path=str(temp_file))
            except OSError as e:
                logger.warning("failed_to_cleanup_temp_file_on_rollback", path=str(temp_file), error=str(e))

        self.temp_files.clear()
        logger.info("export_rollback_complete", success=success)
        return success


class SafeCardExporter:
    """Safe card exporter with validation, cleanup, and progress tracking."""

    def __init__(self) -> None:
        """Initialize safe exporter."""

    def export_to_yaml(
        self,
        cards: list[Card],
        output_path: str | Path,
        deck_name: str,
        deck_description: str = "",
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        include_note_id: bool = True,
        create_backup: bool = True,
    ) -> dict[str, Any]:
        """Safely export cards to YAML file.

        Args:
            cards: List of Card objects to export
            output_path: Path to output YAML file
            deck_name: Deck name for metadata
            deck_description: Optional deck description
            progress_callback: Optional callback for progress updates
            include_note_id: Whether to include noteId field
            create_backup: Whether to create backup of existing file

        Returns:
            Dictionary with operation results and metadata

        Raises:
            DeckExportError: If export fails
        """
        # Validate inputs
        output_path = PathValidator.validate_file_path(output_path)
        output_dir = output_path.parent
        ContentValidator.validate_output_directory(output_dir)
        deck_name = ContentValidator.validate_deck_name(deck_name)
        cards = DataValidator.validate_cards_data(cards)

        progress = ExportProgress(len(cards), progress_callback)
        transaction = ExportTransaction(output_path)

        # Create backup if requested and file exists
        if create_backup and output_path.exists():
            backup_path = output_path.with_suffix(f"{output_path.suffix}.backup")
            try:
                output_path.rename(backup_path)
                transaction.set_backup(backup_path)
                logger.debug("created_backup", backup=str(backup_path))
            except OSError as e:
                logger.warning("failed_to_create_backup", error=str(e))

        try:
            # Export with atomic write
            self._export_cards_to_yaml(
                cards=cards,
                output_path=output_path,
                deck_name=deck_name,
                deck_description=deck_description,
                include_note_id=include_note_id,
                transaction=transaction,
                progress=progress,
            )

            # Commit transaction
            transaction.commit()

            result = progress.get_summary()
            result.update({
                'output_path': str(output_path),
                'format': 'yaml',
                'deck_name': deck_name,
                'backup_created': transaction.backup_path is not None,
            })

            logger.info("safe_yaml_export_complete", **result)
            return result

        except Exception as e:
            # Rollback on failure
            logger.error("yaml_export_failed_rolling_back", error=str(e))
            transaction.rollback()

            msg = f"YAML export failed and was rolled back: {e}"
            raise DeckExportError(msg) from e

    def export_to_csv(
        self,
        cards: list[Card],
        output_path: str | Path,
        deck_name: str,
        deck_description: str = "",
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        include_note_id: bool = True,
        create_backup: bool = True,
    ) -> dict[str, Any]:
        """Safely export cards to CSV file.

        Args:
            cards: List of Card objects to export
            output_path: Path to output CSV file
            deck_name: Deck name for metadata
            deck_description: Optional deck description
            progress_callback: Optional callback for progress updates
            include_note_id: Whether to include noteId field
            create_backup: Whether to create backup of existing file

        Returns:
            Dictionary with operation results and metadata

        Raises:
            DeckExportError: If export fails
        """
        # Validate inputs
        output_path = PathValidator.validate_file_path(output_path)
        output_dir = output_path.parent
        ContentValidator.validate_output_directory(output_dir)
        deck_name = ContentValidator.validate_deck_name(deck_name)
        cards = DataValidator.validate_cards_data(cards)

        progress = ExportProgress(len(cards), progress_callback)
        transaction = ExportTransaction(output_path)

        # Create backup if requested and file exists
        if create_backup and output_path.exists():
            backup_path = output_path.with_suffix(f"{output_path.suffix}.backup")
            try:
                output_path.rename(backup_path)
                transaction.set_backup(backup_path)
                logger.debug("created_backup", backup=str(backup_path))
            except OSError as e:
                logger.warning("failed_to_create_backup", error=str(e))

        try:
            # Export with atomic write
            self._export_cards_to_csv(
                cards=cards,
                output_path=output_path,
                deck_name=deck_name,
                deck_description=deck_description,
                include_note_id=include_note_id,
                transaction=transaction,
                progress=progress,
            )

            # Commit transaction
            transaction.commit()

            result = progress.get_summary()
            result.update({
                'output_path': str(output_path),
                'format': 'csv',
                'deck_name': deck_name,
                'backup_created': transaction.backup_path is not None,
            })

            logger.info("safe_csv_export_complete", **result)
            return result

        except Exception as e:
            # Rollback on failure
            logger.error("csv_export_failed_rolling_back", error=str(e))
            transaction.rollback()

            msg = f"CSV export failed and was rolled back: {e}"
            raise DeckExportError(msg) from e

    def export_to_apkg(
        self,
        cards: list[Card],
        output_path: str | Path,
        deck_name: str,
        deck_description: str = "",
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        media_files: list[str] | None = None,
    ) -> dict[str, Any]:
        """Safely export cards to APKG file.

        Args:
            cards: List of Card objects to export
            output_path: Path to output APKG file
            deck_name: Deck name
            deck_description: Optional deck description
            progress_callback: Optional callback for progress updates
            media_files: Optional list of media file paths

        Returns:
            Dictionary with operation results and metadata

        Raises:
            DeckExportError: If export fails
        """
        # Validate inputs
        output_path = PathValidator.validate_file_path(output_path)
        output_dir = output_path.parent
        ContentValidator.validate_output_directory(output_dir)
        deck_name = ContentValidator.validate_deck_name(deck_name)
        cards = DataValidator.validate_cards_data(cards)

        progress = ExportProgress(len(cards), progress_callback)
        transaction = ExportTransaction(output_path)

        # Create backup if file exists
        if output_path.exists():
            backup_path = output_path.with_suffix(f"{output_path.suffix}.backup")
            try:
                output_path.rename(backup_path)
                transaction.set_backup(backup_path)
                logger.debug("created_backup", backup=str(backup_path))
            except OSError as e:
                logger.warning("failed_to_create_backup", error=str(e))

        try:
            # Export to temporary file first
            with tempfile.NamedTemporaryFile(
                suffix='.apkg',
                dir=output_dir,
                delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
                transaction.add_temp_file(temp_path)

                # Import here to avoid circular imports
                from .exporter import create_deck, export_deck

                # Create and export deck
                deck = create_deck(cards, deck_name, deck_description)
                export_deck(deck, temp_path, media_files)

                progress.update(successful=len(cards))

            # Move temporary file to final location
            temp_path.rename(output_path)

            # Commit transaction
            transaction.commit()

            result = progress.get_summary()
            result.update({
                'output_path': str(output_path),
                'format': 'apkg',
                'deck_name': deck_name,
                'media_files': len(media_files) if media_files else 0,
                'backup_created': transaction.backup_path is not None,
            })

            logger.info("safe_apkg_export_complete", **result)
            return result

        except Exception as e:
            # Rollback on failure
            logger.error("apkg_export_failed_rolling_back", error=str(e))
            transaction.rollback()

            msg = f"APKG export failed and was rolled back: {e}"
            raise DeckExportError(msg) from e

    def _export_cards_to_yaml(
        self,
        cards: list[Card],
        output_path: Path,
        deck_name: str,
        deck_description: str,
        include_note_id: bool,
        transaction: ExportTransaction,
        progress: ExportProgress,
    ) -> None:
        """Export cards to YAML format.

        Args:
            cards: Cards to export
            output_path: Output path
            deck_name: Deck name
            deck_description: Deck description
            include_note_id: Whether to include note IDs
            transaction: Export transaction
            progress: Progress tracker
        """
        import yaml

        # Convert cards to YAML-serializable format
        yaml_data = []
        for card in cards:
            try:
                # Import here to avoid circular imports
                from .exporter import map_apf_to_anki_fields

                # Map APF HTML to Anki fields
                fields = map_apf_to_anki_fields(card.apf_html, card.note_type)

                card_data: dict[str, Any] = {
                    "slug": card.slug,
                    "noteType": card.note_type,
                    "tags": card.tags,
                }

                # Add noteId if available and requested
                if include_note_id and card.manifest.guid:
                    card_data["noteId"] = card.manifest.guid

                # Add all fields
                card_data.update(fields)

                # Add manifest data
                card_data["manifest"] = {
                    "slug": card.manifest.slug,
                    "slug_base": card.manifest.slug_base,
                    "lang": card.manifest.lang,
                    "source_path": card.manifest.source_path,
                    "source_anchor": card.manifest.source_anchor,
                    "note_id": card.manifest.note_id,
                    "note_title": card.manifest.note_title,
                    "card_index": card.manifest.card_index,
                    "guid": card.manifest.guid,
                }

                if card.manifest.hash6:
                    card_data["manifest"]["hash6"] = card.manifest.hash6

                yaml_data.append(card_data)
                progress.update(successful=1)

            except Exception as e:
                progress.update(errors=1)
                logger.error(
                    "card_yaml_export_error",
                    error=str(e),
                    slug=card.slug,
                )

        # Write YAML file
        SafeFileOperations.safe_write_file(
            output_path,
            yaml.dump(
                yaml_data,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        )

    def _export_cards_to_csv(
        self,
        cards: list[Card],
        output_path: Path,
        deck_name: str,
        deck_description: str,
        include_note_id: bool,
        transaction: ExportTransaction,
        progress: ExportProgress,
    ) -> None:
        """Export cards to CSV format.

        Args:
            cards: Cards to export
            output_path: Output path
            deck_name: Deck name
            deck_description: Deck description
            include_note_id: Whether to include note IDs
            transaction: Export transaction
            progress: Progress tracker
        """
        if not cards:
            # Create empty CSV with headers
            SafeFileOperations.safe_write_file(output_path, "noteId,slug,noteType,tags,fields\n")
            return

        # Collect all unique field names across all cards
        all_field_names: set[str] = set()
        card_data_list = []

        for card in cards:
            try:
                # Import here to avoid circular imports
                from .exporter import map_apf_to_anki_fields

                fields = map_apf_to_anki_fields(card.apf_html, card.note_type)
                all_field_names.update(fields.keys())

                card_data = {
                    "slug": card.slug,
                    "noteType": card.note_type,
                    "tags": " ".join(card.tags) if card.tags else "",
                    **fields,
                }

                if include_note_id and card.manifest.guid:
                    card_data["noteId"] = card.manifest.guid

                card_data_list.append(card_data)
                progress.update(successful=1)

            except Exception as e:
                progress.update(errors=1)
                logger.error(
                    "card_csv_export_error",
                    error=str(e),
                    slug=card.slug,
                )

        # Sort field names for consistent column order
        common_fields = ["noteId", "slug", "noteType", "tags"]
        field_names = [
            f
            for f in common_fields
            if f in all_field_names or any(f in d for d in card_data_list)
        ]
        field_names.extend(sorted(all_field_names - set(common_fields)))

        # Write CSV file
        content_lines = []
        content_lines.append(",".join(f'"{name}"' for name in field_names))

        for card_data in card_data_list:
            row_values = []
            for field_name in field_names:
                value = str(card_data.get(field_name, "")).replace('"', '""')
                row_values.append(f'"{value}"')
            content_lines.append(",".join(row_values))

        SafeFileOperations.safe_write_file(output_path, "\n".join(content_lines) + "\n")


@contextlib.contextmanager
def safe_export_context():
    """Context manager for safe export operations.

    Yields:
        SafeCardExporter: Configured exporter instance
    """
    exporter = SafeCardExporter()
    try:
        yield exporter
    except Exception as e:
        logger.error("safe_export_context_failed", error=str(e))
        raise
