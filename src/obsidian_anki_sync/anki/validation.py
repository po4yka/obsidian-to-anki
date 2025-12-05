"""Comprehensive validation utilities for Anki import/export operations.

This module provides safety-focused validation for all input data used in
import/export operations, including file paths, data formats, and content.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml

from obsidian_anki_sync.exceptions import (
    DeckExportError,
    DeckImportError,
    ValidationError,
)
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationLimits:
    """Validation limits and constants."""

    # File size limits (in bytes)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_CSV_ROWS = 10000
    MAX_YAML_SIZE = 50 * 1024 * 1024  # 50MB

    # Content limits
    MAX_FIELD_LENGTH = 65536  # Anki's field size limit
    MAX_TAGS_PER_CARD = 100
    MAX_CARDS_PER_OPERATION = 1000

    # Path validation
    FORBIDDEN_PATH_CHARS = ['..', '<', '>', ':', '"', '|', '?', '*']
    ALLOWED_EXTENSIONS = {'.yaml', '.yml', '.csv', '.apkg', '.json'}


class PathValidator:
    """Path validation utilities."""

    @staticmethod
    def validate_file_path(file_path: str | Path) -> Path:
        """Validate file path for security and existence.

        Args:
            file_path: Path to validate

        Returns:
            Resolved Path object

        Raises:
            ValidationError: If path is invalid or insecure
        """
        try:
            path = Path(file_path).resolve()
        except (OSError, ValueError) as e:
            msg = f"Invalid file path: {file_path}"
            raise ValidationError(msg, suggestion="Check path format and permissions") from e

        # Check for path traversal attempts
        path_str = str(path)
        if any(forbidden in path_str for forbidden in ValidationLimits.FORBIDDEN_PATH_CHARS):
            msg = f"Path contains forbidden characters: {file_path}"
            raise ValidationError(msg, suggestion="Remove special characters from path")

        # Check if path is absolute (security risk)
        if path.is_absolute():
            msg = f"Absolute paths not allowed: {file_path}"
            raise ValidationError(msg, suggestion="Use relative paths")

        # Validate extension
        if path.suffix.lower() not in ValidationLimits.ALLOWED_EXTENSIONS:
            msg = f"Unsupported file extension: {path.suffix}"
            raise ValidationError(msg, suggestion=f"Use one of: {ValidationLimits.ALLOWED_EXTENSIONS}")

        return path

    @staticmethod
    def validate_file_exists(file_path: Path) -> None:
        """Validate that file exists and is readable.

        Args:
            file_path: Path to validate

        Raises:
            ValidationError: If file doesn't exist or isn't readable
        """
        if not file_path.exists():
            msg = f"File not found: {file_path}"
            raise ValidationError(msg, suggestion="Check file path and permissions")

        if not file_path.is_file():
            msg = f"Path is not a file: {file_path}"
            raise ValidationError(msg, suggestion="Provide path to a file")

        if not os.access(file_path, os.R_OK):
            msg = f"File not readable: {file_path}"
            raise ValidationError(msg, suggestion="Check file permissions")

    @staticmethod
    def validate_file_size(file_path: Path, max_size: int = ValidationLimits.MAX_FILE_SIZE) -> int:
        """Validate file size.

        Args:
            file_path: Path to validate
            max_size: Maximum allowed size in bytes

        Returns:
            File size in bytes

        Raises:
            ValidationError: If file is too large
        """
        try:
            size = file_path.stat().st_size
        except OSError as e:
            msg = f"Cannot get file size: {file_path}"
            raise ValidationError(msg, suggestion="Check file permissions") from e

        if size > max_size:
            msg = f"File too large: {size} bytes (max: {max_size})"
            raise ValidationError(msg, suggestion="Split large files or reduce content")

        if size == 0:
            msg = f"File is empty: {file_path}"
            raise ValidationError(msg, suggestion="File must contain data")

        return size


class DataValidator:
    """Data validation utilities."""

    @staticmethod
    def validate_yaml_data(data: Any) -> list[dict[str, Any]]:
        """Validate YAML data structure for card import.

        Args:
            data: Parsed YAML data

        Returns:
            Validated card data list

        Raises:
            DeckImportError: If data format is invalid
        """
        if not isinstance(data, list):
            msg = "YAML data must be a list of cards"
            raise DeckImportError(msg, suggestion="Ensure YAML contains a list at root level")

        if not data:
            msg = "YAML data is empty"
            raise DeckImportError(msg, suggestion="Add card data to the YAML file")

        if len(data) > ValidationLimits.MAX_CARDS_PER_OPERATION:
            msg = f"Too many cards: {len(data)} (max: {ValidationLimits.MAX_CARDS_PER_OPERATION})"
            raise DeckImportError(msg, suggestion="Split into smaller batches")

        for i, card in enumerate(data):
            DataValidator._validate_card_data(card, i)

        return data

    @staticmethod
    def validate_csv_data(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate CSV data for card import.

        Args:
            rows: CSV rows as dictionaries

        Returns:
            Validated card data list

        Raises:
            DeckImportError: If data format is invalid
        """
        if not rows:
            msg = "CSV data is empty"
            raise DeckImportError(msg, suggestion="Add card data to the CSV file")

        if len(rows) > ValidationLimits.MAX_CARDS_PER_OPERATION:
            msg = f"Too many cards: {len(rows)} (max: {ValidationLimits.MAX_CARDS_PER_OPERATION})"
            raise DeckImportError(msg, suggestion="Split into smaller batches")

        for i, row in enumerate(rows):
            DataValidator._validate_card_data(row, i)

        return rows

    @staticmethod
    def _validate_card_data(card: dict[str, Any], index: int) -> None:
        """Validate individual card data.

        Args:
            card: Card data dictionary
            index: Card index for error reporting

        Raises:
            DeckImportError: If card data is invalid
        """
        if not isinstance(card, dict):
            msg = f"Card {index} must be a dictionary"
            raise DeckImportError(msg, suggestion="Ensure each card is properly formatted")

        # Required fields validation
        required_fields = ['slug']
        for field in required_fields:
            if field not in card:
                msg = f"Card {index} missing required field: {field}"
                raise DeckImportError(msg, suggestion=f"Add '{field}' field to card")

        # Field length validation
        for key, value in card.items():
            if isinstance(value, str) and len(value) > ValidationLimits.MAX_FIELD_LENGTH:
                msg = f"Card {index} field '{key}' too long: {len(value)} chars"
                raise DeckImportError(msg, suggestion="Shorten field content")

        # Tags validation
        if 'tags' in card:
            tags = card['tags']
            if isinstance(tags, str):
                tags = tags.split()
            elif not isinstance(tags, list):
                msg = f"Card {index} tags must be string or list"
                raise DeckImportError(msg, suggestion="Format tags as space-separated string or list")

            if len(tags) > ValidationLimits.MAX_TAGS_PER_CARD:
                msg = f"Card {index} has too many tags: {len(tags)}"
                raise DeckImportError(msg, suggestion="Reduce number of tags per card")

            # Validate tag format
            for tag in tags:
                if not isinstance(tag, str) or not tag.strip():
                    msg = f"Card {index} has invalid tag: {tag}"
                    raise DeckImportError(msg, suggestion="Tags must be non-empty strings")

        # Slug validation
        slug = card.get('slug', '')
        if not isinstance(slug, str) or not slug.strip():
            msg = f"Card {index} has invalid slug: {slug}"
            raise DeckImportError(msg, suggestion="Slug must be non-empty string")

        # Basic slug format validation
        if not re.match(r'^[a-zA-Z0-9_-]+$', slug):
            msg = f"Card {index} slug contains invalid characters: {slug}"
            raise DeckImportError(msg, suggestion="Use only letters, numbers, hyphens, and underscores")

    @staticmethod
    def validate_cards_data(cards: list[Any]) -> list[Any]:
        """Validate list of Card objects for export.

        Args:
            cards: List of Card objects

        Returns:
            Validated cards list

        Raises:
            DeckExportError: If card data is invalid
        """
        if not cards:
            msg = "No cards to export"
            raise DeckExportError(msg, suggestion="Provide cards to export")

        if len(cards) > ValidationLimits.MAX_CARDS_PER_OPERATION:
            msg = f"Too many cards: {len(cards)} (max: {ValidationLimits.MAX_CARDS_PER_OPERATION})"
            raise DeckExportError(msg, suggestion="Split into smaller batches")

        for i, card in enumerate(cards):
            DataValidator._validate_export_card(card, i)

        return cards

    @staticmethod
    def _validate_export_card(card: Any, index: int) -> None:
        """Validate individual Card object for export.

        Args:
            card: Card object
            index: Card index for error reporting

        Raises:
            DeckExportError: If card data is invalid
        """
        required_attrs = ['slug', 'note_type', 'apf_html', 'tags', 'manifest']
        for attr in required_attrs:
            if not hasattr(card, attr):
                msg = f"Card {index} missing required attribute: {attr}"
                raise DeckExportError(msg, suggestion="Ensure card has all required attributes")

        # Validate slug
        slug = getattr(card, 'slug', '')
        if not isinstance(slug, str) or not slug.strip():
            msg = f"Card {index} has invalid slug: {slug}"
            raise DeckExportError(msg, suggestion="Slug must be non-empty string")

        # Validate note type
        note_type = getattr(card, 'note_type', '')
        if not isinstance(note_type, str) or not note_type.strip():
            msg = f"Card {index} has invalid note_type: {note_type}"
            raise DeckExportError(msg, suggestion="Note type must be non-empty string")

        # Validate APF HTML
        apf_html = getattr(card, 'apf_html', '')
        if not isinstance(apf_html, str):
            msg = f"Card {index} has invalid apf_html: {type(apf_html)}"
            raise DeckExportError(msg, suggestion="APF HTML must be string")

        if len(apf_html) > ValidationLimits.MAX_FIELD_LENGTH * 10:  # Allow larger for HTML
            msg = f"Card {index} APF HTML too long: {len(apf_html)} chars"
            raise DeckExportError(msg, suggestion="Reduce card content size")


class ContentValidator:
    """Content validation utilities."""

    @staticmethod
    def validate_deck_name(deck_name: str) -> str:
        """Validate deck name.

        Args:
            deck_name: Deck name to validate

        Returns:
            Validated deck name

        Raises:
            ValidationError: If deck name is invalid
        """
        if not isinstance(deck_name, str) or not deck_name.strip():
            msg = "Deck name must be non-empty string"
            raise ValidationError(msg, suggestion="Provide a valid deck name")

        if len(deck_name) > 255:  # Anki's limit
            msg = f"Deck name too long: {len(deck_name)} chars (max: 255)"
            raise ValidationError(msg, suggestion="Shorten deck name")

        # Basic validation for problematic characters
        forbidden_chars = [':', '"', '|', '<', '>', '*', '?']
        if any(char in deck_name for char in forbidden_chars):
            msg = f"Deck name contains forbidden characters: {deck_name}"
            raise ValidationError(msg, suggestion="Remove special characters from deck name")

        return deck_name.strip()

    @staticmethod
    def validate_note_type(note_type: str) -> str:
        """Validate note type name.

        Args:
            note_type: Note type name to validate

        Returns:
            Validated note type

        Raises:
            ValidationError: If note type is invalid
        """
        if not isinstance(note_type, str) or not note_type.strip():
            msg = "Note type must be non-empty string"
            raise ValidationError(msg, suggestion="Provide a valid note type name")

        if len(note_type) > 255:  # Anki's limit
            msg = f"Note type name too long: {len(note_type)} chars (max: 255)"
            raise ValidationError(msg, suggestion="Shorten note type name")

        return note_type.strip()

    @staticmethod
    def validate_output_directory(output_dir: str | Path) -> Path:
        """Validate output directory for writing.

        Args:
            output_dir: Directory path to validate

        Returns:
            Validated directory path

        Raises:
            ValidationError: If directory is invalid or not writable
        """
        try:
            path = Path(output_dir).resolve()
            path.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as e:
            msg = f"Cannot create output directory: {output_dir}"
            raise ValidationError(msg, suggestion="Check directory permissions") from e

        if not os.access(path, os.W_OK):
            msg = f"Directory not writable: {path}"
            raise ValidationError(msg, suggestion="Check directory permissions")

        return path


class SafeFileOperations:
    """Safe file operation utilities."""

    @staticmethod
    def safe_load_yaml(file_path: Path) -> Any:
        """Safely load YAML file with size and content validation.

        Args:
            file_path: Path to YAML file

        Returns:
            Parsed YAML data

        Raises:
            DeckImportError: If file cannot be loaded or parsed
        """
        # Validate file
        PathValidator.validate_file_exists(file_path)
        file_size = PathValidator.validate_file_size(file_path, ValidationLimits.MAX_YAML_SIZE)

        try:
            with file_path.open('r', encoding='utf-8') as f:
                # Read with size limit
                content = f.read(ValidationLimits.MAX_YAML_SIZE)
                if len(content) >= ValidationLimits.MAX_YAML_SIZE:
                    msg = f"YAML file too large: {file_size} bytes"
                    raise DeckImportError(msg, suggestion="Reduce file size or split content")

                data = yaml.safe_load(content)

        except (OSError, UnicodeDecodeError) as e:
            msg = f"Cannot read YAML file: {file_path}"
            raise DeckImportError(msg, suggestion="Check file encoding and permissions") from e
        except yaml.YAMLError as e:
            msg = f"Invalid YAML format: {file_path}"
            raise DeckImportError(msg, suggestion="Fix YAML syntax errors") from e

        logger.debug("yaml_file_loaded", path=str(file_path), size=file_size)
        return data

    @staticmethod
    def safe_load_csv(file_path: Path) -> list[dict[str, str]]:
        """Safely load CSV file with validation.

        Args:
            file_path: Path to CSV file

        Returns:
            List of CSV rows as dictionaries

        Raises:
            DeckImportError: If file cannot be loaded or parsed
        """
        import csv

        # Validate file
        PathValidator.validate_file_exists(file_path)
        PathValidator.validate_file_size(file_path)

        try:
            rows = []
            with file_path.open('r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)

                for i, row in enumerate(reader):
                    if i >= ValidationLimits.MAX_CSV_ROWS:
                        msg = f"CSV file has too many rows: {i + 1}"
                        raise DeckImportError(msg, suggestion="Limit to 10,000 rows or split file")

                    # Convert all values to strings and strip whitespace
                    cleaned_row = {k: v.strip() if v else '' for k, v in row.items()}
                    rows.append(cleaned_row)

        except (OSError, UnicodeDecodeError) as e:
            msg = f"Cannot read CSV file: {file_path}"
            raise DeckImportError(msg, suggestion="Check file encoding and permissions") from e
        except csv.Error as e:
            msg = f"Invalid CSV format: {file_path}"
            raise DeckImportError(msg, suggestion="Fix CSV format errors") from e

        logger.debug("csv_file_loaded", path=str(file_path), rows=len(rows))
        return rows

    @staticmethod
    def safe_write_file(file_path: Path, content: str, atomic: bool = True) -> None:
        """Safely write content to file.

        Args:
            file_path: Path to write to
            content: Content to write
            atomic: Whether to use atomic write

        Raises:
            DeckExportError: If file cannot be written
        """
        try:
            if atomic:
                from obsidian_anki_sync.utils.io import atomic_write
                with atomic_write(file_path) as f:
                    f.write(content)
            else:
                with file_path.open('w', encoding='utf-8') as f:
                    f.write(content)

        except OSError as e:
            msg = f"Cannot write to file: {file_path}"
            raise DeckExportError(msg, suggestion="Check file permissions and disk space") from e

        logger.debug("file_written", path=str(file_path), size=len(content))
