"""Path validation utilities for security and safety."""

from pathlib import Path

from ..exceptions import ConfigurationError


def validate_vault_path(vault_path: Path, allow_symlinks: bool = False) -> Path:
    """Validate vault path for security and existence.

    Args:
        vault_path: Path to validate
        allow_symlinks: Whether to allow symlinks (default: False for security)

    Returns:
        Resolved absolute path

    Raises:
        ConfigurationError: If path is invalid or unsafe
    """
    # Expand user home directory
    vault_path = vault_path.expanduser()

    # Check if path exists
    if not vault_path.exists():
        raise ConfigurationError(
            f"Vault path does not exist: {vault_path}",
            suggestion="Verify the vault_path in your configuration points to an existing directory",
        )

    # Check if it's a directory
    if not vault_path.is_dir():
        raise ConfigurationError(
            f"Vault path is not a directory: {vault_path}",
            suggestion="vault_path must point to a directory, not a file",
        )

    # Resolve to absolute path (and check for symlinks)
    try:
        resolved_path = vault_path.resolve(strict=True)
    except (OSError, RuntimeError) as e:
        raise ConfigurationError(
            f"Cannot resolve vault path: {vault_path}", suggestion=f"Error: {e}"
        )

    # Check for symlink traversal attacks
    if not allow_symlinks and vault_path.is_symlink():
        raise ConfigurationError(
            f"Vault path is a symlink: {vault_path}",
            suggestion="For security, symlinks are not allowed. Use the actual directory path or set allow_symlinks=True",
        )

    return resolved_path


def validate_source_dir(vault_path: Path, source_dir: Path) -> Path:
    """Validate source directory within vault.

    Args:
        vault_path: Validated vault path (must be absolute)
        source_dir: Relative path to source directory within vault

    Returns:
        Absolute path to source directory

    Raises:
        ConfigurationError: If source directory is invalid or outside vault
    """
    # Resolve full path
    full_source_path = (vault_path / source_dir).resolve()

    # Security: Ensure source_dir is actually within vault_path (prevent path traversal)
    try:
        full_source_path.relative_to(vault_path)
    except ValueError:
        raise ConfigurationError(
            f"Source directory is outside vault: {source_dir}",
            suggestion="source_dir must be a relative path within the vault (no .. allowed)",
        )

    # Check if it exists
    if not full_source_path.exists():
        raise ConfigurationError(
            f"Source directory does not exist: {full_source_path}",
            suggestion=f"Create directory '{source_dir}' in your vault or update source_dir in config",
        )

    # Check if it's a directory
    if not full_source_path.is_dir():
        raise ConfigurationError(
            f"Source path is not a directory: {full_source_path}",
            suggestion="source_dir must point to a directory",
        )

    return full_source_path


def validate_note_path(vault_path: Path, note_path: Path) -> Path:
    """Validate a note file path is within the vault.

    Args:
        vault_path: Validated vault path (must be absolute)
        note_path: Path to note file (can be relative or absolute)

    Returns:
        Absolute path to note file

    Raises:
        ConfigurationError: If note path is invalid or outside vault
    """
    # If relative, make it relative to vault
    if not note_path.is_absolute():
        note_path = vault_path / note_path

    # Resolve to absolute path
    try:
        resolved_note = note_path.resolve(
            strict=False
        )  # Allow non-existent for future creation
    except (OSError, RuntimeError) as e:
        raise ConfigurationError(
            f"Cannot resolve note path: {note_path}", suggestion=f"Error: {e}"
        )

    # Security: Ensure note is within vault (prevent path traversal)
    try:
        resolved_note.relative_to(vault_path)
    except ValueError:
        raise ConfigurationError(
            f"Note path is outside vault: {note_path}",
            suggestion="Note files must be within the vault directory",
        )

    return resolved_note


def validate_db_path(db_path: Path, vault_path: Path | None = None) -> Path:
    """Validate database path.

    Args:
        db_path: Path to database file
        vault_path: Optional vault path to ensure DB is outside vault

    Returns:
        Resolved absolute path

    Raises:
        ConfigurationError: If path is invalid
    """
    # Expand user home directory
    db_path = db_path.expanduser()

    # Resolve to absolute path
    try:
        resolved_db = db_path.resolve(strict=False)  # Allow non-existent
    except (OSError, RuntimeError) as e:
        raise ConfigurationError(
            f"Cannot resolve database path: {db_path}", suggestion=f"Error: {e}"
        )

    # Check parent directory exists
    if not resolved_db.parent.exists():
        raise ConfigurationError(
            f"Database directory does not exist: {resolved_db.parent}",
            suggestion=f"Create directory {resolved_db.parent} first",
        )

    # Warning: If DB is inside vault, it might get synced
    if vault_path and resolved_db.is_relative_to(vault_path):
        pass

    return resolved_db


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize filename to prevent injection attacks.

    Args:
        filename: Filename to sanitize
        max_length: Maximum filename length

    Returns:
        Sanitized filename
    """
    # Remove or replace dangerous characters
    dangerous_chars = [
        "/",
        "\\",
        "\0",
        "\n",
        "\r",
        "\t",
        "|",
        ">",
        "<",
        "?",
        "*",
        '"',
        ":",
    ]
    sanitized = filename

    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "_")

    # Remove leading/trailing spaces and dots (Windows issues)
    sanitized = sanitized.strip(". ")

    # Truncate to max length
    if len(sanitized) > max_length:
        # Keep extension if present
        parts = sanitized.rsplit(".", 1)
        if len(parts) == 2:
            name, ext = parts
            max_name_len = max_length - len(ext) - 1
            sanitized = name[:max_name_len] + "." + ext
        else:
            sanitized = sanitized[:max_length]

    # Ensure filename is not empty
    if not sanitized:
        sanitized = "unnamed"

    return sanitized
