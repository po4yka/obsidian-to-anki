"""Centralized exception hierarchy for obsidian-anki-sync.

This module defines a clear exception hierarchy for all errors that can occur
during the synchronization process. All custom exceptions inherit from
ObsidianAnkiSyncError, making it easy to catch all sync-related errors.

Exception Hierarchy:
    ObsidianAnkiSyncError (base)
    ├── ConfigurationError - Configuration loading/validation errors
    ├── ProviderError - LLM provider communication errors
    │   ├── ProviderConnectionError - Provider connection failures
    │   └── ProviderTimeoutError - Provider request timeouts
    ├── ValidationError - Card/note validation errors
    │   └── ParserError - Obsidian note parsing errors
    ├── SyncError - Synchronization operation errors
    │   ├── IndexingError - Index building/reading errors
    │   └── StateError - State database errors
    ├── AnkiError - Anki-related errors
    │   ├── AnkiConnectError - AnkiConnect communication errors
    │   ├── FieldMappingError - Field mapping errors
    │   └── DeckExportError - Deck export errors
    └── AgentError - Multi-agent system errors
        ├── PreValidationError - Pre-validator agent errors
        ├── GeneratorError - Generator agent errors
        └── PostValidationError - Post-validator agent errors

Usage Examples:
    # Catch all sync errors
    try:
        engine.sync()
    except ObsidianAnkiSyncError as e:
        logger.error("Sync failed", error=str(e))

    # Catch specific error types
    try:
        config = load_config()
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        print(f"Suggestion: {e.suggestion}")
"""


class ObsidianAnkiSyncError(Exception):
    """Base exception for all sync-related errors.

    All custom exceptions in this module inherit from this base class,
    allowing users to catch all sync errors with a single except clause.

    Attributes:
        message: Human-readable error message
        suggestion: Optional suggestion for resolving the error
    """

    def __init__(self, message: str, suggestion: str | None = None):
        """Initialize the exception.

        Args:
            message: Human-readable error message
            suggestion: Optional suggestion for resolving the error
        """
        self.message = message
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with suggestion if available."""
        if self.suggestion:
            return f"{self.message}\n\nSuggestion: {self.suggestion}"
        return self.message


# Configuration Errors


class ConfigurationError(ObsidianAnkiSyncError):
    """Configuration loading or validation errors.

    Raised when:
    - Config file is missing or malformed
    - Required configuration values are missing
    - Configuration values fail validation
    - Environment variables are invalid
    """

    pass


# Provider Errors


class ProviderError(ObsidianAnkiSyncError):
    """LLM provider communication errors.

    Base class for all errors related to LLM providers (Ollama, LM Studio, OpenRouter).
    """

    pass


class ProviderConnectionError(ProviderError):
    """Provider connection failures.

    Raised when:
    - Cannot connect to provider service
    - Provider service is not running
    - Network connectivity issues
    """

    pass


class ProviderTimeoutError(ProviderError):
    """Provider request timeouts.

    Raised when:
    - Provider takes too long to respond
    - Model loading times out
    - Generation request exceeds timeout
    """

    pass


# Validation Errors


class ValidationError(ObsidianAnkiSyncError):
    """Card or note validation errors.

    Base class for validation-related errors.
    """

    pass


class ParserError(ValidationError):
    """Obsidian note parsing errors.

    Raised when:
    - Note format is invalid
    - YAML frontmatter is malformed
    - Required fields are missing
    - Q&A pairs are incorrectly formatted
    """

    pass


# Sync Errors


class SyncError(ObsidianAnkiSyncError):
    """Synchronization operation errors.

    Base class for errors during the sync process.
    """

    pass


class IndexingError(SyncError):
    """Index building or reading errors.

    Raised when:
    - Failed to build vault index
    - Failed to read Anki index
    - Index corruption detected
    """

    pass


class StateError(SyncError):
    """State database errors.

    Raised when:
    - Failed to read/write state database
    - Database corruption detected
    - Schema migration failed
    """

    pass


# Anki Errors


class AnkiError(ObsidianAnkiSyncError):
    """Base class for Anki-related errors."""

    pass


class AnkiConnectError(AnkiError):
    """AnkiConnect communication errors.

    Raised when:
    - Cannot connect to AnkiConnect
    - Anki is not running
    - AnkiConnect addon is not installed
    - AnkiConnect API returns an error
    """

    pass


class FieldMappingError(AnkiError):
    """Field mapping errors.

    Raised when:
    - Cannot map APF fields to Anki note type
    - Required fields are missing in note type
    - Field types are incompatible
    """

    pass


class DeckExportError(AnkiError):
    """Deck export errors.

    Raised when:
    - Failed to export deck to .apkg file
    - Invalid deck structure
    - File system errors during export
    """

    pass


# Agent System Errors


class AgentError(ObsidianAnkiSyncError):
    """Multi-agent system errors.

    Base class for errors in the multi-agent card generation pipeline.
    """

    pass


class PreValidationError(AgentError):
    """Pre-validator agent errors.

    Raised when:
    - Pre-validator rejects note structure
    - Pre-validator fails to execute
    - Pre-validator timeout
    """

    pass


class GeneratorError(AgentError):
    """Generator agent errors.

    Raised when:
    - Generator fails to create cards
    - Generator produces invalid output
    - Generator timeout
    """

    pass


class PostValidationError(AgentError):
    """Post-validator agent errors.

    Raised when:
    - Post-validator rejects generated cards
    - Post-validator fails to execute
    - Auto-fix attempts exhausted
    - Post-validator timeout
    """

    pass


# Export helper functions for backward compatibility


def get_exception_hierarchy() -> dict[str, list[str]]:
    """Get the exception hierarchy as a dictionary.

    Returns:
        Dictionary mapping base exceptions to their subclasses
    """
    return {
        "ObsidianAnkiSyncError": [
            "ConfigurationError",
            "ProviderError",
            "ValidationError",
            "SyncError",
            "AnkiError",
            "AgentError",
        ],
        "ProviderError": [
            "ProviderConnectionError",
            "ProviderTimeoutError",
        ],
        "ValidationError": [
            "ParserError",
        ],
        "SyncError": [
            "IndexingError",
            "StateError",
        ],
        "AnkiError": [
            "AnkiConnectError",
            "FieldMappingError",
            "DeckExportError",
        ],
        "AgentError": [
            "PreValidationError",
            "GeneratorError",
            "PostValidationError",
        ],
    }


def is_retriable_error(error: Exception) -> bool:
    """Check if an error is retriable.

    Some errors are transient and can be retried (network issues, timeouts),
    while others are permanent (configuration errors, validation errors).

    Args:
        error: The exception to check

    Returns:
        True if the error is retriable, False otherwise
    """
    retriable_types = (
        ProviderConnectionError,
        ProviderTimeoutError,
        AnkiConnectError,
        IndexingError,
    )
    return isinstance(error, retriable_types)
