"""Centralized exception hierarchy for obsidian-anki-sync.

This module defines a clear exception hierarchy for all errors that can occur
during the synchronization process. All custom exceptions inherit from
ObsidianAnkiSyncError, making it easy to catch all sync-related errors.

Exception Hierarchy:
    ObsidianAnkiSyncError (base)
     ConfigurationError - Configuration loading/validation errors
     ProviderError - LLM provider communication errors
        ProviderConnectionError - Provider connection failures
        ProviderTimeoutError - Provider request timeouts
     ValidationError - Card/note validation errors
        ParserError - Obsidian note parsing errors
     SyncError - Synchronization operation errors
        IndexingError - Index building/reading errors
        StateError - State database errors
     AnkiError - Anki-related errors
        AnkiConnectError - AnkiConnect communication errors
        FieldMappingError - Field mapping errors
        DeckExportError - Deck export errors
     AgentError - Multi-agent system errors
         PreValidationError - Pre-validator agent errors
         GeneratorError - Generator agent errors
         PostValidationError - Post-validator agent errors

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

    # Use structured error codes
    from obsidian_anki_sync.error_codes import ErrorCode

    raise GeneratorError(
        "Generation timed out",
        error_code=ErrorCode.GEN_TIMEOUT_PRIMARY.value,
        context={"timeout_seconds": 300, "model": "gpt-4"},
    )
"""

from typing import Any


class ObsidianAnkiSyncError(Exception):
    """Base exception for all sync-related errors.

    All custom exceptions in this module inherit from this base class,
    allowing users to catch all sync errors with a single except clause.

    Attributes:
        message: Human-readable error message
        suggestion: Optional suggestion for resolving the error
        error_code: Structured error code for machine-readable handling
        context: Additional context for debugging (e.g., file paths, model names)
    """

    def __init__(
        self,
        message: str,
        suggestion: str | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize the exception.

        Args:
            message: Human-readable error message
            suggestion: Optional suggestion for resolving the error
            error_code: Structured error code (e.g., "GEN-TIMEOUT-001")
            context: Additional context for debugging
        """
        self.message = message
        self.suggestion = suggestion
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with error code and suggestion if available."""
        parts = []
        if self.error_code:
            parts.append(f"[{self.error_code}] {self.message}")
        else:
            parts.append(self.message)
        if self.suggestion:
            parts.append(f"\nSuggestion: {self.suggestion}")
        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for structured logging.

        Returns:
            Dictionary with error details
        """
        return {
            "message": self.message,
            "error_code": self.error_code,
            "suggestion": self.suggestion,
            "context": self.context,
            "type": type(self).__name__,
        }


# Configuration Errors


class ConfigurationError(ObsidianAnkiSyncError):
    """Configuration loading or validation errors.

    Raised when:
    - Config file is missing or malformed
    - Required configuration values are missing
    - Configuration values fail validation
    - Environment variables are invalid
    """


# Provider Errors


class ProviderError(ObsidianAnkiSyncError):
    """LLM provider communication errors.

    Base class for all errors related to LLM providers (Ollama, LM Studio, OpenRouter).
    """


class ProviderConnectionError(ProviderError):
    """Provider connection failures.

    Raised when:
    - Cannot connect to provider service
    - Provider service is not running
    - Network connectivity issues
    """


class ProviderTimeoutError(ProviderError):
    """Provider request timeouts.

    Raised when:
    - Provider takes too long to respond
    - Model loading times out
    - Generation request exceeds timeout
    """


# Validation Errors


class ValidationError(ObsidianAnkiSyncError):
    """Card or note validation errors.

    Base class for validation-related errors.
    """


class ParserError(ValidationError):
    """Obsidian note parsing errors.

    Raised when:
    - Note format is invalid
    - YAML frontmatter is malformed
    - Required fields are missing
    - Q&A pairs are incorrectly formatted
    """


# Sync Errors


class SyncError(ObsidianAnkiSyncError):
    """Synchronization operation errors.

    Base class for errors during the sync process.
    """


class IndexingError(SyncError):
    """Index building or reading errors.

    Raised when:
    - Failed to build vault index
    - Failed to read Anki index
    - Index corruption detected
    """


class StateError(SyncError):
    """State database errors.

    Raised when:
    - Failed to read/write state database
    - Database corruption detected
    - Schema migration failed
    """


# Anki Errors


class AnkiError(ObsidianAnkiSyncError):
    """Base class for Anki-related errors."""


class AnkiConnectError(AnkiError):
    """AnkiConnect communication errors.

    Raised when:
    - Cannot connect to AnkiConnect
    - Anki is not running
    - AnkiConnect addon is not installed
    - AnkiConnect API returns an error
    """


class FieldMappingError(AnkiError):
    """Field mapping errors.

    Raised when:
    - Cannot map APF fields to Anki note type
    - Required fields are missing in note type
    - Field types are incompatible
    """


class DeckExportError(AnkiError):
    """Deck export errors.

    Raised when:
    - Failed to export deck to .apkg file
    - Invalid deck structure
    - File system errors during export
    """


class DeckImportError(AnkiError):
    """Deck import errors.

    Raised when:
    - Failed to import deck from file
    - Invalid file format
    - File system errors during import
    - Invalid card data in file
    """


# Agent System Errors


class AgentError(ObsidianAnkiSyncError):
    """Multi-agent system errors.

    Base class for errors in the multi-agent card generation pipeline.
    """


class PreValidationError(AgentError):
    """Pre-validator agent errors.

    Raised when:
    - Pre-validator rejects note structure
    - Pre-validator fails to execute
    - Pre-validator timeout
    """


class GeneratorError(AgentError):
    """Generator agent errors.

    Raised when:
    - Generator fails to create cards
    - Generator produces invalid output
    - Generator timeout
    """


class PostValidationError(AgentError):
    """Post-validator agent errors.

    Raised when:
    - Post-validator rejects generated cards
    - Post-validator fails to execute
    - Auto-fix attempts exhausted
    - Post-validator timeout
    """


class CardGenerationError(AgentError):
    """Card generation pipeline error with detailed context.

    Raised when the card generation pipeline fails. Carries the full
    pipeline result for diagnostic purposes.

    Attributes:
        error_type: Category of failure (pre_validation, generation, post_validation)
        error_details: Specific error details from the failing stage
        note_path: Path to the note that failed
    """

    def __init__(
        self,
        message: str,
        *,
        error_type: str | None = None,
        error_details: str | None = None,
        note_path: str | None = None,
        suggestion: str | None = None,
    ):
        """Initialize card generation error.

        Args:
            message: Human-readable error message
            error_type: Category of failure
            error_details: Specific error details
            note_path: Path to the note that failed
            suggestion: Optional suggestion for resolving the error
        """
        self.error_type = error_type
        self.error_details = error_details
        self.note_path = note_path
        super().__init__(message, suggestion)


class CircuitBreakerOpenError(SyncError):
    """Circuit breaker is open due to too many failures.

    Raised when consecutive failures exceed the threshold and the
    circuit breaker prevents further operations.

    Attributes:
        consecutive_failures: Number of consecutive failures
        threshold: The failure threshold that was exceeded
    """

    def __init__(
        self,
        message: str,
        *,
        consecutive_failures: int,
        threshold: int,
        suggestion: str | None = None,
    ):
        """Initialize circuit breaker error.

        Args:
            message: Human-readable error message
            consecutive_failures: Number of consecutive failures
            threshold: The failure threshold
            suggestion: Optional suggestion for resolving the error
        """
        self.consecutive_failures = consecutive_failures
        self.threshold = threshold
        super().__init__(message, suggestion)


class APFValidationError(ValidationError):
    """APF HTML validation failed after retries.

    Raised when APF validation cannot produce valid HTML even after
    all retry attempts are exhausted.

    Attributes:
        slug: The card slug that failed validation
        validation_errors: List of validation errors
        attempts: Number of validation attempts made
    """

    def __init__(
        self,
        message: str,
        *,
        slug: str,
        validation_errors: list[str],
        attempts: int,
        suggestion: str | None = None,
    ):
        """Initialize APF validation error.

        Args:
            message: Human-readable error message
            slug: The card slug that failed
            validation_errors: List of validation errors
            attempts: Number of attempts made
            suggestion: Optional suggestion for resolving the error
        """
        self.slug = slug
        self.validation_errors = validation_errors
        self.attempts = attempts
        super().__init__(message, suggestion)


class ConcurrencyTimeoutError(SyncError):
    """Timed out waiting for concurrency slot.

    Raised when the system cannot acquire a concurrency slot within
    the allowed timeout period.

    Attributes:
        timeout: The timeout value in seconds
        max_concurrent: The maximum concurrent operations allowed
    """

    def __init__(
        self,
        message: str,
        *,
        timeout: float,
        max_concurrent: int,
        suggestion: str | None = None,
    ):
        """Initialize concurrency timeout error.

        Args:
            message: Human-readable error message
            timeout: The timeout value in seconds
            max_concurrent: Maximum concurrent operations
            suggestion: Optional suggestion for resolving the error
        """
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        super().__init__(message, suggestion)


class TruncationError(ProviderError):
    """Content exceeds model output token limits.

    Raised when note content is too large to process without truncation,
    even after chunking attempts.

    Attributes:
        content_tokens: Estimated input content tokens
        required_output_tokens: Estimated required output tokens
        model_limit: Model's maximum output token limit
        note_path: Path to the note that caused truncation
    """

    def __init__(
        self,
        message: str,
        *,
        content_tokens: int | None = None,
        required_output_tokens: int | None = None,
        model_limit: int | None = None,
        note_path: str | None = None,
        suggestion: str | None = None,
    ):
        """Initialize truncation error.

        Args:
            message: Human-readable error message
            content_tokens: Estimated input content tokens
            required_output_tokens: Estimated required output tokens
            model_limit: Model's maximum output token limit
            note_path: Path to the note that caused truncation
            suggestion: Optional suggestion for resolving the error
        """
        self.content_tokens = content_tokens
        self.required_output_tokens = required_output_tokens
        self.model_limit = model_limit
        self.note_path = note_path
        super().__init__(message, suggestion)


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
            "TruncationError",
        ],
        "ValidationError": [
            "ParserError",
            "APFValidationError",
        ],
        "SyncError": [
            "IndexingError",
            "StateError",
            "CircuitBreakerOpenError",
            "ConcurrencyTimeoutError",
        ],
        "AnkiError": [
            "AnkiConnectError",
            "FieldMappingError",
            "DeckExportError",
            "DeckImportError",
        ],
        "AgentError": [
            "PreValidationError",
            "GeneratorError",
            "PostValidationError",
            "CardGenerationError",
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
