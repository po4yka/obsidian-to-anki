"""Custom exceptions for the agent system.

This module provides a hierarchy of exceptions for better error handling
and debugging in the agent pipeline.
"""


class AgentError(Exception):
    """Base exception for all agent-related errors."""

    def __init__(self, message: str, details: dict | None = None):
        """Initialize agent error.

        Args:
            message: Error message
            details: Optional dict with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ValidationError(AgentError):
    """Raised when validation fails."""


class PreValidationError(ValidationError):
    """Raised when pre-validation fails."""


class PostValidationError(ValidationError):
    """Raised when post-validation fails."""


class HighlightError(AgentError):
    """Raised when highlight agent fails."""


class GenerationError(AgentError):
    """Raised when card generation fails."""


class ModelError(AgentError):
    """Raised when LLM model operations fail."""


class ModelNotAvailableError(ModelError):
    """Raised when a required model is not available."""


class ModelTimeoutError(ModelError):
    """Raised when model request times out."""


class StructuredOutputError(AgentError):
    """Raised when structured output parsing fails."""


class ConfigurationError(AgentError):
    """Raised when agent configuration is invalid."""


class WorkflowError(AgentError):
    """Raised when LangGraph workflow execution fails."""


class RetryExhaustedError(AgentError):
    """Raised when max retries are exhausted."""

    def __init__(
        self, message: str, retry_count: int, last_error: Exception | None = None
    ):
        """Initialize retry exhausted error.

        Args:
            message: Error message
            retry_count: Number of retries attempted
            last_error: Last exception that occurred
        """
        details = {
            "retry_count": retry_count,
            "last_error": str(last_error) if last_error else None,
        }
        super().__init__(message, details)
        self.retry_count = retry_count
        self.last_error = last_error
