from typing import Any

from pydantic import BaseModel, Field

from .error_categories import ErrorCategory


class ValidationError(BaseModel):
    """Structured validation error."""

    category: ErrorCategory
    message: str
    code: str = Field(
        description="Short unique code for the error type (e.g., 'html_tag_unclosed')"
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Additional context for fixing (e.g., {'tag': 'div'})",
    )
    fixable: bool = True

    def __str__(self) -> str:
        return f"[{self.category.value.upper()}] {self.message} ({self.code})"


class ValidationResult(BaseModel):
    """Result of a validation step."""

    is_valid: bool
    errors: list[ValidationError] = Field(default_factory=list)
    validation_time: float = Field(default=0.0, ge=0.0)
