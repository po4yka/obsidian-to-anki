from dataclasses import dataclass
from typing import Optional

from .error_categories import ErrorCategory


@dataclass
class ValidationError:
    """Structured validation error."""

    category: ErrorCategory
    message: str
    code: str  # Short unique code for the error type (e.g., "html_tag_unclosed")
    context: Optional[dict] = None  # Additional context for fixing (e.g., {"tag": "div"})
    fixable: bool = True

    def __str__(self) -> str:
        return f"[{self.category.value.upper()}] {self.message} ({self.code})"


@dataclass
class ValidationResult:
    """Result of a validation step."""

    is_valid: bool
    errors: list[ValidationError]
    validation_time: float = 0.0
