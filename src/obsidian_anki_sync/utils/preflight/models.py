"""Models for preflight checks."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CheckResult(BaseModel):
    """Result of a pre-flight check."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, description="Check name")
    passed: bool = Field(description="Whether the check passed")
    message: str = Field(min_length=1, description="Result message")
    severity: str = Field(
        description="Severity level: 'error', 'blocking_warning', 'warning', 'info'. "
        "'error' and 'blocking_warning' halt execution in strict mode."
    )
    fixable: bool = Field(default=False, description="Whether the issue is fixable")
    fix_suggestion: str | None = Field(
        default=None, description="Suggestion for fixing the issue"
    )


__all__ = ["CheckResult"]
