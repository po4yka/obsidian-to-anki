"""Base validator classes and types for vault validation."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class Severity(Enum):
    """Validation issue severity levels."""

    CRITICAL = "CRITICAL"  # Must fix (breaks required rules)
    WARNING = "WARNING"  # Should fix (missing recommended)
    ERROR = "ERROR"  # Factual/technical issues
    INFO = "INFO"  # Style suggestions


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a note."""

    severity: Severity
    message: str
    line: int | None = None
    section: str | None = None

    def __str__(self) -> str:
        """Format the issue for display."""
        location = ""
        if self.line:
            location = f"[Line {self.line}] "
        elif self.section:
            location = f"[Section: {self.section}] "
        return f"{location}{self.message}"


@dataclass
class AutoFix:
    """Represents an automatic fix for a validation issue.

    Fix functions can either:
    - Take no args and use captured self.content/frontmatter (legacy)
    - Take (content, frontmatter) args for cumulative fixes (preferred)
    """

    description: str
    fix_function: Callable[..., tuple[str, dict[str, Any]]]
    severity: Severity
    safe: bool = True  # Safe fixes can be applied without user confirmation


class BaseValidator:
    """Base class for all validators.

    Validators check specific aspects of Q&A notes and report issues.
    They can also provide auto-fix functions for common problems.
    """

    def __init__(
        self, content: str, frontmatter: dict[str, Any], filepath: str
    ) -> None:
        """Initialize the validator.

        Args:
            content: Full note content including frontmatter
            frontmatter: Parsed YAML frontmatter as dict
            filepath: Path to the note file
        """
        self.content = content
        self.frontmatter = frontmatter
        self.filepath = filepath
        self.issues: list[ValidationIssue] = []
        self.passed_checks: list[str] = []
        self.fixes: list[AutoFix] = []

    def add_issue(
        self,
        severity: Severity,
        message: str,
        line: int | None = None,
        section: str | None = None,
    ) -> None:
        """Add a validation issue.

        Args:
            severity: Issue severity level
            message: Description of the issue
            line: Optional line number where issue was found
            section: Optional section name where issue was found
        """
        self.issues.append(ValidationIssue(severity, message, line, section))

    def add_passed(self, check_name: str) -> None:
        """Record a passed check.

        Args:
            check_name: Name of the check that passed
        """
        self.passed_checks.append(check_name)

    def add_fix(
        self,
        description: str,
        fix_function: Callable[..., tuple[str, dict[str, Any]]],
        severity: Severity,
        safe: bool = True,
    ) -> None:
        """Add an automatic fix for an issue.

        Args:
            description: Description of what the fix does
            fix_function: Function that returns (new_content, new_frontmatter)
            severity: Severity of the issue this fix addresses
            safe: Whether fix can be applied without user confirmation
        """
        self.fixes.append(AutoFix(description, fix_function, severity, safe))

    def validate(self) -> list[ValidationIssue]:
        """Perform validation. Override in subclasses.

        Returns:
            List of validation issues found
        """
        raise NotImplementedError("Subclasses must implement validate()")

    def get_issues_by_severity(self, severity: Severity) -> list[ValidationIssue]:
        """Get all issues of a specific severity.

        Args:
            severity: Severity level to filter by

        Returns:
            List of issues matching the severity
        """
        return [issue for issue in self.issues if issue.severity == severity]

    def has_critical_issues(self) -> bool:
        """Check if any critical issues were found.

        Returns:
            True if critical issues exist
        """
        return any(issue.severity == Severity.CRITICAL for issue in self.issues)

    def get_safe_fixes(self) -> list[AutoFix]:
        """Get all fixes that are safe to apply automatically.

        Returns:
            List of safe auto-fixes
        """
        return [fix for fix in self.fixes if fix.safe]
