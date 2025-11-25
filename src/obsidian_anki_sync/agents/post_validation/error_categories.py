"""Error categorization for validation errors."""

from enum import Enum


class ErrorCategory(str, Enum):
    """Error categories for validation errors."""

    SYNTAX = "syntax"  # APF format or HTML syntax errors (most fixable)
    HTML = "html"  # HTML structure validation errors (fixable)
    APF_FORMAT = "template"  # APF v2.1 template compliance (fixable)
    MANIFEST = "manifest"  # Manifest slug/format mismatches (fixable)
    # Question-answer mismatch or coherence issues (harder to fix)
    SEMANTIC = "semantic"
    # Information inaccuracies or hallucinations (hardest to fix)
    FACTUAL = "factual"
    NONE = "none"  # No errors

    @classmethod
    def from_error_string(cls, error_str: str) -> "ErrorCategory":
        """Categorize error from error string.

        Args:
            error_str: Error description string

        Returns:
            ErrorCategory enum value
        """
        error_lower = error_str.lower()
        if "html" in error_lower or "inline" in error_lower or "<code>" in error_lower:
            return cls.HTML
        elif (
            "apf format" in error_lower
            or "template" in error_lower
            or "format" in error_lower
        ):
            return cls.APF_FORMAT
        elif "manifest" in error_lower or "slug" in error_lower:
            return cls.MANIFEST
        elif (
            "factual" in error_lower
            or "hallucination" in error_lower
            or "inaccurate" in error_lower
        ):
            return cls.FACTUAL
        elif (
            "semantic" in error_lower
            or "mismatch" in error_lower
            or "coherence" in error_lower
        ):
            return cls.SEMANTIC
        elif (
            "syntax" in error_lower
            or "parse" in error_lower
            or "invalid" in error_lower
        ):
            return cls.SYNTAX
        else:
            return cls.NONE
