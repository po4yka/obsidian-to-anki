"""APF card generation and validation.

This module provides APF (Active Prompt Format) card generation and validation.
Supports both HTML and Markdown content formats, with Markdown being the preferred
format for LLM generation. Markdown is converted to HTML at the Anki boundary.

Key features:
- Fast Markdown parsing with mistune (4-5x faster than python-markdown)
- Syntax highlighting with Pygments
- HTML sanitization with nh3 (20x faster than deprecated bleach)
"""

from obsidian_anki_sync.apf.markdown_converter import (
    convert_apf_field_to_html,
    convert_apf_markdown_to_html,
    convert_markdown_to_html,
    get_pygments_css,
    highlight_code,
    sanitize_html,
)
from obsidian_anki_sync.apf.markdown_validator import (
    MarkdownValidationResult,
    validate_apf_markdown,
    validate_markdown,
)

__all__ = [
    "MarkdownValidationResult",
    "convert_apf_field_to_html",
    "convert_apf_markdown_to_html",
    "convert_markdown_to_html",
    "get_pygments_css",
    "highlight_code",
    "sanitize_html",
    "validate_apf_markdown",
    "validate_markdown",
]
