"""Convert APF Markdown content to HTML for Anki.

This module converts Markdown-formatted field content to HTML while preserving
APF structure (comment markers, sentinels, card headers).

Uses mistune for fast Markdown parsing (4-5x faster than python-markdown),
Pygments for syntax highlighting, and nh3 for HTML sanitization.
"""

import re

import mistune
import nh3
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)

# Allowed HTML tags for Anki cards (used by nh3 sanitizer)
ALLOWED_TAGS = {
    "p",
    "br",
    "strong",
    "b",
    "em",
    "i",
    "u",
    "s",
    "strike",
    "code",
    "pre",
    "ul",
    "ol",
    "li",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "a",
    "img",
    "div",
    "span",
    "sup",
    "sub",
    "hr",
    "figure",
    "figcaption",
}

# Global attributes allowed on all tags
_GLOBAL_ATTRIBUTES = {"class", "id", "style"}

# Tag-specific attributes (will be merged with global attributes)
# Note: "rel" is excluded from "a" tag because nh3.clean() uses link_rel parameter
_TAG_SPECIFIC_ATTRIBUTES = {
    "a": {"href", "title", "target"},  # rel is set via link_rel parameter
    "img": {"src", "alt", "title", "width", "height"},
    "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan"},
}


def _build_allowed_attributes() -> dict[str, set[str]]:
    """Build allowed attributes dict with global attrs applied to all tags."""
    result: dict[str, set[str]] = {}
    for tag in ALLOWED_TAGS:
        tag_attrs = _TAG_SPECIFIC_ATTRIBUTES.get(tag, set())
        result[tag] = _GLOBAL_ATTRIBUTES | tag_attrs
    return result


ALLOWED_ATTRIBUTES = _build_allowed_attributes()


class AnkiHighlightRenderer(mistune.HTMLRenderer):
    """Custom mistune renderer with Pygments syntax highlighting for Anki."""

    def __init__(self, escape: bool = True) -> None:
        super().__init__(escape=escape)
        self._formatter = HtmlFormatter(
            cssclass="codehilite",
            linenos=False,
            nowrap=False,
        )

    def block_code(self, code: str, info: str | None = None) -> str:
        """Render code block with syntax highlighting."""
        if info:
            # Extract language from info string (e.g., "python" from "python title")
            lang = info.split()[0] if info else None
        else:
            lang = None

        try:
            if lang:
                lexer = get_lexer_by_name(lang, stripall=True)
            else:
                # Try to guess the language
                lexer = guess_lexer(code)
        except ClassNotFound:
            # Fallback to plain text
            lang_class = f"language-{lang}" if lang else "language-text"
            escaped_code = mistune.html(code.strip())
            return f'<pre><code class="{lang_class}">{escaped_code}</code></pre>\n'

        # Use Pygments for highlighting
        highlighted: str = highlight(code, lexer, self._formatter)
        return highlighted

    def codespan(self, text: str) -> str:
        """Render inline code with language-text class."""
        escaped = mistune.html(text)
        return f'<code class="language-text">{escaped}</code>'

    def paragraph(self, text: str) -> str:
        """Render paragraph, preserving cloze syntax."""
        # Preserve cloze syntax that might have been escaped
        text = _restore_cloze_syntax(text)
        return f"<p>{text}</p>\n"


def _create_mistune_converter() -> mistune.Markdown:
    """Create a configured mistune Markdown converter."""
    renderer = AnkiHighlightRenderer()
    # Use create_markdown with custom renderer and built-in plugins
    return mistune.create_markdown(
        renderer=renderer,
        plugins=[
            "strikethrough",
            "table",
            "task_lists",
            "footnotes",
        ],
    )


def _restore_cloze_syntax(text: str) -> str:
    """Restore cloze syntax that may have been escaped during conversion."""
    # Handle various escaped forms of cloze syntax
    # {{c1::text}} or {{c1::text::hint}}
    patterns = [
        (r"\{\{c(\d+)::(.*?)\}\}", r"{{c\1::\2}}"),
        (r"&#123;&#123;c(\d+)::(.*?)&#125;&#125;", r"{{c\1::\2}}"),
        (r"&lbrace;&lbrace;c(\d+)::(.*?)&rbrace;&rbrace;", r"{{c\1::\2}}"),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.DOTALL)
    return text


def convert_markdown_to_html(md_content: str, sanitize: bool = True) -> str:
    """
    Convert Markdown content to HTML using mistune (fast parser).

    Args:
        md_content: Markdown-formatted text
        sanitize: Whether to sanitize HTML output (default True)

    Returns:
        HTML-formatted text suitable for Anki
    """
    if not md_content or not md_content.strip():
        return md_content

    try:
        converter = _create_mistune_converter()
        result = converter(md_content)
        # With HTMLRenderer, mistune returns a string
        html: str = result if isinstance(result, str) else str(result)
    except Exception as e:
        logger.warning("mistune_conversion_failed", error=str(e))
        # Fallback to basic conversion
        html = _basic_markdown_to_html(md_content)

    # Post-process
    html = _ensure_code_language_classes(html)
    html = _restore_cloze_syntax(html)

    # Sanitize HTML to prevent XSS and ensure Anki compatibility
    if sanitize:
        html = sanitize_html(html)

    return html


def _basic_markdown_to_html(md_content: str) -> str:
    """Basic Markdown to HTML conversion fallback."""
    # Simple regex-based conversion for common patterns
    html = md_content

    # Code blocks (must be first to avoid conflicts)
    html = re.sub(
        r"```(\w*)\n(.*?)\n```",
        lambda m: f'<pre><code class="language-{m.group(1) or "text"}">{mistune.html(m.group(2))}</code></pre>',
        html,
        flags=re.DOTALL,
    )

    # Inline code
    html = re.sub(r"`([^`]+)`", r'<code class="language-text">\1</code>', html)

    # Bold
    html = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html)

    # Italic
    html = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", html)

    # Unordered lists
    html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
    html = re.sub(r"(<li>.*</li>\n?)+", r"<ul>\g<0></ul>", html)

    # Line breaks
    html = html.replace("\n\n", "</p><p>")
    html = f"<p>{html}</p>"
    html = html.replace("<p></p>", "")

    return html


def sanitize_html(html: str) -> str:
    """
    Sanitize HTML using nh3 (fast, modern HTML sanitizer).

    This ensures the HTML is safe and compatible with Anki.

    Args:
        html: Raw HTML string

    Returns:
        Sanitized HTML string
    """
    if not html:
        return html

    try:
        # Use nh3 for fast, safe sanitization
        sanitized = nh3.clean(
            html,
            tags=ALLOWED_TAGS,
            attributes=ALLOWED_ATTRIBUTES,
            link_rel="noopener noreferrer",
            strip_comments=False,  # Keep APF comments
        )
        return sanitized
    except Exception as e:
        logger.warning("html_sanitization_failed", error=str(e))
        return html


def _ensure_code_language_classes(html: str) -> str:
    """Ensure all code blocks have language-* classes for Anki styling."""

    def add_language_class(match: re.Match[str]) -> str:
        tag = match.group(0)
        # Skip if already has a language class
        if "language-" in tag or 'class="' in tag:
            return tag
        return '<code class="language-text">'

    html = re.sub(r"<code(?:\s[^>]*)?>", add_language_class, html)
    return html


def _is_already_html(content: str) -> bool:
    """Check if content is already HTML (not Markdown)."""
    html_indicators = [
        "<pre>",
        "<code",
        "<strong>",
        "<em>",
        "<ul>",
        "<ol>",
        "<li>",
        "<table>",
        "<div>",
        "<span>",
        "<p>",
    ]
    content_lower = content.lower()
    return any(indicator in content_lower for indicator in html_indicators)


def convert_apf_field_to_html(field_content: str) -> str:
    """
    Convert a single APF field from Markdown to HTML.

    This handles special cases for APF fields:
    - Preserves cloze syntax
    - Handles code blocks with language hints
    - Converts lists, bold, italic, etc.

    Args:
        field_content: Raw Markdown content from APF field

    Returns:
        HTML content suitable for Anki
    """
    if not field_content:
        return ""

    # Check if content is already HTML (has significant HTML tags)
    if _is_already_html(field_content):
        logger.debug("field_already_html", preview=field_content[:100])
        # Still sanitize even if already HTML
        return sanitize_html(field_content)

    return convert_markdown_to_html(field_content)


def convert_apf_markdown_to_html(apf_markdown: str) -> str:
    """
    Convert APF Markdown document to APF HTML document.

    Preserves APF structure (sentinels, headers, field markers) while
    converting field content from Markdown to HTML.

    Args:
        apf_markdown: Full APF document with Markdown content

    Returns:
        Full APF document with HTML content
    """
    if not apf_markdown:
        return apf_markdown

    # Check if already HTML
    if _is_already_html(apf_markdown):
        logger.debug("apf_already_html")
        return apf_markdown

    # Split into sections by field markers
    result_parts: list[str] = []
    current_pos = 0

    # Pattern for APF comment markers: <!-- Field Name -->
    field_pattern = re.compile(r"(<!--\s*[^>]+\s*-->)")

    # Process document, converting content between markers
    for match in field_pattern.finditer(apf_markdown):
        # Get content before this marker
        content_before = apf_markdown[current_pos : match.start()]
        if content_before.strip():
            # Convert the content section from Markdown to HTML
            converted = convert_apf_field_to_html(content_before.strip())
            # Preserve leading/trailing whitespace
            leading_ws = len(content_before) - len(content_before.lstrip())
            trailing_ws = len(content_before) - len(content_before.rstrip())
            result_parts.append(content_before[:leading_ws])
            result_parts.append(converted)
            if trailing_ws > 0:
                result_parts.append(content_before[-trailing_ws:])
        else:
            result_parts.append(content_before)

        # Add the marker itself (unchanged)
        result_parts.append(match.group(0))
        current_pos = match.end()

    # Handle content after the last marker
    remaining = apf_markdown[current_pos:]
    if remaining.strip():
        converted = convert_apf_field_to_html(remaining.strip())
        leading_ws = len(remaining) - len(remaining.lstrip())
        result_parts.append(remaining[:leading_ws])
        result_parts.append(converted)
    else:
        result_parts.append(remaining)

    return "".join(result_parts)


def highlight_code(code: str, language: str | None = None) -> str:
    """
    Highlight code using Pygments.

    Args:
        code: Source code to highlight
        language: Programming language (optional, will be guessed if not provided)

    Returns:
        HTML with syntax highlighting
    """
    try:
        if language:
            lexer = get_lexer_by_name(language, stripall=True)
        else:
            lexer = guess_lexer(code)

        formatter = HtmlFormatter(
            cssclass="codehilite",
            linenos=False,
            nowrap=False,
        )
        result: str = highlight(code, lexer, formatter)
        return result
    except ClassNotFound:
        escaped = mistune.html(code)
        lang_class = f"language-{language}" if language else "language-text"
        return f'<pre><code class="{lang_class}">{escaped}</code></pre>'


def get_pygments_css(style: str = "default") -> str:
    """
    Get CSS for Pygments syntax highlighting.

    Args:
        style: Pygments style name (default, monokai, github-dark, etc.)

    Returns:
        CSS string for the specified style
    """
    formatter = HtmlFormatter(style=style, cssclass="codehilite")
    result: str = formatter.get_style_defs()
    return result
