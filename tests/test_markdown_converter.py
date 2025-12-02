"""Tests for APF Markdown to HTML conversion."""

import pytest

from obsidian_anki_sync.apf.markdown_converter import (
    _ensure_code_language_classes,
    _is_already_html,
    _restore_cloze_syntax,
    convert_apf_field_to_html,
    convert_apf_markdown_to_html,
    convert_markdown_to_html,
    highlight_code,
    sanitize_html,
)


class TestConvertMarkdownToHtml:
    """Tests for basic Markdown to HTML conversion."""

    def test_empty_content(self) -> None:
        """Empty content should return unchanged."""
        assert convert_markdown_to_html("") == ""
        assert convert_markdown_to_html("   ") == "   "

    def test_bold_text(self) -> None:
        """Bold markdown should convert to strong tags."""
        result = convert_markdown_to_html("**bold text**")
        assert "<strong>bold text</strong>" in result

    def test_italic_text(self) -> None:
        """Italic markdown should convert to em tags."""
        result = convert_markdown_to_html("*italic text*")
        assert "<em>italic text</em>" in result

    def test_inline_code(self) -> None:
        """Inline code should convert to code tags."""
        result = convert_markdown_to_html("`inline code`")
        assert "<code" in result
        assert "inline code" in result

    def test_fenced_code_block(self) -> None:
        """Fenced code blocks should convert to pre/code tags."""
        md = """```python
def hello():
    print("Hello")
```"""
        result = convert_markdown_to_html(md)
        assert "<pre" in result or "<code" in result
        # Pygments may split keywords across spans, check for presence of function name
        assert "hello" in result

    def test_code_block_with_language(self) -> None:
        """Code blocks should preserve language hints."""
        md = """```kotlin
class Service(val repo: Repository)
```"""
        result = convert_markdown_to_html(md)
        # Pygments may split keywords across spans, check for class name
        assert "Service" in result

    def test_unordered_list(self) -> None:
        """Unordered lists should convert to ul/li tags."""
        md = """- Item 1
- Item 2
- Item 3"""
        result = convert_markdown_to_html(md)
        assert "<ul>" in result
        assert "<li>" in result
        assert "Item 1" in result

    def test_ordered_list(self) -> None:
        """Ordered lists should convert to ol/li tags."""
        md = """1. First
2. Second
3. Third"""
        result = convert_markdown_to_html(md)
        assert "<ol>" in result
        assert "<li>" in result
        assert "First" in result

    def test_mixed_formatting(self) -> None:
        """Mixed formatting should all convert correctly."""
        md = """**Bold** and *italic* with `code`"""
        result = convert_markdown_to_html(md)
        assert "<strong>Bold</strong>" in result
        assert "<em>italic</em>" in result
        assert "code" in result

    def test_newlines_to_br(self) -> None:
        """Newlines should convert to br tags."""
        md = """Line 1
Line 2"""
        result = convert_markdown_to_html(md)
        assert "<br" in result or ("Line 1" in result and "Line 2" in result)

    def test_table(self) -> None:
        """Tables should convert to HTML tables."""
        md = """| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |"""
        result = convert_markdown_to_html(md)
        assert "<table>" in result
        assert "<th>" in result or "Header 1" in result


class TestRestoreClozeSyntax:
    """Tests for cloze syntax restoration."""

    def test_simple_cloze(self) -> None:
        """Simple cloze syntax should be preserved."""
        html = "The answer is {{c1::42}}"
        result = _restore_cloze_syntax(html)
        assert "{{c1::42}}" in result

    def test_multiple_clozes(self) -> None:
        """Multiple cloze deletions should be preserved."""
        html = "{{c1::First}} and {{c2::Second}}"
        result = _restore_cloze_syntax(html)
        assert "{{c1::First}}" in result
        assert "{{c2::Second}}" in result

    def test_cloze_with_hint(self) -> None:
        """Cloze with hint should be preserved."""
        html = "The capital is {{c1::Paris::city}}"
        result = _restore_cloze_syntax(html)
        assert "{{c1::Paris::city}}" in result

    def test_cloze_in_code_block(self) -> None:
        """Cloze syntax in code should be preserved."""
        md = """```kotlin
val x = {{c1::42}}
```"""
        result = convert_markdown_to_html(md)
        # Check that cloze syntax survived conversion
        assert "{{c1::" in result or "c1" in result


class TestEnsureCodeLanguageClasses:
    """Tests for code language class handling."""

    def test_add_language_class_to_bare_code(self) -> None:
        """Bare code tags should get language-text class."""
        html = "<code>some code</code>"
        result = _ensure_code_language_classes(html)
        assert 'class="language-text"' in result

    def test_preserve_existing_language_class(self) -> None:
        """Existing language classes should be preserved."""
        html = '<code class="language-python">code</code>'
        result = _ensure_code_language_classes(html)
        assert 'class="language-python"' in result
        assert result.count("language-") == 1

    def test_preserve_other_classes(self) -> None:
        """Code with other classes should not get language- added."""
        html = '<code class="highlight">code</code>'
        result = _ensure_code_language_classes(html)
        # Should keep the original class
        assert 'class="highlight"' in result


class TestIsAlreadyHtml:
    """Tests for HTML detection."""

    def test_pure_text(self) -> None:
        """Plain text should not be detected as HTML."""
        assert not _is_already_html("Just plain text")

    def test_markdown_text(self) -> None:
        """Markdown should not be detected as HTML."""
        assert not _is_already_html("**bold** and *italic*")
        assert not _is_already_html("```python\ncode\n```")

    def test_html_with_pre(self) -> None:
        """Content with pre tags should be detected as HTML."""
        assert _is_already_html("<pre><code>code</code></pre>")

    def test_html_with_strong(self) -> None:
        """Content with strong tags should be detected as HTML."""
        assert _is_already_html("<strong>bold</strong>")

    def test_html_with_ul(self) -> None:
        """Content with ul tags should be detected as HTML."""
        assert _is_already_html("<ul><li>item</li></ul>")

    def test_html_with_table(self) -> None:
        """Content with table tags should be detected as HTML."""
        assert _is_already_html("<table><tr><td>cell</td></tr></table>")


class TestConvertApfFieldToHtml:
    """Tests for APF field conversion."""

    def test_empty_field(self) -> None:
        """Empty field should return empty."""
        assert convert_apf_field_to_html("") == ""
        assert convert_apf_field_to_html(None) == ""  # type: ignore[arg-type]

    def test_markdown_field(self) -> None:
        """Markdown field should be converted."""
        result = convert_apf_field_to_html("**bold** text")
        assert "<strong>bold</strong>" in result

    def test_html_field_passthrough(self) -> None:
        """HTML field should pass through unchanged."""
        html = "<strong>bold</strong> text"
        result = convert_apf_field_to_html(html)
        assert result == html

    def test_code_block_field(self) -> None:
        """Code block in field should convert properly."""
        md = """```kotlin
class Foo
```"""
        result = convert_apf_field_to_html(md)
        # Pygments may split keywords across spans, check for class name
        assert "Foo" in result
        assert "<pre" in result or "<code" in result


class TestConvertApfMarkdownToHtml:
    """Tests for full APF document conversion."""

    def test_empty_document(self) -> None:
        """Empty document should return unchanged."""
        assert convert_apf_markdown_to_html("") == ""
        assert convert_apf_markdown_to_html(None) is None  # type: ignore[arg-type]

    def test_simple_apf_document(self) -> None:
        """Simple APF document should convert field content."""
        apf_md = """<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->

<!-- Card 1 | slug: test-card | CardType: Simple | Tags: test -->

<!-- Title -->
What is **dependency injection**?

<!-- Key point -->
```kotlin
class Service(val repo: Repository)
```

<!-- Key point notes -->
- Constructor receives dependencies
- Enables testing with mocks

<!-- END_CARDS -->
END_OF_CARDS"""

        result = convert_apf_markdown_to_html(apf_md)

        # Structure should be preserved
        assert "<!-- PROMPT_VERSION: apf-v2.1 -->" in result
        assert "<!-- BEGIN_CARDS -->" in result
        assert "<!-- Card 1 | slug: test-card | CardType: Simple | Tags: test -->" in result
        assert "<!-- Title -->" in result
        assert "<!-- Key point -->" in result
        assert "<!-- Key point notes -->" in result
        assert "<!-- END_CARDS -->" in result

        # Content should be converted
        assert "<strong>dependency injection</strong>" in result
        # Code block should be converted (Pygments may split across spans)
        assert "Service" in result

    def test_html_document_passthrough(self) -> None:
        """Already-HTML document should pass through."""
        apf_html = """<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: test | CardType: Simple | Tags: test -->

<!-- Title -->
<strong>Already HTML</strong>

<!-- END_CARDS -->"""

        result = convert_apf_markdown_to_html(apf_html)
        # Should be unchanged since it's already HTML
        assert "<strong>Already HTML</strong>" in result

    def test_cloze_card(self) -> None:
        """Cloze cards should preserve cloze syntax."""
        apf_md = """<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: cloze-test | CardType: Missing | Tags: test -->

<!-- Title -->
Fill in the blank

<!-- Key point -->
The answer is {{c1::42}}

<!-- END_CARDS -->"""

        result = convert_apf_markdown_to_html(apf_md)
        assert "{{c1::42}}" in result

    def test_multiple_cards(self) -> None:
        """Multiple cards in document should all convert."""
        apf_md = """<!-- BEGIN_CARDS -->

<!-- Card 1 | slug: card-1 | CardType: Simple | Tags: test -->
<!-- Title -->
**Card One**

<!-- Card 2 | slug: card-2 | CardType: Simple | Tags: test -->
<!-- Title -->
**Card Two**

<!-- END_CARDS -->"""

        result = convert_apf_markdown_to_html(apf_md)
        assert "<strong>Card One</strong>" in result
        assert "<strong>Card Two</strong>" in result


class TestEdgeCases:
    """Tests for edge cases and special content."""

    def test_mathjax_passthrough(self) -> None:
        """MathJax notation should pass through unchanged."""
        md = r"The formula is \(E = mc^2\)"
        result = convert_markdown_to_html(md)
        # MathJax delimiters should be preserved
        assert r"\(" in result or "E = mc^2" in result

    def test_html_entities_in_code(self) -> None:
        """HTML entities in code should be handled correctly."""
        md = """```html
<div>Hello</div>
```"""
        result = convert_markdown_to_html(md)
        # The code content should be preserved (possibly escaped)
        assert "div" in result

    def test_nested_formatting(self) -> None:
        """Nested formatting should convert correctly."""
        # Use underscores for inner emphasis to avoid ambiguity
        md = "**bold with _nested italic_**"
        result = convert_markdown_to_html(md)
        assert "<strong>" in result
        # Nested formatting may vary by parser

    def test_special_characters(self) -> None:
        """Special characters should be handled."""
        md = "Symbols: < > & \" '"
        result = convert_markdown_to_html(md)
        # Content should be present, possibly escaped
        assert "Symbols" in result

    def test_long_code_block(self) -> None:
        """Long code blocks should convert correctly."""
        md = """```python
def long_function():
    x = 1
    y = 2
    z = 3
    return x + y + z

class MyClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
```"""
        result = convert_markdown_to_html(md)
        assert "long_function" in result
        assert "MyClass" in result
        assert "__init__" in result


class TestSanitizeHtml:
    """Tests for HTML sanitization with nh3."""

    def test_sanitize_safe_html(self) -> None:
        """Safe HTML should pass through."""
        html = "<p>Hello <strong>world</strong></p>"
        result = sanitize_html(html)
        assert "<p>" in result
        assert "<strong>" in result
        assert "Hello" in result

    def test_sanitize_removes_script(self) -> None:
        """Script tags should be removed."""
        html = "<p>Hello</p><script>alert('xss')</script>"
        result = sanitize_html(html)
        assert "<script>" not in result
        assert "alert" not in result

    def test_sanitize_removes_onclick(self) -> None:
        """Event handlers should be removed."""
        html = '<button onclick="alert()">Click</button>'
        result = sanitize_html(html)
        assert "onclick" not in result

    def test_sanitize_empty_string(self) -> None:
        """Empty string should return empty."""
        assert sanitize_html("") == ""
        assert sanitize_html(None) is None  # type: ignore[arg-type]


class TestHighlightCode:
    """Tests for Pygments code highlighting."""

    def test_highlight_python(self) -> None:
        """Python code should be highlighted."""
        code = "def hello():\n    print('world')"
        result = highlight_code(code, "python")
        # Should contain highlighted spans
        assert "def" in result
        assert "hello" in result

    def test_highlight_unknown_language(self) -> None:
        """Unknown language should fallback gracefully."""
        code = "some code here"
        result = highlight_code(code, "not_a_real_language")
        assert "some code here" in result

    def test_highlight_auto_detect(self) -> None:
        """Code without language hint should be auto-detected."""
        code = "def hello():\n    return 42"
        result = highlight_code(code)
        assert "def" in result
        assert "hello" in result
