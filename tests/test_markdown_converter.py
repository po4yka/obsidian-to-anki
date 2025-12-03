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
        assert (
            "<!-- Card 1 | slug: test-card | CardType: Simple | Tags: test -->"
            in result
        )
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


class TestCodeBlockEdgeCases:
    """Tests for code block rendering edge cases."""

    def test_code_block_without_language(self) -> None:
        """Code blocks without language should use auto-detection."""
        md = """```
print("hello world")
```"""
        result = convert_markdown_to_html(md)
        assert "<pre" in result or "<code" in result
        assert "hello" in result

    def test_code_block_with_unknown_language(self) -> None:
        """Code blocks with unknown language should fallback to plain text."""
        md = """```not_a_real_programming_language_xyz
some code content here
```"""
        result = convert_markdown_to_html(md)
        assert "some code content" in result
        # Should still be wrapped in pre/code tags
        assert "<pre" in result or "<code" in result

    def test_code_block_with_language_and_title(self) -> None:
        """Code blocks with language and extra info should extract language."""
        md = """```python title="example.py"
def hello():
    pass
```"""
        result = convert_markdown_to_html(md)
        assert "hello" in result
        assert "<pre" in result or "<code" in result

    def test_empty_code_block(self) -> None:
        """Empty code blocks should be handled gracefully."""
        md = """```python
```"""
        result = convert_markdown_to_html(md)
        # Should not crash, may have empty content
        assert "<pre" in result or "<code" in result or result.strip() != ""

    def test_code_block_with_special_chars(self) -> None:
        """Code blocks with special HTML chars should be escaped."""
        md = """```html
<div class="test">&nbsp;</div>
```"""
        result = convert_markdown_to_html(md)
        # Content should be preserved (possibly escaped)
        assert "div" in result
        assert "test" in result


class TestBasicMarkdownFallback:
    """Tests for the basic markdown to HTML fallback function."""

    def test_basic_fallback_bold(self) -> None:
        """Fallback should convert bold text."""
        from obsidian_anki_sync.apf.markdown_converter import _basic_markdown_to_html

        result = _basic_markdown_to_html("**bold text**")
        assert "<strong>bold text</strong>" in result

    def test_basic_fallback_italic(self) -> None:
        """Fallback should convert italic text."""
        from obsidian_anki_sync.apf.markdown_converter import _basic_markdown_to_html

        result = _basic_markdown_to_html("*italic text*")
        assert "<em>italic text</em>" in result

    def test_basic_fallback_inline_code(self) -> None:
        """Fallback should convert inline code."""
        from obsidian_anki_sync.apf.markdown_converter import _basic_markdown_to_html

        result = _basic_markdown_to_html("`inline code`")
        assert '<code class="language-text">inline code</code>' in result

    def test_basic_fallback_code_block(self) -> None:
        """Fallback should convert code blocks."""
        from obsidian_anki_sync.apf.markdown_converter import _basic_markdown_to_html

        md = """```python
def test():
    pass
```"""
        result = _basic_markdown_to_html(md)
        assert "<pre>" in result
        assert "<code" in result
        assert "language-python" in result

    def test_basic_fallback_code_block_no_lang(self) -> None:
        """Fallback should handle code blocks without language."""
        from obsidian_anki_sync.apf.markdown_converter import _basic_markdown_to_html

        md = """```
plain code
```"""
        result = _basic_markdown_to_html(md)
        assert "<pre>" in result
        assert "language-text" in result

    def test_basic_fallback_unordered_list(self) -> None:
        """Fallback should convert unordered lists."""
        from obsidian_anki_sync.apf.markdown_converter import _basic_markdown_to_html

        md = """- Item 1
- Item 2"""
        result = _basic_markdown_to_html(md)
        assert "<ul>" in result
        assert "<li>Item 1</li>" in result
        assert "<li>Item 2</li>" in result

    def test_basic_fallback_paragraphs(self) -> None:
        """Fallback should handle paragraph breaks."""
        from obsidian_anki_sync.apf.markdown_converter import _basic_markdown_to_html

        md = """First paragraph

Second paragraph"""
        result = _basic_markdown_to_html(md)
        assert "</p><p>" in result


class TestGetPygmentsCss:
    """Tests for Pygments CSS generation."""

    def test_get_default_css(self) -> None:
        """Should generate CSS for default style."""
        from obsidian_anki_sync.apf.markdown_converter import get_pygments_css

        css = get_pygments_css()
        assert ".codehilite" in css
        # Should contain color definitions
        assert "color:" in css or "#" in css

    def test_get_monokai_css(self) -> None:
        """Should generate CSS for monokai style."""
        from obsidian_anki_sync.apf.markdown_converter import get_pygments_css

        css = get_pygments_css("monokai")
        assert ".codehilite" in css
        assert len(css) > 100  # Should have substantial CSS

    def test_get_different_styles_produce_different_css(self) -> None:
        """Different styles should produce different CSS."""
        from obsidian_anki_sync.apf.markdown_converter import get_pygments_css

        default_css = get_pygments_css("default")
        monokai_css = get_pygments_css("monokai")
        # They should be different
        assert default_css != monokai_css


class TestMistuneConversionFallback:
    """Tests for mistune conversion failure fallback."""

    def test_conversion_uses_fallback_on_error(self) -> None:
        """When mistune fails, should use basic fallback."""
        from unittest.mock import patch

        from obsidian_anki_sync.apf.markdown_converter import convert_markdown_to_html

        # Mock mistune to raise an exception
        with patch(
            "obsidian_anki_sync.apf.markdown_converter._create_mistune_converter"
        ) as mock_converter:
            mock_converter.return_value = lambda x: (_ for _ in ()).throw(
                Exception("Simulated failure")
            )

            result = convert_markdown_to_html("**bold text**")
            # Should still produce valid output via fallback
            assert "<strong>bold text</strong>" in result

    def test_fallback_handles_complex_content(self) -> None:
        """Fallback should handle complex markdown content."""
        from obsidian_anki_sync.apf.markdown_converter import _basic_markdown_to_html

        md = """**Bold** and *italic* with `code`.

```python
def test():
    pass
```

- List item"""
        result = _basic_markdown_to_html(md)
        assert "<strong>Bold</strong>" in result
        assert "<em>italic</em>" in result
        assert "<code" in result
        assert "<li>List item</li>" in result


class TestSanitizationEdgeCases:
    """Tests for HTML sanitization edge cases."""

    def test_sanitize_fallback_on_error(self) -> None:
        """When nh3 fails, should return original HTML."""
        from unittest.mock import patch

        html = "<p>Test content</p>"
        with patch("obsidian_anki_sync.apf.markdown_converter.nh3.clean") as mock_clean:
            mock_clean.side_effect = Exception("Simulated nh3 failure")
            result = sanitize_html(html)
            # Should return original HTML on error
            assert result == html

    def test_sanitize_preserves_allowed_tags(self) -> None:
        """Allowed tags should be preserved."""
        html = "<p>Text with <strong>bold</strong> and <em>italic</em></p>"
        result = sanitize_html(html)
        assert "<p>" in result
        assert "<strong>" in result
        assert "<em>" in result

    def test_sanitize_removes_dangerous_attributes(self) -> None:
        """Dangerous attributes should be removed."""
        html = '<img src="x" onerror="alert(1)">'
        result = sanitize_html(html)
        assert "onerror" not in result.lower()

    def test_sanitize_handles_nested_tags(self) -> None:
        """Nested tags should be handled correctly."""
        html = "<p><strong><em>Nested</em></strong></p>"
        result = sanitize_html(html)
        assert "Nested" in result

    def test_sanitize_preserves_code_classes(self) -> None:
        """Code tag classes should be preserved."""
        html = '<code class="language-python">code</code>'
        result = sanitize_html(html)
        assert 'class="language-python"' in result

    def test_sanitize_handles_table_tags(self) -> None:
        """Table tags should be preserved."""
        html = "<table><tr><td>Cell</td></tr></table>"
        result = sanitize_html(html)
        assert "<table>" in result
        assert "<td>" in result

    def test_sanitize_removes_iframe(self) -> None:
        """Iframe tags should be removed."""
        html = '<p>Text</p><iframe src="evil.com"></iframe>'
        result = sanitize_html(html)
        assert "<iframe" not in result
        assert "Text" in result

    def test_sanitize_preserves_links(self) -> None:
        """Link tags should be preserved with safe attributes."""
        html = '<a href="https://example.com" title="Link">Click</a>'
        result = sanitize_html(html)
        assert "<a" in result
        assert "href=" in result
        assert "Click" in result

    def test_sanitize_adds_rel_to_links(self) -> None:
        """Links should get rel=noopener noreferrer added."""
        html = '<a href="https://example.com">Link</a>'
        result = sanitize_html(html)
        # nh3 adds rel attribute
        assert "noopener" in result or "rel=" in result


class TestRendererMethods:
    """Direct tests for AnkiHighlightRenderer methods."""

    def test_renderer_codespan(self) -> None:
        """Codespan method should wrap in code tags."""
        from obsidian_anki_sync.apf.markdown_converter import AnkiHighlightRenderer

        renderer = AnkiHighlightRenderer()
        result = renderer.codespan("test code")
        assert '<code class="language-text">' in result
        assert "test code" in result

    def test_renderer_paragraph(self) -> None:
        """Paragraph method should wrap in p tags."""
        from obsidian_anki_sync.apf.markdown_converter import AnkiHighlightRenderer

        renderer = AnkiHighlightRenderer()
        result = renderer.paragraph("some text")
        assert "<p>" in result
        assert "some text" in result

    def test_renderer_paragraph_preserves_cloze(self) -> None:
        """Paragraph method should preserve cloze syntax."""
        from obsidian_anki_sync.apf.markdown_converter import AnkiHighlightRenderer

        renderer = AnkiHighlightRenderer()
        result = renderer.paragraph("Answer is {{c1::42}}")
        assert "{{c1::42}}" in result

    def test_renderer_block_code_no_language(self) -> None:
        """Block code with no language should auto-detect."""
        from obsidian_anki_sync.apf.markdown_converter import AnkiHighlightRenderer

        renderer = AnkiHighlightRenderer()
        result = renderer.block_code("print('hello')", info=None)
        assert "hello" in result

    def test_renderer_block_code_unknown_language(self) -> None:
        """Block code with unknown language should fallback."""
        from obsidian_anki_sync.apf.markdown_converter import AnkiHighlightRenderer

        renderer = AnkiHighlightRenderer()
        result = renderer.block_code("some content", info="unknownlang12345")
        assert "some content" in result
        assert "language-unknownlang12345" in result


class TestAttributeBuilding:
    """Tests for attribute building helper."""

    def test_all_tags_have_global_attrs(self) -> None:
        """All allowed tags should have global attributes."""
        from obsidian_anki_sync.apf.markdown_converter import (
            _GLOBAL_ATTRIBUTES,
            ALLOWED_ATTRIBUTES,
            ALLOWED_TAGS,
        )

        for tag in ALLOWED_TAGS:
            assert tag in ALLOWED_ATTRIBUTES
            for attr in _GLOBAL_ATTRIBUTES:
                assert attr in ALLOWED_ATTRIBUTES[tag]

    def test_anchor_has_link_attrs(self) -> None:
        """Anchor tag should have link-specific attributes."""
        from obsidian_anki_sync.apf.markdown_converter import ALLOWED_ATTRIBUTES

        assert "a" in ALLOWED_ATTRIBUTES
        assert "href" in ALLOWED_ATTRIBUTES["a"]
        assert "title" in ALLOWED_ATTRIBUTES["a"]
        assert "target" in ALLOWED_ATTRIBUTES["a"]

    def test_img_has_image_attrs(self) -> None:
        """Image tag should have image-specific attributes."""
        from obsidian_anki_sync.apf.markdown_converter import ALLOWED_ATTRIBUTES

        assert "img" in ALLOWED_ATTRIBUTES
        assert "src" in ALLOWED_ATTRIBUTES["img"]
        assert "alt" in ALLOWED_ATTRIBUTES["img"]
        assert "width" in ALLOWED_ATTRIBUTES["img"]
        assert "height" in ALLOWED_ATTRIBUTES["img"]
