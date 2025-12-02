"""Tests for APF Markdown validation."""

import pytest

from obsidian_anki_sync.apf.markdown_validator import (
    _check_common_issues,
    _remove_code_blocks,
    _validate_code_fences,
    _validate_formatting_markers,
    validate_apf_markdown,
    validate_markdown,
)


class TestValidateMarkdown:
    """Tests for basic Markdown validation."""

    def test_empty_content_is_valid(self) -> None:
        """Empty content should be valid."""
        result = validate_markdown("")
        assert result.is_valid
        assert len(result.errors) == 0

    def test_whitespace_only_is_valid(self) -> None:
        """Whitespace-only content should be valid."""
        result = validate_markdown("   \n\t  ")
        assert result.is_valid

    def test_valid_markdown(self) -> None:
        """Valid Markdown should pass validation."""
        md = """**Bold** and *italic* with `code`.

```python
def hello():
    pass
```

- List item 1
- List item 2
"""
        result = validate_markdown(md)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_unclosed_code_fence(self) -> None:
        """Unclosed code fence should be detected."""
        md = """```python
def hello():
    pass
"""
        result = validate_markdown(md)
        assert not result.is_valid
        assert any("Unclosed code fence" in e for e in result.errors)

    def test_balanced_code_fences(self) -> None:
        """Balanced code fences should pass."""
        md = """```python
code here
```

More text

```javascript
more code
```
"""
        result = validate_markdown(md)
        assert result.is_valid


class TestValidateCodeFences:
    """Tests for code fence validation."""

    def test_no_code_fences(self) -> None:
        """Content without code fences should pass."""
        errors = _validate_code_fences("Just plain text")
        assert len(errors) == 0

    def test_balanced_fence(self) -> None:
        """Balanced fence should pass."""
        errors = _validate_code_fences("```\ncode\n```")
        assert len(errors) == 0

    def test_fence_with_language(self) -> None:
        """Fence with language should pass."""
        errors = _validate_code_fences("```python\ncode\n```")
        assert len(errors) == 0

    def test_unclosed_fence(self) -> None:
        """Unclosed fence should report error."""
        errors = _validate_code_fences("```python\ncode")
        assert len(errors) == 1
        assert "Unclosed code fence" in errors[0]

    def test_multiple_balanced_fences(self) -> None:
        """Multiple balanced fences should pass."""
        content = """```
first
```
text
```python
second
```
"""
        errors = _validate_code_fences(content)
        assert len(errors) == 0


class TestValidateFormattingMarkers:
    """Tests for formatting marker validation."""

    def test_balanced_backticks(self) -> None:
        """Balanced backticks should pass."""
        errors = _validate_formatting_markers("`code` and `more code`")
        assert len(errors) == 0

    def test_unbalanced_backticks(self) -> None:
        """Unbalanced backticks should report error."""
        errors = _validate_formatting_markers("This has `unbalanced code")
        assert any("Unbalanced inline code" in e for e in errors)

    def test_balanced_bold(self) -> None:
        """Balanced bold markers should pass."""
        errors = _validate_formatting_markers("**bold** and **more bold**")
        assert len(errors) == 0

    def test_unbalanced_bold(self) -> None:
        """Unbalanced bold markers should report error."""
        errors = _validate_formatting_markers("This has **unbalanced bold")
        assert any("Unbalanced bold" in e for e in errors)

    def test_backticks_in_code_block_ignored(self) -> None:
        """Backticks inside code blocks should be ignored."""
        content = """```
This has ` backtick
```
"""
        errors = _validate_formatting_markers(content)
        # Code blocks are removed, so no error
        assert len(errors) == 0


class TestRemoveCodeBlocks:
    """Tests for code block removal helper."""

    def test_removes_fenced_blocks(self) -> None:
        """Fenced code blocks should be removed."""
        content = """Before
```python
code
```
After"""
        result = _remove_code_blocks(content)
        assert "Before" in result
        assert "After" in result
        assert "code" not in result

    def test_removes_inline_code(self) -> None:
        """Inline code should be removed."""
        content = "Text with `inline code` here"
        result = _remove_code_blocks(content)
        assert "Text with" in result
        assert "inline code" not in result
        assert "here" in result


class TestCheckCommonIssues:
    """Tests for common issue detection."""

    def test_no_issues_in_clean_markdown(self) -> None:
        """Clean Markdown should have no warnings."""
        warnings = _check_common_issues("**Bold** and `code`")
        assert len(warnings) == 0

    def test_detects_html_tags(self) -> None:
        """HTML tags in Markdown should generate warning."""
        warnings = _check_common_issues("<pre>code</pre>")
        assert any("HTML tag" in w for w in warnings)

    def test_long_code_line_warning(self) -> None:
        """Very long lines in code blocks should warn."""
        long_line = "x" * 250
        content = f"""```
{long_line}
```"""
        warnings = _check_common_issues(content)
        assert any("exceeds 200 characters" in w for w in warnings)


class TestValidateApfMarkdown:
    """Tests for full APF Markdown validation."""

    def test_empty_content_invalid(self) -> None:
        """Empty APF content should be invalid."""
        result = validate_apf_markdown("")
        assert not result.is_valid
        assert any("Empty" in e for e in result.errors)

    def test_missing_begin_sentinel(self) -> None:
        """Missing BEGIN_CARDS sentinel should be detected."""
        apf = """<!-- END_CARDS -->
END_OF_CARDS"""
        result = validate_apf_markdown(apf)
        assert not result.is_valid
        assert any("BEGIN_CARDS" in e for e in result.errors)

    def test_missing_end_sentinel(self) -> None:
        """Missing END_CARDS sentinel should be detected."""
        apf = """<!-- BEGIN_CARDS -->
content
"""
        result = validate_apf_markdown(apf)
        assert not result.is_valid
        assert any("END_CARDS" in e for e in result.errors)

    def test_missing_card_header(self) -> None:
        """Missing card header should be detected."""
        apf = """<!-- BEGIN_CARDS -->
<!-- Title -->
Some title
<!-- END_CARDS -->"""
        result = validate_apf_markdown(apf)
        assert not result.is_valid
        assert any("card header" in e.lower() for e in result.errors)

    def test_missing_title(self) -> None:
        """Missing Title section should be detected."""
        apf = """<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: test | CardType: Simple | Tags: test -->
<!-- Key point -->
content
<!-- END_CARDS -->"""
        result = validate_apf_markdown(apf)
        assert not result.is_valid
        assert any("Title" in e for e in result.errors)

    def test_valid_simple_card(self) -> None:
        """Valid simple card should pass."""
        apf = """<!-- PROMPT_VERSION: apf-v2.1 -->
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
        result = validate_apf_markdown(apf)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_valid_cloze_card(self) -> None:
        """Valid cloze card should pass."""
        apf = """<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: cloze-test | CardType: Missing | Tags: test -->

<!-- Title -->
Fill in the blank

<!-- Key point -->
The answer is {{c1::42}}

<!-- Key point notes -->
Remember the number

<!-- END_CARDS -->"""
        result = validate_apf_markdown(apf)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_detects_unclosed_fence_in_section(self) -> None:
        """Unclosed code fence in section should be detected."""
        apf = """<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: test | CardType: Simple | Tags: test -->

<!-- Title -->
Test title

<!-- Key point -->
```python
def broken():
    pass

<!-- Key point notes -->
Notes here

<!-- END_CARDS -->"""
        result = validate_apf_markdown(apf)
        # Should detect the unclosed fence
        assert any("code fence" in e.lower() for e in result.errors) or not result.is_valid

    def test_multiple_cards(self) -> None:
        """Multiple cards in document should all be validated."""
        apf = """<!-- BEGIN_CARDS -->

<!-- Card 1 | slug: card-1 | CardType: Simple | Tags: test -->
<!-- Title -->
First card

<!-- Key point notes -->
Notes for first

<!-- Card 2 | slug: card-2 | CardType: Simple | Tags: test -->
<!-- Title -->
Second card

<!-- Key point notes -->
Notes for second

<!-- END_CARDS -->"""
        result = validate_apf_markdown(apf)
        assert result.is_valid, f"Errors: {result.errors}"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_card_with_optional_sections(self) -> None:
        """Card with optional sections should be valid."""
        apf = """<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: test | CardType: Simple | Tags: test -->

<!-- Title -->
Test title

<!-- Subtitle (optional) -->
Optional subtitle

<!-- Sample (caption) (optional) -->
Caption text

<!-- Key point (code block / image) -->
Key point content

<!-- Key point notes -->
Notes here

<!-- Other notes (optional) -->
Extra notes

<!-- END_CARDS -->"""
        result = validate_apf_markdown(apf)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_mathjax_content(self) -> None:
        """MathJax content should not cause issues."""
        apf = r"""<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: math | CardType: Simple | Tags: math -->

<!-- Title -->
What is the formula for \(E = mc^2\)?

<!-- Key point notes -->
Energy equals mass times speed of light squared

<!-- END_CARDS -->"""
        result = validate_apf_markdown(apf)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_special_characters_in_content(self) -> None:
        """Special characters should not cause issues."""
        apf = """<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: special | CardType: Simple | Tags: test -->

<!-- Title -->
What about < > & " ' characters?

<!-- Key point notes -->
They should work fine

<!-- END_CARDS -->"""
        result = validate_apf_markdown(apf)
        assert result.is_valid, f"Errors: {result.errors}"
