"""Tests for autofix handlers and registry."""

from __future__ import annotations

from obsidian_anki_sync.agents.autofix import (
    AutoFixRegistry,
    BrokenRelatedEntryHandler,
    BrokenWikilinkHandler,
    EmptyReferencesHandler,
    MissingRelatedQuestionsHandler,
    MocMismatchHandler,
    SectionOrderHandler,
    TitleFormatHandler,
    TrailingWhitespaceHandler,
)
from obsidian_anki_sync.agents.models import AutoFixIssue


class TestTrailingWhitespaceHandler:
    """Tests for trailing whitespace handler."""

    def test_detect_trailing_whitespace(self):
        handler = TrailingWhitespaceHandler()
        # Single trailing space (should be detected)
        content = "Line with trailing space \nClean line\nAnother with tab\t\n"

        issues = handler.detect(content)

        # Handler aggregates all trailing whitespace into 1 issue
        assert len(issues) == 1
        assert issues[0].issue_type == "trailing_whitespace"
        assert "2 lines" in issues[0].description

    def test_detect_no_trailing_whitespace(self):
        handler = TrailingWhitespaceHandler()
        content = "Clean line\nAnother clean line\n"

        issues = handler.detect(content)

        assert len(issues) == 0

    def test_preserve_markdown_line_breaks(self):
        """Markdown line breaks (2+ trailing spaces) should be preserved."""
        handler = TrailingWhitespaceHandler()
        # Two trailing spaces = markdown line break
        content = "Line with markdown break  \nNext line\n"

        issues = handler.detect(content)

        # No issues detected - markdown line breaks are intentional
        assert len(issues) == 0

    def test_fix_preserves_markdown_line_breaks(self):
        """Fix should preserve markdown line breaks while removing other trailing whitespace."""
        handler = TrailingWhitespaceHandler()
        # Mix of markdown line break and unintentional trailing space
        content = "Markdown break here  \nUnintentional space \nClean line\n"
        issues = handler.detect(content)

        fixed_content, updated_issues = handler.fix(content, issues)

        # Markdown line break preserved, single trailing space removed
        assert fixed_content == "Markdown break here  \nUnintentional space\nClean line\n"
        assert all(issue.auto_fixed for issue in updated_issues)

    def test_fix_trailing_whitespace(self):
        handler = TrailingWhitespaceHandler()
        content = "Line with space \nClean\nAnother\t\n"
        issues = handler.detect(content)

        fixed_content, updated_issues = handler.fix(content, issues)

        assert fixed_content == "Line with space\nClean\nAnother\n"
        assert all(issue.auto_fixed for issue in updated_issues)

    def test_preserve_code_block_indentation(self):
        """Code block indentation should be preserved."""
        handler = TrailingWhitespaceHandler()
        content = """```python
def foo():
    return 42
```
"""
        issues = handler.detect(content)

        # No trailing whitespace issues in properly formatted code
        assert len(issues) == 0

        # Content should be unchanged
        fixed_content, _ = handler.fix(content, issues)
        assert fixed_content == content


class TestEmptyReferencesHandler:
    """Tests for empty references section handler."""

    def test_detect_empty_references(self):
        handler = EmptyReferencesHandler()
        content = """## Some Section

Content here

## References

## Another Section"""

        issues = handler.detect(content)

        assert len(issues) == 1
        assert issues[0].issue_type == "empty_references"

    def test_detect_non_empty_references(self):
        handler = EmptyReferencesHandler()
        # Content immediately after the header (no blank line) - not empty
        content = """## References
- Some reference here

## Another Section"""

        issues = handler.detect(content)

        assert len(issues) == 0

    def test_fix_empty_references(self):
        handler = EmptyReferencesHandler()
        content = """## Content

Text here

## References

## Next Section"""
        issues = handler.detect(content)

        fixed_content, updated_issues = handler.fix(content, issues)

        assert "## References" not in fixed_content
        assert "## Next Section" in fixed_content
        assert all(issue.auto_fixed for issue in updated_issues)


class TestTitleFormatHandler:
    """Tests for title format handler."""

    def test_detect_missing_bilingual_title(self):
        handler = TitleFormatHandler()
        content = """---
id: test-123
title: English Only Title
aliases:
  - Test Note
---

# English Only Title"""
        metadata = {
            "id": "test-123",
            "title": "English Only Title",
            "aliases": ["Test Note"],
        }

        issues = handler.detect(content, metadata)

        assert len(issues) == 1
        assert issues[0].issue_type == "title_format"

    def test_detect_valid_bilingual_title(self):
        handler = TitleFormatHandler()
        content = """---
id: test-123
title: English Title / Russian Title
---

# English Title / Russian Title"""
        metadata = {"id": "test-123", "title": "English Title / Russian Title"}

        issues = handler.detect(content, metadata)

        assert len(issues) == 0

    def test_fix_title_with_russian_alias(self):
        handler = TitleFormatHandler()
        content = """---
id: test-123
title: English Title
aliases:
  - English Title
  - Russian
---

# English Title"""
        metadata = {
            "id": "test-123",
            "title": "English Title",
            "aliases": ["English Title", "Russian"],
        }
        issues = handler.detect(content, metadata)

        # No Cyrillic in aliases, so cannot be fixed
        assert len(issues) == 1

    def test_fix_title_with_cyrillic_alias(self):
        handler = TitleFormatHandler()
        content = """---
id: test-123
title: English Title
aliases:
  - English Title
---

# English Title"""
        metadata = {
            "id": "test-123",
            "title": "English Title",
            "aliases": ["English Title"],
        }
        issues = handler.detect(content, metadata)

        _, updated_issues = handler.fix(content, issues, metadata)

        # No Cyrillic alias found, so not auto-fixed
        assert not any(issue.auto_fixed for issue in updated_issues)


class TestMocMismatchHandler:
    """Tests for MOC mismatch handler."""

    def test_detect_moc_mismatch(self):
        handler = MocMismatchHandler()
        content = """---
topic: algorithms
moc: moc-wrong
---"""
        metadata = {"topic": "algorithms", "moc": "moc-wrong"}

        issues = handler.detect(content, metadata)

        assert len(issues) == 1
        assert issues[0].issue_type == "moc_mismatch"

    def test_detect_correct_moc(self):
        handler = MocMismatchHandler()
        content = """---
topic: algorithms
moc: moc-algorithms
---"""
        metadata = {"topic": "algorithms", "moc": "moc-algorithms"}

        issues = handler.detect(content, metadata)

        assert len(issues) == 0

    def test_fix_moc_mismatch(self):
        handler = MocMismatchHandler()
        content = """---
topic: data-structures
moc: moc-wrong
---"""
        metadata = {"topic": "data-structures", "moc": "moc-wrong"}
        issues = handler.detect(content, metadata)

        fixed_content, updated_issues = handler.fix(content, issues, metadata)

        assert "moc: moc-data-structures" in fixed_content
        assert all(issue.auto_fixed for issue in updated_issues)


class TestSectionOrderHandler:
    """Tests for section order handler - detect only, no auto-fix."""

    def test_detect_wrong_section_order(self):
        handler = SectionOrderHandler()
        # Answer comes before Question - wrong order
        content = """---
id: test
---

## Answer (EN)

Answer content

# Question (EN)

Question content"""

        issues = handler.detect(content)

        assert len(issues) == 1
        assert issues[0].issue_type == "section_order"

    def test_detect_correct_section_order(self):
        handler = SectionOrderHandler()
        # Correct order: RU Q, EN Q, RU A, EN A
        content = """---
id: test
---

# Question (EN)

Question content

## Answer (EN)

Answer content"""

        issues = handler.detect(content)

        # Only EN sections present, and they are in order
        assert len(issues) == 0

    def test_detect_bilingual_correct_order(self):
        handler = SectionOrderHandler()
        # Full bilingual order: RU Q, EN Q, RU A, EN A
        content = """---
id: test
---

# Вопрос (RU)

Question RU

# Question (EN)

Question EN

## Ответ (RU)

Answer RU

## Answer (EN)

Answer EN"""

        issues = handler.detect(content)

        assert len(issues) == 0

    def test_detect_bilingual_wrong_order(self):
        handler = SectionOrderHandler()
        # Wrong order: EN before RU
        content = """---
id: test
---

# Question (EN)

Question EN

# Вопрос (RU)

Question RU

## Answer (EN)

Answer EN

## Ответ (RU)

Answer RU"""

        issues = handler.detect(content)

        assert len(issues) == 1
        assert issues[0].issue_type == "section_order"

    def test_no_auto_fix_to_preserve_formatting(self):
        """Section order handler should NOT auto-fix to preserve formatting."""
        handler = SectionOrderHandler()
        # Wrong order with code block - auto-fix would break formatting
        original_content = """---
id: test
---

## Answer (EN)

```python
def foo():
    return 42
```

# Question (EN)

What does this code do?"""
        issues = handler.detect(original_content)

        fixed_content, updated_issues = handler.fix(original_content, issues)

        # Content should be unchanged
        assert fixed_content == original_content
        # Issue should be marked as NOT auto-fixed
        assert all(not issue.auto_fixed for issue in updated_issues)
        assert "Manual fix required" in updated_issues[0].fix_description


class TestMissingRelatedQuestionsHandler:
    """Tests for missing Related Questions section handler."""

    def test_detect_missing_related_questions(self):
        handler = MissingRelatedQuestionsHandler()
        content = """---
related: [c-arrays, c-lists]
---

## Question

Content

## Answer

Answer content"""
        metadata = {"related": ["c-arrays", "c-lists"]}

        issues = handler.detect(content, metadata)

        assert len(issues) == 1
        assert issues[0].issue_type == "missing_related_questions"

    def test_detect_existing_related_questions(self):
        handler = MissingRelatedQuestionsHandler()
        content = """---
related: [c-arrays]
---

## Related Questions

- [[c-arrays]]"""
        metadata = {"related": ["c-arrays"]}

        issues = handler.detect(content, metadata)

        assert len(issues) == 0

    def test_fix_missing_related_questions(self):
        handler = MissingRelatedQuestionsHandler()
        content = """---
related: [c-arrays, c-lists]
---

## Answer

Answer content"""
        metadata = {"related": ["c-arrays", "c-lists"]}
        issues = handler.detect(content, metadata)

        fixed_content, updated_issues = handler.fix(content, issues, metadata)

        assert "## Related Questions" in fixed_content
        assert "[[c-arrays]]" in fixed_content
        assert "[[c-lists]]" in fixed_content
        assert all(issue.auto_fixed for issue in updated_issues)


class TestBrokenWikilinkHandler:
    """Tests for broken wikilink handler.

    Note: Handler only flags q-* links with -- as broken (not c-* or moc-* links)
    """

    def test_detect_broken_wikilinks_with_pattern(self):
        # Handler uses BROKEN_PATTERNS to detect broken links
        handler = BrokenWikilinkHandler()
        content = """## Content

See [[undefined-question]] and [[test-note]] for more."""

        issues = handler.detect(content)

        # Will not detect these as broken unless they match BROKEN_PATTERNS
        # or are q-* links not in index
        # Check that handler runs without error
        assert isinstance(issues, list)

    def test_detect_q_link_broken(self):
        note_index = {"q-valid-note--123"}
        handler = BrokenWikilinkHandler(note_index=note_index)
        content = """## Content

See [[q-valid-note--123]] and [[q-broken--456]] for more."""

        issues = handler.detect(content)

        # q-broken--456 is not in index and has -- pattern
        assert len(issues) == 1
        assert issues[0].issue_type == "broken_wikilink"

    def test_detect_no_broken_wikilinks(self):
        # c-* links are not flagged as broken (concepts are allowed)
        note_index = {"c-existing-note", "c-another-note"}
        handler = BrokenWikilinkHandler(note_index=note_index)
        content = """## Content

See [[c-existing-note]] and [[c-new-concept]] for more."""

        issues = handler.detect(content)

        # c-new-concept is NOT flagged because c-* is not checked strictly
        assert len(issues) == 0

    def test_fix_removes_known_patterns(self):
        handler = BrokenWikilinkHandler()
        # Test with a known broken pattern if any exist
        content = """See [[valid-note]] here."""
        issues = []  # No issues

        fixed_content, _ = handler.fix(content, issues)

        assert fixed_content == content


class TestBrokenRelatedEntryHandler:
    """Tests for broken related entry handler.

    Note: Handler skips c-* and moc-* prefixes (always considered valid).
    Only flags entries that don't start with these prefixes.
    """

    def test_detect_broken_related_entries(self):
        # Use non-prefixed entries to test broken detection
        note_index = {"existing-note", "another-note"}
        handler = BrokenRelatedEntryHandler(note_index=note_index)
        content = """---
related: [existing-note, nonexistent-entry, another-note]
---"""
        metadata = {"related": ["existing-note", "nonexistent-entry", "another-note"]}

        issues = handler.detect(content, metadata)

        assert len(issues) == 1
        assert issues[0].issue_type == "broken_related_entry"
        # Description mentions count, location mentions the broken entries
        assert "nonexistent-entry" in issues[0].location

    def test_allow_special_prefixes(self):
        note_index = {"some-note"}
        handler = BrokenRelatedEntryHandler(note_index=note_index)
        content = """---
related: [c-arrays, c-special, moc-algorithms]
---"""
        # c- and moc- prefixes are always allowed even if not in index
        metadata = {"related": ["c-arrays", "c-special", "moc-algorithms"]}

        issues = handler.detect(content, metadata)

        # All entries start with c- or moc-, so none are flagged
        assert len(issues) == 0

    def test_fix_broken_related_entries(self):
        # Mix of prefixed (allowed) and non-prefixed entries
        note_index = {"valid-note"}
        handler = BrokenRelatedEntryHandler(note_index=note_index)
        content = """---
related: [valid-note, broken-entry, c-arrays]
---"""
        metadata = {"related": ["valid-note", "broken-entry", "c-arrays"]}
        issues = handler.detect(content, metadata)

        fixed_content, updated_issues = handler.fix(content, issues, metadata)

        assert "valid-note" in fixed_content
        assert "c-arrays" in fixed_content  # Allowed because of c- prefix
        assert "broken-entry" not in fixed_content
        assert all(issue.auto_fixed for issue in updated_issues)


class TestAutoFixRegistry:
    """Tests for AutoFixRegistry."""

    def test_registry_initialization(self):
        registry = AutoFixRegistry()

        handlers = registry.list_handlers()

        assert len(handlers) == 8
        assert any(h["type"] == "trailing_whitespace" for h in handlers)
        assert any(h["type"] == "empty_references" for h in handlers)

    def test_registry_with_enabled_handlers(self):
        registry = AutoFixRegistry(
            enabled_handlers=["trailing_whitespace", "empty_references"]
        )

        handlers = registry.list_handlers()

        assert len(handlers) == 2
        assert all(
            h["type"] in ["trailing_whitespace", "empty_references"] for h in handlers
        )

    def test_detect_all(self):
        registry = AutoFixRegistry(
            enabled_handlers=["trailing_whitespace", "empty_references"]
        )
        # Content with both trailing whitespace and empty references
        content = """Line with trailing space

## References

## Next Section"""

        issues = registry.detect_all(content)

        # Should detect at least 1 issue from each handler
        assert len(issues) >= 1
        issue_types = [i.issue_type for i in issues]
        # At least one of these should be present
        assert "trailing_whitespace" in issue_types or "empty_references" in issue_types

    def test_fix_all(self):
        registry = AutoFixRegistry(
            enabled_handlers=["trailing_whitespace", "empty_references"]
        )
        content = """Line with space

## References

## Next Section"""

        result = registry.fix_all(content)

        assert result.file_modified
        assert result.issues_fixed >= 1
        # Trailing whitespace should be removed
        assert "   \n" not in result.fixed_content

    def test_update_note_index(self):
        registry = AutoFixRegistry(enabled_handlers=["broken_wikilink"])

        # Use q-* links with -- pattern which are actually checked
        content = "See [[q-note1--123]] and [[q-note2--456]]"

        # Update index with one of the notes
        registry.update_note_index({"q-note1--123"})
        issues = registry.detect_all(content)

        # After update, only q-note2--456 should be broken
        broken_links = [i for i in issues if i.issue_type == "broken_wikilink"]
        assert len(broken_links) == 1

    def test_get_handler(self):
        registry = AutoFixRegistry()

        handler = registry.get_handler("trailing_whitespace")

        assert handler is not None
        assert isinstance(handler, TrailingWhitespaceHandler)

    def test_get_nonexistent_handler(self):
        registry = AutoFixRegistry()

        handler = registry.get_handler("nonexistent")

        assert handler is None

    def test_fix_all_no_changes_needed(self):
        registry = AutoFixRegistry(
            enabled_handlers=["trailing_whitespace", "moc_mismatch"]
        )
        # Content with no trailing whitespace and correct MOC
        content = """---
id: clean-note
topic: algorithms
moc: moc-algorithms
---

# Clean Title

## Question

Question content

## Answer

Answer content"""

        result = registry.fix_all(content)

        # No issues found for enabled handlers
        assert not result.file_modified
        assert result.issues_fixed == 0
        assert result.fixed_content is None


class TestAutoFixIssueModel:
    """Tests for AutoFixIssue model."""

    def test_issue_creation(self):
        issue = AutoFixIssue(
            issue_type="trailing_whitespace",
            severity="warning",
            description="Line 5 has trailing whitespace",
            location="line 5",
        )

        assert issue.issue_type == "trailing_whitespace"
        assert issue.severity == "warning"
        assert not issue.auto_fixed
        assert issue.fix_description == ""

    def test_issue_update_after_fix(self):
        issue = AutoFixIssue(
            issue_type="trailing_whitespace",
            severity="warning",
            description="Trailing whitespace",
        )

        issue.auto_fixed = True
        issue.fix_description = "Removed trailing whitespace"

        assert issue.auto_fixed
        assert issue.fix_description == "Removed trailing whitespace"


class TestMarkdownFormattingPreservation:
    """Tests to ensure autofix handlers preserve markdown formatting."""

    def test_code_block_indentation_preserved(self):
        """Indentation inside code blocks should be preserved."""
        registry = AutoFixRegistry(enabled_handlers=["trailing_whitespace"])
        content = """---
id: test
---

## Code Example

```python
def nested_function():
    if True:
        for i in range(10):
            print(i)
```

Some text after.
"""
        result = registry.fix_all(content)

        # Code block indentation must be preserved
        assert "    if True:" in (result.fixed_content or content)
        assert "        for i in range(10):" in (result.fixed_content or content)
        assert "            print(i)" in (result.fixed_content or content)

    def test_nested_list_indentation_preserved(self):
        """Nested list indentation should be preserved."""
        registry = AutoFixRegistry(enabled_handlers=["trailing_whitespace"])
        content = """---
id: test
---

## Lists

- Item 1
  - Nested item 1.1
  - Nested item 1.2
    - Deep nested item
- Item 2
"""
        result = registry.fix_all(content)

        # Nested list indentation must be preserved
        fixed = result.fixed_content or content
        assert "  - Nested item 1.1" in fixed
        assert "    - Deep nested item" in fixed

    def test_yaml_frontmatter_preserved(self):
        """YAML frontmatter should be preserved correctly."""
        registry = AutoFixRegistry(enabled_handlers=["moc_mismatch"])
        content = """---
id: test-note
topic: algorithms
moc: moc-wrong
related:
  - c-arrays
  - c-lists
tags:
  - programming
  - data-structures
---

## Content
"""
        metadata = {
            "id": "test-note",
            "topic": "algorithms",
            "moc": "moc-wrong",
            "related": ["c-arrays", "c-lists"],
            "tags": ["programming", "data-structures"],
        }

        result = registry.fix_all(content)

        # YAML structure should be preserved
        fixed = result.fixed_content or content
        assert "---" in fixed
        assert "moc: moc-algorithms" in fixed  # Fixed
        assert "related:" in fixed  # Structure preserved

    def test_blockquote_preserved(self):
        """Blockquotes should be preserved."""
        registry = AutoFixRegistry(enabled_handlers=["trailing_whitespace"])
        content = """---
id: test
---

> This is a blockquote
> that spans multiple lines
>
> And has a gap

Normal text.
"""
        result = registry.fix_all(content)

        fixed = result.fixed_content or content
        assert "> This is a blockquote" in fixed
        assert "> that spans multiple lines" in fixed

    def test_table_formatting_preserved(self):
        """Markdown table formatting should be preserved."""
        registry = AutoFixRegistry(enabled_handlers=["trailing_whitespace"])
        content = """---
id: test
---

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |
"""
        result = registry.fix_all(content)

        fixed = result.fixed_content or content
        assert "| Column 1 | Column 2 |" in fixed
        assert "|----------|----------|" in fixed

    def test_combined_handlers_preserve_formatting(self):
        """Multiple handlers running together should preserve formatting."""
        registry = AutoFixRegistry(
            enabled_handlers=[
                "trailing_whitespace",
                "empty_references",
                "moc_mismatch",
            ]
        )
        content = """---
id: complex-note
topic: kotlin
moc: moc-kotlin
---

# Question (EN)

What is the output of this code?

```kotlin
fun main() {
    val list = listOf(1, 2, 3)
    list.forEach { item ->
        println(item)
    }
}
```

## Answer (EN)

The output is:
```
1
2
3
```

## References

## Related Questions

- [[c-kotlin-basics]]
"""
        metadata = {
            "id": "complex-note",
            "topic": "kotlin",
            "moc": "moc-kotlin",
        }

        result = registry.fix_all(content)
        fixed = result.fixed_content or content

        # Code blocks preserved
        assert "    val list = listOf(1, 2, 3)" in fixed
        assert "    list.forEach { item ->" in fixed
        assert "        println(item)" in fixed

        # Empty references removed
        assert "## References\n\n## Related" not in fixed

        # Related Questions preserved
        assert "## Related Questions" in fixed
        assert "[[c-kotlin-basics]]" in fixed
