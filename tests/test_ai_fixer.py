"""Tests for AI-powered validation and auto-fixing."""

from obsidian_anki_sync.validation import AIFixer, AIFixerValidator


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(self):
        self.call_count = 0

    def get_provider_name(self) -> str:
        return "MockProvider"

    def generate_json(self, model, prompt, system, temperature, json_schema):
        """Mock JSON generation."""
        self.call_count += 1

        # Detect if this is code language detection
        if "programming language" in prompt.lower():
            return {"language": "python", "confidence": "high"}

        # Detect if this is title generation
        if "bilingual title" in prompt.lower():
            return {
                "en_title": "Recursion in Programming",
                "ru_title": "Рекурсия в программировании",
            }

        return {}


class TestAIFixer:
    """Test AIFixer class."""

    def test_init_with_provider(self):
        """Test initialization with provider."""
        provider = MockProvider()
        ai_fixer = AIFixer(provider=provider, model="test-model")

        assert ai_fixer.available is True
        assert ai_fixer.provider == provider
        assert ai_fixer.model == "test-model"

    def test_init_without_provider(self):
        """Test initialization without provider."""
        ai_fixer = AIFixer(provider=None)

        assert ai_fixer.available is False
        assert ai_fixer.provider is None

    def test_detect_code_language_success(self):
        """Test code language detection success."""
        provider = MockProvider()
        ai_fixer = AIFixer(provider=provider)

        code = "def foo():\n    pass"
        language = ai_fixer.detect_code_language(code)

        assert language == "python"
        assert provider.call_count == 1

    def test_detect_code_language_no_provider(self):
        """Test code language detection without provider."""
        ai_fixer = AIFixer(provider=None)

        code = "def foo():\n    pass"
        language = ai_fixer.detect_code_language(code)

        assert language is None

    def test_detect_code_language_empty_code(self):
        """Test code language detection with empty code."""
        provider = MockProvider()
        ai_fixer = AIFixer(provider=provider)

        language = ai_fixer.detect_code_language("")

        assert language is None
        assert provider.call_count == 0

    def test_generate_bilingual_title_success(self):
        """Test bilingual title generation success."""
        provider = MockProvider()
        ai_fixer = AIFixer(provider=provider)

        content = "# Question (EN)\n\nWhat is recursion?"
        current_title = "Recursion"
        result = ai_fixer.generate_bilingual_title(content, current_title)

        assert result is not None
        en_title, ru_title = result
        assert en_title == "Recursion in Programming"
        assert ru_title == "Рекурсия в программировании"
        assert provider.call_count == 1

    def test_generate_bilingual_title_no_provider(self):
        """Test bilingual title generation without provider."""
        ai_fixer = AIFixer(provider=None)

        content = "# Question (EN)\n\nWhat is recursion?"
        current_title = "Recursion"
        result = ai_fixer.generate_bilingual_title(content, current_title)

        assert result is None


class TestAIFixerValidator:
    """Test AIFixerValidator class."""

    def test_init_with_ai_fixer(self):
        """Test initialization with AI fixer."""
        provider = MockProvider()
        ai_fixer = AIFixer(provider=provider)

        content = "test content"
        frontmatter = {"title": "Test"}
        filepath = "/test/note.md"

        validator = AIFixerValidator(
            content=content,
            frontmatter=frontmatter,
            filepath=filepath,
            ai_fixer=ai_fixer,
            enable_ai_fixes=True,
        )

        assert validator.ai_fixer == ai_fixer
        assert validator.enable_ai_fixes is True

    def test_init_without_ai_fixer(self):
        """Test initialization without AI fixer."""
        content = "test content"
        frontmatter = {"title": "Test"}
        filepath = "/test/note.md"

        validator = AIFixerValidator(
            content=content,
            frontmatter=frontmatter,
            filepath=filepath,
            ai_fixer=None,
            enable_ai_fixes=True,
        )

        assert validator.ai_fixer is None
        assert validator.enable_ai_fixes is False

    def test_validate_code_blocks_without_language(self):
        """Test validation of code blocks without language."""
        content = """# Test

Here's some code:

```
def foo():
    pass
```

More text.
"""
        frontmatter = {"title": "Test / Тест"}
        filepath = "/test/note.md"

        provider = MockProvider()
        ai_fixer = AIFixer(provider=provider)

        validator = AIFixerValidator(
            content=content,
            frontmatter=frontmatter,
            filepath=filepath,
            ai_fixer=ai_fixer,
            enable_ai_fixes=True,
        )

        validator.validate()

        # Should find fixes for code blocks
        assert len(validator.fixes) > 0
        fix_descriptions = [f.description for f in validator.fixes]
        assert any("AI-detect language" in desc for desc in fix_descriptions)

    def test_validate_missing_bilingual_title(self):
        """Test validation of missing bilingual title."""
        content = "# Test content"
        frontmatter = {"title": "Single Title"}
        filepath = "/test/note.md"

        provider = MockProvider()
        ai_fixer = AIFixer(provider=provider)

        validator = AIFixerValidator(
            content=content,
            frontmatter=frontmatter,
            filepath=filepath,
            ai_fixer=ai_fixer,
            enable_ai_fixes=True,
        )

        validator.validate()

        # Should find fix for bilingual title
        assert len(validator.fixes) > 0
        fix_descriptions = [f.description for f in validator.fixes]
        assert any("bilingual title" in desc.lower() for desc in fix_descriptions)

    def test_validate_list_formatting(self):
        """Test validation of list formatting."""
        content = """# Test

-  Item 1
-   Item 2
"""
        frontmatter = {"title": "Test / Тест"}
        filepath = "/test/note.md"

        validator = AIFixerValidator(
            content=content,
            frontmatter=frontmatter,
            filepath=filepath,
            ai_fixer=None,
            enable_ai_fixes=False,
        )

        validator.validate()

        # Should find fix for list spacing
        assert len(validator.fixes) > 0
        fix_descriptions = [f.description for f in validator.fixes]
        assert any("list spacing" in desc.lower() for desc in fix_descriptions)

    def test_fix_code_block_languages(self):
        """Test fixing code block languages."""
        content = """# Test

```
def foo():
    pass
```
"""
        frontmatter = {"title": "Test / Тест"}
        filepath = "/test/note.md"

        provider = MockProvider()
        ai_fixer = AIFixer(provider=provider)

        validator = AIFixerValidator(
            content=content,
            frontmatter=frontmatter,
            filepath=filepath,
            ai_fixer=ai_fixer,
            enable_ai_fixes=True,
        )

        validator.validate()

        # Apply the fix
        for fix in validator.fixes:
            if "AI-detect language" in fix.description:
                new_content, new_frontmatter = fix.fix_function()

                # Should add language to code block
                assert "```python" in new_content
                assert new_frontmatter == frontmatter

    def test_fix_bilingual_title(self):
        """Test fixing bilingual title."""
        content = "# Question (EN)\n\nWhat is recursion?"
        frontmatter = {"title": "Recursion"}
        filepath = "/test/note.md"

        provider = MockProvider()
        ai_fixer = AIFixer(provider=provider)

        validator = AIFixerValidator(
            content=content,
            frontmatter=frontmatter,
            filepath=filepath,
            ai_fixer=ai_fixer,
            enable_ai_fixes=True,
        )

        validator.validate()

        # Apply the fix
        for fix in validator.fixes:
            if "bilingual title" in fix.description.lower():
                new_content, new_frontmatter = fix.fix_function()

                # Should update title
                assert " / " in new_frontmatter["title"]
                assert "Recursion in Programming" in new_frontmatter["title"]
                assert "Рекурсия в программировании" in new_frontmatter["title"]
                assert new_content == content

    def test_fix_list_spacing(self):
        """Test fixing list spacing."""
        content = """# Test

-  Item 1
-   Item 2
"""
        frontmatter = {"title": "Test / Тест"}
        filepath = "/test/note.md"

        validator = AIFixerValidator(
            content=content,
            frontmatter=frontmatter,
            filepath=filepath,
            ai_fixer=None,
            enable_ai_fixes=False,
        )

        validator.validate()

        # Apply the fix
        for fix in validator.fixes:
            if "list spacing" in fix.description.lower():
                new_content, new_frontmatter = fix.fix_function()

                # Should normalize spacing
                assert "- Item 1" in new_content
                assert "- Item 2" in new_content
                assert "-  Item" not in new_content
                assert new_frontmatter == frontmatter

    def test_safe_fixes(self):
        """Test safe vs unsafe fixes."""
        content = """# Test

```
def foo():
    pass
```
"""
        frontmatter = {"title": "Single Title"}
        filepath = "/test/note.md"

        provider = MockProvider()
        ai_fixer = AIFixer(provider=provider)

        validator = AIFixerValidator(
            content=content,
            frontmatter=frontmatter,
            filepath=filepath,
            ai_fixer=ai_fixer,
            enable_ai_fixes=True,
        )

        validator.validate()

        # Code block fix should be safe
        safe_fixes = validator.get_safe_fixes()
        assert any("AI-detect language" in f.description for f in safe_fixes)

        # Title fix should be unsafe
        unsafe_fixes = [f for f in validator.fixes if not f.safe]
        assert any("bilingual title" in f.description.lower() for f in unsafe_fixes)

    def test_disabled_ai_adds_issues_not_fixes(self):
        """Test that disabled AI adds issues instead of fixes."""
        content = """# Test

```
def foo():
    pass
```
"""
        frontmatter = {"title": "Test / Тест"}
        filepath = "/test/note.md"

        # No AI fixer provided
        validator = AIFixerValidator(
            content=content,
            frontmatter=frontmatter,
            filepath=filepath,
            ai_fixer=None,
            enable_ai_fixes=False,
        )

        issues = validator.validate()

        # Should add issue about code block, but no AI fix
        issue_messages = [issue.message for issue in issues]
        assert any("code block" in msg.lower() for msg in issue_messages)

        # Should still have non-AI fixes (like list spacing)
        fix_descriptions = [f.description for f in validator.fixes]
        assert not any("AI-detect language" in desc for desc in fix_descriptions)
