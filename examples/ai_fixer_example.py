"""Example usage of the AI validation system.

This script demonstrates how to use AIFixer and AIFixerValidator
for AI-powered validation and auto-fixing of Q&A notes.
"""

from pathlib import Path

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.validation import AIFixer, AIFixerValidator


def example_basic_usage():
    """Example 1: Basic usage with config."""

    # Create a minimal config (AI disabled by default)
    config = Config(
        vault_path=Path.home() / "Documents" / "InterviewQuestions",
        enable_ai_validation=False,  # Explicitly disable for this example
    )

    # Create AIFixer from config
    ai_fixer = AIFixer.from_config(config)


def example_with_provider():
    """Example 2: Using AIFixer with a provider."""

    try:
        from obsidian_anki_sync.providers.factory import ProviderFactory

        # Create a config with AI enabled
        config = Config(
            vault_path=Path.home() / "Documents" / "InterviewQuestions",
            enable_ai_validation=True,
            llm_provider="ollama",
            ollama_base_url="http://localhost:11434",
            ai_validation_model="qwen/qwen-2.5-14b-instruct",
        )

        # Create provider and AIFixer
        provider = ProviderFactory.create_from_config(config)
        ai_fixer = AIFixer(
            provider=provider,
            model="qwen/qwen-2.5-14b-instruct",
            temperature=0.1,
        )

        # Example: Detect code language
        code_sample = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        language = ai_fixer.detect_code_language(code_sample)

    except Exception as e:
        pass


def example_validator():
    """Example 3: Using AIFixerValidator."""

    # Sample note content with issues
    content = """---
title: Sample Question
tags: [programming, python]
---

# Question (EN)

What is recursion?

# Answer

```
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
```

The code above shows recursion.
"""

    frontmatter = {
        "title": "Sample Question",
        "tags": ["programming", "python"],
    }

    # Create validator without AI (AI disabled by default)
    validator = AIFixerValidator(
        content=content,
        frontmatter=frontmatter,
        filepath="/path/to/note.md",
        ai_fixer=None,  # No AI fixer
        enable_ai_fixes=False,
    )

    # Run validation
    issues = validator.validate()
    for issue in issues:
        pass

    for fix in validator.fixes:
        pass


def example_with_ai_enabled():
    """Example 4: AIFixerValidator with AI enabled."""

    try:
        # Create config with AI enabled
        config = Config(
            vault_path=Path.home() / "Documents" / "InterviewQuestions",
            enable_ai_validation=True,
            llm_provider="ollama",
        )

        # Create AI fixer from config
        ai_fixer = AIFixer.from_config(config)

        # Sample note content with issues
        content = """---
title: Recursion Example
tags: [programming]
---

# Question (EN)

What is recursion?

# Answer

```
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
```
"""

        frontmatter = {
            "title": "Recursion Example",
            "tags": ["programming"],
        }

        # Create validator with AI
        validator = AIFixerValidator(
            content=content,
            frontmatter=frontmatter,
            filepath="/path/to/note.md",
            ai_fixer=ai_fixer,
            enable_ai_fixes=True,
        )

        # Run validation
        issues = validator.validate()

        # Show available fixes
        for fix in validator.fixes:
            pass

        # Apply safe fixes
        if validator.get_safe_fixes():
            for fix in validator.get_safe_fixes():
                _new_content, _new_frontmatter = fix.fix_function()

    except Exception as e:
        pass


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_with_provider()
    example_validator()
    example_with_ai_enabled()
