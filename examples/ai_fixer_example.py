"""Example usage of the AI validation system.

This script demonstrates how to use AIFixer and AIFixerValidator
for AI-powered validation and auto-fixing of Q&A notes.
"""

from pathlib import Path

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.validation import AIFixer, AIFixerValidator


def example_basic_usage():
    """Example 1: Basic usage with config."""
    print("Example 1: Basic AIFixer usage")
    print("=" * 60)

    # Create a minimal config (AI disabled by default)
    config = Config(
        vault_path=Path.home() / "Documents" / "InterviewQuestions",
        enable_ai_validation=False,  # Explicitly disable for this example
    )

    # Create AIFixer from config
    ai_fixer = AIFixer.from_config(config)
    print(f"AI Fixer available: {ai_fixer.available}")
    print()


def example_with_provider():
    """Example 2: Using AIFixer with a provider."""
    print("Example 2: AIFixer with explicit provider")
    print("=" * 60)

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

        print(f"AI Fixer available: {ai_fixer.available}")
        print(f"Provider: {provider.get_provider_name()}")
        print()

        # Example: Detect code language
        code_sample = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        language = ai_fixer.detect_code_language(code_sample)
        print(f"Detected language: {language}")
        print()

    except Exception as e:
        print(f"Error: {e}")
        print("(This is expected if provider is not available)")
        print()


def example_validator():
    """Example 3: Using AIFixerValidator."""
    print("Example 3: AIFixerValidator usage")
    print("=" * 60)

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
    print(f"Found {len(issues)} issue(s)")
    for issue in issues:
        print(f"  - {issue}")

    print(f"\nAvailable fixes: {len(validator.fixes)}")
    for fix in validator.fixes:
        print(f"  - {fix.description}")
    print()


def example_with_ai_enabled():
    """Example 4: AIFixerValidator with AI enabled."""
    print("Example 4: AIFixerValidator with AI enabled")
    print("=" * 60)

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
        print(f"Found {len(issues)} issue(s)")

        # Show available fixes
        print(f"\nAvailable fixes: {len(validator.fixes)}")
        for fix in validator.fixes:
            print(f"  - {fix.description} (safe: {fix.safe})")

        # Apply safe fixes
        if validator.get_safe_fixes():
            print("\nApplying safe fixes...")
            for fix in validator.get_safe_fixes():
                new_content, new_frontmatter = fix.fix_function()
                print(f"Applied: {fix.description}")

        print()

    except Exception as e:
        print(f"Error: {e}")
        print("(This is expected if provider is not available)")
        print()


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_with_provider()
    example_validator()
    example_with_ai_enabled()

    print("\nNote: Some examples may fail if providers are not configured.")
    print("Configure your LLM provider in config.yaml or .env file.")
