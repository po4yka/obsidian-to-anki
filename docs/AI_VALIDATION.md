# AI-Powered Validation System

This document describes the AI-powered validation and auto-fixing system for Q&A notes.

## Overview

The AI validation system provides intelligent auto-fixes for common validation issues using LLM providers. It integrates seamlessly with the existing provider system and follows the BaseValidator pattern.

## Components

### AIFixer

The `AIFixer` class is the core AI-powered fixing engine that uses LLM providers to:

- Detect programming language for code blocks
- Generate bilingual titles (EN / RU)
- Fix common formatting issues

### AIFixerValidator

The `AIFixerValidator` is a validator that inherits from `BaseValidator` and provides:

- Code block language detection and fixing
- Bilingual title validation and generation
- List formatting fixes

## Configuration

Add the following to your `config.yaml`:

```yaml
# Enable AI-powered validation
enable_ai_validation: true

# Model to use for AI validation (optional, uses llm_provider if not set)
ai_validation_model: "qwen/qwen-2.5-14b-instruct"

# Temperature for AI validation (default: 0.1)
ai_validation_temperature: 0.1

# LLM Provider configuration (reuses existing provider system)
llm_provider: "ollama"  # or "openrouter", "openai", "anthropic"
ollama_base_url: "http://localhost:11434"
```

Or use environment variables:

```bash
ENABLE_AI_VALIDATION=true
AI_VALIDATION_MODEL=qwen/qwen-2.5-14b-instruct
LLM_PROVIDER=ollama
```

## Features

### 1. Code Block Language Detection

Automatically detects and adds language specifications to code blocks.

**Before:**
```markdown
```
def factorial(n):
    return 1 if n <= 1 else n * factorial(n-1)
```
```

**After:**
```markdown
```python
def factorial(n):
    return 1 if n <= 1 else n * factorial(n-1)
```
```

**How it works:**
- Analyzes code syntax and structure
- Returns language with confidence level (high/medium/low)
- Only applies fix if confidence is high or medium
- Safe auto-fix (can be applied without confirmation)

### 2. Bilingual Title Generation

Generates or fixes titles in bilingual format (English / Russian).

**Before:**
```yaml
title: Recursion Example
```

**After:**
```yaml
title: Recursion in Programming / Рекурсия в программировании
```

**How it works:**
- Analyzes question content for context
- Generates concise technical title in English
- Provides accurate Russian translation
- Maintains technical terminology in both languages
- Not safe (requires confirmation before applying)

### 3. List Formatting Fixes

Normalizes list formatting (single space after dash).

**Before:**
```markdown
-  Item 1
-   Item 2
```

**After:**
```markdown
- Item 1
- Item 2
```

**How it works:**
- Regex-based fix (no AI required)
- Safe auto-fix
- Normalizes multiple spaces to single space

## Usage

### Basic Usage

```python
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.validation import AIFixer, AIFixerValidator

# Create config with AI enabled
config = Config(
    vault_path="~/Documents/InterviewQuestions",
    enable_ai_validation=True,
)

# Create AI fixer from config
ai_fixer = AIFixer.from_config(config)

# Create validator
validator = AIFixerValidator(
    content=note_content,
    frontmatter=frontmatter,
    filepath=note_path,
    ai_fixer=ai_fixer,
    enable_ai_fixes=True,
)

# Run validation
issues = validator.validate()

# Apply safe fixes automatically
for fix in validator.get_safe_fixes():
    new_content, new_frontmatter = fix.fix_function()
```

### Manual AI Operations

```python
from obsidian_anki_sync.providers.factory import ProviderFactory
from obsidian_anki_sync.validation import AIFixer

# Create provider
provider = ProviderFactory.create_from_config(config)

# Create AI fixer
ai_fixer = AIFixer(
    provider=provider,
    model="qwen/qwen-2.5-14b-instruct",
    temperature=0.1,
)

# Detect code language
language = ai_fixer.detect_code_language(code_block)

# Generate bilingual title
en_title, ru_title = ai_fixer.generate_bilingual_title(
    content=note_content,
    current_title="Current Title"
)
```

## Integration with Existing Providers

The AI validation system reuses the existing provider infrastructure:

- Uses `ProviderFactory.create_from_config()` for provider initialization
- Supports all existing providers (Ollama, OpenRouter, OpenAI, Anthropic, LM Studio)
- No duplicate LLM provider code
- Consistent configuration across all LLM features

## Error Handling

The AI fixer is designed to gracefully handle errors:

- **Provider unavailable**: AI features are disabled, falls back to basic validation
- **API errors**: Logs error and returns None (no fix applied)
- **Invalid responses**: Validates response structure, logs error if malformed
- **Timeout**: Respects provider timeout settings

## Safety Levels

Fixes are categorized by safety:

**Safe fixes** (auto-applicable):
- Code block language detection
- List formatting fixes

**Unsafe fixes** (require confirmation):
- Bilingual title generation (modifies metadata)

## Performance Considerations

- Code block detection: ~1-2 seconds per code block
- Title generation: ~2-3 seconds per title
- Uses structured output (JSON schema) for reliability
- Batches multiple code blocks in single note
- Logging at DEBUG level to minimize overhead

## Logging

The AI fixer provides detailed logging:

```
ai_fixer_initialized: provider=Ollama, model=qwen/qwen-2.5-14b-instruct
code_language_detected: language=python, confidence=high
bilingual_title_generated: en_title=..., ru_title=...
code_language_fixed: line=42, language=python, filepath=...
```

## Troubleshooting

### AI features not working

Check:
1. `enable_ai_validation` is set to `true` in config
2. Provider is properly configured (API key, base URL)
3. Provider is accessible (`ProviderFactory.create_from_config(config).check_connection()`)
4. Model supports JSON output

### Low quality fixes

Try:
1. Use a more capable model (e.g., GPT-4, Claude)
2. Increase temperature (0.2-0.3) for more creative titles
3. Provide better context in note content

### Performance issues

Consider:
1. Use faster models (e.g., Qwen 2.5 7B/14B)
2. Reduce timeout values
3. Disable AI validation for bulk operations
4. Use local providers (Ollama, LM Studio)

## Best Practices

1. **Enable selectively**: Use AI validation for high-value notes
2. **Review unsafe fixes**: Always review title generation before applying
3. **Monitor costs**: Track API usage for cloud providers
4. **Use appropriate models**: Balance quality vs. speed/cost
5. **Fallback gracefully**: System works without AI if provider unavailable

## Examples

See `/Users/npochaev/GitHub/obsidian-to-anki/examples/ai_fixer_example.py` for complete working examples.

## Future Enhancements

Potential improvements:

- Batch processing for multiple notes
- Custom prompt templates
- Additional AI-powered validators
- Translation quality validation
- Code snippet explanation generation
- Question quality scoring
