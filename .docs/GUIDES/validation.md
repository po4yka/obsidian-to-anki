# AI-Powered Validation Guide

This guide covers the AI-powered validation and auto-fixing system for Q&A notes in the Obsidian to Anki sync service.

## Overview

The AI validation system provides intelligent fixes for common validation issues using LLM providers. It seamlessly integrates with the existing provider system and provides three main capabilities:

-   **Code block language detection** - Automatically identifies programming languages
-   **Bilingual title generation** - Creates English/Russian title pairs
-   **List formatting fixes** - Normalizes markdown list formatting

## Quick Start

### Enable AI Validation

Add to your `config.yaml`:

```yaml
# Enable AI-powered validation
enable_ai_validation: true

# Optional: Specify model (uses default_llm_model if not set)
ai_validation_model: "openai/gpt-4o-mini"

# Optional: Adjust temperature (default: 0.1)
ai_validation_temperature: 0.1
```

Or use environment variables:

```bash
export ENABLE_AI_VALIDATION=true
export AI_VALIDATION_MODEL="openai/gpt-4o-mini"
```

### Basic Usage

```python
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.validation import AIFixer, AIFixerValidator

# Create config with AI enabled
config = Config(enable_ai_validation=True)
ai_fixer = AIFixer.from_config(config)

# Validate a note
validator = AIFixerValidator(
    content=note_content,
    frontmatter=frontmatter,
    filepath=note_path,
    ai_fixer=ai_fixer,
    enable_ai_fixes=True
)

issues = validator.validate()

# Apply safe fixes automatically
for fix in validator.get_safe_fixes():
    new_content, new_frontmatter = fix.fix_function()
```

## Features

### 1. Code Block Language Detection

Automatically detects and adds language specifications to code blocks.

**Before:**

```markdown

```

def factorial(n):
return 1 if n <= 1 else n \* factorial(n-1)

```

```

**After:**

````markdown
```python
def factorial(n):
    return 1 if n <= 1 else n * factorial(n-1)
```
````

````

**Characteristics:**
- Analyzes code syntax and structure
- Returns confidence level (high/medium/low)
- Only applies fix if confidence is high or medium
- Safe auto-fix (no confirmation required)

### 2. Bilingual Title Generation

Generates or fixes titles in bilingual format (English / Russian).

**Before:**
```yaml
title: Recursion Example
````

**After:**

```yaml
title: Recursion in Programming / Рекурсия в программировании
```

**Characteristics:**

-   Analyzes question content for context
-   Generates concise technical titles
-   Provides accurate Russian translations
-   Maintains technical terminology consistency
-   **Requires confirmation** (not auto-applied)

### 3. List Formatting Fixes

Normalizes list formatting to single space after dash markers.

**Before:**

```markdown
-   Item 1
-   Item 2
```

**After:**

```markdown
-   Item 1
-   Item 2
```

**Characteristics:**

-   Regex-based fix (no AI required)
-   Safe auto-fix
-   Normalizes multiple spaces to single space

## Configuration Options

### Basic Configuration

```yaml
# Enable/disable AI validation
enable_ai_validation: true

# Model selection (optional, uses default_llm_model if not set)
ai_validation_model: "qwen/qwen-2.5-14b-instruct"

# Temperature for creativity vs consistency (default: 0.1)
ai_validation_temperature: 0.1
# Provider reuse (uses existing llm_provider configuration)
# No additional provider config needed
```

### Advanced Configuration

```yaml
# Validation behavior
ai_validation:
    # Enable specific fixes
    enable_code_language_detection: true
    enable_bilingual_titles: true
    enable_list_formatting: true

    # Confidence thresholds
    min_confidence_code_detection: "medium" # low|medium|high
    max_retries: 3

    # Performance tuning
    timeout: 30 # seconds
    batch_size: 5 # notes per batch
```

## Manual AI Operations

### Code Language Detection

```python
from obsidian_anki_sync.providers.factory import ProviderFactory
from obsidian_anki_sync.validation import AIFixer

# Create provider and fixer
provider = ProviderFactory.create_from_config(config)
ai_fixer = AIFixer(provider=provider, model="qwen/qwen-2.5-14b-instruct")

# Detect language
language_result = await ai_fixer.detect_code_language(code_block)
print(f"Language: {language_result.language}, Confidence: {language_result.confidence}")
```

### Bilingual Title Generation

```python
# Generate title pair
en_title, ru_title = await ai_fixer.generate_bilingual_title(
    content=note_content,
    current_title="Current Title"
)
print(f"English: {en_title}")
print(f"Russian: {ru_title}")
```

## Integration with Validation Pipeline

The AI validation system integrates with the standard validation pipeline:

```python
from obsidian_anki_sync.validation import ValidationPipeline

# Create pipeline with AI validation enabled
pipeline = ValidationPipeline(config)

# Process note (includes AI validation if enabled)
result = await pipeline.validate_note(note_content, frontmatter)

# Access AI fixes
if result.ai_fixes_available:
    safe_fixes = result.get_safe_fixes()
    unsafe_fixes = result.get_unsafe_fixes()  # Require confirmation

    # Apply safe fixes
    for fix in safe_fixes:
        note_content, frontmatter = fix.apply()
```

## Safety Levels

### Safe Fixes (Auto-applicable)

-   Code block language detection
-   List formatting normalization
-   Minor syntax corrections

### Unsafe Fixes (Require Confirmation)

-   Bilingual title generation (modifies metadata)
-   Content restructuring
-   Major formatting changes

## Performance Considerations

### Expected Performance

-   **Code block detection:** ~1-2 seconds per code block
-   **Title generation:** ~2-3 seconds per title
-   **List formatting:** < 0.1 seconds (regex-based)

### Optimization Tips

-   **Batch processing:** Process multiple notes together
-   **Confidence filtering:** Only apply high-confidence fixes
-   **Caching:** Cache results for repeated content
-   **Selective enabling:** Enable only needed AI features

## Error Handling

The AI validation system gracefully handles errors:

```python
try:
    fixes = await ai_fixer.detect_code_language(code_block)
except ProviderError:
    # Provider unavailable, skip AI validation
    logger.warning("AI provider unavailable, falling back to basic validation")
    fixes = []
except TimeoutError:
    # Request timed out
    logger.warning("AI validation timed out, skipping")
    fixes = []
except ValidationError as e:
    # Invalid response from AI
    logger.error(f"AI validation error: {e}")
    fixes = []
```

### Fallback Behavior

-   **Provider unavailable:** AI features disabled, basic validation continues
-   **API errors:** Logged and skipped, no failure
-   **Invalid responses:** Logged, fix not applied
-   **Timeout:** Request abandoned, no blocking

## Troubleshooting

### AI Features Not Working

**Check configuration:**

1. `enable_ai_validation: true` in config
2. Valid LLM provider configuration
3. Provider connectivity (`provider.validate_connection()`)

**Test provider:**

```bash
# Test Ollama
curl http://localhost:11434/api/tags

# Test OpenRouter
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/models
```

### Low Quality Fixes

**Adjust settings:**

1. Use more capable model (GPT-4, Claude-3 instead of GPT-3.5)
2. Increase temperature (0.2-0.3) for creative titles
3. Provide better context in note content

### Performance Issues

**Optimize:**

1. Use faster models (GPT-4o-mini instead of GPT-4)
2. Reduce timeout values
3. Disable AI validation for bulk operations
4. Use local providers (Ollama, LM Studio)

## Best Practices

### When to Use AI Validation

-   **High-value notes:** Important or complex content
-   **Bilingual content:** Notes requiring translation
-   **Code-heavy notes:** Multiple programming languages
-   **Quality assurance:** Final review before sync

### Configuration Recommendations

-   **Development:** Enable all features, use capable models
-   **Production:** Balance quality vs cost, enable selectively
-   **Bulk operations:** Disable AI features for speed

### Monitoring

-   Track fix acceptance rates
-   Monitor API usage and costs
-   Log failures for pattern analysis
-   Review applied fixes for quality

## Related Documentation

-   **[Configuration Guide](configuration.md)** - Complete setup instructions
-   **[Provider Guide](../ARCHITECTURE/providers.md)** - LLM provider configuration
-   **[Agent System](../ARCHITECTURE/agents.md)** - How AI validation integrates with agents

---

**Version**: 1.0
**Last Updated**: November 28, 2025
