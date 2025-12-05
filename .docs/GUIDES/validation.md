# AI-Powered Validation Guide

AI validation provides intelligent fixes for Q&A notes using LLM providers.

## Features

-   **Code block language detection** - Identifies programming languages
-   **Bilingual title generation** - Creates English/Russian title pairs
-   **List formatting fixes** - Normalizes markdown list formatting

## Configuration

```yaml
enable_ai_validation: true
ai_validation_model: "qwen/qwen-2.5-32b-instruct"
ai_validation_temperature: 0.1

ai_validation:
    enable_code_language_detection: true
    enable_bilingual_titles: true
    enable_list_formatting: true
    min_confidence_code_detection: "medium"
```

Or via environment:

```bash
export ENABLE_AI_VALIDATION=true
export AI_VALIDATION_MODEL="qwen/qwen-2.5-32b-instruct"
```

## Usage

```python
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.validation import AIFixer, AIFixerValidator

config = Config(enable_ai_validation=True)
ai_fixer = AIFixer.from_config(config)

validator = AIFixerValidator(
    content=note_content,
    frontmatter=frontmatter,
    filepath=note_path,
    ai_fixer=ai_fixer,
    enable_ai_fixes=True
)

issues = validator.validate()

# Apply safe fixes
for fix in validator.get_safe_fixes():
    new_content, new_frontmatter = fix.fix_function()
```

## Safety Levels

**Safe fixes (auto-apply):**

-   Code block language detection
-   List formatting normalization

**Unsafe fixes (require confirmation):**

-   Bilingual title generation
-   Content restructuring

## Performance

| Operation        | Time           |
| ---------------- | -------------- |
| Code detection   | 1-2s per block |
| Title generation | 2-3s per title |
| List formatting  | <0.1s (regex)  |

**Tips:** Use faster models (qwen/qwen-2.5-32b-instruct), batch processing, caching.

## Error Handling

Errors are gracefully handled - AI features disable on failure, basic validation continues.

## Post-Validation Resilience

-   `post_validator_timeout_seconds` (default **45s**) caps every validator call so hung LLMs cannot block the worker queue.
-   `post_validator_retry_backoff_seconds` and `post_validator_retry_jitter_seconds` drive the exponential backoff helper in `utils/resilience.py`, spacing retries with bounded jitter to avoid thundering herds.
-   When the validator emits `corrected_cards`, the pipeline now preserves those cards even if the final validation status is `False`, ensuring highlight/fix agents can inspect the best-known HTML.

---

**Related**: [Configuration](configuration.md) | [Providers](../ARCHITECTURE/providers.md)
