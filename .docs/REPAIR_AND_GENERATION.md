# Repair and Content Generation

## Overview

The system now supports intelligent repair and content generation for imperfect notes. When notes have missing language sections, malformed markdown, or incomplete Q&A pairs, the system can automatically fix structural issues and generate missing content using LLM.

## Features

### 1. Tolerant Parsing

By default, the parser operates in tolerant mode, allowing notes with minor issues to proceed through the pipeline. Validation errors are logged as warnings instead of blocking the sync process.

**Configuration:**
```yaml
tolerant_parsing: true  # Allow notes with minor issues to proceed
```

### 2. Parser Repair Agent

The ParserRepairAgent automatically repairs notes that fail parsing:

- **YAML frontmatter errors**: Fixes syntax issues, missing required fields
- **Malformed markdown**: Corrects header levels, fixes unbalanced code fences
- **Missing language sections**: Generates missing questions/answers by translating from existing content
- **Incomplete Q&A pairs**: Completes truncated content based on context

**Configuration:**
```yaml
parser_repair_enabled: true
parser_repair_model: "x-ai/grok-4.1-fast"
parser_repair_generate_content: true  # Enable content generation
```

### 3. Content Generation

When enabled, the system can generate missing content:

- **Translation**: Translates questions/answers from one language to another
- **Inference**: Infers missing questions from answers (or vice versa)
- **Completion**: Completes truncated sections based on context

**Configuration:**
```yaml
enable_content_generation: true  # Allow LLM to generate missing content
repair_missing_sections: true  # Generate missing language sections
```

## How It Works

### Parsing Flow

1. **Initial Parse**: System attempts to parse the note using rule-based parser
2. **Tolerant Validation**: If `tolerant_parsing=true`, validation errors are warnings, not errors
3. **Repair Attempt**: If parsing fails, ParserRepairAgent is invoked
4. **Content Generation**: If missing sections are detected and `enable_content_generation=true`, LLM generates them
5. **Re-parse**: Repaired content is parsed again
6. **Success or Failure**: If repair succeeds, processing continues; otherwise, note is marked as error

### Content Generation Process

When a note specifies multiple languages (e.g., `language_tags: [en, ru]`) but is missing content for one language:

1. **Detection**: System detects missing language sections
2. **Translation**: If content exists in one language, LLM translates to missing language
3. **Preservation**: All existing content is preserved exactly as-is
4. **Quality**: Generated content maintains technical accuracy and formatting

## Configuration Options

### Core Settings

```yaml
# Enable/disable features
enable_content_generation: true
repair_missing_sections: true
tolerant_parsing: true
parser_repair_enabled: true
parser_repair_generate_content: true

# Validation behavior
enforce_bilingual_validation: false  # Default to false - validation done by LLM repair instead
```

### Model Configuration

```yaml
# Parser Repair Agent
parser_repair_model: "x-ai/grok-4.1-fast"
parser_repair_temperature: null  # Use preset default
```

## Examples

### Example 1: Missing English Answer

**Input Note:**
```markdown
---
language_tags: [en, ru]
---

# Question (EN)
What is polymorphism?

# Вопрос (RU)
Что такое полиморфизм?

## Ответ (RU)
Полиморфизм - это способность объектов принимать различные формы.
```

**Repaired Note:**
```markdown
---
language_tags: [en, ru]
---

# Question (EN)
What is polymorphism?

# Вопрос (RU)
Что такое полиморфизм?

## Answer (EN)
Polymorphism is the ability of objects to take on multiple forms.

## Ответ (RU)
Полиморфизм - это способность объектов принимать различные формы.
```

### Example 2: Unbalanced Code Fence

**Input Note:**
```markdown
## Answer (EN)
Here's some code:
```kotlin
fun example() {
    // code here
```

**Repaired Note:**
```markdown
## Answer (EN)
Here's some code:
```kotlin
fun example() {
    // code here
```
```

### Example 3: Malformed YAML

**Input Note:**
```markdown
---
language_tags: (en, ru)  # Invalid syntax
---
```

**Repaired Note:**
```markdown
---
language_tags: [en, ru]  # Fixed syntax
---
```

## Logging

The system logs repair and generation activities:

- `parser_repair_attempt`: Repair agent invoked
- `parser_repair_applied`: Repairs successfully applied
- `parser_repair_content_generated`: Missing content generated
- `parser_repair_succeeded`: Note successfully repaired and parsed
- `parser_repair_failed`: Repair unsuccessful

## Limitations

1. **Empty Notes**: Completely empty notes cannot be repaired
2. **Corrupted Files**: Severely corrupted files may be unrepairable
3. **No Source Content**: If both languages are missing, generation may not be possible
4. **Quality**: Generated content quality depends on the LLM model used

## Best Practices

1. **Review Generated Content**: Always review LLM-generated content for accuracy
2. **Source Quality**: Better source content leads to better generated content
3. **Model Selection**: Use powerful models (e.g., Grok 4.1 Fast) for better generation quality
4. **Monitoring**: Monitor repair logs to identify patterns in note issues

## Troubleshooting

### Notes Still Failing After Repair

- Check logs for `parser_repair_failed` messages
- Verify LLM provider is accessible and configured correctly
- Ensure `parser_repair_enabled: true` in config
- Check that model specified in `parser_repair_model` is available

### Generated Content Quality Issues

- Use a more powerful model (e.g., `x-ai/grok-4.1-fast`)
- Ensure source content is complete and accurate
- Review generated content and manually correct if needed

### Performance Concerns

- Repair adds LLM calls, but only for notes that would otherwise fail
- Consider using faster models for repair if speed is critical
- Monitor repair statistics to identify frequently failing notes

