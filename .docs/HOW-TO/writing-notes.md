# Writing Notes for Obsidian-to-Anki

How to write Obsidian markdown notes that the sync pipeline can discover,
parse, and convert into Anki flashcards.

## File naming and placement

Notes must match the `q-*.md` glob pattern (discovered via `rglob("q-*.md")`).
Directories starting with `c-`, `moc-`, or `template` are ignored entirely.
Maximum file size: 10 MB.

## Frontmatter

Every note starts with a YAML frontmatter block delimited by `---`.

### Required fields (6)

| Field            | Type         | Description                              |
|------------------|--------------|------------------------------------------|
| `id`             | string       | Unique note identifier                   |
| `title`          | string       | Note title                               |
| `topic`          | string       | Main topic                               |
| `language_tags`  | list[string] | Language codes, e.g. `["en", "ru"]`      |
| `created`        | date/datetime| Creation date                            |
| `updated`        | date/datetime| Last update date                         |

### Optional fields

| Field                | Type         | Description                                  |
|----------------------|--------------|----------------------------------------------|
| `aliases`            | list[string] | Alternative names for the note               |
| `subtopics`          | list[string] | Subtopics covered                            |
| `question_kind`      | string       | Type of question                             |
| `difficulty`         | string       | Difficulty level                             |
| `original_language`  | string       | Primary language (must be in `language_tags`) |
| `source`             | string       | Source reference                             |
| `source_note`        | string       | Source note reference                        |
| `status`             | string       | Note status                                  |
| `moc`                | string       | Map of Contents (wikilink format)            |
| `related`            | list[string] | Related notes (wikilink list)                |
| `tags`               | list[string] | Tags                                         |
| `sources`            | list[dict]   | Source dicts, each with a `url` key          |
| `anki_note_type`     | string       | Anki model name                              |
| `anki_slugs`         | list[string] | Anki slugs                                   |

### Date formats

Accepted: `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM:SS`, `YYYY-MM-DD HH:MM:SS`.

### Automatic normalization

- Wikilinks (`[[...]]`) are stripped automatically from string values.
- String lists handle comma and newline separators.
- Bullet points are removed from link lists.
- Sources given as plain strings are converted to dicts with a `url` key.

## Q&A section structure

Each language in `language_tags` requires both a question and an answer section.

| Section                  | Level | Examples                              |
|--------------------------|-------|---------------------------------------|
| Question                 | `#`   | `# Question (EN)`, `# Вопрос (RU)`   |
| Answer                   | `##`  | `## Answer (EN)`, `## Ответ (RU)`    |
| Follow-ups (optional)    | `##`  | `## Follow-ups`                       |
| References (optional)    | `##`  | `## References`                       |
| Related Questions (opt.) | `##`  | `## Related Questions`                |

Use a `---` separator between the question body and the answer heading.

## Complete working example

Save the following as `q-python-decorators.md` in your vault:

```markdown
---
id: q-python-decorators-001
title: Python @property Decorator
topic: Python
language_tags: ["en", "ru"]
created: 2025-06-15
updated: 2025-06-15
subtopics: ["decorators", "OOP"]
difficulty: intermediate
original_language: en
tags: ["python", "oop"]
sources:
  - url: https://docs.python.org/3/library/functions.html#property
moc: "[[moc-python]]"
related:
  - "[[q-python-descriptors]]"
---

# Question (EN)

What does the `@property` decorator do in Python, and when should you use it
instead of a regular attribute?

---

## Answer (EN)

The `@property` decorator converts a method into a read-only attribute.
Use it when you need computed attributes or validation logic on access.

```python
class Circle:
    def __init__(self, radius: float) -> None:
        self._radius = radius

    @property
    def area(self) -> float:
        return 3.14159 * self._radius ** 2
```

## Follow-ups

- How do you define a setter with `@property`?

# Вопрос (RU)

Что делает декоратор `@property` в Python и когда его следует использовать
вместо обычного атрибута?

---

## Ответ (RU)

Декоратор `@property` превращает метод в атрибут только для чтения.
Используйте его, когда нужны вычисляемые атрибуты или валидация доступа.
```

Verify the file is discoverable:

```bash
find /path/to/vault -name 'q-*.md' | head -5
```

## Tolerant parsing and auto-repair

Two settings (both on by default) make the parser forgiving:

- **`tolerant_parsing: true`** -- allows partial or incomplete sections.
- **`parser_repair_enabled: true`** -- uses an LLM agent (`qwen3:8b` by
  default) to fix broken note structure automatically.

Set both to `false` in your configuration to enforce strict parsing (e.g. in CI).

## Common mistakes

**Backticks in YAML frontmatter** -- use YAML quoting, not backticks:

```yaml
# Wrong                          # Correct
title: `Python Decorators`       title: "Python Decorators"
```

**Wrong heading levels** -- questions are `#` (h1), answers are `##` (h2).
Swapping them causes the parser to miss sections.

**Unbalanced code fences** -- every opening triple-backtick needs a matching
close. An unclosed fence swallows everything after it, including headings.

**Missing `---` separator** -- the horizontal rule between question body and
answer heading is required. Without it, the answer may merge into the question.

## Validate and sync

```bash
uv run python -m obsidian_anki_sync.validation.validate_notes /path/to/vault
```

See [validating-notes.md](validating-notes.md) for the full validation
workflow and [first-sync.md](first-sync.md) for running your first sync.
