# Templates Usage Guide

**Last Updated**: 2025-10-06

---

## Overview

The `_templates` directory contains Templater-compatible templates for creating new content in the Interview Questions vault. These templates work with the Obsidian Templater plugin.

---

## Available Templates

### 1. `_tpl-qna.md` - Question & Answer Template

**Use for**: Creating bilingual interview questions with answers.

**File naming convention**:
```
q-<slug>--<topic>--<difficulty>.md
```

**Examples**:
- `q-kotlin-coroutines-basics--kotlin--medium.md`
- `q-mvvm-pattern--android--medium.md`
- `q-binary-search--algorithms--easy.md`

**Sections included**:
- YAML frontmatter with full metadata
- Question (EN) and Вопрос (RU)
- Answer (EN) and Ответ (RU)
- Follow-ups
- References
- Related Questions

**Required heading format**:
```markdown
## Question (EN)
> Your question here

## Вопрос (RU)
> Ваш вопрос здесь

---

## Answer (EN)
Your answer here

## Ответ (RU)
Ваш ответ здесь
```

**Important**: All question and answer headings must use `##` (h2 level).

---

### 2. `_tpl-concept.md` - Concept Note Template

**Use for**: Creating reusable concept explanations referenced by multiple questions.

**File naming convention**:
```
c-<concept-name>.md
```

**Examples**:
- `c-dependency-injection.md`
- `c-reactive-programming.md`

**Sections included**:
- Summary (EN) and Сводка (RU)
- Use Cases / Trade-offs
- References

---

### 3. `_tpl-moc.md` - Map of Content Template

**Use for**: Creating topic overview pages with Dataview queries.

**File naming convention**:
```
moc-<topic-name>.md
```

**Examples**:
- `moc-kotlin.md`
- `moc-android.md`

**Sections included**:
- Start Here section
- Example Dataview queries for filtering by difficulty

---

## How to Use Templates

### With Templater Plugin (Recommended)

1. **Install Templater** in Obsidian (if not already installed)

2. **Configure Templater**:
   - Go to Settings → Templater
   - Set "Template folder location" to: `_templates`
   - Enable "Trigger Templater on new file creation"

3. **Create new file**:
   - Create a new note in the appropriate folder (e.g., `70-Kotlin/`)
   - Name it following the convention (e.g., `q-kotlin-flow-operators--kotlin--medium.md`)
   - Run Templater command: `Templater: Insert Template`
   - Select `_tpl-qna.md`

4. **Fill in template**:
   - Replace placeholder values (marked with `<%...%>` or `...`)
   - Update topic, subtopics, difficulty
   - Write question and answer in both languages

### Manual Copy-Paste Method

If you don't use Templater:

1. **Copy template content**:
   - Open the appropriate template file
   - Copy all content

2. **Create new file**:
   - Create a new file in the target folder
   - Use proper naming convention

3. **Paste and update**:
   - Paste template content
   - Manually replace all placeholders:
     - `<% tp.date.now("YYYYMMDD-HHmmss") %>` → Current timestamp (e.g., `20251006-143022`)
     - `<% tp.date.now("YYYY-MM-DD") %>` → Current date (e.g., `2025-10-06`)
     - `<%- tp.file.title %>` → Your file title
     - `...` → Your content

4. **Update metadata**:
   - Set `topic` from TAXONOMY.md
   - Choose 0-3 `subtopics`
   - Set `difficulty`: `easy`, `medium`, or `hard`
   - Set `status`: `draft`, `reviewed`, `ready`, or `retired`
   - Add appropriate tags

---

## YAML Frontmatter Fields

### Required Fields (Q&A Template)

```yaml
id: 20251006-143022            # Unique timestamp ID
title: "Question Title"        # Bilingual or EN title
topic: kotlin                  # From TAXONOMY.md
difficulty: medium             # easy | medium | hard
language_tags: [en, ru]        # Languages present
original_language: en          # Which was written first
status: draft                  # draft | reviewed | ready | retired
tags: [kotlin, coroutines]     # Related tags
```

### Optional Fields

```yaml
subtopics: [coroutines, flow]  # 0-3 subtopics
question_kind: theory          # coding | theory | system-design
source: https://...            # Original source URL
source_note: "Description"     # Source description
moc: moc-kotlin               # Related MOC (without brackets)
related: [c-reactive]          # Related concepts
created: 2025-10-06           # Creation date
updated: 2025-10-06           # Last update date
```

---

## Topic Selection

Choose topic from TAXONOMY.md:

**Common topics**:
- `android` - Android-specific questions
- `kotlin` - Kotlin language questions
- `algorithms` - Algorithm questions
- `data-structures` - Data structure questions
- `architecture-patterns` - Design patterns (MVVM, MVP, etc.)
- `system-design` - System design questions
- `git` - Git version control

**See**: `/00-Administration/TAXONOMY.md` for complete list

---

## Subtopics (Android-specific)

For Android questions, choose 1-3 subtopics from TAXONOMY.md:

**Examples**:
- `coroutines`, `flow` - For coroutine-related questions
- `ui-compose` - For Jetpack Compose UI
- `architecture-mvvm` - For MVVM pattern
- `testing-unit` - For unit testing
- `room` - For Room database

**See**: `/00-Administration/TAXONOMY.md` for complete Android subtopics list

---

## Best Practices

### 1. File Naming
- **Always** use kebab-case
- **Always** include topic and difficulty
- **Be specific** in the slug (e.g., `kotlin-flow-operators` not just `flow`)

### 2. Content Structure
- **Write in both languages** (EN and RU)
- **Use code examples** for coding questions
- **Include comparison tables** when comparing concepts
- **Add references** to official documentation

### 3. Heading Levels
- **Questions**: Use `## Question (EN)` and `## Вопрос (RU)`
- **Answers**: Use `## Answer (EN)` and `## Ответ (RU)`
- **Subsections**: Use `###` for subsections within answers

### 4. Tags
- **Topic tags**: Main topic (e.g., `kotlin`, `android`)
- **Subtopic tags**: Specific areas (e.g., `coroutines`, `flow`)
- **Difficulty tag**: Always include (e.g., `difficulty/medium`)
- **Keep consistent**: Check existing files for tag patterns

### 5. Cross-References
- **Link related questions**: Use `[[q-...]]` syntax
- **Link concepts**: Use `[[c-...]]` syntax
- **Link to MOCs**: Use `[[moc-...]]` syntax

---

## Examples

### Example 1: Creating a Kotlin Question

**File name**: `70-Kotlin/q-kotlin-scope-functions--kotlin--medium.md`

**Frontmatter**:
```yaml
---
id: 20251006-143022
title: Kotlin Scope Functions / Функции Области Видимости Kotlin
topic: kotlin
subtopics:
  - scope-functions
  - lambda
difficulty: medium
language_tags: [en, ru]
original_language: en
status: draft
source: https://kotlinlang.org/docs/scope-functions.html
tags:
  - kotlin
  - scope-functions
  - lambda
  - difficulty/medium
---
```

**Content**:
```markdown
## Question (EN)
> What are Kotlin scope functions and when should you use each one?

## Вопрос (RU)
> Что такое функции области видимости в Kotlin и когда следует использовать каждую?

---

## Answer (EN)

Kotlin provides five scope functions: `let`, `run`, `with`, `apply`, and `also`...

## Ответ (RU)

Kotlin предоставляет пять функций области видимости: `let`, `run`, `with`, `apply` и `also`...

---

## References
- [Kotlin Scope Functions](https://kotlinlang.org/docs/scope-functions.html)

## Related Questions
- [[q-kotlin-lambda-expressions--kotlin--medium]]
```

### Example 2: Creating an Android Question

**File name**: `40-Android/q-compose-remember--android--medium.md`

**Frontmatter**:
```yaml
---
id: 20251006-144530
title: Remember in Jetpack Compose / Remember в Jetpack Compose
topic: android
subtopics:
  - ui-compose
  - ui-state
difficulty: medium
language_tags: [en, ru]
original_language: en
status: draft
tags:
  - android
  - compose
  - state
  - remember
  - difficulty/medium
---
```

---

## Validation

Before committing a new file, verify:

1. **File naming** follows convention
2. **YAML frontmatter** is complete and valid
3. **Both languages** (EN and RU) are present
4. **Heading levels** use `##` for questions and answers
5. **Code examples** are properly formatted
6. **Tags** include topic and `difficulty/X`
7. **No emoji** in content (unless explicitly requested)

---

## Common Issues

### Issue 1: Templater Variables Not Replaced

**Problem**: Template shows `<% tp.date.now("YYYY-MM-DD") %>`

**Solution**:
- Either use Templater plugin and run "Insert Template" command
- Or manually replace with current date/time

### Issue 2: Wrong Heading Levels

**Problem**: Questions use `#` instead of `##`

**Solution**: Always use `##` for all question and answer headings

### Issue 3: Missing Language Sections

**Problem**: Only English or only Russian content

**Solution**: All Q&A files must be bilingual with both EN and RU sections

### Issue 4: Wrong File Location

**Problem**: File in wrong folder

**Solution**:
- Kotlin questions → `70-Kotlin/`
- Android questions → `40-Android/`
- Algorithm questions → `20-Algorithms/`
- Git questions → `10-Git/`
- Backend questions → `50-Backend/`
- Tools questions → `80-Tools/`
- CS theory → `60-CompSci/`

---

## Additional Resources

- **Taxonomy**: `/00-Administration/TAXONOMY.md` - Complete list of topics and subtopics
- **File Naming**: `/00-Administration/FILE-NAMING-RULES.md` - Detailed naming rules
- **Existing Files**: Check similar files in target folder for examples

---

**For questions or issues, refer to existing files as examples.**
