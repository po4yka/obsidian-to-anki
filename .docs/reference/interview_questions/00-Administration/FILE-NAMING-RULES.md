# File Naming Rules

**Date Established**: 2025-10-03
**Last Updated**: 2025-10-05

---

## Core Rule: English-Only Filenames

**ALL note files MUST be named in English**, regardless of the content language.

### ✅ Correct Examples

```
q-what-is-viewmodel--android--medium.md
q-how-to-create-coroutine--programming-languages--easy.md
q-sealed-vs-enum-classes--programming-languages--medium.md
```

### ❌ Incorrect Examples

```
q-что-такое-viewmodel--android--medium.md              # Russian characters
q-как-создать-корутину--programming-languages--easy.md  # Russian characters
q-запечатанные-vs-enum--programming-languages--medium.md # Russian characters
```

---

## File Naming Pattern

### Question Notes
```
q-[english-slug]--[topic]--[difficulty].md
```

**Components:**
- `q-` prefix for questions
- `[english-slug]`: Short, descriptive, lowercase, hyphen-separated English phrase
- `--[topic]--`: Topic category (android, kotlin, algorithms, system-design, cs, etc.)
- `--[difficulty]`: Difficulty level (easy, medium, hard)

**Examples:**
```
q-recyclerview-vs-listview--android--easy.md
q-kotlin-coroutine-context--programming-languages--medium.md
q-sql-join-types--backend--medium.md
q-git-merge-vs-rebase--tools--hard.md
```

### Concept Notes
```
c-[english-concept-name].md
```

**Examples:**
```
c-jetpack-compose.md
c-kotlin-coroutines.md
c-mvvm-architecture.md
```

### MOC (Map of Content) Notes
```
moc-[english-topic-name].md
```

**Examples:**
```
moc-android.md
moc-kotlin.md
moc-backend.md
```

---

## Slug Creation Guidelines

### 1. Use English Words
- Translate or transliterate Russian to English
- Use common English terms from documentation

**Examples:**
```
# Russian: "Что такое ViewModel"
# English slug: what-is-viewmodel

# Russian: "Как создать корутину"
# English slug: how-to-create-coroutine

# Russian: "Для чего нужны фрагменты"
# English slug: why-are-fragments-needed
```

### 2. Keep It Concise
- 3-8 words maximum
- Remove articles (a, an, the) when possible
- Use abbreviations for well-known terms

**Examples:**
```
q-what-is-mvvm--android--easy.md                    # Good
q-what-is-the-mvvm-architecture-pattern--android--easy.md  # Too long

q-recyclerview-optimization--android--medium.md     # Good
q-how-to-optimize-recyclerview-performance--android--medium.md # Too long
```

### 3. Lowercase with Hyphens
- All lowercase letters
- Use hyphens (-) to separate words
- No spaces, underscores, or special characters

**Examples:**
```
✅ q-viewmodel-vs-savedstate--android--medium.md
❌ q-ViewModel-vs-SavedState--android--medium.md    # Uppercase
❌ q-viewmodel_vs_savedstate--android--medium.md    # Underscores
❌ q-viewmodel vs savedstate--android--medium.md    # Spaces
```

### 4. Be Descriptive
- Slug should hint at the question content
- Use specific terms, not generic ones

**Examples:**
```
✅ q-kotlin-null-safety-operators--programming-languages--easy.md
❌ q-kotlin-operators--programming-languages--easy.md  # Too generic

✅ q-recyclerview-diffutil-usage--android--medium.md
❌ q-list-optimization--android--medium.md  # Too vague
```

---

## Language Handling

### Content vs Filename

| Aspect | Language | Rule |
|--------|----------|------|
| **Filename** | English ONLY | Always use English slugs |
| **YAML title** | Bilingual | Include both EN and RU |
| **Content** | Bilingual | Both EN and RU sections |
| **Tags** | English | Use English keywords |

### Example Note Structure

**Filename:** `q-what-is-viewmodel--android--medium.md`

```yaml
---
id: 20251003141234
title: What is ViewModel / Что такое ViewModel
aliases: []

# Classification
topic: android
subtopics: [android, architecture, viewmodel]
question_kind: theory
difficulty: medium

# Language & provenance
original_language: ru
language_tags: [en, ru]
source: https://t.me/easy_kotlin/123
source_note: easy_kotlin Telegram channel
---

# Question (EN)
> What is ViewModel and why is it used in Android?

# Вопрос (RU)
> Что такое ViewModel и для чего используется в Android?

---

## Answer (EN)
[English answer content...]

## Ответ (RU)
[Russian answer content...]
```

---

## Vault Statistics

**As of 2025-10-06**: 840+ notes total
- **40-Android**: ~280+ notes (includes moved Android questions from CompSci)
- **70-Kotlin**: ~70+ notes (includes moved Kotlin questions from CompSci)
- **60-CompSci**: ~90+ notes (design patterns, architecture patterns, OOP)
- **50-Backend**: 4 notes
- **80-Tools**: 3+ notes (includes Git questions)
- **20-Algorithms**: 1 note

All files follow the English-only naming convention.

**Note**: Exact counts may vary as questions are reorganized from `60-CompSci` into appropriate topic folders (`70-Kotlin` and `40-Android`).

---

## Implementation Checklist

When creating new notes, ensure:

- [ ] Filename uses English characters only (a-z, 0-9, hyphens)
- [ ] Filename follows pattern: `q-[slug]--[topic]--[difficulty].md`
- [ ] Slug is 3-8 words, lowercase, hyphen-separated
- [ ] Slug is descriptive and specific
- [ ] Title in frontmatter includes both EN and RU
- [ ] Content includes both English and Russian sections
- [ ] No Russian characters in filename

---

## Tools for Validation

### Check for Russian Filenames
```bash
# Find any files with Cyrillic characters
find . -name "*.md" -type f | grep -E '[а-яА-ЯёЁ]'

# Should return nothing if all files are properly named
```

### Rename Script Template
```bash
# For future use if Russian filenames are accidentally created
# Extract English title from frontmatter and rename
grep "title:" file.md | sed -e 's/^title: //' -e 's/ \/ .*//'
```

---

## Why English-Only Filenames?

1. **Cross-platform compatibility**: Some systems have issues with Unicode filenames
2. **Git compatibility**: Better handling in version control
3. **URL safety**: Can be used in web URLs without encoding
4. **Search efficiency**: Easier to search and reference
5. **Tool compatibility**: Works with all development tools and scripts
6. **Consistency**: Maintains uniform naming across the knowledge base
7. **Accessibility**: Easier for international collaboration

---

**This rule is mandatory for all new notes and must be enforced in import scripts.**
