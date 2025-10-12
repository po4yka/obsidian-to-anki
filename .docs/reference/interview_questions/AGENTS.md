# LLM Agent Instructions for Interview Vault

This document provides **normative instructions** for LLM agents (Claude Code, ChatGPT, etc.) working with this Obsidian vault for technical interview preparation.

## Overview

This is a **bilingual (EN/RU)** Obsidian vault for interview prep covering:
- **Algorithms** (LeetCode-style coding problems)
- **System Design** (scalability, trade-offs, components)
- **Android** (platform, Jetpack, Compose, performance)
- **CS Fundamentals** (OS, networking, databases, concurrency)
- **Behavioral** (non-technical interviews)

**Key Principle**: Both languages live in the **same note**. Use YAML frontmatter for metadata, tags, and linking.

---

## Canonical Rules (MUST Follow)

### 1. One Note = One Q&A + Both Languages
- **MUST**: Keep EN and RU in the **same file** (never split)
- **MUST**: Use complete YAML frontmatter (see templates in `_templates/`)
- **MUST**: All tags are **English-only**
- **MUST**: Choose **exactly one** `topic` from TAXONOMY.md
- **SHOULD**: Add 1–3 `subtopics` (see TAXONOMY.md for Android-specific list)
- **SHOULD**: Link to ≥1 Concept and ≥1 MOC

### 2. Folder Structure
```
00-Administration/  # vault docs (README, TAXONOMY, this file)
10-Concepts/        # reusable theory notes (c-<slug>.md)
20-Algorithms/      # coding problems (q-<slug>--algorithms--<difficulty>.md)
30-System-Design/   # design questions (q-<slug>--system-design--<difficulty>.md)
40-Android/         # Android Q&As (q-<slug>--android--<difficulty>.md)
50-Behavioral/      # behavioral Q&As
60-CompSci/         # CS fundamentals (OS, networking, etc.)
90-MOCs/            # Maps of Content (moc-<topic>.md)
_templates/         # Templater templates
```

**Rule**: File goes into folder matching its `topic` field.

### 3. File Naming
- **Q&A**: `q-<slug>--<topic>--<difficulty>.md` (e.g., `q-two-sum--algorithms--easy.md`)
- **Concept**: `c-<slug>.md` (e.g., `c-hash-map.md`)
- **MOC**: `moc-<topic>.md` (e.g., `moc-android.md`)

Use **English**, **kebab-case**, short, stable. Add Russian/English titles to `aliases`.

### 4. Status Workflow
- New/LLM-modified notes: `status: draft`
- After human review: `status: reviewed` or `status: ready`
- Deprecated: `status: retired` (move to `99-Archive/` if created)

**IMPORTANT**: Always set `status: draft` for notes you create/modify. Let humans promote to `reviewed`/`ready`.

---

## Agent Tasks & Workflows

### Task 1: Create New Q&A Note

1. **Choose folder** based on topic (20-Algorithms, 30-System-Design, 40-Android, etc.)
2. **Use template**: Copy from `_templates/_tpl-qna.md`
3. **Fill YAML**:
   - `id`: Use format `YYYYMMDD-HHmmss` (e.g., `20251003-143022`)
   - `title`: "Question Title EN / Заголовок RU"
   - `topic`: Pick **one** from TAXONOMY.md
   - `subtopics`: Pick 1–3 (Android: use Android subtopics list)
   - `difficulty`: `easy` | `medium` | `hard`
   - `question_kind`: `coding` | `theory` | `system-design` | `android`
   - `original_language`: `en` | `ru`
   - `language_tags`: `[en, ru]` if both present
   - `status`: `draft`
   - `moc`: Link to relevant MOC (e.g., `[[moc-algorithms]]`)
   - `related`: Link 2–5 related concepts/questions
   - `tags`: Include `difficulty/<level>`, topic tags, and `android/<subtopic>` if applicable
4. **Write content**:
   - Question in EN and RU
   - Answer in EN and RU (explain approach, complexity, trade-offs)
   - Code examples (when relevant)
   - Follow-ups, references, related questions
5. **Validate** (see checklist below)

### Task 2: Translate Existing Note

1. Read the note
2. Identify missing language section (EN or RU)
3. Translate preserving:
   - Technical terms accuracy
   - Code examples (keep same)
   - Links and formatting
4. Update `language_tags` to `[en, ru]`
5. Keep `status: draft` until human review

### Task 3: Normalize/Validate Note

1. Check YAML completeness:
   - All required fields present
   - `topic` is valid (matches TAXONOMY.md)
   - `subtopics` valid (for Android: use Android subtopics list)
   - `difficulty`, `question_kind`, `original_language` use controlled values
2. Check tags:
   - English-only
   - Include `difficulty/<level>`
   - For Android: mirror subtopics as `android/<subtopic>` tags
3. Check links:
   - At least 1 concept link
   - At least 1 MOC link
   - `related` field populated
4. Check folder placement matches `topic`
5. Suggest fixes or apply them (keep `status: draft`)

### Task 4: Create Concept Note

1. **Use template**: `_templates/_tpl-concept.md`
2. **Fill YAML**:
   - `id`: `ivc-YYYYMMDD-HHmmss`
   - `title`: "Concept Name EN / Название RU"
   - `aliases`: Include both languages
   - `summary`: 1-2 sentence TL;DR
   - `tags`: `[concept, <topic-related-tags>]`
3. **Write content**:
   - Summary (EN/RU)
   - Use cases / trade-offs
   - References (links, Wikipedia, textbooks)
4. Place in `10-Concepts/`

### Task 5: Create/Update MOC

1. **Use template**: `_templates/_tpl-moc.md`
2. **Fill YAML**:
   - `id`: `ivm-YYYYMMDD-HHmmss`
   - `title`: "<Topic> — MOC"
   - `tags`: `[moc, topic/<topic-name>]`
3. **Write content**:
   - "Start Here" section with key concepts
   - Dataview queries (by difficulty, tags, subtopics)
   - Manual curated lists
4. Place in `90-MOCs/`

### Task 6: Suggest Cross-Links

1. Read the note
2. Search vault for related:
   - Concepts (similar techniques, data structures)
   - Other Q&As (variations, follow-ups)
   - Relevant MOC
3. Add to `related` field in YAML
4. Add inline links in content
5. Keep `status: draft`

---

## Controlled Vocabularies (Reference TAXONOMY.md)

### Topics (Pick ONE)
```
algorithms | data-structures | system-design | android | programming-languages |
architecture-patterns | concurrency | distributed-systems | databases | networking |
operating-systems | security | performance | testing | devops-ci-cd | cloud |
debugging | ui-ux-accessibility | behavioral | cs
```

### Difficulty
```
easy | medium | hard
```

### Question Kind
```
coding | theory | system-design | android
```

### Original Language
```
en | ru
```

### Status
```
draft | reviewed | ready | retired
```

### Android Subtopics (when topic=android)
See TAXONOMY.md for full list. Examples:
```
ui-compose | lifecycle | coroutines | room | testing-unit | gradle | performance-startup
```

**Rule**: Mirror Android subtopics into tags as `android/<subtopic>`.

---

## Tag Conventions

- **English only** (no Russian)
- **Namespaced** for clarity:
  - `difficulty/easy` | `difficulty/medium` | `difficulty/hard`
  - `lang/en` | `lang/ru` | `lang/kotlin` | `lang/java`
  - `platform/android` | `platform/web` | `platform/backend`
  - `android/<subtopic>` (e.g., `android/ui-compose`, `android/coroutines`)
  - `topic/<topic-name>` (e.g., `topic/algorithms`)
- Include source tags: `leetcode`, `neetcode`, `system-design-primer`, etc.
- Include technique tags: `two-pointers`, `dp`, `binary-search`, `graph-bfs`, etc.

---

## Quality Checklist (Before Finalizing)

- [ ] YAML frontmatter complete and valid
- [ ] `topic` matches one from TAXONOMY.md
- [ ] File is in correct folder (matches `topic`)
- [ ] 1–3 `subtopics` set (Android: from Android subtopics list)
- [ ] Tags include `difficulty/<level>` and any `android/<subtopic>` tags
- [ ] Both EN and RU sections present and equivalent
- [ ] Linked to ≥1 Concept (in YAML `related` and content)
- [ ] Linked to ≥1 MOC (in YAML `moc`)
- [ ] Code examples compile/run (if applicable)
- [ ] `status: draft` set (let human promote to `reviewed`/`ready`)
- [ ] Timestamps `created` and `updated` set (YYYY-MM-DD format)

---

## Dataview Query Examples

Use these in MOCs or for ad-hoc queries:

**All LeetCode problems by difficulty:**
```dataview
TABLE difficulty, subtopics, status
FROM "20-Algorithms"
WHERE contains(tags, "leetcode")
SORT difficulty ASC, file.name ASC
```

**Android Compose notes:**
```dataview
LIST file.link
FROM "40-Android"
WHERE contains(tags, "android/ui-compose")
```

**Recently updated (30 days):**
```dataview
TABLE updated, topic, difficulty
FROM ""
WHERE updated >= date(today) - dur(30 days)
SORT updated DESC
```

**Draft notes needing review:**
```dataview
LIST file.link
FROM ""
WHERE status = "draft"
SORT updated DESC
```

---

## Common Mistakes to Avoid

1. Splitting EN/RU into separate notes → Keep both in one note
2. Using Russian in tags → English-only tags, RU in `aliases` and content
3. Multiple `topic` values → Exactly one topic
4. Forgetting `android/<subtopic>` tags → Mirror Android subtopics to tags
5. Setting `status: ready` → Use `status: draft` (let human review)
6. No MOC link → Always link to relevant MOC
7. File in wrong folder → Folder must match `topic` field
8. Invalid topic name → Use exact values from TAXONOMY.md

---

## When Uncertain

1. **Check TAXONOMY.md** for valid topic/subtopic values
2. **Check templates** in `_templates/` for correct YAML structure
3. **Check README.md** (00-Administration/) for detailed schema/rules
4. **Set `status: draft`** and let human decide
5. **Ask the user** if ambiguous (topic, difficulty, placement)

---

## Example: Creating a Q&A Note

**User request**: "Add a note about Two Sum problem from LeetCode"

**Agent actions**:
1. Determine: `topic: algorithms`, `difficulty: easy`, `question_kind: coding`
2. Filename: `q-two-sum--algorithms--easy.md`
3. Folder: `20-Algorithms/`
4. Use `_templates/_tpl-qna.md` template
5. Fill YAML:
   ```yaml
   id: 20251003-143500
   title: Two Sum / Два слагаемых
   aliases: [Two Sum, Два слагаемых]
   topic: algorithms
   subtopics: [arrays, hash-map]
   question_kind: coding
   difficulty: easy
   original_language: en
   language_tags: [en, ru]
   source: https://leetcode.com/problems/two-sum/
   source_note: LeetCode original problem
   status: draft
   moc: moc-algorithms
   related:
     - c-hash-map
     - c-array
     - q-three-sum--algorithms--medium
   created: 2025-10-03
   updated: 2025-10-03
   tags: [leetcode, arrays, hash-map, difficulty/easy]
   ```
6. Write question in EN and RU
7. Write answer in EN and RU (approach, complexity, code)
8. Add follow-ups, references
9. Validate using checklist
10. Confirm with user: "Created q-two-sum--algorithms--easy.md in 20-Algorithms/ with status: draft. Ready for review."

---

## File References

- **Full schema & rules**: `00-Administration/README.md`
- **Controlled vocabularies**: `00-Administration/TAXONOMY.md`
- **Templates**:
  - `_templates/_tpl-qna.md` (Q&A notes)
  - `_templates/_tpl-concept.md` (Concept notes)
  - `_templates/_tpl-moc.md` (MOC notes)
- **Agent instructions**: `00-Administration/AGENTS.md` (this file)

---

**Summary**: Always use controlled vocabularies, validate YAML, keep both languages in one note, set `status: draft`, and link to concepts + MOCs. When in doubt, ask the user or check TAXONOMY.md.
