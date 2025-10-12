# Quick Agent Checklist

**Use this checklist before creating/modifying notes. See AGENTS.md for detailed instructions.**

---

## Before Creating a Note

- [ ] Read TAXONOMY.md for valid `topic` and `subtopics`
- [ ] Determine correct folder (matches `topic`)
- [ ] Choose correct template (_tpl-qna, _tpl-concept, _tpl-moc)
- [ ] Determine filename format (q-/c-/moc- prefix)

---

## YAML Frontmatter (Q&A)

- [ ] `id`: YYYYMMDD-HHmmss format
- [ ] `title`: "EN Title / RU Заголовок"
- [ ] `aliases`: Include both languages
- [ ] `topic`: ONE value from TAXONOMY.md
- [ ] `subtopics`: 1–3 values (Android: use Android list)
- [ ] `question_kind`: coding | theory | system-design | android
- [ ] `difficulty`: easy | medium | hard
- [ ] `original_language`: en | ru
- [ ] `language_tags`: [en] or [en, ru]
- [ ] `sources`: [{url, note}] if applicable
- [ ] `status`: **draft** (always for agent-created)
- [ ] `moc`: Link to ≥1 MOC
- [ ] `related`: Link to 2–5 concepts/questions
- [ ] `created`: YYYY-MM-DD
- [ ] `updated`: YYYY-MM-DD
- [ ] `tags`: English only, include difficulty/<level>

---

## Content Sections (Q&A)

- [ ] `# Question (EN)` - Clear English version
- [ ] `# Вопрос (RU)` - Clear Russian version
- [ ] `## Answer (EN)` - Approach, complexity, code
- [ ] `## Ответ (RU)` - Same content in Russian
- [ ] `## Follow-ups` - Variations, edge cases
- [ ] `## References` - Links to concepts, external sources
- [ ] `## Related Questions` - Links to related Q&As

---

## Tags

- [ ] All tags in **English** (no Russian)
- [ ] Include `difficulty/easy|medium|hard`
- [ ] For Android: Include `android/<subtopic>` tags
- [ ] Include source tag if applicable (leetcode, neetcode, etc.)
- [ ] Include technique tags (two-pointers, dp, etc.)
- [ ] Use namespaces: `lang/`, `platform/`, `topic/`

---

## Links

- [ ] Link to ≥1 Concept in `related` YAML field
- [ ] Link to ≥1 MOC in `moc` YAML field
- [ ] Inline concept links in answer: [[c-hash-map]]
- [ ] Related questions in content section

---

## Android-Specific (if topic=android)

- [ ] `subtopics`: 1–3 from Android subtopics list (TAXONOMY.md)
- [ ] Mirror each subtopic to tag: `android/<subtopic>`
- [ ] `question_kind`: usually `android` or `theory`
- [ ] Link to [[moc-android]]

---

## Final Validation

- [ ] File in correct folder (20-Algorithms, 40-Android, etc.)
- [ ] Filename follows convention (q-/c-/moc- + kebab-case)
- [ ] Both EN and RU sections present
- [ ] Code examples valid (if applicable)
- [ ] No Russian in tags
- [ ] `status: draft` (let human review)
- [ ] All wikilinks use correct format [[note-name]]

---

## Common Values Quick Reference

**Topics** (pick ONE):
```
algorithms, data-structures, system-design, android, kotlin,
programming-languages, architecture-patterns, concurrency,
distributed-systems, databases, networking, operating-systems,
security, performance, testing, devops-ci-cd, cloud, debugging,
ui-ux-accessibility, behavioral, cs
```

**Difficulty**: `easy` | `medium` | `hard`

**Question Kind**: `coding` | `theory` | `system-design` | `android`

**Status**: `draft` (always for agents) | reviewed | ready | retired

**Folders**:
- Algorithms → `20-Algorithms/`
- System Design → `30-System-Design/`
- Android → `40-Android/`
- Kotlin → `70-Kotlin/`
- CompSci → `60-CompSci/`
- Backend → `50-Backend/`
- Tools → `80-Tools/`
- Git → `10-Git/`
- Concepts → `10-Concepts/`
- MOCs → `90-MOCs/`

---

**When in doubt**: Check TAXONOMY.md, use `status: draft`, ask the user.
