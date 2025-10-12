---
id: <% tp.date.now("YYYYMMDD-HHmmss") %>
title: <%- tp.file.title %>
aliases: []

# Classification
topic: <%* /* choose one from TAXONOMY.md */ %>
subtopics: []                  # 0–3; для Android — из списка Android subtopics
question_kind: coding          # coding | theory | system-design | android
difficulty: easy               # easy | medium | hard

# Language & provenance
original_language: en          # en | ru
language_tags: [en]            # какие языки уже есть в заметке
source:                        # URL источника
source_note:                   # Описание источника

# Workflow & relations
status: draft                  # draft | reviewed | ready | retired
moc:                           # moc-<topic> (без скобок)
related: []                    # список: c-concept-name (без скобок [[]])

# Timestamps
created: <% tp.date.now("YYYY-MM-DD") %>
updated: <% tp.date.now("YYYY-MM-DD") %>

# Tags (EN only; no leading #)
tags: []
---

# Question (EN)
> ...

# Вопрос (RU)
> ...

---

## Answer (EN)
...

## Ответ (RU)
...

---

## Follow-ups
- ...

## References
- ...

## Related Questions
- ...
