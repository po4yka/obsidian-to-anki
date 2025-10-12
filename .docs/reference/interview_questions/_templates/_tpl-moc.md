---
id: ivm-<% tp.date.now("YYYYMMDD-HHmmss") %>
title: <%- tp.file.title %> — MOC
kind: moc
created: <% tp.date.now("YYYY-MM-DD") %>
updated: <% tp.date.now("YYYY-MM-DD") %>
tags: [moc]
---

# Start Here
- ...

## Auto: by difficulty (example)
```dataview
TABLE difficulty, file.link, subtopics
FROM "20-Algorithms"
WHERE difficulty = "easy"
SORT file.name ASC
```
