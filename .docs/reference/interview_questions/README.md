# Obsidian Interview Vault — Full README

A complete, **bilingual (EN/RU)**, personal Obsidian vault for technical interview preparation across **Android**, **Kotlin**, **CompSci** (design patterns, architecture), **Algorithms**, **System Design**, **Backend**, and **Tools**. This README describes the **normative rules**, folder layout, metadata schema, templates, tagging and linking conventions, MOCs, Dataview queries, LLM-assisted workflows, and maintenance practices.

> **For LLM Agents**:
> - **Cursor AI**: Rules are in `../.cursorrules` (auto-loaded)
> - **General agents**: [AGENTS.md](AGENTS.md) for task instructions
> - **Quick reference**: [AGENT-CHECKLIST.md](AGENT-CHECKLIST.md) for validation
> - **Gemini CLI**: [GEMINI.md](GEMINI.md) for command-line workflows

---

## 0) Design Goals & Non‑Goals

**Goals:**
* Keep all information **in one place** per question: EN+RU in the same note.
* Make notes **queryable** via YAML for Dataview dashboards.
* Use **folders for coarse topics**; **tags + links** for facets and relationships.
* Maintain a **controlled vocabulary** for topics, subtopics, difficulty, language, and Android areas.
* Support **LLM-assisted** translation/normalization with **human review**.

**Non‑Goals:**
* Flashcard/Spaced repetition (Anki) is out of scope.
* Team collaboration features are not primary (personal vault).

---

## 1) Canonical Rules (MUST / SHOULD / MAY)

* **MUST**: One note == one Q&A item; **both languages (EN/RU)** live in the **same file**.
* **MUST**: Every note starts with **complete YAML frontmatter** (schema below).
* **MUST**: **English-only tags**; Russian appears in content and `aliases`.
* **MUST**: Choose **exactly one** `topic` (maps 1:1 to a top-level folder).
* **SHOULD**: Add **1–3 `subtopics`**; prefer namespaced tags for clarity.
* **SHOULD**: Link each Q&A to at least one **Concept** and one **MOC** hub.
* **MAY**: Use LLMs to translate/normalize/suggest links; mark `status: draft` until reviewed.

---

## 2) Folder Layout (Shallow, Topic‑first)

> Folders encode coarse topics; use YAML/tags/links for everything else.

```
📄 Homepage.md                # vault entry point
📁 _templates                 # note templates for Q&A, concepts, MOCs
📁 00-Administration          # vault documentation, README, taxonomy
📁 10-Concepts                # theory/glossary/definitions referenced across notes
📁 20-Algorithms              # coding problems incl. LeetCode-style
📁 30-System-Design           # large-scale design, components, trade-offs
📁 40-Android                 # platform APIs, lifecycle, Compose, perf, tooling
📁 50-Backend                 # backend development, APIs, databases
📁 50-Behavioral              # optional, non-technical interview topics
📁 60-CompSci                 # CS fundamentals, design patterns, architecture patterns
📁 70-Kotlin                  # Kotlin language: coroutines, syntax, idioms, Flow
📁 80-Tools                   # development tools: Git, build systems, CI/CD, IDEs
📁 90-MOCs                    # Maps of Content (hub/overview notes)
```

**Rules:**
* A file **belongs to exactly one** top-level topic folder.
* Use **10-Concepts** for reusable theory; **90-MOCs** for hub pages per topic.
* Folders use numeric prefixes (00, 10, 20, etc.) for consistent sorting.

---

## 3) Topic Taxonomy (for `topic:`)

Use **one** of the following canonical values (lower kebab-case):

* `algorithms` — problem solving, techniques (two-pointers, DP, greedy), complexity.
* `data-structures` — arrays, lists, trees, heaps, hash maps, graphs.
* `system-design` — large-scale design, scalability, availability, consistency.
* `android` — platform, lifecycle, Jetpack, Compose, performance, tooling.
* `kotlin` — Kotlin language: coroutines, flow, syntax, idioms, stdlib.
* `programming-languages` — other languages (Java, etc.), language comparisons.
* `architecture-patterns` — MVVM/MVI/Clean, SOLID, modularization.
* `design-patterns` — creational, structural, behavioral patterns (Singleton, Factory, Observer, etc.).
* `concurrency` — threads, coroutines, synchronization, actors.
* `distributed-systems` — consensus, partitioning, replication, queues.
* `databases` — SQL/NoSQL, indexing, transactions, query plans, ORMs.
* `networking` — TCP/UDP/HTTP, REST, gRPC, caching, CDNs.
* `operating-systems` — processes, scheduling, memory, filesystems.
* `security` — authN/authZ, crypto basics, OWASP, mobile app security.
* `performance` — profiling, memory/CPU, startup, rendering.
* `testing` — unit/integration/UI, strategy, TDD, coverage.
* `devops-ci-cd` — build systems, pipelines, artifacts, release engineering.
* `cloud` — AWS/GCP/Azure, containers, serverless, infra basics.
* `debugging` — troubleshooting, logs, tracing, crash analysis.
* `ui-ux-accessibility` — UI principles, accessibility, navigation patterns.
* `behavioral` — collaboration, leadership, estimation, culture.
* `tools` — Git, build systems, CI/CD, IDEs, development tools.
* `cs` — catch-all CS fundamentals **only if nothing else fits**.

> Prefer the **most specific** topic (`networking` over `cs`). Expand detail with `subtopics` and tags.

---

## 4) Android Subtopics (for `subtopics:` when `topic: android`)

Pick **1–3** values. Mirror each into a tag `android/<subtopic>`.

**UI & UX:**
* `ui-compose`, `ui-views`, `ui-navigation`, `ui-state`, `ui-animation`, `ui-theming`, `ui-accessibility`, `ui-graphics`, `ui-widgets`

**Architecture & Modularity:**
* `architecture-mvvm`, `architecture-mvi`, `architecture-clean`, `architecture-modularization`, `di-hilt`, `di-koin`, `feature-flags-remote-config`

**Lifecycle & Components:**
* `lifecycle`, `activity`, `fragment`, `service`, `broadcast-receiver`, `content-provider`, `app-startup`, `processes`

**Concurrency & Reactivity:**
* `coroutines`, `flow`, `threads-sync`, `background-execution`

**Data & Storage:**
* `room`, `sqldelight`, `datastore`, `files-media`, `serialization`, `cache-offline`

**Networking & APIs:**
* `networking-http`, `websockets`, `grpc`, `graphql`, `connectivity-caching`

**Performance & Reliability:**
* `performance-startup`, `performance-rendering`, `performance-memory`, `performance-battery`, `strictmode-anr`, `profiling`

**Testing & QA:**
* `testing-unit`, `testing-instrumented`, `testing-ui`, `testing-screenshot`, `testing-benchmark`, `testing-mocks`

**Build, Tooling & CI/CD:**
* `gradle`, `build-variants`, `dependency-management`, `static-analysis`, `ci-cd`, `versioning`

**Distribution & Play:**
* `app-bundle`, `play-console`, `in-app-updates`, `in-app-review`, `billing`, `instant-apps`

**Security & Privacy:**
* `permissions`, `keystore-crypto`, `obfuscation`, `network-security-config`, `privacy-sdks`

**Platform APIs & Hardware:**
* `camera`, `media`, `location`, `bluetooth`, `nfc`, `sensors`, `notifications`, `intents-deeplinks`, `shortcuts-widgets`

**I18n & A11y:**
* `i18n-l10n`, `a11y`

**Kotlin Multiplatform:**
* `kmp`, `compose-multiplatform`

**Form Factors:**
* `wear`, `tv`, `auto`, `foldables-chromeos`

**Analytics & Observability:**
* `analytics`, `logging-tracing`, `crash-reporting`, `monitoring-slo`

**Monetization & Growth:**
* `ads`, `engagement-retention`, `ab-testing`

---

## 5) File Naming

* **Q&A notes**: `q-<slug>--<topic>--<difficulty>.md` → `q-two-sum--algorithms--easy.md`
* **Concepts**: `c-<slug>.md` → `c-hash-map.md`
* **MOCs**: `moc-<topic>.md` → `moc-android.md`

Rules: English, **kebab-case**, short, stable. Use `aliases` for RU/EN titles.

---

## 6) YAML Schemas

### 6.1 Q&A Note (Required fields)

```yaml
---
# Identity
id: iv-2025-0001
title: Two Sum / Два слагаемых
aliases: [Two Sum, Два слагаемых]

# Classification
topic: algorithms                  # one canonical topic
subtopics: [arrays, hash-map]      # 0–3; android-specific when topic=android
question_kind: coding              # coding | theory | system-design | android
difficulty: easy                   # easy | medium | hard

# Language & provenance
original_language: en              # en | ru
language_tags: [en, ru]            # which languages are present here
source: https://leetcode.com/problems/two-sum/
source_note: LeetCode original problem

# Workflow & relations
status: draft                      # draft | reviewed | ready | retired
moc: moc-algorithms                # without brackets
related:                           # list without brackets
  - c-hash-map
  - c-array

# Timestamps (ISO8601)
created: 2025-10-03
updated: 2025-10-03

# Tags (English only; no leading # in YAML)
tags: [leetcode, arrays, hash-map, difficulty/easy]
---
```

### 6.2 Concept Note

```yaml
---
id: ivc-2025-0001
title: Hash Map / Хеш-таблица
aliases: [Hash Map, Hash Table, Хеш-таблица]
kind: concept
summary: Constant-time average lookups via hashing; collisions via chaining/open addressing.
links:
  - url: https://en.wikipedia.org/wiki/Hash_table
  - url: https://algs4.cs.princeton.edu/34hash/
created: 2025-10-03
updated: 2025-10-03
tags: [concept, data-structures, hashing]
---
```

### 6.3 MOC Note

```yaml
---
id: ivm-2025-0001
title: Algorithms — MOC
kind: moc
created: 2025-10-03
updated: 2025-10-03
tags: [moc, topic/algorithms]
---
```

---

## 7) Note Body Templates

### 7.1 Q&A (Bilingual in one note)

```markdown
# Question (EN)
> Clear, concise English version of the prompt.

# Вопрос (RU)
> Точная русская формулировка задачи.

---

## Answer (EN)
Explain approach, complexity, trade-offs, pitfalls. Include code when relevant.

## Ответ (RU)
Подробное объяснение на русском. Код при необходимости.

---

## Follow-ups
- Variation A ...
- Edge cases ...

## References
- [[c-hash-map]]
- External links (also listed in YAML `sources`).

## Related Questions
- [[q-three-sum--algorithms--medium]]
```

### 7.2 Concept

```markdown
# Summary (EN)
Short definition, key properties, diagrams.

# Сводка (RU)
Краткое определение, ключевые свойства, схемы.

## Use Cases / Trade-offs
- …

## References
- …
```

### 7.3 MOC (Hub)

````markdown
# Start Here
- [[c-complexity-analysis]]
- [[c-hash-map]]

## Easy Starters (Dataview)
```dataview
TABLE difficulty, file.link, subtopics
FROM "20-Algorithms"
WHERE difficulty = "easy"
SORT file.name ASC
````

## By Technique

```dataview
LIST FROM "20-Algorithms"
WHERE contains(tags, "two-pointers")
```

````

---

## 8) Tagging & Namespaces
- **English-only**, short, reusable: `arrays`, `graphs`, `dp`, `kotlin`, `android/compose`.
- Prefer **namespaces** for controlled vocabularies:
  - `difficulty/easy|medium|hard`
  - `lang/en|ru|kotlin|java`
  - `platform/android|web|backend`
  - `android/<subtopic>` for Android specifics
- Do **not** duplicate the folder name as a tag unless helpful (e.g., `topic/algorithms`).

---

## 9) Linking Strategy (Backlinks, Concepts, MOCs)
- Each Q&A **SHOULD** link to at least one **Concept** and **MOC**.
- Use concept links for shared theory (e.g., `[[c-binary-tree]]`).
- For sequences/follow-ups, link forward/back and list in YAML `related`.
- Maintain a few curated **MOCs** per topic as navigational hubs.

---

## 10) Dataview Dashboards (Examples)

**All LeetCode by difficulty**
```dataview
TABLE difficulty, subtopics, status
FROM "20-Algorithms"
WHERE contains(tags, "leetcode")
SORT difficulty ASC, file.name ASC
````

**Android notes touching Compose**

```dataview
LIST file.link
FROM "40-Android"
WHERE contains(tags, "android/ui-compose")
```

**Recently Updated (30 days)**

```dataview
TABLE updated, topic, difficulty
FROM ""
WHERE updated >= date(today) - dur(30 days)
SORT updated DESC
```

---
## 11) LLM‑Assisted Workflows (Human‑Reviewed)

* **Translate sections**: Generate RU from EN (and vice versa); keep both; review.
* **Normalize YAML**: Ask the model to validate keys/values; ensure canonical topics/tags.
* **Suggest cross‑links**: Request 3–5 related concepts/questions; add to `related`.
* **Summaries**: Produce 1–2 line TL;DR for answer sections.

**Rule**: Keep `status: draft` for LLM-modified notes until reviewed; then set `reviewed`/`ready`.

---

## 12) Maintenance & Hygiene

* **Statuses**: `draft` → `reviewed` → `ready` (→ `retired` in Archive).
* **Timestamps**: Update `updated` on meaningful edits.
* **Renames**: Prefer `aliases` over filename changes; if renaming, fix backlinks.
* **Archive**: Move deprecated/duplicates to `99-Archive/` (when created) with `status: retired`.
* **Tag Health**: Periodically dedupe (`hashmap` vs `hash-map`) and standardize.

**Quality Checklist (per note)**

* YAML complete and valid; topic set and folder matches.
* 1–3 subtopics; tags include `difficulty/*` and any `android/*` derived tags.
* Linked to ≥1 Concept and ≥1 MOC; `related` populated if applicable.
* EN/RU sections accurate and equivalent; examples compile.

---

## 13) Appendix — Controlled Vocabularies

**Difficulty**: `easy | medium | hard`

**Question Kind**: `coding | theory | system-design | android`

**Languages** (tags): `lang/en | lang/ru | lang/kotlin | lang/java`

**Android Subtopics**: see §4; mirror to `android/<subtopic>` tags.

**Examples (Tags)**

```
# topic/algorithms, difficulty/medium, arrays, two-pointers
# android/ui-compose, android/lifecycle, lang/kotlin
```
