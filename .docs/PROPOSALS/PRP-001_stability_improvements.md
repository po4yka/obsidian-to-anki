# Proposal Template

## Metadata
- **ID:** `PRP-001`
- **Title:** Stabilize Bilingual Content Flow & Post-Validation
- **Author:** GPT-5.1 Codex
- **Date:** 2025-11-23
- **Status:** Draft

## Problem Statement
- **Context:** Latest applied run (`test-run --count 5 --no-dry-run --index`) spent 12 minutes and ~$0.03 on Grok calls yet produced zero cards. Root causes:
  1. **Content gaps:** sampled notes set `language_tags: [en, ru]` but only contain Russian answers. Pre-validation repeatedly failed, wasting extraction time.
  2. **Formatting defects:** malformed code fences (` ```kotli`) and truncated Kotlin blocks prevent validation.
  3. **Post-validation defects:** generated slugs such as `system-design-microservices-vs-monolith-architectur-...` violate naming rules, and post-validator cannot inspect APF HTML because the agent response omits it.
  4. **Auto-fix instability:** retries crashed with `">" not supported between instances of 'NoneType' and 'float'`, leaving the pipeline stuck.
- **Impact:** High latency, high LLM cost, and no card creation. Engineers lack early signals to repair notes before expensive Grok calls. Post-validation cannot certify even good generations.
- **Goals:**
  - Enforce bilingual completeness and Markdown integrity before Grok extraction.
  - Reduce retries and surface actionable errors to authors.
  - Ensure post-validator receives full HTML plus valid slugs so it can pass notes that conform.
  - Harden auto-fix logic so a single failure does not abort the pipeline.
- **Non-Goals:**
  - Rewriting agent orchestration architecture.
  - Changing APF template semantics.

## Success Metrics
- ≥90% of sampled notes either pass pre-validation on first attempt or fail fast before triggering card generation.
- At least one bilingual note proceeds through post-validation without manual intervention during regression runs.
- No occurrences of `NoneType` comparison errors in auto-fix during smoke tests.
- Mean `test-run --count 5` duration reduced by 30% compared to current ~12 minutes (measured after note repairs).

## Proposed Solution
1. **Pre-flight bilingual validator:**
   - Extend `parse_note` or a new `note_health_check.py` to inspect markdown before scheduling QA extraction.
   - Rules:
     - For each language in `language_tags`, require matching `## Answer (<LANG>)` section with non-empty content.
     - Validate code fences (matching `` ``` ``) using simple stack parser.
   - Failed notes are logged and excluded from sampling, emitting actionable diagnostics.
2. **Author feedback tooling:**
   - Add CLI command `obsidian-anki-sync lint-note <path>` to run the validator locally and show bilingual/markdown issues.
3. **Slug integrity fix:**
   - Update `generate_slug()` (and agent slug builder) to preserve full topic names (e.g., `architecture`). Add unit tests to prevent silent truncation.
4. **Post-validator payload improvements:**
   - Ensure generator attaches full APF HTML (front/back/extra) to the structure passed into post-validator.
   - Modify post-validator schema to include slug metadata for verification.
5. **Auto-fix hardening:**
   - Guard comparisons against `None` latencies; when missing, default to 0 or skip ordering.
   - Wrap auto-fix steps in per-card try/except so one failure does not abort the note.

## Risks & Mitigations
- **Risk:** Rejecting too many notes may slow content throughput.
  **Mitigation:** Provide override flag (`--allow-partial-language`) for emergency runs.
- **Risk:** Added validation increases CLI latency.
  **Mitigation:** Validator operates on plain text before LLM calls; complexity is O(n) per note.
- **Risk:** Changes to slug logic might break existing Anki references.
  **Mitigation:** Keep deterministic hash suffix; only fix truncation bug and add migration to map old slugs when encountered.
- **Risk:** Post-validator schema changes may be incompatible with cached cards.
  **Mitigation:** Version the schema and support both old/new formats during rollout.

## Rollout Plan
1. **Week 1:** Implement bilingual validator + CLI lint command. Ship behind `--enforce-bilingual` flag, default on for test-run.
2. **Week 2:** Patch slug generation and add regression tests.
3. **Week 3:** Update agent generator to include HTML payload; adjust post-validator schema.
4. **Week 4:** Harden auto-fix, add telemetry, and run end-to-end dry-run + applied tests.

## Open Questions
- Should bilingual enforcement support asymmetric languages (e.g., RU-only cards) by automatically downgrading `language_tags`?
- Do we need a migration job to rename existing slugs in Anki if truncation is fixed?

