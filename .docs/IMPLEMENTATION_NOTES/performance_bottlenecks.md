# Performance Bottleneck Review (non-LLM)

**Date**: 2025-11-27
**Status**: Analysis/Recommendations

## Summary

This document highlights non-LLM performance risks identified during a code review and suggests mitigations.

## Findings

# Performance Bottleneck Review (non-LLM)

## Summary

This document highlights non-LLM performance risks identified during a code review and suggests mitigations.

## Findings

1. **Duplicate file reads during agent-enabled note processing**

    - When the agent system is enabled, `NoteScanner.process_note` reads the full note content to supply agents and then calls `parse_note`, which reads the file again. This doubles disk I/O per note and scales poorly for large files or many concurrent workers. 【F:src/obsidian_anki_sync/sync/note_scanner.py†L629-L659】【F:src/obsidian_anki_sync/obsidian/parser.py†L310-L339】
    - _Mitigation_: Pass the already-read content into parsing or provide a parse function that accepts preloaded text to avoid redundant reads.

2. **Repeated copying of the slug set for every card generation**

    - Each card generation call receives `existing_slugs.copy()`, cloning the full slug set for every QA pair and language. For large vaults this becomes O(N²) memory churn and CPU overhead. 【F:src/obsidian_anki_sync/sync/note_scanner.py†L660-L720】
    - _Mitigation_: Pass the shared set with appropriate locking or use a lightweight read-only view instead of copying per call.

3. **Vault indexing runs strictly single-threaded**

    - `VaultIndexer.index_vault` walks and parses notes sequentially, even though parsing and metadata extraction are CPU and I/O heavy. Large vaults will see long indexing times compared to the parallel note scanning path. 【F:src/obsidian_anki_sync/sync/indexer.py†L54-L200】
    - _Mitigation_: Reuse the parallel note-processing strategy from `NoteScanner` or add batching/worker pools to index in parallel while keeping database writes serialized.

4. **Worker timeouts lacked stage-level attribution**
    - When generation or post-validation exceeded their SLA the ARQ layer emitted a generic `TimeoutError`, offering no hint about which stage was responsible. Downstream recovery (highlight/fix routing) could not differentiate between slow generation and a hung validator.
    - _Mitigation_: Introduced `worker_generation_timeout_seconds` and `worker_validation_timeout_seconds` budgets enforced inside `obsidian_anki_sync.worker`. Stage overruns now short-circuit the job with a descriptive error and the retry handler propagates the stage context so the highlight/fix flow can be triggered immediately.
