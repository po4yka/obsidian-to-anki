# Technical Design Template

## Metadata

-   **ID:** `TD-011`
-   **Title:** Console Noise Filter for Logging
-   **Author:** GPT-5.1 Codex
-   **Date:** 2025-11-25
-   **Status:** Implemented

## Context

-   Recent sync logs (see `.docs/LOG_ANALYSIS_2025-11-19.md` and the latest `test_run` output from 2025-11-25) show hundreds of high-frequency informational events (e.g., provider factory lifecycle messages, note discovery listings) overwhelming the console.
-   Developers need high-signal warnings (LLM slow requests, validation failures) without scrolling through repetitive boilerplate. File logs must still capture everything for deep forensics.
-   We therefore need a console-specific noise filter that suppresses low-value spam while preserving structured context for debugging.

## Objectives & Non-Goals

-   Reduce console clutter by suppressing repeated low-signal events while keeping full fidelity in rotating log files.
-   Make console output highlight meaningful warnings/errors plus aggregated session stats (LLM summaries already exist in `SyncEngine` and should remain visible).
-   Provide deterministic behavior that is testable and can be toggled if deeper debugging is required.
-   Non-goals: replacing Loguru, changing log formats, or altering existing file/error log retention policies.

## Architecture

-   Extend `obsidian_anki_sync.utils.logging` with:
    -   `HighVolumeEventPolicy` dataclass describing max console occurrences + sliding window.
    -   `ConsoleNoiseFilter` callable that wraps `_add_formatted_extra`, enforces module-level minimum levels, and rate-limits specific event names.
-   Console handler (`sys.stderr`) uses `ConsoleNoiseFilter`; file handlers retain `_add_formatted_extra` to capture all events.
-   Configuration exposes `enable_console_noise_filter` flag (default `True`) plus overridable module-level thresholds/event policies for future tuning.

## Detailed Design

### Flow / Sequence

1. CLI bootstraps logging via `configure_logging(...)`.
2. When `enable_console_noise_filter=True`, instantiate `ConsoleNoiseFilter` with:
    - `DEFAULT_CONSOLE_LEVEL_OVERRIDES = {"obsidian_anki_sync.providers": "WARNING"}` (squelches provider factory spam while keeping warnings).
    - `DEFAULT_HIGH_VOLUME_EVENTS` with policies for `creating_provider`, `creating_provider_from_config`, `provider_created_successfully`, and `discover_notes_in_dir` (first N occurrences per window still emitted).
3. Console handler logs only messages passing the filter; files/error sinks remain unchanged.

### Default Policies (Implemented)

-   Module-level override: `obsidian_anki_sync.providers` must emit `WARNING` or higher on console.
-   High-volume rate limits (per sliding window):
    -   `creating_provider_from_config`: 2 events / 60 seconds
    -   `creating_provider`: 3 events / 60 seconds
    -   `provider_created_successfully`: 3 events / 60 seconds
    -   `discover_notes_in_dir`: 5 events / 10 seconds

### Algorithms / Logic

```python
@dataclass
class HighVolumeEventPolicy:
    max_occurrences: int
    window_seconds: int

class ConsoleNoiseFilter:
    def __call__(self, record: dict) -> bool:
        # 1. Enrich record extras (reuse `_add_formatted_extra`).
        # 2. Enforce module-specific min levels (compare `record["level"].no`).
        # 3. For high-volume events, track timestamps in deque and drop events
        #    once `max_occurrences` is exceeded within the sliding window.
```

Thread-safe via `threading.Lock`.

### API / Interface Changes

-   `configure_logging(...)` gains `enable_console_noise_filter: bool = True` parameter.
-   Internal helper classes exported for testing (`HighVolumeEventPolicy`, `ConsoleNoiseFilter`).

## Deployment & Migration

-   No migrations. Feature flag defaults to on.
-   File logs remain exhaustive, so existing monitoring tooling unaffected.

## Observability

-   Console now surfaces fewer noisy INFO messages; warnings/errors remain untouched.
-   Rate-limited event counters prevent spinner/progress output from being pushed off-screen.
-   Developers can disable the filter (future hook) by calling `configure_logging(..., enable_console_noise_filter=False)` if deeper tracing is needed.

## Security & Privacy

-   No new data collected. Filter operates entirely in-process on log metadata.

## Risks & Mitigations

| Risk                                                     | Impact | Mitigation                                                                                                         |
| -------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------ |
| Legitimate INFO events hidden unintentionally            | Medium | Start with conservative overrides (providers + known spam events) and keep file logs complete. Allow opt-out flag. |
| Race conditions when rate-limiting from multiple threads | Low    | Use `threading.Lock` around policy window pruning/appends.                                                         |

## Testing Strategy

-   Unit tests covering:
    -   Module-level override: INFO from `obsidian_anki_sync.providers.factory` is dropped while WARNING passes (`test_module_level_override_blocks_low_priority_provider_logs`).
    -   High-volume event suppression: After exceeding `max_occurrences`, further logs within the window are filtered out (`test_high_volume_policy_rate_limits_events`).
-   Existing `tests/test_logging_enhanced.py` extended to validate new behaviors without touching disk logs.

## Rollout Plan

-   Land feature guarded by default-on flag.
-   Verify locally by running `test_run` to confirm console shrinkage.
-   Monitor `LOG_ANALYSIS` doc for future tuning recommendations.

## Open Questions

-   Should overrides be configurable via `config.yaml`? (Not needed now but noted for potential follow-up.)
-   Do we need per-command CLI switches (`--verbose-provider-logs`)? Evaluate after feedback.
