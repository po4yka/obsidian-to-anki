# OpenRouter Integration Baseline

**Date**: 2025-12-02
**Status**: Analysis
**Owner**: AI Engineering Partner

## Overview

This note captures how OpenRouter is wired into the sync pipeline today and lists the
capability gaps we observed while preparing the OpenRouter upgrade initiative. It serves as
the reference checklist for the documentation, streaming, reasoning, and rate-limit
improvements tracked in the current plan.

## Entry Points

-   `src/obsidian_anki_sync/apf/generator.py` — Uses the OpenAI SDK pointed at
    `https://openrouter.ai/api/v1` to generate APF cards inside CLI sync commands. No
    streaming or response-format controls are exposed here.
-   `src/obsidian_anki_sync/providers/openrouter/provider.py` — Main provider invoked by
    agents and other services. Handles retries, JSON mode fallback, and Grok reasoning
    toggles but currently raises `NotImplementedError` for streaming requests.
-   `src/obsidian_anki_sync/providers/pydantic_ai_models.py` — Wraps OpenRouter via
    `OpenAIChatModel` for LangGraph/PydanticAI nodes. Uses custom retry transport but does
    not surface reasoning-effort controls or telemetry for rate limiting.
-   `src/obsidian_anki_sync/utils/preflight.py` — Performs connectivity checks against the
    `/models` endpoint and surfaces coarse-grained success/failure messaging during `check`
    and `sync` commands.

## Current Behavior

-   Structured outputs rely on JSON Schema payloads. When `response_format` fails,
    `OpenRouterProvider.generate_json` replays the request without schema enforcement.
-   Reasoning toggles are Grok-specific. Other models cannot request different reasoning
    budgets even though OpenRouter now supports the generalized `reasoning.effort` contract.
-   Streaming is disabled everywhere. Both the CLI generator and the provider raise errors if
    `stream=True` is passed. Long generations therefore block until completion and prevent
    incremental logging.
-   The CLI and docs only mention the API key. Optional attribution headers (`HTTP-Referer`,
    `X-Title`), rate-limit telemetry, and prompt-caching practices are undocumented.
-   Prompt reuse now emits `apf_prompt_cache_hit` log events so we can monitor when OpenRouter
    cache savings are possible.

## Known Gaps / Risks

1. **No streaming support**
    - Impact: Users receive no partial output or progress updates for multi-minute
      generation calls, and we cannot pipe tokens to stdout or future UI surfaces.
    - Evidence: `OpenRouterProvider.generate()` raises `NotImplementedError` when
      `stream=True`; `apf/generator.py` has no streaming branch.
2. **Reasoning controls limited to heuristics**
    - Impact: `llm_reasoning_effort` now maps to OpenRouter's reasoning-effort contract, but
      we still need stage-aware overrides to avoid enabling high-effort reasoning for every
      agent by default.
    - Evidence: `payload_builder` now honors explicit effort, yet config override coverage
      remains sparse outside generation/post-validation.
3. **Rate-limit telemetry surfaced but not automated**
    - Impact: `/key` probing now reports remaining credits/RPM, yet concurrency still relies
      on manual tuning of `max_concurrent_generations`.
    - Evidence: Preflight emits warnings when concurrency exceeds RPM, but runtime throttling
      still depends on user intervention.
4. **Docs omit attribution headers and schema fallbacks**
    - Impact: Users are unaware of best practices (site URL/name, structured-output
      behavior, prompt caching) and face needless troubleshooting.
    - Evidence: `.docs/ARCHITECTURE/providers.md` and `.docs/GUIDES/configuration.md`
      mention only API keys.

## Proposed Actions

-   **P0** `doc-audit`: Update providers/config docs plus this note to describe current flow,
    attribution headers, and fallback behavior.
-   **P0** `streaming-support`: Add SSE streaming plumbing in `OpenRouterProvider`,
    `apf/generator.py`, and LangGraph nodes with tests and telemetry.
-   **P0** `reasoning-controls`: Generalize reasoning metadata and expose config knobs for
    the LangGraph agents and CLI.
-   **P1** `rate-limit-cache`: Query `/api/v1/key`, surface results in preflight checks, and
    document prompt-caching guidance.

## Validation Strategy

-   Unit tests in `tests/test_openrouter_provider.py` cover JSON fallback paths today and
    will be extended for streaming, reasoning-effort payloads, and rate-limit probes.
-   CLI smoke tests (`obsidian-anki-sync test-run`) will verify doc instructions and
    preflight flags once the new features ship.

## References

-   `.docs/ARCHITECTURE/providers.md`
-   `.docs/GUIDES/configuration.md`
-   `src/obsidian_anki_sync/providers/openrouter/provider.py`
-   `tests/test_openrouter_provider.py`
