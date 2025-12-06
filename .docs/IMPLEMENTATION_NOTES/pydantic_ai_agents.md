# PydanticAI Agents Module Split

**Date**: 2025-12-06
**Status**: Implemented
**Owner**: Assistant

## Overview

The monolithic `src/obsidian_anki_sync/agents/pydantic_ai_agents.py` has been
modularized to separate shared utilities, dependency models, and structured
outputs. This reduces file size, clarifies responsibilities, and improves reuse
across PydanticAI agent implementations (pre-validation, generation,
post-validation, memorization quality, card splitting, duplicate detection, and
context enrichment).

## Module Map

- `agents/pydantic_ai/streaming.py`
  - `run_agent_with_streaming`: shared streaming runner with progress logging.
  - `_decode_html_encoded_apf`, `_truncate_for_log`, model name/token estimate
    helpers.

- `agents/pydantic_ai/deps.py`
  - Dependency models per agent family: `PreValidationDeps`, `GenerationDeps`,
    `PostValidationDeps`, `CardSplittingDeps`.

- `agents/pydantic_ai/outputs.py`
  - Structured outputs used by agents:
    - Validation: `PreValidationOutput`, `PostValidationOutput`
    - Generation: `CardGenerationOutput`
    - Memorization: `MemorizationIssue`, `MemorizationQualityOutput`
    - Splitting: `CardSplitPlanOutput`, `CardSplittingOutput`
    - Dedup: `DuplicateMatchOutput`, `DuplicateDetectionOutput`
    - Enrichment: `ContextEnrichmentOutput`
  - APF HTML decode + APF linter validation inside `CardGenerationOutput`.

- `agents/pydantic_ai/__init__.py`
  - Re-exports shared utilities, deps, and outputs for concise imports.

- `agents/pydantic_ai_agents.py`
  - Agent classes now import shared pieces from `agents.pydantic_ai.*` while
    retaining existing public APIs and behaviors.

## Notes

- No behavior changes intended; class names and signatures remain stable.
- APF validation still enforced via `CardGenerationOutput` validator.
- Streaming logging unchanged; shared in `streaming.py`.
- Further candidates for extraction: prompt builders and APF HTML extraction
  helpers (currently remain in `pydantic_ai_agents.py`).

## Validation Strategy

- Run `uv run ruff check .` and `pytest` to ensure imports and behavior remain
  intact after modularization. Current change was structural; tests not run yet.


