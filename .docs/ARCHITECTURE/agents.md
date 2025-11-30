# Agent System Architecture

## Overview

Multi-agent system for card generation using LangGraph orchestration, PydanticAI structured outputs, and OpenRouter model access.

## Pipeline

```
Input Note -> Pre-Validator -> Generator -> Post-Validator -> Output Cards
```

**Benefits:** 15-20% faster (early rejection), higher quality, automatic error correction.

## Agent Types

| Agent | Purpose | Models | Output |
|-------|---------|--------|--------|
| Pre-Validator | Early input validation | gpt-4o-mini, qwen3:8b | `PreValidationResult` |
| Highlight | Analyze failed notes, suggest fixes | grok-4.1-fast | `HighlightResult` |
| Generator | Card generation | claude-3-5-sonnet, gpt-4o | `CardGenerationResult` |
| Post-Validator | Quality assurance | gpt-4o, claude-3-haiku | `PostValidationResult` |

## Configuration

```yaml
use_langgraph: true
use_pydantic_ai: true

pre_validator_model: "openai/gpt-4o-mini"
generator_model: "anthropic/claude-3-5-sonnet-20241022"
post_validator_model: "openai/gpt-4o"

agent_config:
    max_retries: 3
    timeout: 60
    temperature: 0.1
```

## Model Presets

| Preset | Pre-validator | Generator | Post-validator |
|--------|---------------|-----------|----------------|
| `cost_effective` | gpt-4o-mini | claude-3-5-sonnet | gpt-4o-mini |
| `balanced` | gpt-4o-mini | claude-3-5-sonnet | gpt-4o |
| `high_quality` | gpt-4o | claude-3-5-sonnet | o1-preview |
| `fast` | gpt-4o-mini | gpt-4o-mini | gpt-4o-mini |

## Usage

```python
from obsidian_anki_sync.agents import LangGraphOrchestrator

orchestrator = LangGraphOrchestrator.from_config(config)
result = await orchestrator.process_note(note_content, frontmatter)
```

```bash
obsidian-anki-sync sync --use-langgraph --use-pydantic-ai
```

## Error Handling

```python
try:
    result = await orchestrator.process_note(note_content, frontmatter)
except PreValidationError as e:
    print(f"Pre-validation failed: {e.issues}")
except GenerationError as e:
    print(f"Generation failed: {e.reason}")
```

---

**Related**: [Providers](providers.md) | [Configuration](../GUIDES/configuration.md)
