# Agent System Architecture

## Overview

Multi-agent system for card generation using LangGraph orchestration, PydanticAI structured outputs, and OpenRouter model access.

## Pipeline

```
Input Note -> Pre-Validator -> Generator -> Post-Validator -> Output Cards
```

**Benefits:** 15-20% faster (early rejection), higher quality, automatic error correction.

## Agent Types

| Agent          | Purpose                             | Models                                                         | Output                 |
| -------------- | ----------------------------------- | -------------------------------------------------------------- | ---------------------- |
| Pre-Validator  | Early input validation              | qwen/qwen-2.5-7b-instruct (latest: fast, cheap)                | `PreValidationResult`  |
| Highlight      | Analyze failed notes, suggest fixes | qwen/qwen-2.5-7b-instruct (latest: cost-effective)             | `HighlightResult`      |
| Generator      | Card generation                     | deepseek/deepseek-v3.2 (latest: excellent reasoning, 163K ctx) | `CardGenerationResult` |
| Post-Validator | Quality assurance                   | deepseek/deepseek-v3.2 (latest: excellent reasoning)           | `PostValidationResult` |

## Configuration

```yaml
use_langgraph: true
use_pydantic_ai: true

pre_validator_model: "qwen/qwen-2.5-7b-instruct" # Latest: Fast, cheap ($0.04/$0.10)
generator_model: "deepseek/deepseek-v3.2" # Latest: Excellent reasoning, 163K context
post_validator_model: "deepseek/deepseek-v3.2" # Latest: Excellent reasoning for validation

agent_config:
    max_retries: 3
    timeout: 60
    temperature: 0.1
```

## Model Presets

| Preset           | Pre-validator             | Generator                  | Post-validator            |
| ---------------- | ------------------------- | -------------------------- | ------------------------- |
| `cost_effective` | qwen/qwen-2.5-7b-instruct | qwen/qwen-2.5-32b-instruct | deepseek/deepseek-v3.2    |
| `balanced`       | qwen/qwen-2.5-7b-instruct | deepseek/deepseek-v3.2     | deepseek/deepseek-v3.2    |
| `high_quality`   | qwen/qwen-2.5-7b-instruct | deepseek/deepseek-v3.2     | deepseek/deepseek-v3.2    |
| `fast`           | qwen/qwen-2.5-7b-instruct | qwen/qwen-2.5-7b-instruct  | qwen/qwen-2.5-7b-instruct |

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
