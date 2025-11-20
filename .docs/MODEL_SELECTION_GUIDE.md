# Model Selection and Customization Guide

**Last Updated**: 2025-11-19

## Overview

The model configuration system provides flexible, preset-based model selection with per-task customization. Choose a preset for automatic optimization, or override specific models for fine-grained control.

## Quick Start

### Option 1: Use a Preset (Recommended)

Simply set `model_preset` in your `config.yaml`:

```yaml
model_preset: "balanced"  # Options: cost_effective, balanced, high_quality, fast
```

That's it! All tasks will use optimized models automatically.

### Option 2: Customize Specific Tasks

Use a preset as a base, then override specific tasks:

```yaml
model_preset: "balanced"

# Override generator to use a larger model
generator_model: "qwen/qwen-2.5-72b-instruct"
generator_temperature: 0.3
```

## Available Presets

### `balanced` (Default)

Best cost/quality balance. Recommended for most users.

**Models:**
- QA Extraction: `qwen/qwen-2.5-32b-instruct`
- Generation: `qwen/qwen-2.5-72b-instruct`
- Post-Validation: `deepseek/deepseek-chat`
- Context Enrichment: `minimax/minimax-m2`
- Memorization Quality: `moonshotai/kimi-k2`
- Card Splitting: `moonshotai/kimi-k2-thinking`

**Use when:** You want good quality without excessive cost.

---

### `cost_effective`

Lower cost, still good quality. Uses smaller models where possible.

**Models:**
- Most tasks: `qwen/qwen-2.5-32b-instruct`
- Post-Validation: `deepseek/deepseek-chat`

**Use when:** Cost is a primary concern, quality is acceptable.

---

### `high_quality`

Maximum quality, higher cost. Uses larger models for all tasks.

**Models:**
- Most tasks: `qwen/qwen-2.5-72b-instruct`
- Post-Validation: `deepseek/deepseek-chat`
- Context Enrichment: `moonshotai/kimi-k2`

**Use when:** Quality is paramount, cost is less important.

---

### `fast`

Fastest models, lower quality. Optimized for speed.

**Models:**
- All tasks: `qwen/qwen-2.5-32b-instruct` or `deepseek/deepseek-chat`
- Lower `max_tokens` limits for faster responses

**Use when:** Speed is critical, quality can be slightly lower.

## Per-Task Customization

Override any task's model, temperature, or max_tokens:

```yaml
model_preset: "balanced"

# Customize QA extraction
qa_extractor_model: "qwen/qwen-2.5-72b-instruct"  # Use larger model
qa_extractor_temperature: 0.0
qa_extractor_max_tokens: 8192

# Customize generation
generator_model: "qwen/qwen-2.5-72b-instruct"
generator_temperature: 0.3
generator_max_tokens: 16384  # Allow longer responses
```

**Note:** Set to empty string (`""`) or `null` to use preset defaults.

## Model Capabilities

Each model has different capabilities:

| Model | Structured Outputs | Reasoning | Max Output | Context Window | Cost (per 1M) |
|-------|-------------------|-----------|------------|----------------|---------------|
| `qwen/qwen-2.5-72b-instruct` | No | No | 8K | 128K | $0.55 |
| `qwen/qwen-2.5-32b-instruct` | No | No | 8K | 128K | $0.20 |
| `deepseek/deepseek-chat` | Yes | No | 8K | 128K | $0.14/$0.28 |
| `minimax/minimax-m2` | Yes | No | 8K | 128K | $0.30 |
| `moonshotai/kimi-k2` | Yes | No | 8K | 128K | $0.25 |
| `moonshotai/kimi-k2-thinking` | Yes | Yes | 8K | 128K | $0.50 |

## Task-Specific Recommendations

### QA Extraction
- **Recommended:** `qwen/qwen-2.5-32b-instruct`
- **Why:** Fast, cost-effective, sufficient for structured extraction
- **Temperature:** 0.0 (deterministic)

### Generation
- **Recommended:** `qwen/qwen-2.5-72b-instruct`
- **Why:** Best quality for content creation
- **Temperature:** 0.3 (balanced creativity)

### Post-Validation
- **Recommended:** `deepseek/deepseek-chat`
- **Why:** Excellent reasoning, supports structured outputs
- **Temperature:** 0.0 (deterministic)

### Context Enrichment
- **Recommended:** `minimax/minimax-m2` or `moonshotai/kimi-k2`
- **Why:** Good at creative examples and code generation
- **Temperature:** 0.4 (more creative)

## Migration from Legacy Config

If you have existing model configuration, it will continue to work. The new system provides:

1. **Backward compatibility:** Legacy `qa_extractor_model`, `generator_model`, etc. still work
2. **Preset support:** Add `model_preset` to get automatic optimization
3. **Gradual migration:** Override specific tasks as needed

### Example Migration

**Before:**
```yaml
qa_extractor_model: "qwen/qwen-2.5-32b-instruct"
generator_model: "qwen/qwen-2.5-72b-instruct"
post_validator_model: "deepseek/deepseek-chat"
```

**After (using preset):**
```yaml
model_preset: "balanced"
# All models set automatically
```

**After (with customizations):**
```yaml
model_preset: "balanced"
generator_model: "qwen/qwen-2.5-72b-instruct"  # Override if needed
```

## Advanced: Programmatic Access

Use the model configuration system in code:

```python
from obsidian_anki_sync.models.config import (
    ModelTask,
    ModelPreset,
    get_model_config,
)

# Get model config for a task
config = get_model_config(
    ModelTask.GENERATION,
    ModelPreset.BALANCED,
    overrides={"temperature": 0.4}  # Optional
)

print(config.model_name)  # "qwen/qwen-2.5-72b-instruct"
print(config.temperature)  # 0.4
print(config.max_tokens)  # 8192
```

## Troubleshooting

### Model Not Found

If a model name is invalid, the system falls back to `default_llm_model`. Check logs for warnings.

### Preset Not Working

Ensure `model_preset` is one of: `cost_effective`, `balanced`, `high_quality`, `fast` (case-insensitive).

### Overrides Not Applied

Check that override values are not empty strings or `null`. Empty strings use preset defaults.

## Best Practices

1. **Start with a preset:** Use `balanced` as a starting point
2. **Override selectively:** Only customize tasks that need different models
3. **Monitor costs:** Use `cost_effective` preset if costs are high
4. **Test quality:** Try `high_quality` preset if results are insufficient
5. **Check logs:** Model selection is logged for debugging

## See Also

- [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) - Detailed model capabilities
- [MODEL_ANALYSIS.md](../MODEL_ANALYSIS.md) - Model analysis and recommendations

