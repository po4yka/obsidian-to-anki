# Switching LLM Providers

This guide shows how to change the LLM provider that generates your Anki
cards. Each switch involves editing `config.yaml` and, for remote providers,
setting an API key.

## Available providers

| Provider     | Type   | Default base URL                     |
|--------------|--------|--------------------------------------|
| `ollama`     | Local  | `http://localhost:11434`             |
| `lm_studio`  | Local  | `http://localhost:1234/v1`           |
| `openrouter` | Remote | `https://openrouter.ai/api/v1`      |
| `openai`     | Remote | (uses OpenAI SDK default)            |
| `anthropic`  | Remote | (uses Anthropic SDK default)         |
| `claude`     | Remote | (uses Anthropic SDK default)         |

## Config fields reference

| Field                  | Default                      | Purpose                        |
|------------------------|------------------------------|--------------------------------|
| `llm_provider`         | `"ollama"`                   | Active provider name           |
| `default_llm_model`    | `"deepseek/deepseek-v3.2"`  | Primary model for generation   |
| `fallback_llm_model`   | `"qwen/qwen3-max"`          | Fallback when primary fails    |
| `ollama_base_url`      | `http://localhost:11434`     | Ollama server address          |
| `lm_studio_base_url`   | `http://localhost:1234/v1`   | LM Studio server address       |
| `openrouter_base_url`  | `https://openrouter.ai/api/v1` | OpenRouter API address      |
| `llm_temperature`      | `0.2`                        | Sampling temperature           |
| `llm_top_p`            | `0.3`                        | Nucleus sampling threshold     |
| `llm_timeout`          | `3600.0`                     | Request timeout in seconds     |
| `llm_max_tokens`       | `8192`                       | Maximum tokens per response    |

## Switching from Ollama to OpenRouter

1. Set your API key:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

2. Update `config.yaml`:

```yaml
# Before (Ollama)
llm_provider: "ollama"
default_llm_model: "qwen3:32b"

# After (OpenRouter)
llm_provider: "openrouter"
default_llm_model: "deepseek/deepseek-v3.2"
```

3. Verify the switch:

```bash
obsidian-anki-sync check --skip-anki
obsidian-anki-sync test-run --count 1
```

## Switching from OpenRouter to Ollama

1. Pull the model you need:

```bash
ollama pull qwen3:32b
```

2. Update `config.yaml`:

```yaml
llm_provider: "ollama"
default_llm_model: "qwen3:32b"
```

3. Verify:

```bash
obsidian-anki-sync check --skip-anki
obsidian-anki-sync test-run --count 1
```

## Setting up LM Studio

1. Start the LM Studio application and load a model.
2. Start the local server from within LM Studio (default port 1234).
3. Update `config.yaml`:

```yaml
llm_provider: "lm_studio"
default_llm_model: "your-loaded-model-name"
lm_studio_base_url: "http://localhost:1234/v1"
```

4. Verify:

```bash
obsidian-anki-sync check --skip-anki
```

## Model presets and per-task overrides

The `model_preset` field selects a predefined configuration profile. The
default preset is `"balanced"`.

For fine-grained control, use `model_overrides` to assign specific models
to individual pipeline tasks:

```yaml
model_preset: "balanced"
model_overrides:
  generator: "deepseek/deepseek-v3.2"
  validator: "qwen/qwen3-max"
```

The selection cascade is: per-task override > preset default > `default_llm_model`.

## Verification after any switch

Always run these two commands after changing providers:

```bash
# Test LLM connectivity (skip Anki check if Anki is not running)
obsidian-anki-sync check --skip-anki

# Generate one card to verify end-to-end
obsidian-anki-sync test-run --count 1
```

## Next steps

- [First sync walkthrough](first-sync.md)
- [Performance tuning](performance-tuning.md)
