# Model Removal Summary

This document summarizes the removal of all models that are NOT qwen, deepseek, kimi, or minimax from the project.

**Status: COMPLETE** - All non-qwen/deepseek/kimi/minimax models have been removed from the project.

## Completed Changes

### 1. Core Model Configuration Files

#### `src/obsidian_anki_sync/models/config.py`

-   ✅ Removed `x-ai/grok-4.1-fast:free` from MODEL_CAPABILITIES
-   ✅ Updated DEFAULT_MODEL from `x-ai/grok-4.1-fast:free` to `qwen/qwen-2.5-32b-instruct`
-   ✅ Updated all MODEL_PRESETS to use `qwen/qwen-2.5-32b-instruct` (all instances of grok replaced)

#### `src/obsidian_anki_sync/providers/openrouter/models.py`

-   ✅ Removed Grok models from MODELS_WITH_REASONING_SUPPORT (kept only deepseek)
-   ✅ Removed OpenAI, Google Gemini, Anthropic Claude, Mistral, and Grok models from MODEL_CONTEXT_WINDOWS
-   ✅ Removed OpenAI, Google Gemini, Anthropic Claude, Mistral, and Grok models from MODEL_MAX_OUTPUT_TOKENS
-   ✅ Updated MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS to only include qwen/deepseek/kimi/minimax models

#### `src/obsidian_anki_sync/config.py`

-   ✅ Updated `default_llm_model` from `x-ai/grok-4.1-fast:free` to `qwen/qwen-2.5-32b-instruct`
-   ✅ Updated `openrouter_model` from `x-ai/grok-4.1-fast:free` to `qwen/qwen-2.5-32b-instruct`

### 4. Configuration Files

-   ✅ Updated `config.yaml.example` - Replaced all model examples with qwen/deepseek/kimi/minimax
-   ✅ Updated `config.example.yaml` - Removed OpenAI and Anthropic sections, updated examples
-   ✅ Updated `config.providers.example.yaml` - Replaced GPT-4 and Llama models with qwen models
-   ✅ Updated `config.agents.example.yaml` - Replaced GPT-4 with qwen model
-   ✅ `config.langgraph.example.yaml` - Already uses qwen/deepseek/kimi/minimax (no changes needed)

### 5. Documentation Files

-   ✅ Updated `.docs/GUIDES/configuration.md` - Replaced all GPT, Claude, O1-preview with qwen/deepseek/kimi/minimax
-   ✅ Updated `.docs/ARCHITECTURE/agents.md` - Replaced all GPT, Claude, O1-preview with qwen/deepseek/kimi/minimax
-   ✅ Updated `.docs/ARCHITECTURE/providers.md` - Removed OpenAI and Anthropic sections, updated model examples
-   ✅ Updated `.docs/GUIDES/validation.md` - Replaced GPT model references with qwen
-   ✅ Updated `README.md` - Removed OpenAI/Anthropic provider references, updated model examples
-   ✅ Updated `CLAUDE.md` - Removed OpenAI/Anthropic references

### 6. Code Files

-   ✅ Updated `src/obsidian_anki_sync/providers/anthropic.py` - Removed all Claude models from list_models()
-   ✅ Updated `src/obsidian_anki_sync/agents/generator.py` - Removed Llama model references from MAX_TOKENS
-   ✅ Updated `src/obsidian_anki_sync/cli.py` - Replaced GPT model defaults with qwen
-   ✅ Updated `src/obsidian_anki_sync/exceptions.py` - Replaced GPT model reference with qwen
-   ✅ Updated `src/obsidian_anki_sync/agents/output_fixing.py` - Replaced GPT model example with qwen
-   ✅ Updated `src/obsidian_anki_sync/utils/quality_check.py` - Replaced GPT default with qwen
-   ✅ Updated `src/obsidian_anki_sync/providers/pydantic_ai_models.py` - Replaced Grok and Claude examples/defaults
-   ✅ Updated `src/obsidian_anki_sync/providers/openrouter/provider.py` - Removed Grok-specific fallback logic
-   ✅ Updated `src/obsidian_anki_sync/providers/openrouter/payload_builder.py` - Removed Grok-specific temperature logic
-   ✅ Updated `src/obsidian_anki_sync/providers/openrouter/api_calls.py` - Updated docstring examples
-   ✅ Updated `src/obsidian_anki_sync/providers/base.py` - Updated docstring example
-   ✅ Updated `src/obsidian_anki_sync/providers/ollama.py` - Updated docstring example
-   ✅ Updated `src/obsidian_anki_sync/providers/anthropic.py` - Updated docstring and test model
-   ✅ Updated `src/obsidian_anki_sync/providers/openai.py` - Updated docstring
-   ✅ Updated `src/obsidian_anki_sync/providers/README.md` - Replaced Llama and Claude examples

### 7. Test Files

-   ✅ Updated `tests/test_providers_unit.py` - Replaced Llama and Mistral models with qwen models
-   ✅ Updated `tests/test_openrouter_api_calls.py` - Replaced GPT, Claude, Gemini models with qwen/deepseek/kimi
-   ✅ Updated `tests/test_openrouter_provider.py` - Replaced Grok models with qwen
-   ✅ Updated `tests/test_config_validation.py` - Replaced Grok model references with qwen

## Models Removed

### OpenAI/GPT Models (11 models)

-   openai/gpt-4, openai/gpt-4o, openai/gpt-4o-mini, openai/gpt-4-turbo, openai/gpt-3.5-turbo
-   openai/o3-mini, openai/o3-mini-high
-   gpt-4, gpt-4o-mini, gpt-4-turbo-preview, gpt-3.5-turbo, o1-preview

### Anthropic/Claude Models (12 models)

-   anthropic/claude-3-opus, anthropic/claude-3-sonnet, anthropic/claude-3.5-sonnet
-   anthropic/claude-3-haiku, anthropic/claude-3.5-haiku, anthropic/claude-3.5-sonnet-20241022
-   claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
-   claude-2.1, claude-2.0, claude-instant-1.2

### Google/Gemini Models (5 models)

-   google/gemini-pro, google/gemini-2.0-flash-001, google/gemini-2.0-flash-exp
-   google/gemini-2.5-flash, google/gemini-pro-1.5

### xAI/Grok Models (3 models)

-   x-ai/grok-4.1-fast:free, x-ai/grok-4.1, x-ai/grok-3

### Mistral Models (3 models)

-   mistralai/mistral-small-3.1-24b-instruct, mistralai/mistral-large-2411, mistral:latest

### Meta/Llama Models (7 models)

-   meta-llama/llama-3.1-70b-instruct
-   llama3:8b, llama3:70b, llama3.2:3b, llama2:7b, llama2
-   lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF

### Other Models (1 model)

-   openrouter/polaris-alpha

## Models Kept (Qwen, Deepseek, Kimi, Minimax)

### Qwen Models

-   qwen/qwen-2.5-72b-instruct
-   qwen/qwen-2.5-32b-instruct
-   qwen/qwen3-235b-a22b-2507
-   qwen/qwen3-next-80b-a3b-instruct
-   qwen/qwen3-max
-   qwen/qwen3-32b
-   qwen/qwen3-30b-a3b

### Deepseek Models

-   deepseek/deepseek-chat
-   deepseek/deepseek-chat-v3.1
-   deepseek/deepseek-v3.2
-   deepseek/deepseek-r1

### Kimi (Moonshot) Models

-   moonshotai/kimi-k2
-   moonshotai/kimi-k2-thinking

### Minimax Models

-   minimax/minimax-m2
