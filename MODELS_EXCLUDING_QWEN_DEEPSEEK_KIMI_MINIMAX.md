# Models Excluding Qwen, Deepseek, Kimi, and Minimax

This document lists all LLM models found in the codebase that are **NOT** qwen, deepseek, kimi, or minimax. These models include GPT, Claude, Gemini, Grok, Mistral, Llama, and other model families.

## OpenAI/GPT Models

### Models Found in Code:

-   `openai/gpt-4` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS)
-   `openai/gpt-4o` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODEL_CONTEXT_WINDOWS, MODEL_MAX_OUTPUT_TOKENS, MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS)
-   `openai/gpt-4o-mini` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODEL_CONTEXT_WINDOWS, MODEL_MAX_OUTPUT_TOKENS, MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS), `config.yaml.example`, `.docs/GUIDES/configuration.md`, `.docs/ARCHITECTURE/agents.md`
-   `openai/gpt-4-turbo` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODEL_CONTEXT_WINDOWS, MODEL_MAX_OUTPUT_TOKENS, MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS)
-   `openai/gpt-3.5-turbo` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS)
-   `openai/o3-mini` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODELS_WITH_REASONING_SUPPORT)
-   `openai/o3-mini-high` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODELS_WITH_REASONING_SUPPORT)
-   `gpt-4` - Found in: `src/obsidian_anki_sync/exceptions.py`, `src/obsidian_anki_sync/cli.py` (as default), `config.providers.example.yaml`
-   `gpt-4o-mini` - Found in: `src/obsidian_anki_sync/cli.py` (as default), `.docs/GUIDES/configuration.md` (presets)
-   `gpt-4-turbo-preview` - Found in: `config.example.yaml` (commented example)
-   `gpt-3.5-turbo` - Found in: `config.example.yaml` (commented example)
-   `o1-preview` - Found in: `.docs/GUIDES/configuration.md` (high_quality preset), `.docs/ARCHITECTURE/agents.md` (high_quality preset)

## Anthropic/Claude Models

### OpenRouter Format Models:

-   `anthropic/claude-3-opus` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS)
-   `anthropic/claude-3-sonnet` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS)
-   `anthropic/claude-3.5-sonnet` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODEL_CONTEXT_WINDOWS, MODEL_MAX_OUTPUT_TOKENS, MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS), `config.yaml.example`, `.docs/GUIDES/configuration.md`, `.docs/ARCHITECTURE/agents.md`
-   `anthropic/claude-3-haiku` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS), `.docs/ARCHITECTURE/agents.md`
-   `anthropic/claude-3.5-haiku` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODEL_CONTEXT_WINDOWS, MODEL_MAX_OUTPUT_TOKENS, MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS)
-   `anthropic/claude-3.5-sonnet-20241022` - Found in: `.docs/GUIDES/configuration.md`, `.docs/ARCHITECTURE/agents.md`

### Anthropic Provider Format Models (from `src/obsidian_anki_sync/providers/anthropic.py`):

-   `claude-3-opus-20240229` - Found in: `src/obsidian_anki_sync/providers/anthropic.py` (list_models method)
-   `claude-3-sonnet-20240229` - Found in: `src/obsidian_anki_sync/providers/anthropic.py` (list_models method)
-   `claude-3-haiku-20240307` - Found in: `src/obsidian_anki_sync/providers/anthropic.py` (list_models method), `config.example.yaml` (commented example)
-   `claude-2.1` - Found in: `src/obsidian_anki_sync/providers/anthropic.py` (list_models method)
-   `claude-2.0` - Found in: `src/obsidian_anki_sync/providers/anthropic.py` (list_models method)
-   `claude-instant-1.2` - Found in: `src/obsidian_anki_sync/providers/anthropic.py` (list_models method)

## Google/Gemini Models

-   `google/gemini-pro` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS)
-   `google/gemini-2.0-flash-001` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODEL_CONTEXT_WINDOWS, MODEL_MAX_OUTPUT_TOKENS, MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS)
-   `google/gemini-2.0-flash-exp` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODEL_CONTEXT_WINDOWS, MODEL_MAX_OUTPUT_TOKENS, MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS)
-   `google/gemini-2.5-flash` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODEL_CONTEXT_WINDOWS, MODEL_MAX_OUTPUT_TOKENS, MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS)
-   `google/gemini-pro-1.5` - Found in: `config.yaml.example` (commented example)

## xAI/Grok Models

-   `x-ai/grok-4.1-fast:free` - Found in: `src/obsidian_anki_sync/models/config.py` (MODEL_CAPABILITIES, DEFAULT_MODEL), `src/obsidian_anki_sync/config.py` (default_llm_model, openrouter_model), `src/obsidian_anki_sync/providers/openrouter/models.py` (MODELS_WITH_REASONING_SUPPORT, MODEL_CONTEXT_WINDOWS, MODEL_MAX_OUTPUT_TOKENS, MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS)
-   `x-ai/grok-4.1` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODELS_WITH_REASONING_SUPPORT)
-   `x-ai/grok-3` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODELS_WITH_REASONING_SUPPORT)

## Mistral Models

-   `mistralai/mistral-small-3.1-24b-instruct` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODEL_CONTEXT_WINDOWS, MODEL_MAX_OUTPUT_TOKENS, MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS)
-   `mistralai/mistral-large-2411` - Found in: `src/obsidian_anki_sync/providers/openrouter/models.py` (MODEL_CONTEXT_WINDOWS, MODEL_MAX_OUTPUT_TOKENS, MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS)
-   `mistral:latest` - Found in: `tests/test_providers_unit.py` (test example)

## Meta/Llama Models

### OpenRouter Format:

-   `meta-llama/llama-3.1-70b-instruct` - Found in: `config.yaml.example` (commented example)

### Ollama Format (local models):

-   `llama3:8b` - Found in: `src/obsidian_anki_sync/agents/generator.py` (MAX_TOKENS mapping)
-   `llama3:70b` - Found in: `src/obsidian_anki_sync/agents/generator.py` (MAX_TOKENS mapping)
-   `llama3.2:3b` - Found in: `README.md` (example pull command), `config.providers.example.yaml` (commented example)
-   `llama2:7b` - Found in: `tests/test_providers_unit.py` (test examples)
-   `llama2` - Found in: `tests/test_providers_unit.py` (test examples)

### LM Studio Format:

-   `lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF` - Found in: `config.providers.example.yaml` (commented example), `src/obsidian_anki_sync/providers/README.md`

## Other Models

-   `openrouter/polaris-alpha` - Found in: `config.yaml.example` (default_llm_model example)

## Summary by Source File

### Primary Model Definition Files:

1. **`src/obsidian_anki_sync/models/config.py`**:

    - `x-ai/grok-4.1-fast:free` (MODEL_CAPABILITIES, DEFAULT_MODEL)

2. **`src/obsidian_anki_sync/providers/openrouter/models.py`**:

    - OpenAI: `openai/gpt-4`, `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-4-turbo`, `openai/gpt-3.5-turbo`, `openai/o3-mini`, `openai/o3-mini-high`
    - Anthropic: `anthropic/claude-3-opus`, `anthropic/claude-3-sonnet`, `anthropic/claude-3.5-sonnet`, `anthropic/claude-3-haiku`, `anthropic/claude-3.5-haiku`
    - Google: `google/gemini-pro`, `google/gemini-2.0-flash-001`, `google/gemini-2.0-flash-exp`, `google/gemini-2.5-flash`
    - xAI: `x-ai/grok-4.1-fast:free`, `x-ai/grok-4.1`, `x-ai/grok-3`
    - Mistral: `mistralai/mistral-small-3.1-24b-instruct`, `mistralai/mistral-large-2411`

3. **`src/obsidian_anki_sync/providers/anthropic.py`**:
    - `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`, `claude-2.1`, `claude-2.0`, `claude-instant-1.2`

### Configuration Files:

4. **`config.yaml.example`**:

    - `openrouter/polaris-alpha`, `openai/gpt-4o-mini`, `openai/gpt-4o`, `anthropic/claude-3-5-sonnet`, `google/gemini-pro-1.5`, `meta-llama/llama-3.1-70b-instruct`

5. **`config.example.yaml`**:

    - `gpt-4-turbo-preview`, `gpt-4`, `gpt-3.5-turbo`, `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`

6. **`config.providers.example.yaml`**:
    - `openai/gpt-4`, `llama3.2:3b`, `llama3:8b`, `llama3:70b`, `lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF`

### Documentation Files:

7. **`.docs/GUIDES/configuration.md`**:

    - `gpt-4o-mini`, `claude-3-5-sonnet`, `gpt-4o`, `o1-preview`, `anthropic/claude-3-5-sonnet-20241022`

8. **`.docs/ARCHITECTURE/agents.md`**:
    - `gpt-4o-mini`, `claude-3-5-sonnet`, `gpt-4o`, `claude-3-haiku`, `o1-preview`, `anthropic/claude-3-5-sonnet-20241022`

### Code Files:

9. **`src/obsidian_anki_sync/agents/generator.py`**:

    - `llama3:8b`, `llama3:70b`

10. **`src/obsidian_anki_sync/config.py`**:

    - `x-ai/grok-4.1-fast:free` (default_llm_model, openrouter_model)

11. **`tests/test_providers_unit.py`**:
    - `llama2`, `llama2:7b`, `mistral:latest`

## Total Count

-   **OpenAI/GPT Models**: 11 unique model identifiers
-   **Anthropic/Claude Models**: 12 unique model identifiers
-   **Google/Gemini Models**: 5 unique model identifiers
-   **xAI/Grok Models**: 3 unique model identifiers
-   **Mistral Models**: 3 unique model identifiers (including `mistral:latest`)
-   **Meta/Llama Models**: 7 unique model identifiers (various formats)
-   **Other Models**: 1 model (`openrouter/polaris-alpha`)

**Grand Total**: 42 unique model identifiers (excluding qwen, deepseek, kimi, and minimax)
