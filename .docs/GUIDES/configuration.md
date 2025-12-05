# Configuration Guide

## Quick Start

```yaml
vault_path: "~/Documents/ObsidianVault"
source_dir: "Notes"
anki_deck_name: "My Deck"

llm_provider: "openrouter" # or "ollama", "openai", "anthropic"
openrouter_api_key: "${OPENROUTER_API_KEY}"

model_preset: "balanced" # cost_effective|balanced|high_quality|fast
use_langgraph: true
use_pydantic_ai: true
```

## Environment Variables

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OLLAMA_BASE_URL="http://localhost:11434"
```

## Model Presets

| Preset           | Pre-validator             | Generator                  | Post-validator            | Best For            |
| ---------------- | ------------------------- | -------------------------- | ------------------------- | ------------------- |
| `cost_effective` | qwen/qwen-2.5-7b-instruct | qwen/qwen-2.5-32b-instruct | deepseek/deepseek-v3.2    | Budget, high volume |
| `balanced`       | qwen/qwen-2.5-7b-instruct | deepseek/deepseek-v3.2     | deepseek/deepseek-v3.2    | Quality + cost      |
| `high_quality`   | qwen/qwen-2.5-7b-instruct | deepseek/deepseek-v3.2     | deepseek/deepseek-v3.2    | Maximum quality     |
| `fast`           | qwen/qwen-2.5-7b-instruct | qwen/qwen-2.5-7b-instruct  | qwen/qwen-2.5-7b-instruct | Testing             |

## Provider Examples

### Ollama (Local)

```yaml
llm_provider: "ollama"
ollama_base_url: "http://localhost:11434"
generator_model: "qwen3:32b"
```

### OpenRouter (Cloud)

```yaml
llm_provider: "openrouter"
openrouter_api_key: "${OPENROUTER_API_KEY}"
generator_model: "deepseek/deepseek-v3.2" # Latest: Excellent reasoning, 163K context
openrouter_site_url: "https://yourapp.example.com" # Optional attribution
openrouter_site_name: "Obsidian → Anki Sync" # Optional attribution
```

#### Notes (Current State)

-   **API headers** – When `openrouter_site_url` / `openrouter_site_name` are set, the
    provider automatically sends `HTTP-Referer` and `X-Title` so your app shows up in
    OpenRouter rankings.
-   **Structured outputs** – JSON Schema enforcement is attempted first. If a model returns
    an empty completion, we transparently retry without `response_format` before surfacing an
    error.
-   **Streaming** – Enable SSE streaming by setting `llm_streaming_enabled: true`. The APF
    generator and LangGraph agents log chunk progress via `log_llm_stream_chunk` while still
    returning the final HTML/JSON output.
-   **Reasoning effort** – Use `llm_reasoning_effort` (`auto|minimal|low|medium|high|none`)
    to control OpenRouter’s `reasoning.effort` parameter. For per-agent tuning, set
    `reasoning_effort_overrides` (e.g., `generation: "high"`).
-   **Prompt caching telemetry** – Look for `apf_prompt_cache_hit` log entries to understand
    when prompts repeat and could benefit from OpenRouter’s cache.
-   **Reasoning controls** – Only Grok models toggle reasoning automatically today. New
    provider/config knobs are being added to expose OpenRouter’s `reasoning.effort`
    capability.
-   **Preflight checks** – `obsidian-anki-sync check` currently validates `/models`. Credit
    telemetry (`/key`) and prompt-cache reporting are part of the ongoing upgrade.

## Agent Configuration

```yaml
# Model overrides (using latest models)
pre_validator_model: "qwen/qwen-2.5-7b-instruct" # Latest: Fast, cheap ($0.04/$0.10)
generator_model: "deepseek/deepseek-v3.2" # Latest: Excellent reasoning, 163K context
post_validator_model: "deepseek/deepseek-v3.2" # Latest: Excellent reasoning for validation

# Optional agents
context_enrichment_model: "minimax/minimax-m2" # Latest: Excellent for creative tasks (204K context)
card_splitting_model: "moonshotai/kimi-k2-thinking" # Latest: 256K context, advanced reasoning
memorization_quality_model: "moonshotai/kimi-k2" # Strong reasoning capabilities
enable_highlight_agent: true

# Performance
agent_config:
    max_retries: 3
    timeout: 60
    temperature: 0.1
```

## Sync Settings

```yaml
sync_mode: "incremental" # incremental|full
dry_run: false
max_concurrent_requests: 5
batch_size: 50
```

## Validation

```yaml
enable_ai_validation: true
ai_validation_model: "qwen/qwen-2.5-7b-instruct" # Latest: Fast, cheap validation
min_qa_score: 0.8
require_frontmatter: true
```

## Performance & Caching

```yaml
cache_config:
    enable_result_cache: true
    cache_ttl: 3600
    cache_dir: "~/.cache/obsidian-anki-sync"

performance:
    max_connections: 10
    requests_per_minute: 60
```

## Logging

```yaml
logging:
    level: "INFO" # DEBUG|INFO|WARNING|ERROR
    file: "logs/sync.log"
```

## Validation

```bash
obsidian-anki-sync validate --config config.yaml
obsidian-anki-sync ping
```

---

**Related**: [Getting Started](../GETTING_STARTED.md) | [Providers](../ARCHITECTURE/providers.md)
