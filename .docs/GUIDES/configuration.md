# Configuration Guide

## Quick Start

```yaml
vault_path: "~/Documents/ObsidianVault"
source_dir: "Notes"
anki_deck_name: "My Deck"

llm_provider: "openrouter"  # or "ollama", "openai", "anthropic"
openrouter_api_key: "${OPENROUTER_API_KEY}"

model_preset: "balanced"  # cost_effective|balanced|high_quality|fast
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

| Preset | Pre-validator | Generator | Post-validator | Best For |
|--------|---------------|-----------|----------------|----------|
| `cost_effective` | gpt-4o-mini | claude-3-5-sonnet | gpt-4o-mini | Budget, high volume |
| `balanced` | gpt-4o-mini | claude-3-5-sonnet | gpt-4o | Quality + cost |
| `high_quality` | gpt-4o | claude-3-5-sonnet | o1-preview | Maximum quality |
| `fast` | gpt-4o-mini | gpt-4o-mini | gpt-4o-mini | Testing |

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
generator_model: "anthropic/claude-3-5-sonnet-20241022"
```

### OpenAI
```yaml
llm_provider: "openai"
openai_api_key: "${OPENAI_API_KEY}"
generator_model: "gpt-4o"
```

## Agent Configuration

```yaml
# Model overrides
pre_validator_model: "openai/gpt-4o-mini"
generator_model: "anthropic/claude-3-5-sonnet-20241022"
post_validator_model: "openai/gpt-4o"

# Optional agents
context_enrichment_model: "minimax/minimax-m2"
enable_highlight_agent: true

# Performance
agent_config:
    max_retries: 3
    timeout: 60
    temperature: 0.1
```

## Sync Settings

```yaml
sync_mode: "incremental"  # incremental|full
dry_run: false
max_concurrent_requests: 5
batch_size: 50
```

## Validation

```yaml
enable_ai_validation: true
ai_validation_model: "openai/gpt-4o-mini"
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
    level: "INFO"  # DEBUG|INFO|WARNING|ERROR
    file: "logs/sync.log"
```

## Validation

```bash
obsidian-anki-sync validate --config config.yaml
obsidian-anki-sync ping
```

---

**Related**: [Getting Started](../GETTING_STARTED.md) | [Providers](../ARCHITECTURE/providers.md)
