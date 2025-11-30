# LLM Providers

## Provider Comparison

| Provider | Type | Best For | Cost |
|----------|------|----------|------|
| **Ollama** | Local/Cloud | Privacy, offline | Free (local) |
| **LM Studio** | Local | GUI, testing | Free |
| **OpenRouter** | Cloud | Latest models | Pay-per-token |
| **OpenAI** | Cloud | GPT models | Pay-per-token |
| **Anthropic** | Cloud | Claude models | Pay-per-token |

## Configuration Examples

### Ollama (Local)
```yaml
llm_provider: "ollama"
ollama_base_url: "http://localhost:11434"
generator_model: "qwen3:32b"
```

```bash
brew install ollama
ollama serve
ollama pull qwen3:32b
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

### Anthropic
```yaml
llm_provider: "anthropic"
anthropic_api_key: "${ANTHROPIC_API_KEY}"
generator_model: "claude-3-5-sonnet-20241022"
```

### LM Studio
```yaml
llm_provider: "lm_studio"
lm_studio_base_url: "http://localhost:1234/v1"
```

## Model Selection

Model cascade: Agent-specific -> Default -> Provider default

```yaml
default_llm_model: "openai/gpt-4o-mini"
generator_model: "anthropic/claude-3-5-sonnet-20241022"  # Override
```

## Recommended Models by Agent

| Agent | Requirement | Recommended |
|-------|-------------|-------------|
| Pre-Validator | Fast, cheap | gpt-4o-mini, claude-3-haiku |
| Generator | High quality | claude-3-5-sonnet, gpt-4o |
| Post-Validator | Balanced | gpt-4o, claude-3-5-sonnet |

## Security

```bash
# Set API keys via environment (never commit)
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Troubleshooting

**Provider not available:** Check API key, network, rate limits
```bash
curl -H "Authorization: Bearer $API_KEY" https://api.provider.com/v1/models
```

**Model not found:** Verify model name (case-sensitive), check availability

**Rate limited:** Increase retry delay, reduce frequency, upgrade plan

---

**Related**: [Agents](agents.md) | [Configuration](../GUIDES/configuration.md)
