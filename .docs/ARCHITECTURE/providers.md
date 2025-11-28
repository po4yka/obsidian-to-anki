# LLM Providers and Model Configuration

This guide covers the LLM provider system and model configuration for the Obsidian to Anki sync service. Choose the right provider for your needs and configure optimal models for each agent.

## Overview

The system supports multiple LLM providers through a unified interface, allowing you to choose based on your requirements for privacy, cost, performance, and available models.

### Provider Comparison

| Provider       | Type        | Best For                          | Setup                 | Cost                       |
| -------------- | ----------- | --------------------------------- | --------------------- | -------------------------- |
| **Ollama**     | Local/Cloud | Privacy, offline, unlimited usage | `brew install ollama` | Free (local) / Low (cloud) |
| **LM Studio**  | Local       | GUI management, model testing     | Download app          | Free                       |
| **OpenRouter** | Cloud       | Latest models, scalability        | API key               | Pay-per-token              |
| **OpenAI**     | Cloud       | GPT models, reliability           | API key               | Pay-per-token              |
| **Anthropic**  | Cloud       | Claude models, reasoning          | API key               | Pay-per-token              |

---

## Supported Providers

### 1. Ollama (Local & Cloud)

**Best for:** Privacy, offline usage, unlimited requests, Apple Silicon optimization

#### Local Ollama Setup

```yaml
llm_provider: "ollama"
ollama_base_url: "http://localhost:11434"
pre_validator_model: "qwen3:8b"
generator_model: "qwen3:32b"
post_validator_model: "qwen3:14b"
```

**Installation:**

```bash
# Install Ollama
brew install ollama

# Start service
ollama serve

# Pull recommended models
ollama pull qwen3:8b
ollama pull qwen3:14b
ollama pull qwen3:32b
```

#### Ollama Cloud Setup

```yaml
llm_provider: "ollama"
ollama_base_url: "https://api.ollama.com"
ollama_api_key: "your-api-key" # Or set OLLAMA_API_KEY env var
```

**Recommended Models:**

-   Pre-validator: `qwen3:8b`, `llama3:8b`, `phi3:mini`
-   Generator: `qwen3:32b`, `llama3:70b`, `mixtral:8x7b`
-   Post-validator: `qwen3:14b`, `llama3:13b`, `mistral:7b`

---

### 2. LM Studio

**Best for:** GUI model management, local deployment, testing different models

```yaml
llm_provider: "lm_studio"
lm_studio_base_url: "http://localhost:1234/v1"
pre_validator_model: "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
generator_model: "lmstudio-community/Qwen2.5-32B-Instruct-GGUF"
post_validator_model: "lmstudio-community/Qwen2.5-14B-Instruct-GGUF"
```

**Setup:**

1. Download LM Studio: https://lmstudio.ai
2. Load a model in LM Studio
3. Start the local server (Developer → Local Server)
4. Use the model identifier shown in LM Studio

**Features:**

-   OpenAI-compatible API
-   Easy model switching through GUI
-   Model performance monitoring
-   GGUF model support

---

### 3. OpenRouter

**Best for:** State-of-the-art models, cloud scalability, no local hardware required

```yaml
llm_provider: "openrouter"
openrouter_api_key: "your-api-key" # Or set OPENROUTER_API_KEY env var
openrouter_base_url: "https://openrouter.ai/api/v1"
pre_validator_model: "openai/gpt-4o-mini"
generator_model: "anthropic/claude-3-5-sonnet-20241022"
post_validator_model: "openai/gpt-4o"
```

**Setup:**

1. Sign up at https://openrouter.ai
2. Get API key from dashboard
3. Set `OPENROUTER_API_KEY` environment variable
4. Choose models from OpenRouter's model list

**Popular Models:**

-   OpenAI: `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-3.5-turbo`
-   Anthropic: `anthropic/claude-3-5-sonnet-20241022`, `anthropic/claude-3-haiku`
-   Meta: `meta-llama/llama-3.1-70b-instruct`
-   Mistral: `mistralai/mixtral-8x7b-instruct`

**Optional Configuration:**

```yaml
openrouter_site_url: "https://yourdomain.com"
openrouter_site_name: "Your App Name"
```

---

### 4. OpenAI

**Best for:** GPT models, reliability, consistent API

```yaml
llm_provider: "openai"
openai_api_key: "sk-..." # Or set OPENAI_API_KEY env var
openai_base_url: "https://api.openai.com/v1"
pre_validator_model: "gpt-4o-mini"
generator_model: "gpt-4o"
post_validator_model: "gpt-4o-mini"
```

**Setup:**

1. Sign up at https://platform.openai.com
2. Get API key from dashboard
3. Set `OPENAI_API_KEY` environment variable

**Available Models:**

-   `gpt-4o` (recommended for generation)
-   `gpt-4o-mini` (fast, cost-effective)
-   `gpt-3.5-turbo` (legacy, inexpensive)

---

### 5. Anthropic

**Best for:** Claude models, advanced reasoning, safety

```yaml
llm_provider: "anthropic"
anthropic_api_key: "sk-ant-..." # Or set ANTHROPIC_API_KEY env var
anthropic_base_url: "https://api.anthropic.com"
pre_validator_model: "claude-3-haiku-20240307"
generator_model: "claude-3-5-sonnet-20241022"
post_validator_model: "claude-3-haiku-20240307"
```

**Setup:**

1. Sign up at https://console.anthropic.com
2. Get API key from dashboard
3. Set `ANTHROPIC_API_KEY` environment variable

**Available Models:**

-   `claude-3-5-sonnet-20241022` (recommended for generation)
-   `claude-3-haiku-20240307` (fast, cost-effective)
-   `claude-3-opus-20240229` (most capable, expensive)

---

## Model Configuration

### Unified Configuration

All model configuration is unified in your `config.yaml`:

```yaml
# Set default model for all agents
default_llm_model: "qwen/qwen-2.5-72b-instruct"

# Optional: Override specific agents
pre_validator_model: "qwen/qwen-2.5-32b-instruct"
generator_model: "anthropic/claude-3-5-sonnet-20241022"
post_validator_model: "openai/gpt-4o-mini"
```

### How It Works

The system uses a cascading model selection:

1. **Agent-specific model** (if set)
2. **Default model** (if agent-specific not set)
3. **Provider default** (fallback)

**Example:**

```yaml
default_llm_model: "openai/gpt-4o-mini"
generator_model: "anthropic/claude-3-5-sonnet-20241022"
# Pre-validator uses gpt-4o-mini, generator uses claude-3-5-sonnet
```

### Model Presets

The system provides optimized presets for different use cases:

#### Cost-Effective Preset

```yaml
model_preset: "cost_effective"
# Uses: GPT-4o-mini, Claude-3-Haiku, GPT-4o-mini
# Best for: Budget-conscious users, high volume
```

#### Balanced Preset

```yaml
model_preset: "balanced"
# Uses: GPT-4o, Claude-3-5-Sonnet, GPT-4o
# Best for: Good quality with reasonable cost
```

#### High Quality Preset

```yaml
model_preset: "high_quality"
# Uses: GPT-4o, Claude-3-5-Sonnet, o1-preview
# Best for: Maximum quality, cost not a concern
```

#### Fast Preset

```yaml
model_preset: "fast"
# Uses: GPT-4o-mini, GPT-4o-mini, GPT-4o-mini
# Best for: Speed over quality, testing
```

### Agent-Specific Model Recommendations

#### Pre-Validator Agent

**Task:** Early validation, reject invalid inputs
**Requirements:** Fast, accurate, cost-effective
**Recommended:** GPT-4o-mini, Claude-3-Haiku, Qwen2.5-32B

#### Generator Agent

**Task:** Core card generation from Q&A pairs
**Requirements:** High quality, creative, comprehensive
**Recommended:** Claude-3-5-Sonnet, GPT-4o, Qwen2.5-72B

#### Post-Validator Agent

**Task:** Quality assurance and error correction
**Requirements:** Analytical, consistent, reliable
**Recommended:** GPT-4o, Claude-3-5-Sonnet, Qwen2.5-32B

#### Context Enrichment Agent

**Task:** Add examples and explanations
**Requirements:** Creative, educational, helpful
**Recommended:** Claude-3-5-Sonnet, GPT-4o, MiniMax M2

#### Memorization Quality Agent

**Task:** Assess learning effectiveness
**Requirements:** Analytical, evidence-based
**Recommended:** GPT-4o, Claude-3-5-Sonnet, Kimi K2

---

## Performance Optimization

### Model Selection Strategy

Choose models based on your priorities:

**For Speed:**

-   Use smaller models (GPT-4o-mini, Claude-3-Haiku)
-   Focus on pre-validation and post-validation
-   Reserve large models only for generation

**For Quality:**

-   Use latest large models (Claude-3-5-Sonnet, GPT-4o)
-   Apply to generation and complex analysis
-   Accept higher latency for better results

**For Cost:**

-   Use OpenRouter for model comparison
-   Optimize with smaller models where possible
-   Monitor usage and switch based on cost trends

### Caching and Reuse

```yaml
# Enable response caching
cache_config:
    enable_result_cache: true
    cache_ttl: 3600 # 1 hour
    max_cache_size: 1000
```

### Batch Processing

```yaml
# Process multiple requests efficiently
batch_config:
    max_batch_size: 10
    batch_timeout: 30
```

---

## Error Handling and Fallbacks

### Automatic Fallbacks

The system automatically falls back on errors:

1. **Model unavailable** → Try alternative model
2. **Provider down** → Try alternative provider
3. **Rate limited** → Wait and retry with backoff
4. **API error** → Fallback to simpler model

### Configuration Example

```yaml
# Primary configuration
llm_provider: "openrouter"
default_llm_model: "anthropic/claude-3-5-sonnet-20241022"

# Fallback configuration
fallback_provider: "openai"
fallback_model: "gpt-4o-mini"

# Retry configuration
max_retries: 3
retry_delay: 2.0
circuit_breaker_threshold: 5
```

### Monitoring

```python
# Check provider health
health = provider.check_connection()
print(f"Provider healthy: {health}")

# Get usage statistics
stats = provider.get_usage_stats()
print(f"Requests: {stats.requests}, Cost: ${stats.cost}")
```

---

## Security Considerations

### API Key Management

**Never commit API keys to version control.**

```bash
# Set in environment (recommended)
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use .env file (gitignored)
echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env
```

### Provider-Specific Security

-   **Ollama (Local):** No API keys, fully private
-   **LM Studio:** Local server, no external access
-   **Cloud Providers:** Use environment variables, rotate keys regularly

### Validation

```python
# Validate configuration on startup
config.validate_providers()

# Test provider connectivity
await provider.validate_connection()

# Check model availability
models = await provider.list_available_models()
```

---

## Troubleshooting

### Common Issues

#### "Provider not available"

**Check:**

-   API key is set correctly
-   Provider service is online
-   Network connectivity
-   Rate limits not exceeded

**Fix:**

```bash
# Test provider
curl -H "Authorization: Bearer $API_KEY" https://api.provider.com/v1/models
```

#### "Model not found"

**Check:**

-   Model name is correct (case-sensitive)
-   Model is available for your account
-   Provider supports the model

**Fix:**

-   Use provider's model list API
-   Switch to alternative model
-   Update configuration

#### "Rate limited"

**Symptoms:** 429 errors, slow responses

**Fix:**

-   Increase retry delay
-   Reduce request frequency
-   Switch to alternative provider
-   Upgrade API plan

#### "High costs"

**Monitor:**

```bash
# Check usage
obsidian-anki-sync usage --last-30-days
```

**Optimize:**

-   Use smaller models for simple tasks
-   Enable caching
-   Switch to cost-effective models

### Debug Commands

```bash
# Test provider connectivity
obsidian-anki-sync ping

# List available models
obsidian-anki-sync models

# Check configuration
obsidian-anki-sync validate

# Monitor usage
obsidian-anki-sync usage
```

---

## Migration Guide

### From Legacy Configuration

**Old style** (deprecated):

```yaml
# Multiple provider configs mixed together
openai_api_key: "..."
anthropic_api_key: "..."
ollama_base_url: "..."
```

**New style** (recommended):

```yaml
# Single provider with unified config
llm_provider: "openrouter"
openrouter_api_key: "sk-or-v1-..."
default_llm_model: "anthropic/claude-3-5-sonnet-20241022"
```

### Provider Migration

**From Ollama to OpenRouter:**

```yaml
# Before
llm_provider: "ollama"
ollama_base_url: "http://localhost:11434"

# After
llm_provider: "openrouter"
openrouter_api_key: "sk-or-v1-..."
```

**From OpenAI to Anthropic:**

```yaml
# Before
llm_provider: "openai"
generator_model: "gpt-4"

# After
llm_provider: "anthropic"
generator_model: "claude-3-5-sonnet-20241022"
```

---

## Related Documentation

-   **[Agent System](agents.md)** - How providers integrate with agents
-   **[Configuration Guide](../GUIDES/configuration.md)** - Complete setup guide
-   **[Getting Started](../GETTING_STARTED.md)** - Quick provider setup

---

**Version**: 2.0
**Last Updated**: November 28, 2025
