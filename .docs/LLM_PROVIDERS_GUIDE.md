# LLM Provider Selection Guide

## Overview

The Obsidian  Anki sync system supports multiple LLM providers, giving you flexibility in choosing the best model for your needs based on cost, performance, privacy, and quality.

## Supported Providers

| Provider | Type | Best For | Cost | Privacy |
|----------|------|----------|------|---------|
| **Ollama** | Local/Cloud | Privacy, offline use, no API costs | Free (local) | Excellent |
| **OpenAI** | Cloud API | Highest quality, GPT-4 | $$ - $$$ | Good |
| **Anthropic** | Cloud API | Strong reasoning (Claude 3) | $$ - $$$ | Good |
| **OpenRouter** | Cloud Gateway | Access to many models | $ - $$$ | Good |
| **LM Studio** | Local GUI | Easy local setup, visual interface | Free (local) | Excellent |

---

## Provider Details

### 1. Ollama (Recommended for Privacy)

**Description**: Run open-source LLMs locally or use Ollama Cloud

**Pros:**
-  Completely private (local execution)
-  No API costs
-  Works offline
-  Large model selection (Qwen, Llama, Mistral, etc.)
-  Fast inference with GPU

**Cons:**
-  Requires powerful hardware (GPU recommended)
-  Initial model download required
-  Quality varies by model

**Configuration:**

```yaml
llm_provider: "ollama"

# Local Ollama
ollama_base_url: "http://localhost:11434"

# OR Ollama Cloud
# ollama_base_url: "https://api.ollama.com"
# ollama_api_key: "your-api-key"

# Recommended models
generator_model: "qwen3:32b"      # High quality for card generation
post_validator_model: "qwen3:14b"  # Good balance for QA
pre_validator_model: "qwen3:8b"    # Fast for validation
```

**Setup Steps:**

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull models
ollama pull qwen3:8b
ollama pull qwen3:14b
ollama pull qwen3:32b

# 3. Verify
ollama list
```

**Recommended Models:**
- **qwen3:32b** - Best quality for complex tasks
- **qwen3:14b** - Good balance of speed/quality
- **qwen3:8b** - Fast for simple tasks
- **llama3:70b** - Very high quality (requires 48GB+ RAM)
- **mistral:7b** - Fast and capable

---

### 2. OpenAI (Recommended for Quality)

**Description**: Use OpenAI's GPT models via API

**Pros:**
-  Highest quality (GPT-4)
-  Excellent reasoning and instruction following
-  JSON mode built-in
-  Fast API response
-  No local setup required

**Cons:**
-  API costs (can add up)
-  Requires internet connection
-  Data sent to OpenAI servers

**Configuration:**

```yaml
llm_provider: "openai"

openai_api_key: "sk-..."  # Get from https://platform.openai.com/api-keys

# Optional settings
# openai_organization: "org-..."
# openai_base_url: "https://api.openai.com/v1"  # For custom endpoints
# openai_max_retries: 3

# Recommended models
generator_model: "gpt-4-turbo-preview"  # Highest quality
post_validator_model: "gpt-4"           # Excellent for QA
pre_validator_model: "gpt-3.5-turbo"    # Fast and cheap
```

**Cost Estimates** (as of 2025):

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Est. per 1000 cards |
|-------|----------------------|------------------------|---------------------|
| GPT-4 Turbo | $10 | $30 | $2-5 |
| GPT-4 | $30 | $60 | $5-15 |
| GPT-3.5 Turbo | $0.50 | $1.50 | $0.10-0.30 |

**Setup Steps:**

```bash
# 1. Get API key from https://platform.openai.com/api-keys

# 2. Add to config.yaml or .env
echo 'openai_api_key: "sk-..."' >> config.yaml

# Or use environment variable
export OPENAI_API_KEY="sk-..."

# 3. Test connection
obsidian-anki-sync test-run --count 1
```

---

### 3. Anthropic (Claude) (Recommended for Reasoning)

**Description**: Use Anthropic's Claude models via API

**Pros:**
-  Excellent reasoning capabilities
-  Long context window (200k tokens for Claude 3)
-  Strong instruction following
-  Good at structured output
-  Competitive pricing

**Cons:**
-  API costs
-  Requires internet connection
-  No native JSON mode (uses system prompt)

**Configuration:**

```yaml
llm_provider: "anthropic"  # or "claude"

anthropic_api_key: "sk-ant-..."  # Get from https://console.anthropic.com/

# Optional settings
# anthropic_base_url: "https://api.anthropic.com"
# anthropic_api_version: "2023-06-01"
# anthropic_max_retries: 3

# Recommended models
generator_model: "claude-3-opus-20240229"    # Highest quality
post_validator_model: "claude-3-sonnet-20240229"  # Good balance
pre_validator_model: "claude-3-haiku-20240307"    # Fast and cheap
```

**Cost Estimates** (as of 2025):

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Est. per 1000 cards |
|-------|----------------------|------------------------|---------------------|
| Claude 3 Opus | $15 | $75 | $5-12 |
| Claude 3 Sonnet | $3 | $15 | $1-3 |
| Claude 3 Haiku | $0.25 | $1.25 | $0.05-0.20 |

**Setup Steps:**

```bash
# 1. Get API key from https://console.anthropic.com/

# 2. Add to config.yaml
echo 'anthropic_api_key: "sk-ant-..."' >> config.yaml

# 3. Test connection
obsidian-anki-sync test-run --count 1
```

---

### 4. OpenRouter (Recommended for Flexibility)

**Description**: Access 100+ models through a unified API

**Pros:**
-  Access to many providers (OpenAI, Anthropic, Google, etc.)
-  Single API key for all models
-  Competitive pricing
-  Easy to switch models
-  Usage credits available

**Cons:**
-  API costs
-  Requires internet connection
-  Slight latency overhead

**Configuration:**

```yaml
llm_provider: "openrouter"

openrouter_api_key: "sk-or-..."  # Get from https://openrouter.ai/keys

# Optional settings
# openrouter_site_url: "https://your-site.com"
# openrouter_site_name: "Your App Name"

# You can use models from any provider
generator_model: "anthropic/claude-3-opus"
post_validator_model: "openai/gpt-4"
pre_validator_model: "google/gemini-pro"
```

**Popular Models on OpenRouter:**

| Model | Provider | Quality | Cost (per 1M tokens) |
|-------|----------|---------|----------------------|
| `anthropic/claude-3-opus` | Anthropic | Excellent | $15/$75 |
| `openai/gpt-4-turbo` | OpenAI | Excellent | $10/$30 |
| `google/gemini-pro-1.5` | Google | Very Good | $3.50/$10.50 |
| `meta-llama/llama-3-70b` | Meta | Good | $0.90/$0.90 |
| `mistralai/mixtral-8x7b` | Mistral | Good | $0.60/$0.60 |

**Setup Steps:**

```bash
# 1. Get API key from https://openrouter.ai/keys

# 2. Add to config.yaml
echo 'openrouter_api_key: "sk-or-..."' >> config.yaml

# 3. Browse models at https://openrouter.ai/models

# 4. Test connection
obsidian-anki-sync test-run --count 1
```

#### Structured Output Compatibility

- Some OpenRouter-hosted models (notably `qwen/qwen-2.5-72b-instruct` and `qwen/qwen-2.5-32b-instruct`) still return empty completions when `response_format.type="json_schema"` is enabled, despite marketing claims about JSON support ([OpenRouter model spec, retrieved 2025-11-18](https://openrouter.ai/qwen/qwen-2.5-72b-instruct)).
- The sync engine now detects this condition and **retries once without JSON schema**, logging `structured_output_retry_without_schema`. The fallback keeps QA extraction running without manual config changes.
- If both the structured call and the fallback fail, the CLI surfaces a `empty_response` error so you can switch models or rerun later. Monitor `.logs/sync.log` for the warning to understand when the fallback triggered.

---

### 5. LM Studio (Recommended for Local GUI)

**Description**: Run local models with a user-friendly GUI

**Pros:**
-  Completely private
-  No API costs
-  Visual model management
-  Easy setup
-  Works offline

**Cons:**
-  Requires powerful hardware
-  Manual model selection
-  Windows/Mac only

**Configuration:**

```yaml
llm_provider: "lm_studio"

lm_studio_base_url: "http://localhost:1234/v1"

# Model names depend on what you loaded in LM Studio
generator_model: "local-model"
post_validator_model: "local-model"
pre_validator_model: "local-model"
```

**Setup Steps:**

1. Download LM Studio from https://lmstudio.ai/
2. Install and launch LM Studio
3. Download a model (e.g., Qwen 2.5 14B)
4. Click "Start Server" in LM Studio
5. Configure `config.yaml` with model name
6. Test sync

---

## Choosing the Right Provider

### Decision Tree

```
Do you need the HIGHEST quality?
 Yes  OpenAI (GPT-4) or Anthropic (Claude 3 Opus)
 No

Is privacy a top concern?
 Yes  Ollama (local) or LM Studio
 No

Do you want to minimize costs?
 Yes  Ollama (local, free) or OpenRouter (cheap models)
 No

Do you want access to many models?
 Yes  OpenRouter
 No  OpenAI or Anthropic
```

### Use Case Recommendations

#### 1. Personal Use (Privacy-First)
```yaml
llm_provider: "ollama"
generator_model: "qwen3:32b"
post_validator_model: "qwen3:14b"
pre_validator_model: "qwen3:8b"
```
**Why**: Free, private, works offline

#### 2. Professional Use (Quality-First)
```yaml
llm_provider: "openai"
generator_model: "gpt-4-turbo-preview"
post_validator_model: "gpt-4"
pre_validator_model: "gpt-3.5-turbo"
```
**Why**: Best quality, reliable, fast

#### 3. Budget-Conscious (Cost-Optimized)
```yaml
llm_provider: "anthropic"
generator_model: "claude-3-sonnet-20240229"
post_validator_model: "claude-3-haiku-20240307"
pre_validator_model: "claude-3-haiku-20240307"
```
**Why**: Good balance of cost and quality

#### 4. Experimenter (Model Explorer)
```yaml
llm_provider: "openrouter"
generator_model: "anthropic/claude-3-opus"
post_validator_model: "google/gemini-pro-1.5"
pre_validator_model: "mistralai/mixtral-8x7b"
```
**Why**: Access to all models, easy to switch

---

## Environment Variables

For security, API keys can be set via environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."

# Ollama Cloud
export OLLAMA_API_KEY="your-key"
```

Or use a `.env` file:

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```

---

## Troubleshooting

### Connection Issues

```bash
# Test provider connection
obsidian-anki-sync test-run --count 1

# Check logs
tail -f .logs/sync.log | grep provider
```

### API Key Not Working

```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Or check config
grep api_key config.yaml

# Test directly with curl
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Model Not Found

```bash
# List available models
obsidian-anki-sync models

# For Ollama
ollama list

# For OpenRouter
curl https://openrouter.ai/api/v1/models
```

### Slow Performance

- **OpenAI/Anthropic**: Check your internet connection
- **Ollama**: Ensure GPU is being used (`nvidia-smi` or Activity Monitor)
- **LM Studio**: Use smaller models or reduce max_tokens

---

## Advanced Configuration

### Mixed Provider Setup

Use different providers for different roles:

```yaml
# Not directly supported, but achievable with custom config

# Option 1: Use OpenRouter with different models
llm_provider: "openrouter"
generator_model: "anthropic/claude-3-opus"  # Expensive but best
post_validator_model: "openai/gpt-3.5-turbo"  # Cheap for validation
pre_validator_model: "google/gemini-flash"  # Very cheap

# Option 2: Switch providers manually for different runs
# llm_provider: "openai"  # For high-quality generation
# llm_provider: "ollama"  # For fast testing
```

### Custom Base URLs

For self-hosted or proxied endpoints:

```yaml
llm_provider: "openai"
openai_base_url: "https://your-proxy.com/v1"
openai_api_key: "your-key"
```

---

## Cost Optimization Tips

1. **Use smaller models for validation**: Pre-validator doesn't need GPT-4
2. **Lower temperature**: Reduces randomness and token usage
3. **Reduce max_tokens**: Limit response length
4. **Use Ollama for testing**: Free local testing before production
5. **Batch processing**: Process multiple cards in one session
6. **Monitor usage**: Track API costs in provider dashboards

---

## Security Best Practices

1. **Never commit API keys to git**
   ```bash
   echo "config.yaml" >> .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use environment variables**
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. **Rotate keys regularly**

4. **Use read-only keys** when possible

5. **Monitor API usage** in provider dashboards

---

## Support

For provider-specific issues:

- **Ollama**: https://github.com/ollama/ollama/issues
- **OpenAI**: https://help.openai.com/
- **Anthropic**: https://console.anthropic.com/support
- **OpenRouter**: https://openrouter.ai/docs
- **LM Studio**: https://lmstudio.ai/support

For integration issues:

- File issue: [GitHub Issues](https://github.com/po4yka/obsidian-to-anki/issues)
- Check logs: `.logs/sync.log`

---

**Last Updated**: 2025-11-10
**Version**: 2.0.0
