# LLM Provider System

The Obsidian to Anki sync service supports multiple LLM providers through a unified provider system. This allows you to choose the best provider for your needs while maintaining consistent behavior across the application.

## Supported Providers

### 1. Ollama (Local & Cloud)

**Best for:** Privacy, offline usage, unlimited requests, Apple Silicon optimization

#### Local Ollama
```yaml
llm_provider: "ollama"
ollama_base_url: "http://localhost:11434"
pre_validator_model: "qwen3:8b"
generator_model: "qwen3:32b"
post_validator_model: "qwen3:14b"
```

**Setup:**
1. Install Ollama: https://ollama.ai
2. Start Ollama: `ollama serve`
3. Pull models:
   ```bash
   ollama pull qwen3:8b
   ollama pull qwen3:14b
   ollama pull qwen3:32b
   ```

#### Ollama Cloud
```yaml
llm_provider: "ollama"
ollama_base_url: "https://api.ollama.com"
ollama_api_key: "your-api-key"  # Or set OLLAMA_API_KEY env var
```

**Recommended Models:**
- Pre-validator: `qwen3:8b`, `llama3:8b`, `phi3:mini`
- Generator: `qwen3:32b`, `llama3:70b`, `mixtral:8x7b`
- Post-validator: `qwen3:14b`, `llama3:13b`, `mistral:7b`

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
- OpenAI-compatible API
- Easy model switching through GUI
- Model performance monitoring
- GGUF model support

---

### 3. OpenRouter

**Best for:** State-of-the-art models, cloud scalability, no local hardware required

```yaml
llm_provider: "openrouter"
openrouter_api_key: "your-api-key"  # Or set OPENROUTER_API_KEY env var
openrouter_base_url: "https://openrouter.ai/api/v1"
pre_validator_model: "openai/gpt-3.5-turbo"
generator_model: "anthropic/claude-3-opus"
post_validator_model: "anthropic/claude-3-sonnet"
```

**Setup:**
1. Sign up at https://openrouter.ai
2. Get API key from dashboard
3. Set `OPENROUTER_API_KEY` environment variable
4. Choose models from OpenRouter's model list

**Popular Models:**
- OpenAI: `openai/gpt-4`, `openai/gpt-3.5-turbo`
- Anthropic: `anthropic/claude-3-opus`, `anthropic/claude-3-sonnet`
- Meta: `meta-llama/llama-3-70b-instruct`
- Mistral: `mistralai/mixtral-8x7b-instruct`

**Optional Configuration:**
```yaml
openrouter_site_url: "https://yourdomain.com"
openrouter_site_name: "Your App Name"
```

---

## Configuration

### Single Point of Specification

All providers share common configuration with provider-specific extensions:

```yaml
# Provider selection
llm_provider: "ollama"  # or "lm_studio" or "openrouter"

# Common settings (all providers)
llm_temperature: 0.2
llm_top_p: 0.3
llm_timeout: 120.0
llm_max_tokens: 2048

# Provider-specific settings
ollama_base_url: "http://localhost:11434"
ollama_api_key: null  # Only for Ollama Cloud

lm_studio_base_url: "http://localhost:1234/v1"

openrouter_api_key: "your-key"
openrouter_base_url: "https://openrouter.ai/api/v1"

# Model specifications (adjust for your provider)
pre_validator_model: "qwen3:8b"
generator_model: "qwen3:32b"
post_validator_model: "qwen3:14b"
```

### Environment Variables

Override any setting via environment variables:

```bash
# Provider selection
export LLM_PROVIDER=lm_studio

# Common settings
export LLM_TEMPERATURE=0.3
export LLM_TIMEOUT=180.0

# Ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_API_KEY=your-key

# LM Studio
export LM_STUDIO_BASE_URL=http://localhost:1234/v1

# OpenRouter
export OPENROUTER_API_KEY=your-key

# Model specifications
export PRE_VALIDATOR_MODEL=qwen3:8b
export GENERATOR_MODEL=qwen3:32b
export POST_VALIDATOR_MODEL=qwen3:14b
```

---

## Provider Comparison

| Feature | Ollama (Local) | LM Studio | OpenRouter | Ollama Cloud |
|---------|---------------|-----------|------------|--------------|
| **Cost** | Free | Free | Pay per token | Pay per usage |
| **Privacy** | 100% private | 100% private | Cloud-based | Cloud-based |
| **Offline** | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| **Hardware** | Required | Required | Not required | Not required |
| **Setup** | Easy (CLI) | Easy (GUI) | Easiest (API key) | Easy (API key) |
| **Model Selection** | Large library | Large library | 100+ models | Growing library |
| **Performance** | Hardware-dependent | Hardware-dependent | Consistent | Consistent |
| **API Latency** | Very low | Very low | Medium | Medium |
| **Best For** | Power users, privacy | GUI preference | Cloud access, SOTA models | Ollama + cloud |

---

## Model Selection Guidelines

### Pre-Validator Agent
- **Role:** Fast structural validation
- **Requirements:** Lightweight, quick inference
- **Recommended Size:** 7-8B parameters
- **Examples:**
  - Ollama: `qwen3:8b`, `llama3:8b`
  - LM Studio: `Meta-Llama-3-8B-Instruct-GGUF`
  - OpenRouter: `openai/gpt-3.5-turbo`

### Generator Agent
- **Role:** High-quality card generation
- **Requirements:** Strong reasoning, creativity
- **Recommended Size:** 30-70B parameters
- **Examples:**
  - Ollama: `qwen3:32b`, `llama3:70b`
  - LM Studio: `Qwen2.5-32B-Instruct-GGUF`
  - OpenRouter: `anthropic/claude-3-opus`, `openai/gpt-4`

### Post-Validator Agent
- **Role:** Semantic validation, error correction
- **Requirements:** Good reasoning, medium speed
- **Recommended Size:** 13-14B parameters
- **Examples:**
  - Ollama: `qwen3:14b`, `llama3:13b`
  - LM Studio: `Qwen2.5-14B-Instruct-GGUF`
  - OpenRouter: `anthropic/claude-3-sonnet`

---

## Hardware Requirements

### Local Providers (Ollama, LM Studio)

**Minimum Configuration:**
- Pre-validator (8B): 8 GB RAM/VRAM
- Generator (32B): 24 GB RAM/VRAM
- Post-validator (14B): 12 GB RAM/VRAM

**Recommended Configuration:**
- **Mac:** M1 Pro/Max/Ultra or M2 Pro/Max/Ultra with 32+ GB unified memory
- **PC:** NVIDIA GPU with 24+ GB VRAM, 64+ GB system RAM
- **Linux:** Same as PC, better CUDA support

**Sequential Mode** (lower memory usage):
```yaml
agent_execution_mode: "sequential"  # Loads one model at a time
```
- Peak memory: ~24 GB (largest model only)
- Throughput: Lower (sequential processing)

**Parallel Mode** (higher throughput):
```yaml
agent_execution_mode: "parallel"  # All models loaded simultaneously
```
- Peak memory: ~44 GB (all models loaded)
- Throughput: Higher (concurrent processing)

### Cloud Providers (OpenRouter, Ollama Cloud)
- **Hardware:** None required
- **Internet:** Stable connection required
- **Latency:** 200-2000ms per request

---

## Troubleshooting

### Connection Issues

**Ollama:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve

# Check available models
ollama list
```

**LM Studio:**
1. Ensure local server is started in LM Studio
2. Check server URL matches configuration
3. Verify model is loaded and ready

**OpenRouter:**
```bash
# Test API key
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

### Model Not Found

**Ollama:**
```bash
# Pull missing model
ollama pull qwen3:8b

# List available models
ollama list
```

**LM Studio:**
1. Download model through LM Studio UI
2. Load model before starting server
3. Use exact model identifier from LM Studio

**OpenRouter:**
- Check model availability: https://openrouter.ai/models
- Verify model ID format: `provider/model-name`

### Timeout Issues

Increase timeout for large models:
```yaml
llm_timeout: 300.0  # 5 minutes
```

Or via environment:
```bash
export LLM_TIMEOUT=300.0
```

### Memory Issues (Local Providers)

**Option 1:** Use sequential execution
```yaml
agent_execution_mode: "sequential"
```

**Option 2:** Use smaller models
```yaml
pre_validator_model: "phi3:mini"  # ~4B
generator_model: "qwen3:14b"      # ~14B
post_validator_model: "mistral:7b" # ~7B
```

**Option 3:** Switch to cloud provider
```yaml
llm_provider: "openrouter"
```

---

## Migration Guide

### From Legacy OpenRouter-only Setup

**Old configuration:**
```yaml
openrouter_api_key: "..."
openrouter_model: "openai/gpt-4"
use_agent_system: false
```

**New configuration (keeps OpenRouter):**
```yaml
llm_provider: "openrouter"
openrouter_api_key: "..."
use_agent_system: true
pre_validator_model: "openai/gpt-3.5-turbo"
generator_model: "openai/gpt-4"
post_validator_model: "openai/gpt-3.5-turbo"
```

### From Ollama-only Setup

**Old configuration:**
```yaml
ollama_base_url: "http://localhost:11434"
use_agent_system: true
```

**New configuration (no changes needed):**
```yaml
llm_provider: "ollama"  # Now explicit
ollama_base_url: "http://localhost:11434"
use_agent_system: true
```

The old configuration remains backward compatible!

---

## Advanced Usage

### Custom Provider Endpoints

```yaml
ollama_base_url: "http://custom-server:11434"
lm_studio_base_url: "http://192.168.1.100:1234/v1"
openrouter_base_url: "https://custom-proxy.com/v1"
```

### Per-Agent Temperature Control

```yaml
pre_validator_temperature: 0.0   # Deterministic validation
generator_temperature: 0.3       # Some creativity
post_validator_temperature: 0.0  # Deterministic validation
```

### Programmatic Provider Creation

```python
from obsidian_anki_sync.providers import ProviderFactory

# Create provider from config
provider = ProviderFactory.create_from_config(config)

# Or create directly
provider = ProviderFactory.create_provider(
    "ollama",
    base_url="http://localhost:11434"
)

# Use with orchestrator
orchestrator = AgentOrchestrator(config, provider=provider)
```

---

## API Reference

See the [Provider API Documentation](../src/obsidian_anki_sync/providers/README.md) for detailed API information.

---

## Contributing

To add a new provider:

1. Create provider class inheriting from `BaseLLMProvider`
2. Implement required methods: `generate()`, `check_connection()`, `list_models()`
3. Add to `ProviderFactory.PROVIDER_MAP`
4. Update configuration schema
5. Add documentation and examples
6. Submit PR with tests

Example:
```python
from obsidian_anki_sync.providers.base import BaseLLMProvider

class MyCustomProvider(BaseLLMProvider):
    def generate(self, model, prompt, **kwargs):
        # Implementation
        pass

    def check_connection(self):
        # Implementation
        pass

    def list_models(self):
        # Implementation
        pass
```

---

## Support

- **Documentation:** https://github.com/po4yka/obsidian-to-anki
- **Issues:** https://github.com/po4yka/obsidian-to-anki/issues
- **Discussions:** https://github.com/po4yka/obsidian-to-anki/discussions

---

## License

Same as parent project.
