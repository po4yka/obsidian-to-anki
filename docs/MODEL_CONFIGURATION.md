# Unified Model Configuration Guide

**Last Updated**: 2025-01-12
**Feature**: Single place to configure LLM models for all agents

## Overview

All LLM model configuration is now **unified in one place** in your `config.yaml`:

```yaml
# Set this ONCE for all agents
default_llm_model: "openrouter/polaris-alpha"
```

That's it! All 5 agents will use this model.

## Quick Start

### Step 1: Set Default Model

Edit `config.yaml`:

```yaml
# Unified model configuration
default_llm_model: "openrouter/polaris-alpha"

# Leave individual agent models EMPTY to use default
pydantic_ai_pre_validator_model: ""
pydantic_ai_generator_model: ""
pydantic_ai_post_validator_model: ""
context_enrichment_model: ""
memorization_quality_model: ""
```

### Step 2: Set API Key

Add to `.env` or set environment variable:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

### Step 3: Done!

All agents now use `openrouter/polaris-alpha`.

## How It Works

### Default Model

The `default_llm_model` is used by **all agents** unless overridden:

- Pre-Validator Agent
- Generator Agent
- Post-Validator Agent
- Context Enrichment Agent
- Memorization Quality Agent
- Card Splitting Agent (future)

### Per-Agent Overrides (Optional)

If you want a specific agent to use a different model, set its individual config:

```yaml
# All use default EXCEPT generator
default_llm_model: "openai/gpt-4o-mini"
pydantic_ai_generator_model: "anthropic/claude-3-5-sonnet"  # Override
```

**Rule**: Empty string (`""`) = use `default_llm_model`

## Configuration Method

The system uses a helper method:

```python
# In config.py
def get_model_for_agent(self, agent_type: str) -> str:
    """Get model for specific agent, falling back to default."""
    agent_model = self.context_enrichment_model  # Example
    return agent_model if agent_model else self.default_llm_model
```

**Called in agents**:
```python
model_name = state["config"].get_model_for_agent("context_enrichment")
model = create_openrouter_model_from_env(model_name=model_name)
```

## Supported Models

### OpenRouter (Recommended)

OpenRouter provides access to many models through one API:

```yaml
default_llm_model: "openrouter/polaris-alpha"  # OpenRouter's own model
default_llm_model: "openai/gpt-4o"  # OpenAI via OpenRouter
default_llm_model: "openai/gpt-4o-mini"  # Cheaper option
default_llm_model: "anthropic/claude-3-5-sonnet"  # Anthropic via OpenRouter
default_llm_model: "google/gemini-pro-1.5"  # Google via OpenRouter
default_llm_model: "meta-llama/llama-3.1-70b-instruct"  # Meta via OpenRouter
default_llm_model: "mistralai/mistral-large"  # Mistral via OpenRouter
```

**Advantages**:
- One API key for all models
- Easy model switching
- Competitive pricing
- No vendor lock-in

**Browse models**: https://openrouter.ai/models

### Direct Providers

You can also use models directly:

```yaml
# OpenAI Direct
llm_provider: "openai"
default_llm_model: "gpt-4o-mini"
openai_api_key: "${OPENAI_API_KEY}"

# Anthropic Direct
llm_provider: "anthropic"
default_llm_model: "claude-3-5-sonnet-20241022"
anthropic_api_key: "${ANTHROPIC_API_KEY}"
```

## Common Configurations

### Configuration 1: Single Model for Everything

**Use case**: Simplest setup, consistent quality

```yaml
default_llm_model: "openrouter/polaris-alpha"
pydantic_ai_pre_validator_model: ""
pydantic_ai_generator_model: ""
pydantic_ai_post_validator_model: ""
context_enrichment_model: ""
memorization_quality_model: ""
```

---

### Configuration 2: High-Quality Generation, Fast Validation

**Use case**: Balance quality with speed

```yaml
default_llm_model: "openai/gpt-4o-mini"  # Fast default
pydantic_ai_generator_model: "anthropic/claude-3-5-sonnet"  # High quality
pydantic_ai_pre_validator_model: ""  # Use fast default
pydantic_ai_post_validator_model: ""  # Use fast default
context_enrichment_model: ""  # Use fast default
memorization_quality_model: ""  # Use fast default
```

---

### Configuration 3: All Validation Disabled

**Use case**: Fast mode for testing

```yaml
default_llm_model: "openrouter/polaris-alpha"
enable_context_enrichment: false
enable_memorization_quality: false
```

---

### Configuration 4: Maximum Quality

**Use case**: Production quality

```yaml
default_llm_model: "anthropic/claude-3-5-sonnet"  # Best quality
pydantic_ai_pre_validator_model: "openai/gpt-4o"
pydantic_ai_generator_model: "anthropic/claude-3-5-sonnet"
pydantic_ai_post_validator_model: "openai/gpt-4o"
context_enrichment_model: "anthropic/claude-3-5-sonnet"
memorization_quality_model: "openai/gpt-4o"
enable_context_enrichment: true
enable_memorization_quality: true
```

## Model Selection Guide

### By Use Case

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **General use** | `openrouter/polaris-alpha` | Good balance |
| **High quality** | `anthropic/claude-3-5-sonnet` | Best reasoning, accuracy |
| **Fast responses** | `openai/gpt-4o-mini` | Quick, efficient |
| **Speed** | `openai/gpt-4o-mini` | Fast responses |
| **Complex reasoning** | `anthropic/claude-3-5-sonnet` | Superior logic |

### By Agent

| Agent | Best Model | Why |
|-------|------------|-----|
| **Pre-Validator** | `gpt-4o-mini` | Simple checks, fast |
| **Generator** | `claude-3-5-sonnet` | Main quality bottleneck |
| **Post-Validator** | `gpt-4o-mini` | Validation logic, fast |
| **Context Enrichment** | `polaris-alpha` | Good creativity balance |
| **Memorization Quality** | `gpt-4o-mini` | Pattern matching, fast |

## Migration from Old Config

### Before (Scattered Models)

```yaml
pydantic_ai_pre_validator_model: "openai/gpt-4o-mini"
pydantic_ai_generator_model: "anthropic/claude-3-5-sonnet"
pydantic_ai_post_validator_model: "openai/gpt-4o-mini"
context_enrichment_model: "openai/gpt-4o-mini"
memorization_quality_model: "openai/gpt-4o-mini"
```

**Problem**: Repetitive, error-prone, hard to change

### After (Unified)

```yaml
default_llm_model: "openrouter/polaris-alpha"
pydantic_ai_generator_model: "anthropic/claude-3-5-sonnet"  # Only override if needed
# All others empty = use default
```

**Benefits**: DRY, easy to change, clear intent

## Environment Variables

### OpenRouter

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### In .env File

```
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Troubleshooting

### Issue: All Agents Using Wrong Model

**Check**:
```yaml
default_llm_model: "openrouter/polaris-alpha"  # Is this set?
```

**Solution**: Set `default_llm_model` in config.yaml

---

### Issue: One Agent Using Wrong Model

**Check**:
```yaml
context_enrichment_model: "wrong-model"  # Is this overriding?
```

**Solution**: Set to empty string to use default:
```yaml
context_enrichment_model: ""
```

---

### Issue: Model Not Found

**Check model name format**:
- OpenRouter: `openrouter/polaris-alpha` ✓
- OpenRouter (other): `openai/gpt-4o` ✓
- Direct OpenAI: `gpt-4o` ✓
- Wrong: `polaris-alpha` ✗ (missing provider)

**Solution**: Use correct format for your provider

---

### Issue: API Key Not Working

**Check environment variable**:
```bash
echo $OPENROUTER_API_KEY  # Should print key
```

**Check config.yaml**:
```yaml
openrouter_api_key: "${OPENROUTER_API_KEY}"  # Uses env var
# OR
openrouter_api_key: "sk-or-..."  # Direct (not recommended)
```

## Programmatic Access

### Getting Model for Agent

```python
from obsidian_anki_sync.config import load_config

config = load_config()
model_name = config.get_model_for_agent("generator")
print(f"Generator using: {model_name}")
```

### Override in Code

```python
# Override config
config.pydantic_ai_generator_model = "anthropic/claude-3-5-sonnet"
model_name = config.get_model_for_agent("generator")
# Returns "anthropic/claude-3-5-sonnet"
```

## Best Practices

### ✅ DO

1. **Set default_llm_model** - Single source of truth
2. **Use environment variables** - Keep API keys secret
3. **Start with one model** - Simplify before optimizing
4. **Test with cheap model** - Use gpt-4o-mini for development
5. **Monitor costs** - Track usage on OpenRouter dashboard

### ❌ DON'T

1. **Don't set all individual models** - Defeats the purpose
2. **Don't commit API keys** - Use .env or environment variables
3. **Don't use expensive models everywhere** - Optimize per agent
4. **Don't skip validation** - Check model names are correct
5. **Don't assume pricing** - Check current rates

## Summary

### Old Way (Complex)
```yaml
# 6 different places to maintain
pydantic_ai_pre_validator_model: "openai/gpt-4o-mini"
pydantic_ai_generator_model: "openai/gpt-4o-mini"
pydantic_ai_post_validator_model: "openai/gpt-4o-mini"
context_enrichment_model: "openai/gpt-4o-mini"
memorization_quality_model: "openai/gpt-4o-mini"
card_splitting_model: "openai/gpt-4o-mini"
```

### New Way (Simple)
```yaml
# 1 place to maintain
default_llm_model: "openrouter/polaris-alpha"
```

**Result**: Easier configuration, fewer errors, faster model switching! 🎉

## Related Documentation

- [LangGraph Integration](LANGGRAPH_INTEGRATION_COMPLETE.md) - Complete pipeline guide
- [Agent Summary](AGENT_SUMMARY.md) - All agents overview
- [config.yaml.example](../config.yaml.example) - Full configuration example

## Support

- **OpenRouter Models**: https://openrouter.ai/models
- **Issues**: https://github.com/po4yka/obsidian-to-anki/issues
- **Configuration**: See `config.yaml.example`

---

**Feature Status**: ✅ Complete
**Default Model**: `openrouter/polaris-alpha`
**Recommended**: Use default for all agents unless specific needs require optimization
