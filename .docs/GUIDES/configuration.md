# Configuration Guide

This guide covers all configuration options for the Obsidian to Anki sync service, including model presets, provider setup, agent configuration, and advanced options.

## Quick Configuration

For most users, this minimal configuration is sufficient:

```yaml
# Basic setup - copy from config.example.yaml
vault_path: "~/Documents/ObsidianVault"
source_dir: "Notes"
anki_deck_name: "My Deck"

# Choose your provider
llm_provider: "openrouter" # or "ollama", "openai", "anthropic"
openrouter_api_key: "sk-or-v1-..." # pragma: allowlist secret

# Use optimized presets
model_preset: "balanced" # cost_effective|balanced|high_quality|fast

# Enable agent system
use_langgraph: true
use_pydantic_ai: true
```

**TIP**: Start with `config.example.yaml` and modify for your needs.

## Configuration Files

### Primary Configuration (`config.yaml`)

The main configuration file supports YAML format with environment variable overrides.

**Location**: Project root or `~/.config/obsidian-anki-sync/config.yaml`

### Environment Variables

Set sensitive information via environment variables:

```bash
# API Keys (recommended)
export OPENROUTER_API_KEY="sk-or-v1-..."  # pragma: allowlist secret
export OPENAI_API_KEY="sk-..."  # pragma: allowlist secret
export ANTHROPIC_API_KEY="sk-ant-..."  # pragma: allowlist secret

# Provider settings
export OLLAMA_BASE_URL="http://localhost:11434"
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"
```

### Example Configurations

#### Ollama (Local, Private)

```yaml
llm_provider: "ollama"
ollama_base_url: "http://localhost:11434"
model_preset: "balanced"

# Agent models (Ollama naming)
pre_validator_model: "qwen3:8b"
generator_model: "qwen3:32b"
post_validator_model: "qwen3:14b"
```

#### OpenRouter (Cloud, Flexible)

```yaml
llm_provider: "openrouter"
openrouter_api_key: "${OPENROUTER_API_KEY}"
model_preset: "balanced"

# Uses OpenRouter model names
pre_validator_model: "openai/gpt-4o-mini"
generator_model: "anthropic/claude-3-5-sonnet-20241022"
post_validator_model: "openai/gpt-4o"
```

#### OpenAI (Reliable, Consistent)

```yaml
llm_provider: "openai"
openai_api_key: "${OPENAI_API_KEY}"
model_preset: "balanced"

# OpenAI model names
pre_validator_model: "gpt-4o-mini"
generator_model: "gpt-4o"
post_validator_model: "gpt-4o-mini"
```

## Model Presets

The system provides optimized model combinations for different use cases:

### Cost-Effective Preset

```yaml
model_preset: "cost_effective"
```

**Best for**: Budget-conscious users, high volume processing

-   **Pre-validator**: GPT-4o-mini / Claude-3-Haiku
-   **Generator**: Claude-3-5-Sonnet / GPT-4o
-   **Post-validator**: GPT-4o-mini / Claude-3-Haiku
-   **Cost**: ~60% of balanced preset
-   **Quality**: Good for most use cases

### Balanced Preset (Recommended)

```yaml
model_preset: "balanced"
```

**Best for**: Good quality with reasonable cost

-   **Pre-validator**: GPT-4o-mini / Claude-3-Haiku
-   **Generator**: Claude-3-5-Sonnet / GPT-4o
-   **Post-validator**: GPT-4o / Claude-3-5-Sonnet
-   **Cost**: Moderate
-   **Quality**: High quality output

### High Quality Preset

```yaml
model_preset: "high_quality"
```

**Best for**: Maximum quality, cost not a concern

-   **Pre-validator**: GPT-4o / Claude-3-5-Sonnet
-   **Generator**: Claude-3-5-Sonnet / GPT-4o
-   **Post-validator**: o1-preview / Claude-3-5-Sonnet
-   **Cost**: High
-   **Quality**: Best possible output

### Fast Preset

```yaml
model_preset: "fast"
```

**Best for**: Speed over quality, testing, development

-   **Pre-validator**: GPT-4o-mini
-   **Generator**: GPT-4o-mini
-   **Post-validator**: GPT-4o-mini
-   **Cost**: Low
-   **Quality**: Sufficient for testing

## Agent Configuration

### Basic Agent Setup

```yaml
# Enable agent system
use_langgraph: true # Use LangGraph for workflow orchestration
use_pydantic_ai: true # Use PydanticAI for structured outputs

# Model overrides (optional, uses preset if not set)
pre_validator_model: "openai/gpt-4o-mini"
generator_model: "anthropic/claude-3-5-sonnet-20241022"
post_validator_model: "openai/gpt-4o"
```

### Advanced Agent Options

```yaml
agent_config:
    # Retry settings
    max_retries: 3
    retry_delay: 2.0

    # Performance tuning
    timeout: 60
    temperature: 0.1
    max_tokens: 4000

    # Workflow settings
    enable_checkpointing: true
    max_workflow_time: 300
```

### Agent-Specific Overrides

```yaml
# Override specific agent models
pre_validator_model: "qwen/qwen-2.5-32b-instruct" # Fast validation
generator_model: "anthropic/claude-3-5-sonnet-20241022" # High quality
post_validator_model: "openai/gpt-4o" # Reliable checking

# Context enrichment (optional agent)
context_enrichment_model: "minimax/minimax-m2"
memorization_quality_model: "moonshotai/kimi-k2"
card_splitting_model: "qwen/qwen-2.5-32b-instruct"
```

### Highlight Agent

```yaml
# Highlight agent (runs when notes fail pre-validation)
enable_highlight_agent: true
highlight_max_candidates: 3
highlight_model: "" # Empty string uses preset/default model
```

When enabled, the highlight agent analyzes incomplete notes and surfaces candidate Q&A pairs plus actionable suggestions. The output is included in sync logs so you know exactly which sections need attention instead of a generic “validation failed” message.

## Sync Configuration

### Vault and Deck Settings

```yaml
# Obsidian vault
vault_path: "~/Documents/ObsidianVault"
source_dir: "Notes" # Subdirectory to scan
ignore_patterns: # Files to skip
    - "*.tmp"
    - "*.bak"
    - ".obsidian/*"

# Anki settings
anki_deck_name: "My Deck"
anki_model_name: "Basic" # Anki note type
anki_base_url: "http://localhost:8765" # AnkiConnect default

# Sync behavior
sync_mode: "incremental" # incremental|full
dry_run: false # Preview changes
force_sync: false # Ignore timestamps
max_concurrent_requests: 5 # Parallel operations
```

### Change Detection

```yaml
# Content hashing
content_hash_algorithm: "sha256"
hash_buffer_size: 65536

# File monitoring
use_file_timestamps: true
timestamp_tolerance: 1.0 # Seconds tolerance

# Performance
batch_size: 50 # Cards per batch
max_concurrent_notes: 3 # Parallel note processing
```

## Validation Configuration

### AI Validation Settings

```yaml
# Enable AI-powered validation
enable_ai_validation: true
ai_validation_model: "openai/gpt-4o-mini"
ai_validation_temperature: 0.1

# Specific validations to enable
ai_validation:
    enable_code_language_detection: true
    enable_bilingual_titles: true
    enable_list_formatting: true
```

### Basic Validation

```yaml
# Content validation
min_qa_score: 0.8 # Quality threshold
require_frontmatter: true # YAML headers required
validate_links: true # Check wiki links

# Format validation
max_line_length: 88
enforce_apf_format: true
strict_tag_taxonomy: true
```

## Provider-Specific Configuration

### OpenRouter

```yaml
llm_provider: "openrouter"
openrouter_api_key: "${OPENROUTER_API_KEY}"
openrouter_base_url: "https://openrouter.ai/api/v1"

# Optional settings
openrouter_site_url: "https://yourdomain.com"
openrouter_site_name: "Your App Name"
openrouter_app_name: "Obsidian to Anki Sync"
```

### Ollama

```yaml
llm_provider: "ollama"
ollama_base_url: "http://localhost:11434"

# For Ollama Cloud
ollama_api_key: "${OLLAMA_API_KEY}"
ollama_base_url: "https://api.ollama.com"
```

### LM Studio

```yaml
llm_provider: "lm_studio"
lm_studio_base_url: "http://localhost:1234/v1"
# Uses OpenAI-compatible API
```

### OpenAI

```yaml
llm_provider: "openai"
openai_api_key: "${OPENAI_API_KEY}"
openai_base_url: "https://api.openai.com/v1"
openai_organization: "" # Optional
```

### Anthropic

```yaml
llm_provider: "anthropic"
anthropic_api_key: "${ANTHROPIC_API_KEY}"
anthropic_base_url: "https://api.anthropic.com"
```

## APF Configuration

### Card Generation Settings

```yaml
apf_config:
    version: "apf-v2.1"
    enable_validation: true
    strict_mode: false
    max_line_length: 88

    # Card type preferences
    default_card_type: "simple" # simple|missing|draw
    enable_bilingual_cards: true
    enable_draw_cards: false

    # Quality settings
    min_card_quality_score: 0.8
    enable_auto_fixes: true
```

### Language and Encoding

```yaml
# Text processing
default_language: "en"
supported_languages: ["en", "ru"]
encoding: "utf-8"

# Markdown processing
enable_wiki_links: true
enable_embeds: false
sanitize_html: true
```

## Performance and Caching

### Caching Configuration

```yaml
cache_config:
    enable_result_cache: true
    cache_ttl: 3600 # 1 hour
    max_cache_size: 1000
    cache_dir: "~/.cache/obsidian-anki-sync"

    # Cache types
    enable_llm_cache: true
    enable_validation_cache: true
    enable_apf_cache: true
```

### Performance Tuning

```yaml
performance:
    # Connection pooling
    max_connections: 10
    connection_timeout: 30

    # Rate limiting
    requests_per_minute: 60
    burst_limit: 10

    # Memory management
    max_memory_mb: 1024
    enable_gc_tuning: true
```

## Logging and Monitoring

### Logging Configuration

```yaml
logging:
    level: "INFO" # DEBUG|INFO|WARNING|ERROR
    format: "structured" # structured|simple
    file: "logs/sync.log"
    max_file_size: "10MB"
    max_files: 5

    # Log categories
    enable_api_logs: true
    enable_performance_logs: true
    enable_error_logs: true
```

### Monitoring

```yaml
monitoring:
    enable_metrics: true
    metrics_port: 9090
    enable_health_checks: true

    # External monitoring
    enable_prometheus: false
    prometheus_port: 9091
```

## Security Configuration

### API Key Management

```yaml
security:
    # Environment variable validation
    validate_api_keys: true
    require_encrypted_keys: false

    # Path security
    enable_path_validation: true
    allow_symlinks: false
    restrict_to_vault: true

    # Network security
    enable_ssl_verification: true
    timeout_hard_limit: 300
```

### Data Protection

```yaml
privacy:
    # Content handling
    enable_content_hashing: true
    hash_pii_data: false
    log_sensitive_data: false

    # Cache security
    encrypt_cache: false
    cache_retention_days: 7
```

## Advanced Configuration

### Custom Model Routing

```yaml
model_routing:
    # Route by content type
    code_content_model: "qwen/qwen-2.5-72b-instruct"
    text_content_model: "anthropic/claude-3-5-sonnet-20241022"

    # Route by complexity
    simple_content_model: "openai/gpt-4o-mini"
    complex_content_model: "anthropic/claude-3-5-sonnet-20241022"

    # Fallback chain
    fallback_models:
        - "openai/gpt-4o"
        - "anthropic/claude-3-5-sonnet-20241022"
        - "qwen/qwen-2.5-72b-instruct"
```

### Experimental Features

```yaml
experimental:
    # New agent types
    enable_enhanced_agents: false
    enable_multi_modal: false

    # Advanced sync features
    enable_bidirectional_sync: false
    enable_smart_merge: false

    # Performance features
    enable_gpu_acceleration: false
    enable_distributed_processing: false
```

## Configuration Validation

### Validate Configuration

```bash
# Check configuration syntax
obsidian-anki-sync validate --config config.yaml

# Test provider connectivity
obsidian-anki-sync ping

# Validate vault access
obsidian-anki-sync validate --vault
```

### Configuration Examples

See `config.example.yaml` for complete examples.

### Common Issues

**Configuration not found**:

```bash
# Check current directory
ls -la config.yaml

# Use explicit path
obsidian-anki-sync --config /path/to/config.yaml
```

**API key issues**:

```bash
# Check environment variable
echo $OPENROUTER_API_KEY

# Test API access
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models
```

## Related Documentation

-   **[Getting Started](../GETTING_STARTED.md)** - Quick setup guide
-   **[Provider Guide](../ARCHITECTURE/providers.md)** - Detailed provider setup
-   **[Agent System](../ARCHITECTURE/agents.md)** - Agent configuration
-   **[Validation Guide](validation.md)** - AI validation setup

---

**Version**: 1.0
**Last Updated**: November 28, 2025
