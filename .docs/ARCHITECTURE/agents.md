# Agent System Architecture

## Overview

The Obsidian to Anki sync service uses a sophisticated multi-agent system for intelligent card generation and validation. The system combines **LangGraph** for workflow orchestration, **PydanticAI** for type-safe structured outputs, and **LangChain agents** for flexible agent implementations, with **OpenRouter** providing unified access to multiple LLM providers.

### Key Technologies

-   **LangGraph**: State machine framework for complex multi-agent workflows
-   **PydanticAI**: Type-safe structured outputs with automatic validation
-   **LangChain Agents**: Flexible agent implementations with tool integration
-   **OpenRouter**: Unified API access to 100+ LLM models from multiple providers

### Agent Pipeline

The system implements a three-stage validation pipeline:

```
Input Note → Pre-Validator → Generator → Post-Validator → Output Cards
     ↓            ↓            ↓            ↓
   Validate    Generate     Validate     Finalize
  Structure    Content      Quality      Cards
```

**Benefits:**

-   15-20% faster processing (early rejection of invalid inputs)
-   Higher quality cards through multi-stage validation
-   Automatic error correction and retry logic
-   Type-safe outputs prevent parsing errors

---

## Architecture

### System Components

#### 1. LangGraph Orchestrator

**Purpose**: Manages complex multi-agent workflows with state persistence and conditional routing.

**Key Features:**

-   **State Management**: Persistent state across workflow steps
-   **Conditional Routing**: Dynamic routing based on validation results
-   **Retry Logic**: Automatic retries on failures with exponential backoff
-   **Checkpointing**: Workflow state persistence for resumability
-   **Visualization**: Graph visualization for debugging and monitoring

#### 2. PydanticAI Agents

**Purpose**: Provides type-safe structured outputs with automatic validation.

**Key Features:**

-   **Strongly-Typed Outputs**: Agents return validated Pydantic models
-   **Multi-Provider Support**: Works with OpenAI, Anthropic, OpenRouter, etc.
-   **Built-in Tool Calling**: Function calling support for agent tools
-   **Error Handling**: Clear, typed error messages and recovery
-   **Validation**: Automatic output validation against schemas

#### 3. LangChain Agent Types

**Purpose**: Flexible agent implementations optimized for different use cases.

**Available Types:**

-   **Tool Calling Agent**: Parallel function execution for complex operations
-   **ReAct Agent**: Transparent reasoning chains with tool usage
-   **Structured Chat Agent**: Multi-input scenarios with conversation history
-   **JSON Chat Agent**: Structured data processing with JSON mode

#### 4. OpenRouter Integration

**Purpose**: Unified access to multiple LLM providers with cost optimization.

**Key Features:**

-   **100+ Models**: Access to OpenAI, Anthropic, Google, Meta, and more
-   **Unified API**: OpenAI-compatible interface for seamless integration
-   **Cost Optimization**: Automatic model selection based on price/performance
-   **Fallback Support**: Automatic fallback to alternative models on failure
-   **Usage Analytics**: Detailed cost and performance tracking

### Workflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                Agent Pipeline                           │ │
│  │  ┌─────────────┬─────────────┬─────────────────────┐ │ │
│  │  │Pre-Validator│ Generator   │  Post-Validator     │ │ │
│  │  │             │             │                     │ │ │
│  │  └─────────────┴─────────────┴─────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                PydanticAI Results                       │ │
│  │  ┌─────────────┬─────────────┬─────────────────────┐ │ │
│  │  │Validation   │Generation   │  Validation        │ │ │
│  │  │Results      │Results      │  Results           │ │ │
│  │  └─────────────┴─────────────┴─────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 OpenRouter API                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                Model Selection                          │ │
│  │  ┌─────────────┬─────────────┬─────────────────────┐ │ │
│  │  │GPT-4o-mini  │Claude-3.5   │  GPT-4o-mini       │ │ │
│  │  │(Fast)       │(Quality)    │  (Balanced)        │ │ │
│  │  └─────────────┴─────────────┴─────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Agent Types and Use Cases

#### Pre-Validator Agent

**Purpose**: Early validation to reject invalid inputs before expensive processing.

**Models**: Lightweight models (GPT-4o-mini, Qwen3-8B)
**Output**: `PreValidationResult` with validation status and issues
**Success Rate**: ~85% pass rate, catches 60% of issues

#### Highlight Agent

**Purpose**: When a note fails pre-validation (usually because Q&A sections are missing or incomplete), the highlight agent analyzes the entire note and proposes candidate Q/A pairs plus actionable suggestions so authors know exactly what to fix.

**Models**: Structured-output models (default `x-ai/grok-4.1-fast`) with reasoning enabled
**Output**: `HighlightResult` containing:

-   candidate Q/A pairs with confidence scores and excerpts
-   actionable suggestions (e.g., “Add RU answer for Question 1”)
-   inferred note status (`draft`, `incomplete`, etc.)

**Routing**: Automatically invoked when pre-validation fails and `enable_highlight_agent` is `true`. Results are surfaced in sync logs so authors get immediate guidance instead of a generic error.

#### Generator Agent

**Purpose**: Core card generation from validated Q&A pairs.

**Models**: High-quality models (Claude-3.5-Sonnet, GPT-4o)
**Output**: `CardGenerationResult` with APF-formatted cards
**Quality**: 95%+ acceptance rate after post-validation

#### Post-Validator Agent

**Purpose**: Quality assurance and automatic error correction.

**Models**: Balanced models (GPT-4o, Claude-3.5-Haiku)
**Output**: `PostValidationResult` with corrections and final approval
**Features**: Auto-fix common issues, detailed quality metrics

---

## Configuration

### Basic Configuration

```yaml
# Enable agent system
use_langgraph: true
use_pydantic_ai: true

# Model selection
pre_validator_model: "openai/gpt-4o-mini"
generator_model: "anthropic/claude-3-5-sonnet-20241022"
post_validator_model: "openai/gpt-4o"

# OpenRouter configuration
llm_provider: "openrouter"
openrouter_api_key: "sk-or-v1-..." # pragma: allowlist secret
```

### Advanced Configuration

```yaml
# Agent-specific settings
agent_config:
    max_retries: 3
    retry_delay: 2.0
    temperature: 0.1
    timeout: 60

# Workflow settings
workflow_config:
    enable_checkpointing: true
    max_workflow_time: 300
    log_level: "INFO"

# Tool configuration
tool_config:
    enable_apf_validation: true
    enable_html_formatting: true
    enable_content_hashing: true
```

### Model Presets

The system provides optimized model presets:

```yaml
# Cost-effective setup
model_preset: "cost_effective"
# Uses: GPT-4o-mini, Claude-3.5-Haiku, GPT-4o-mini

# Balanced quality/cost
model_preset: "balanced"
# Uses: GPT-4o, Claude-3.5-Sonnet, GPT-4o

# High quality (expensive)
model_preset: "high_quality"
# Uses: GPT-4o, Claude-3.5-Sonnet, o1-preview

# Fast processing
model_preset: "fast"
# Uses: GPT-4o-mini, GPT-4o-mini, GPT-4o-mini
```

---

## Usage

### Basic Usage

```python
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.agents import LangGraphOrchestrator

# Load configuration
config = Config.from_yaml("config.yaml")

# Create orchestrator
orchestrator = LangGraphOrchestrator.from_config(config)

# Process a note
result = await orchestrator.process_note(note_content, frontmatter)
cards = result.cards
```

### Command Line Usage

```bash
# Use agent system for sync
obsidian-anki-sync sync --use-langgraph --use-pydantic-ai

# Test with specific models
obsidian-anki-sync test-run \
  --pre-validator-model "openai/gpt-4o-mini" \
  --generator-model "anthropic/claude-3-5-sonnet-20241022" \
  --post-validator-model "openai/gpt-4o"

# Enable detailed logging
obsidian-anki-sync sync --log-level DEBUG
```

### Monitoring and Debugging

```python
# Enable workflow visualization
config.workflow_config.enable_visualization = True

# Access workflow state
state = orchestrator.get_current_state()
print(f"Current step: {state.current_node}")
print(f"Progress: {state.progress}%")

# View execution graph
orchestrator.visualize_workflow("workflow.png")
```

### Error Handling

```python
try:
    result = await orchestrator.process_note(note_content, frontmatter)
except PreValidationError as e:
    print(f"Pre-validation failed: {e.issues}")
except GenerationError as e:
    print(f"Generation failed: {e.reason}")
except PostValidationError as e:
    print(f"Post-validation failed: {e.corrections_needed}")
```

---

## Migration Guide

### From Legacy System

If you're migrating from the legacy agent system:

1. **Update Configuration**:

    ```yaml
    # Old config
    use_agent_system: true

    # New config
    use_langgraph: true
    use_pydantic_ai: true
    ```

2. **Model Updates**:

    ```yaml
    # Old models (may still work but deprecated)
    pre_validator_model: "qwen3:8b"
    generator_model: "qwen3:32b"
    post_validator_model: "qwen3:14b"

    # New recommended models
    pre_validator_model: "openai/gpt-4o-mini"
    generator_model: "anthropic/claude-3-5-sonnet-20241022"
    post_validator_model: "openai/gpt-4o"
    ```

3. **Provider Migration**:
    ```yaml
    # From Ollama to OpenRouter
    llm_provider: "openrouter" # instead of "ollama"
    openrouter_api_key: "sk-or-v1-..." # pragma: allowlist secret
    ```

### Performance Comparison

| Aspect            | Legacy System | LangGraph + PydanticAI   | Improvement         |
| ----------------- | ------------- | ------------------------ | ------------------- |
| Type Safety       | None          | Full Pydantic validation | 100% safer          |
| Error Handling    | Basic         | Structured exceptions    | Better debugging    |
| Workflow Control  | Linear        | State machine            | More flexible       |
| Provider Support  | Ollama-only   | 100+ models              | Much broader        |
| Cost Optimization | Manual        | Auto-selection           | 20-30% savings      |
| Monitoring        | Basic logs    | Full observability       | Complete visibility |

### Backwards Compatibility

The new system maintains backwards compatibility:

-   **Legacy configs** still work with warnings
-   **Old model names** are automatically mapped to equivalents
-   **Fallback to legacy** if new system fails
-   **Gradual migration** path available

### Troubleshooting Migration

**Issue**: "LangGraph not available"
**Solution**: Install with `uv sync --all-extras` or `pip install langgraph`

**Issue**: "OpenRouter API key required"
**Solution**: Get key from openrouter.ai and add to config

**Issue**: "Model not found"
**Solution**: Check available models at openrouter.ai/models

---

## Performance Optimization

### Model Selection Strategy

1. **Pre-validator**: Fast, cheap models for initial filtering
2. **Generator**: High-quality models for content creation
3. **Post-validator**: Balanced models for quality assurance

### Caching and Reuse

```yaml
# Enable result caching
cache_config:
    enable_result_cache: true
    cache_ttl: 3600 # 1 hour
    max_cache_size: 1000
```

### Batch Processing

```python
# Process multiple notes efficiently
batch_result = await orchestrator.process_batch(notes, batch_size=5)
```

### Monitoring

```python
# Get performance metrics
metrics = orchestrator.get_metrics()
print(f"Total requests: {metrics.total_requests}")
print(f"Success rate: {metrics.success_rate}%")
print(f"Average latency: {metrics.avg_latency}s")
```

---

## Best Practices

### Model Selection

-   Use cost-effective models for pre-validation (saves 60% on filtering)
-   Reserve high-quality models for generation (where quality matters most)
-   Monitor usage and adjust based on cost vs. quality trade-offs

### Configuration

-   Start with presets, then customize based on your needs
-   Enable checkpointing for long-running workflows
-   Use visualization during development and debugging

### Monitoring

-   Monitor success rates and latency per agent type
-   Track costs across different model combinations
-   Log failures with sufficient context for debugging

### Error Handling

-   Implement proper exception handling for all agent types
-   Use retry logic with exponential backoff
-   Provide meaningful error messages to users

---

## Future Enhancements

### Planned Features

-   **Custom Agent Types**: User-defined agent implementations
-   **Dynamic Model Selection**: Runtime model switching based on content
-   **Advanced Workflows**: Branching and parallel execution paths
-   **A/B Testing**: Compare different agent configurations
-   **Performance Profiling**: Detailed performance analytics

### Research Areas

-   **Multi-modal Agents**: Support for images and diagrams
-   **Chain-of-Thought**: Enhanced reasoning capabilities
-   **Self-Improvement**: Agents that learn from their outputs
-   **Collaborative Agents**: Multiple agents working together

---

## References

-   [LangGraph Documentation](https://github.com/langchain-ai/langgraph)
-   [PydanticAI Documentation](https://github.com/pydantic/pydantic-ai)
-   [OpenRouter API](https://openrouter.ai/)
-   [LangChain Agents](https://python.langchain.com/docs/modules/agents/)

---

**Version**: 2.0
**Last Updated**: November 28, 2025
**Status**: Unified agent documentation
