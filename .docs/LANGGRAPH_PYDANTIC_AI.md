# LangGraph + PydanticAI Integration

## Overview

This document describes the new agentic system for card generation using **LangGraph** for workflow orchestration and **PydanticAI** for type-safe structured outputs with OpenRouter as the LLM provider.

## Table of Contents

1. [What's New](#whats-new)
2. [Architecture](#architecture)
3. [Key Features](#key-features)
4. [Configuration](#configuration)
5. [Components](#components)
6. [Workflow](#workflow)
7. [Usage Examples](#usage-examples)
8. [Migration Guide](#migration-guide)
9. [Troubleshooting](#troubleshooting)

---

## What's New

### PydanticAI

[PydanticAI](https://github.com/pydantic/pydantic-ai) is the official AI agent framework from Pydantic, providing:

-   **Type-Safe Structured Outputs**: Agents return strongly-typed Pydantic models
-   **Multi-Provider Support**: Works with OpenAI, Anthropic, Gemini, OpenRouter, and more
-   **Tool/Function Calling**: Built-in support for agent tools
-   **Validation**: Automatic output validation using Pydantic schemas
-   **Better Error Handling**: Clear, typed error messages

### LangGraph

[LangGraph](https://github.com/langchain-ai/langgraph) is a state machine framework for building multi-agent workflows:

-   **State Management**: Persistent state across workflow steps
-   **Conditional Routing**: Dynamic routing based on validation results
-   **Cycles & Loops**: Support for retry logic and iterative workflows
-   **Checkpointing**: State persistence for resumable workflows
-   **Visualization**: Graph visualization for debugging

### OpenRouter Integration

[OpenRouter](https://openrouter.ai/) provides unified access to multiple LLM providers:

-   **100+ Models**: Access OpenAI, Anthropic, Google, Meta, and more
-   **Unified API**: OpenAI-compatible interface
-   **Cost Optimization**: Choose models based on price/performance
-   **Fallback Support**: Automatic fallback to alternative models
-   **Usage Tracking**: Detailed cost and usage analytics

---

## Architecture

### System Diagram

```

‚                    LangGraph Workflow                       ‚
‚                                                             ‚
‚               ‚
‚  ‚ Pre-Validator ‚>‚   Generator   ‚>‚  Post-  ‚ ‚
‚  ‚    Agent      ‚      ‚     Agent     ‚      ‚Validator‚ ‚
‚  ¬˜      ˜      ¬˜ ‚
‚          ‚                                           ‚      ‚
‚          ‚                                           ‚      ‚
‚                        ˜      ‚
‚                          ‚              ‚                   ‚
‚                     ¼¼              ‚
‚                     ‚  Conditional Routing   ‚              ‚
‚                     ‚  (Retry/Complete/Fail) ‚              ‚
‚                     ˜              ‚
˜
                              ‚
                              ¼

‚                     PydanticAI Agents                       ‚
‚                                                             ‚
‚       ‚
‚  ‚ PreValidatorAI   ‚  ‚  GeneratorAI     ‚  ‚PostValidatorAI‚
‚  ‚                  ‚  ‚                  ‚  ‚           ‚ ‚
‚  ‚ Result:          ‚  ‚ Result:          ‚  ‚ Result:   ‚ ‚
‚  ‚ PreValidation    ‚  ‚ CardGeneration   ‚  ‚PostValidation‚
‚  ‚ Output           ‚  ‚ Output           ‚  ‚ Output    ‚ ‚
‚  ¬˜  ¬˜  ¬˜ ‚
‚           ‚                     ‚                   ‚       ‚
¼¼¼˜
            ‚                     ‚                   ‚
            ¼                     ¼                   ¼

‚                      OpenRouter API                         ‚
‚                                                             ‚
‚            ‚
‚  ‚ GPT-4o-mini  ‚  ‚ Claude 3.5   ‚  ‚ GPT-4o-mini  ‚      ‚
‚  ‚ (Validator)  ‚  ‚  (Generator) ‚  ‚ (Validator)  ‚      ‚
‚  ˜  ˜  ˜      ‚
˜
```

### Component Layers

1. **Workflow Layer (LangGraph)**

    - State management
    - Node execution
    - Conditional routing
    - Checkpoint persistence

2. **Agent Layer (PydanticAI)**

    - Type-safe agents
    - Structured outputs
    - Validation logic
    - Error handling

3. **Provider Layer (OpenRouter)**
    - Model selection
    - API communication
    - Cost optimization
    - Fallback handling

---

## Key Features

### 1. Type-Safe Structured Outputs

All agent outputs are validated Pydantic models:

```python
class PreValidationOutput(BaseModel):
    is_valid: bool
    error_type: str
    error_details: str = ""
    suggested_fixes: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
```

### 2. Automatic Retry Logic

The workflow automatically retries failed validations:

```
Pre-Validation > Generation > Post-Validation
                         ²                    ‚
                         ‚                    ‚ (failed)
                         ˜
                         (retry with fixes)
```

### 3. State Persistence

Workflows can be resumed from checkpoints:

```python
workflow.invoke(
    state,
    config={"configurable": {"thread_id": "note-123"}}
)
```

### 4. Multi-Model Support

Different agents can use different models:

-   **Pre-Validator**: `gpt-4o-mini` (fast, cheap)
-   **Generator**: `claude-3-5-sonnet` (powerful, accurate)
-   **Post-Validator**: `gpt-4o-mini` (fast, cheap)

### 5. Conditional Routing

Workflow routes based on validation results:

```python
def should_continue_after_pre_validation(state):
    if state["pre_validation"]["is_valid"]:
        return "generation"
    return "failed"
```

---

## Configuration

### Environment Variables

```bash
# Required: OpenRouter API key
export OPENROUTER_API_KEY="sk-or-v1-..."

# Optional: Site information for rankings
export OPENROUTER_SITE_URL="https://yoursite.com"
export OPENROUTER_SITE_NAME="Your App Name"
```

### config.yaml

```yaml
# Enable new system
use_langgraph: true
use_pydantic_ai: true

# LLM provider
llm_provider: openrouter

# Model selection
pydantic_ai_pre_validator_model: openai/gpt-4o-mini
pydantic_ai_generator_model: anthropic/claude-3-5-sonnet
pydantic_ai_post_validator_model: openai/gpt-4o-mini

# Workflow settings
langgraph_max_retries: 3
langgraph_auto_fix: true
langgraph_strict_mode: true
langgraph_checkpoint_enabled: true

# Enhanced agents (2025)
enable_note_correction: false # Optional proactive correction
note_correction_model: "qwen/qwen-2.5-32b-instruct"

# Card splitting preferences
card_splitting_preferred_size: "medium" # small, medium, large
card_splitting_prefer_splitting: true
card_splitting_min_confidence: 0.7
card_splitting_max_cards_per_note: 10
```

See `config.langgraph.example.yaml` for full configuration.

---

## Components

### PydanticAI Agents

#### PreValidatorAgentAI

Validates note structure before generation:

```python
from obsidian_anki_sync.agents import PreValidatorAgentAI
from obsidian_anki_sync.providers import create_openrouter_model_from_env

model = create_openrouter_model_from_env("openai/gpt-4o-mini")
agent = PreValidatorAgentAI(model=model, temperature=0.0)

result = await agent.validate(
    note_content=content,
    metadata=metadata,
    qa_pairs=qa_pairs,
)
```

#### GeneratorAgentAI

Generates APF cards with structured output:

```python
from obsidian_anki_sync.agents import GeneratorAgentAI

model = create_openrouter_model_from_env("anthropic/claude-3-5-sonnet")
agent = GeneratorAgentAI(model=model, temperature=0.3)

result = await agent.generate_cards(
    note_content=content,
    metadata=metadata,
    qa_pairs=qa_pairs,
    slug_base="android-recyclerview",
)
```

#### PostValidatorAgentAI

Validates generated cards for quality:

```python
from obsidian_anki_sync.agents import PostValidatorAgentAI

model = create_openrouter_model_from_env("openai/gpt-4o-mini")
agent = PostValidatorAgentAI(model=model, temperature=0.0)

result = await agent.validate(
    cards=generated_cards,
    metadata=metadata,
    strict_mode=True,
)
```

### LangGraph Orchestrator

Coordinates the entire workflow:

```python
from obsidian_anki_sync.agents import LangGraphOrchestrator
from obsidian_anki_sync.config import load_config

config = load_config()
orchestrator = LangGraphOrchestrator(
    config=config,
    max_retries=3,
    auto_fix_enabled=True,
    strict_mode=True,
)

result = orchestrator.process_note(
    note_content=content,
    metadata=metadata,
    qa_pairs=qa_pairs,
)

if result.success:
    print(f"Generated {result.generation.total_cards} cards")
else:
    print(f"Failed: {result.post_validation.error_details}")
```

---

## Workflow

### State Transitions

```

‚    START     ‚
¬˜
       ‚
       ¼

‚Pre-Validation‚
¬˜
       ‚
       (valid)
       ‚                                  ‚
       (invalid)> FAILED               ‚
       ‚                                  ‚
       ¼                                  ¼

‚  Generation  ‚                   ‚ Generation  ‚
¬˜                   ¬˜
       ‚                                  ‚
       ¼                                  ¼

‚Post-Validation‚                  ‚Post-Validation‚
¬˜                   ¬˜
       ‚                                  ‚
       (valid)> COMPLETE               ‚
       ‚                                  ‚
       (retry<max)> Generation         ‚
       ‚              (with fixes)         ‚
       ‚                                  ‚
       (retry‰¥max)> FAILED             ‚
                                          ‚
                                    (successful path)
```

### Enhanced Pipeline Stages (2025)

1. **Note Correction** (Optional)

    - Proactive quality improvement before parsing
    - Grammar, clarity, and completeness checks
    - Only runs if `enable_note_correction: true`
    - Default: disabled (reactive repair handles corrections)

2. **Pre-Validation**

    - Checks note structure
    - Validates frontmatter
    - Verifies Q/A pairs
    - Result: `PreValidationOutput`

3. **Generation**

    - Generates APF cards
    - Applies language hints
    - Creates slugs
    - Result: `CardGenerationOutput`

4. **Post-Validation**

    - Validates card syntax
    - Checks factual accuracy
    - Verifies template compliance
    - Result: `PostValidationOutput`

5. **Retry Loop**
    - Applies suggested corrections
    - Re-runs validation
    - Max retries: configurable
    - Auto-fix: optional

---

## Usage Examples

### Basic Usage

```python
from pathlib import Path
from obsidian_anki_sync.config import load_config
from obsidian_anki_sync.agents import LangGraphOrchestrator
from obsidian_anki_sync.obsidian.parser import parse_note

# Load configuration
config = load_config(Path("config.yaml"))

# Create orchestrator
orchestrator = LangGraphOrchestrator(config)

# Parse note
note_path = Path("notes/android-recyclerview.md")
metadata, qa_pairs = parse_note(note_path)

# Process note through workflow
result = orchestrator.process_note(
    note_content=note_path.read_text(),
    metadata=metadata,
    qa_pairs=qa_pairs,
    file_path=note_path,
)

# Check result
if result.success:
    print(f"Success! Generated {result.generation.total_cards} cards")
    print(f"Total time: {result.total_time:.2f}s")
    print(f"Retries: {result.retry_count}")
else:
    print(f"Failed: {result.post_validation.error_details}")
```

### Custom Model Selection

```python
from obsidian_anki_sync.providers import PydanticAIModelFactory

# Use different models for different agents
pre_val_model = PydanticAIModelFactory.create_openrouter_model(
    model_name="openai/gpt-4o-mini",
    api_key="your-key",
)

gen_model = PydanticAIModelFactory.create_openrouter_model(
    model_name="anthropic/claude-3-opus",  # Most powerful
    api_key="your-key",
)

post_val_model = PydanticAIModelFactory.create_openrouter_model(
    model_name="openai/gpt-4o-mini",
    api_key="your-key",
)
```

### Debugging with State Inspection

```python
# Access workflow state for debugging
orchestrator = LangGraphOrchestrator(config)
result = orchestrator.process_note(...)

# Inspect state
print("Pre-validation:", result.pre_validation.model_dump())
print("Generation:", result.generation.model_dump())
print("Post-validation:", result.post_validation.model_dump())
print("Messages:", result.pipeline_state.get("messages", []))
```

---

## Migration Guide

### From Legacy Agent System

#### Before (Legacy)

```python
from obsidian_anki_sync.agents import AgentOrchestrator

orchestrator = AgentOrchestrator(config)
result = orchestrator.process_note(...)
```

#### After (LangGraph + PydanticAI)

```python
from obsidian_anki_sync.agents import LangGraphOrchestrator

orchestrator = LangGraphOrchestrator(
    config,
    max_retries=config.langgraph_max_retries,
    auto_fix_enabled=config.langgraph_auto_fix,
    strict_mode=config.langgraph_strict_mode,
)
result = orchestrator.process_note(...)
```

### Configuration Changes

```yaml
# Old
use_agent_system: true
pre_validator_model: qwen3:8b
generator_model: qwen3:32b
post_validator_model: qwen3:14b

# New
use_langgraph: true
use_pydantic_ai: true
pydantic_ai_pre_validator_model: openai/gpt-4o-mini
pydantic_ai_generator_model: anthropic/claude-3-5-sonnet
pydantic_ai_post_validator_model: openai/gpt-4o-mini
```

### API Compatibility

The `AgentPipelineResult` is the same, so existing code using results continues to work:

```python
# Works with both orchestrators
result = orchestrator.process_note(...)
if result.success:
    cards = result.generation.cards
```

---

## Troubleshooting

### Common Issues

#### 1. OpenRouter API Key Not Found

**Error**: `ValueError: OpenRouter API key is required`

**Solution**:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

#### 2. Model Not Available

**Error**: `Model 'xyz' not found`

**Solution**: Check available models:

```python
from obsidian_anki_sync.providers import OpenRouterProvider

provider = OpenRouterProvider(api_key="your-key")
models = provider.list_models()
print(models)
```

#### 3. Async Runtime Error

**Error**: `RuntimeError: asyncio.run() cannot be called from a running event loop`

**Solution**: The orchestrator handles async internally. Don't call from async context:

```python
# Don't do this
async def main():
    result = orchestrator.process_note(...)  # Already handles async

# Do this instead
def main():
    result = orchestrator.process_note(...)  # Correct
```

#### 4. High API Costs

**Solution**: Use cheaper models for validation:

```yaml
pydantic_ai_pre_validator_model: openai/gpt-4o-mini # Cheap
pydantic_ai_generator_model: anthropic/claude-3-5-sonnet # Expensive but good
pydantic_ai_post_validator_model: openai/gpt-4o-mini # Cheap
langgraph_max_retries: 1 # Reduce retries
```

### Debug Mode

Enable debug logging:

```yaml
log_level: DEBUG
```

Check workflow execution:

```python
result = orchestrator.process_note(...)
print(f"Retries: {result.retry_count}")
print(f"Stage times: {result.pipeline_state['stage_times']}")
print(f"Messages: {result.pipeline_state['messages']}")
```

---

## Performance & Cost

### Model Recommendations

| Use Case     | Pre-Validator | Generator         | Post-Validator | Est. Cost/Card |
| ------------ | ------------- | ----------------- | -------------- | -------------- |
| **Budget**   | gpt-4o-mini   | gpt-4o-mini       | gpt-4o-mini    | $0.001         |
| **Balanced** | gpt-4o-mini   | claude-3-5-sonnet | gpt-4o-mini    | $0.01          |
| **Premium**  | gpt-4o        | claude-3-opus     | gpt-4o         | $0.05          |

### Optimization Tips

1. **Use fast models for validation**: Pre/post validators don't need expensive models
2. **Reduce retries**: Set `langgraph_max_retries: 1` for lower costs
3. **Batch processing**: Process multiple notes in parallel
4. **Cache results**: Enable checkpointing to resume failed workflows
5. **Monitor usage**: Check OpenRouter dashboard for cost tracking

---

## Future Enhancements

-   [ ] Multi-agent collaboration (agents can communicate)
-   [ ] Human-in-the-loop approval steps
-   [ ] Parallel card generation for speed
-   [ ] Cost tracking and budgets
-   [ ] Workflow visualization UI
-   [ ] Custom agent tools/functions
-   [ ] Local model support (Ollama with PydanticAI)

---

## References

-   [PydanticAI Documentation](https://ai.pydantic.dev/)
-   [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
-   [OpenRouter Documentation](https://openrouter.ai/docs)
-   [Project Repository](https://github.com/po4yka/obsidian-to-anki)

---

For questions or issues, please open an issue on GitHub.
