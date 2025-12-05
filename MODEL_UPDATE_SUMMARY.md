# Model Update Summary - December 2025

This document summarizes the update to use the latest and most suitable qwen/deepseek/kimi/minimax models based on December 2025 research.

## Latest Models Integrated

### Qwen Models

-   **qwen/qwen-2.5-7b-instruct** (NEW)
    -   Context: 33K tokens
    -   Cost: $0.04/$0.10 per M tokens (very cost-effective)
    -   Use case: Pre-validation, fast tasks
    -   Speed: Tier 2 (fast)
    -   Quality: Tier 4

### DeepSeek Models

-   **deepseek/deepseek-v3.2** (LATEST - December 2025)
    -   Context: 163K tokens (largest among DeepSeek models)
    -   Cost: $0.14/$0.56 per M tokens
    -   Reasoning: Excellent (rivals GPT-5)
    -   Use case: Generation, post-validation, QA extraction, parser repair
    -   Speed: Tier 3
    -   Quality: Tier 5 (highest)

### Minimax Models

-   **minimax/minimax-m2** (LATEST - December 2025)
    -   Context: 204K tokens
    -   Cost: $0.10/$0.40 per M tokens
    -   Use case: Context enrichment, creative tasks, agentic workflows
    -   Speed: Tier 3
    -   Quality: Tier 5

### Kimi Models

-   **moonshotai/kimi-k2-thinking** (LATEST - November 2025)
    -   Context: 256K tokens (largest context window)
    -   Cost: $0.12/$0.48 per M tokens
    -   Reasoning: Advanced thinking model
    -   Use case: Card splitting, complex decision-making
    -   Speed: Tier 4 (slower due to thinking process)
    -   Quality: Tier 5

## Updated Model Presets

### Cost Effective Preset

-   **Pre-Validation**: `qwen/qwen-2.5-7b-instruct` (NEW: fast, cheap)
-   **Generation**: `qwen/qwen-2.5-32b-instruct` (cost-effective)
-   **Post-Validation**: `deepseek/deepseek-v3.2` (latest: excellent reasoning)
-   **Context Enrichment**: `minimax/minimax-m2` (latest: creative tasks)
-   **Card Splitting**: `moonshotai/kimi-k2-thinking` (latest: 256K context)

### Balanced Preset (Default)

-   **Pre-Validation**: `qwen/qwen-2.5-7b-instruct` (NEW: fast, cheap)
-   **Generation**: `deepseek/deepseek-v3.2` (latest: excellent reasoning, 163K context)
-   **Post-Validation**: `deepseek/deepseek-v3.2` (latest: excellent reasoning)
-   **QA Extraction**: `deepseek/deepseek-v3.2` (latest: excellent reasoning)
-   **Parser Repair**: `deepseek/deepseek-v3.2` (latest: excellent reasoning)
-   **Context Enrichment**: `minimax/minimax-m2` (latest: creative tasks)
-   **Card Splitting**: `moonshotai/kimi-k2-thinking` (latest: 256K context)

### High Quality Preset

-   **Pre-Validation**: `qwen/qwen-2.5-7b-instruct` (NEW: fast, cheap)
-   **Generation**: `deepseek/deepseek-v3.2` (latest: premium reasoning)
-   **Post-Validation**: `deepseek/deepseek-v3.2` (latest: premium reasoning)
-   All other tasks use latest models optimized for quality

### Fast Preset

-   **All tasks**: `qwen/qwen-2.5-7b-instruct` (NEW: smallest, fastest model)

## Default Model Changes

-   **Previous**: `qwen/qwen-2.5-32b-instruct`
-   **New**: `deepseek/deepseek-v3.2` (latest: excellent reasoning, 163K context)

## Key Improvements

1. **Cost Optimization**: Using `qwen/qwen-2.5-7b-instruct` for pre-validation reduces costs by ~80% ($0.04 vs $0.20 per M tokens)

2. **Quality Enhancement**: Using `deepseek/deepseek-v3.2` for generation and validation provides GPT-5 class reasoning capabilities

3. **Context Window**:

    - DeepSeek V3.2: 163K tokens (vs 128K standard)
    - Minimax M2: 204K tokens
    - Kimi K2 Thinking: 256K tokens (largest)

4. **Task-Specific Optimization**:
    - Pre-validation: Fast, cheap models
    - Generation: High-quality reasoning models
    - Post-validation: Excellent reasoning models
    - Context enrichment: Creative/agentic models
    - Card splitting: Large context thinking models

## Files Updated

1. `src/obsidian_anki_sync/models/config.py`

    - Added latest model capabilities
    - Updated all model presets
    - Changed DEFAULT_MODEL to `deepseek/deepseek-v3.2`

2. `src/obsidian_anki_sync/providers/openrouter/models.py`

    - Added latest models to context windows
    - Added latest models to max output tokens
    - Added latest models to reasoning support
    - Added latest models to structured outputs

3. `src/obsidian_anki_sync/config.py`

    - Updated `default_llm_model` to `deepseek/deepseek-v3.2`
    - Updated `openrouter_model` to `deepseek/deepseek-v3.2`

4. Configuration Files

    - `config.yaml.example`: Updated with latest model recommendations
    - Documentation: Updated all model references

5. Documentation
    - `.docs/ARCHITECTURE/agents.md`: Updated model recommendations
    - `.docs/ARCHITECTURE/providers.md`: Updated model recommendations
    - `.docs/GUIDES/configuration.md`: Updated model presets and examples

## Model Capabilities Summary

| Model                       | Context | Reasoning | Structured Output | Best For                          |
| --------------------------- | ------- | --------- | ----------------- | --------------------------------- |
| qwen/qwen-2.5-7b-instruct   | 33K     | No        | Yes               | Fast, cheap validation            |
| deepseek/deepseek-v3.2      | 163K    | Yes       | Yes               | Generation, validation, reasoning |
| minimax/minimax-m2          | 204K    | No        | Yes               | Creative tasks, agentic workflows |
| moonshotai/kimi-k2-thinking | 256K    | Yes       | Yes               | Complex decisions, card splitting |

## Benefits

1. **Lower Costs**: Using smaller models for validation tasks reduces overall costs
2. **Higher Quality**: Latest reasoning models improve generation and validation quality
3. **Larger Context**: New models support much larger context windows
4. **Better Task Matching**: Each task now uses the most suitable model
5. **Future-Proof**: Using latest models ensures access to newest capabilities
