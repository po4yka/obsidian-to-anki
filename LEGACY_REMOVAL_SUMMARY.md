# Legacy Code Removal Summary

This document summarizes the removal of all legacy files, settings, and outdated configurations from the project.

**Status: COMPLETE** - All legacy code, settings, and outdated configurations have been removed.

## Removed Legacy Settings

### 1. Legacy OpenRouter Settings

-   **Removed**: `openrouter_model` field from `Config` class
-   **Reason**: Replaced by unified `default_llm_model` configuration
-   **Impact**: CLI now uses `default_llm_model` instead of `openrouter_model`

### 2. Legacy LangChain Agent System Configuration

-   **Removed**: LangChain-specific type settings (not used in current implementation)
-   **Removed Settings**:
    -   `langchain_generator_type`
    -   `langchain_pre_validator_type`
    -   `langchain_post_validator_type`
    -   `langchain_enrichment_type`
    -   `agent_fallback_on_error`
    -   `agent_fallback_on_timeout`
-   **Updated**: `agent_framework` validator now only supports `pydantic_ai` and `memory_enhanced` (removed `langchain` option)
-   **Reason**: Unified agent system uses PydanticAI, LangChain option was not implemented

### 3. Legacy OpenRouter Configuration Section

-   **Removed**: Entire "LEGACY OPENROUTER CONFIGURATION" section from `config.agents.example.yaml`
-   **Removed Settings**:
    -   `openrouter_model`
    -   `llm_temperature`
    -   `llm_top_p`
-   **Reason**: Replaced by unified model configuration system

## Removed Legacy Code

### 1. Deprecated `configure_llm_extraction()` Function

-   **Removed**: Entire function from `src/obsidian_anki_sync/obsidian/parser.py`
-   **Reason**: Not thread-safe, replaced by `create_qa_extractor()` and explicit parameter passing
-   **Replacement**: Use `create_qa_extractor()` and pass extractor explicitly to parse functions

### 2. Legacy Global State Variables

-   **Removed**: Global state variables for LLM extraction:
    -   `_USE_LLM_EXTRACTION`
    -   `_QA_EXTRACTOR_AGENT`
    -   `_ENFORCE_LANGUAGE_VALIDATION`
    -   `_GLOBAL_STATE_LOCK`
-   **Reason**: Not thread-safe, replaced by thread-local state
-   **Impact**: All extraction now uses thread-local state or explicit parameters

### 3. Legacy APFGenerator References

-   **Removed**: `apf_gen` parameter from `CardGenerator.__init__()`
-   **Removed**: All references to legacy APFGenerator in code and comments
-   **Reason**: APFGenerator has been completely replaced by agent system
-   **Impact**: All card generation now uses LangGraphOrchestrator with PydanticAI agents

### 4. Legacy Non-Agent Mode Fallback

-   **Removed**: Automatic fallback code that enabled agent system when non-agent mode was detected
-   **Removed**: Warning messages about deprecated non-agent mode
-   **Reason**: Agent system is now required, no fallback needed
-   **Impact**: Clearer error messages when agent system is not available

## Updated Comments and Documentation

### 1. Removed "Legacy" and "Deprecated" Labels

-   Updated all comments to remove "legacy", "deprecated", "backward compatibility" references
-   Simplified documentation to reflect current architecture

### 2. Updated Function Documentation

-   Removed references to deprecated functions
-   Updated docstrings to reflect current usage patterns
-   Removed backward compatibility notes

### 3. Updated Configuration Examples

-   Removed legacy configuration sections from example files
-   Updated all examples to use current configuration system

## Files Modified

1. **src/obsidian_anki_sync/config.py**

    - Removed `openrouter_model` field
    - Updated comments to remove legacy references

2. **src/obsidian_anki_sync/cli.py**

    - Updated to use `default_llm_model` instead of `openrouter_model`
    - Removed legacy non-agent mode fallback code

3. **src/obsidian_anki_sync/obsidian/parser.py**

    - Removed `configure_llm_extraction()` function
    - Removed global state variables
    - Updated `_get_qa_extractor()` to only use thread-local state
    - Updated `_get_enforce_language_validation()` to only use thread-local state
    - Updated docstrings to remove legacy references

4. **src/obsidian_anki_sync/sync/card_generator.py**

    - Removed `apf_gen` parameter
    - Removed legacy APFGenerator references
    - Updated error messages

5. **src/obsidian_anki_sync/sync/engine.py**

    - Removed legacy non-agent mode fallback
    - Removed `apf_gen` parameter references
    - Updated comments

6. **src/obsidian_anki_sync/application/container.py**

    - Updated comments about ICardGenerator

7. **src/obsidian_anki_sync/agents/generator.py**

    - Removed legacy generator note from docstring

8. **src/obsidian_anki_sync/providers/README.md**

    - Updated "legacy" references to current terminology

9. **config.example.yaml**

    - Removed entire "LEGACY LANGCHAIN AGENT SYSTEM" section

10. **config.agents.example.yaml**
    - Removed "LEGACY OPENROUTER CONFIGURATION" section

## Migration Guide

### For Users Updating Configuration

1. **Replace `openrouter_model`**:

    ```yaml
    # Old (removed)
    openrouter_model: "qwen/qwen-2.5-32b-instruct"

    # New
    default_llm_model: "deepseek/deepseek-v3.2"
    ```

2. **Remove LangChain Settings**:

    ```yaml
    # Old (removed)
    use_langchain_agents: false
    langchain_generator_type: "tool_calling"

    # New - use unified agent system
    use_langgraph: true
    use_pydantic_ai: true
    ```

3. **Use Model Presets**:

    ```yaml
    # Old (removed)
    openrouter_model: "qwen/qwen-2.5-32b-instruct"
    llm_temperature: 0.2

    # New
    model_preset: "balanced" # or "cost_effective", "high_quality", "fast"
    ```

### For Developers

1. **LLM Extraction**:

    ```python
    # Old (removed)
    configure_llm_extraction(provider, model="qwen3:8b")
    metadata, qa_pairs = parse_note(file_path)

    # New
    extractor = create_qa_extractor(provider, model="qwen3:8b")
    metadata, qa_pairs = parse_note(file_path, qa_extractor=extractor)
    ```

2. **Card Generation**:

    ```python
    # Old (removed)
    generator = CardGenerator(config, apf_gen=apf_generator)

    # New
    generator = CardGenerator(config, agent_orchestrator=orchestrator)
    ```

## Benefits

1. **Simplified Configuration**: Single unified model configuration system
2. **Thread Safety**: Removed global state, using thread-local state only
3. **Clearer Architecture**: No legacy code paths, single modern agent system
4. **Better Maintainability**: Less code to maintain, clearer intent
5. **Improved Performance**: No fallback code paths, direct agent system usage

## Verification

All legacy code has been removed. The project now uses:

-   Unified model configuration with presets
-   Thread-safe LLM extraction with explicit parameters
-   Modern agent system (LangGraph + PydanticAI) exclusively
-   No backward compatibility code paths
