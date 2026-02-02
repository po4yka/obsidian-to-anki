# Architecture

## Core Components

**Sync Engine** (`sync/engine.py`):
- Orchestrates the entire sync pipeline
- Coordinates note discovery, parsing, card generation, and Anki updates
- Handles incremental sync, resumable syncs, and batch operations
- Manages state via SQLite database (`state_db.py`)

**Parser System** (`obsidian/parser.py`):
- Extracts Q&A pairs from Obsidian markdown notes
- Supports both regex-based and LLM-based extraction (configurable via `use_agent_system`)
- Handles YAML frontmatter, multi-pair blocks, and bilingual content

**Agent Orchestration** (`agents/`):
- LangGraph orchestrator (`langgraph_orchestrator.py`) uses state machine workflow with conditional routing and retries
- Multi-stage pipeline: Pre-Validator -> Generator -> Post-Validator
- Optional enhancement agents: Context Enrichment, Memorization Quality, Card Splitting, Duplicate Detection

**LLM Provider System** (`providers/`):
- Unified interface via `BaseLLMProvider` (`base.py`)
- Factory pattern (`factory.py`) creates provider instances from config
- Providers: Ollama, LM Studio, OpenRouter
- PydanticAI integration for structured outputs (`pydantic_ai_models.py`)

**Configuration** (`config.py`):
- Pydantic-based settings with YAML + environment variable support
- Model preset system: "cost_effective", "balanced", "high_quality", "fast"
- Per-task model overrides via `get_model_for_agent()` and `get_model_config_for_task()`
- Security: path validation, API key validation, symlink prevention

**State Management** (`sync/state_db.py`):
- SQLite with WAL mode for concurrency
- Tracks note hashes, card metadata, sync sessions
- Supports resumable syncs via progress tracking

## Data Flow

1. **Discovery**: `discover_notes()` finds markdown files in vault
2. **Parsing**: `parse_note()` extracts Q&A pairs and metadata
3. **Slug Generation**: `generate_slug()` creates stable IDs with collision resolution
4. **Card Generation**:
   - Agent System: Pre-Validator -> Generator -> Post-Validator
   - Direct: `APFGenerator` creates cards from Q&A pairs
5. **Anki Sync**: `AnkiClient` communicates with AnkiConnect API
6. **State Update**: `StateDB` persists sync state for incremental/resumable syncs

## Key Design Patterns

**Provider Factory**: `ProviderFactory.create_from_config()` instantiates LLM providers based on `config.llm_provider`

**Transaction System**: `CardTransaction` (`sync/transactions.py`) provides rollback capability for Anki operations

**Content Hashing**: `compute_content_hash()` detects changes for incremental sync

**Deterministic GUIDs**: `deterministic_guid()` ensures stable Anki note IDs across syncs

**Model Configuration Cascade**:
1. Check explicit per-task override (e.g., `generator_model`)
2. Use preset default (`model_preset` -> `ModelTask` -> model name)
3. Fallback to `default_llm_model`

## Agent System

The agent system uses LangGraph orchestrator with state machine architecture:
- **LangGraph** (`agents/langgraph_orchestrator.py`): State machine with conditional routing, persistence
- PydanticAI for structured outputs

Enable via config:
```yaml
use_agent_system: true
```

LangGraph workflow nodes:
- `pre_validate_node`: Validates note structure
- `card_splitting_node`: Analyzes if note should be split (optional)
- `generate_node`: Creates flashcard content
- `post_validate_node`: Quality validation with auto-fix and retry
- `context_enrichment_node`: Adds examples/mnemonics (optional)
- `memorization_quality_node`: Checks SRS effectiveness (optional)

## Model Selection

Models are configured via preset + optional overrides:

```yaml
model_preset: "balanced"  # or "cost_effective", "high_quality", "fast"
generator_model: "qwen/qwen-2.5-72b-instruct"
pre_validator_model: "qwen/qwen-2.5-14b-instruct"
```

Access in code:
```python
model = config.get_model_for_agent("generator")
model_config = config.get_model_config_for_task("generation")
```

## APF Format

Cards follow APF v2.1 specification:
- `APFGenerator` (`apf/generator.py`): Converts Q&A pairs to APF cards
- `validate_apf()` (`apf/linter.py`): Validates against APF spec
- `validate_card_html()` (`apf/html_validator.py`): HTML syntax validation
- `map_apf_to_anki_fields()` (`anki/field_mapper.py`): Maps APF to Anki note type

## Sync Modes

- **Incremental**: `--incremental` only processes notes not yet in `StateDB`
- **Resumable**: Session ID + checkpoint system allows recovering from interruptions
- **Dry Run**: `--dry-run` previews all changes without modifying Anki

## Performance

- **Batch Operations**: Enable `enable_batch_operations: true` for batch Anki/DB writes
- **Concurrent Generations**: Set `max_concurrent_generations: 5` for parallel card generation
- **Connection Pooling**: HTTP clients use connection pooling (see `providers/base.py`)
- **Caching**: LLM responses cached with `diskcache` (see `sync/engine.py`)
- **Indexing**: Full index built once per sync (`sync/indexer.py`), use `--no-index` to skip
