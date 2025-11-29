# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Skills

This project uses specialized Claude Skills for enhanced Python development, testing, and LLM integration. See `.claude/SKILLS.md` for:

-   Recommended skills (async-python-patterns, python-testing-patterns, prompt-engineering-patterns, etc.)
-   Installation instructions
-   Usage examples for this project

Quick install:

```bash
/plugin install python-development@wshobson/claude-code-workflows
/plugin install llm-application-dev@wshobson/claude-code-workflows
```

## Project Overview

See `README.md` for project overview and setup instructions.

## Development Commands

See `README.md` for common commands.

### Code Quality

```bash
# Format code (REQUIRED before commits)
uv run ruff format . && uv run isort .

# Linting
uv run ruff check .

# Type checking
uv run mypy src/

# Run all quality checks
uv run ruff format . && uv run isort . && uv run ruff check . && uv run mypy src/
```

**IMPORTANT**: After every atomic change (function, class, or module modification), run `uv run ruff check .` and fix any linting issues immediately. This ensures code quality is maintained incrementally and prevents large refactoring sessions later.

### Testing

```bash
# Run all tests with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_parser.py

# Run specific test
uv run pytest tests/test_parser.py::test_parse_note_with_qa_pairs

# Run with verbose output
uv run pytest -v

# Run only fast tests (skip integration)
uv run pytest -m "not integration"

# Run integration tests only
uv run pytest tests/integration/

# Run Queue Worker
uv run arq obsidian_anki_sync.worker.WorkerSettings

```

## Architecture

### Core Components

**Sync Engine** (`sync/engine.py`):

-   Orchestrates the entire sync pipeline
-   Coordinates note discovery, parsing, card generation, and Anki updates
-   Handles incremental sync, resumable syncs, and batch operations
-   Manages state via SQLite database (`state_db.py`)

**Parser System** (`obsidian/parser.py`):

-   Extracts Q&A pairs from Obsidian markdown notes
-   Supports both regex-based and LLM-based extraction (configurable via `use_agent_system`)
-   Handles YAML frontmatter, multi-pair blocks, and bilingual content

**Agent Orchestration** (`agents/`):

-   Two orchestrator options: Legacy (`orchestrator.py`) or LangGraph (`langgraph_orchestrator.py`)
-   LangGraph uses state machine workflow with conditional routing and retries
-   Multi-stage pipeline: Pre-Validator → Generator → Post-Validator
-   Optional enhancement agents: Context Enrichment, Memorization Quality, Card Splitting, Duplicate Detection

**LLM Provider System** (`providers/`):

-   Unified interface via `BaseLLMProvider` (`base.py`)
-   Factory pattern (`factory.py`) creates provider instances from config
-   Providers: Ollama, OpenAI, Anthropic, OpenRouter, LM Studio
-   PydanticAI integration for structured outputs (`pydantic_ai_models.py`)

**Configuration** (`config.py`):

-   Pydantic-based settings with YAML + environment variable support
-   Model preset system: "cost_effective", "balanced", "high_quality", "fast"
-   Per-task model overrides via `get_model_for_agent()` and `get_model_config_for_task()`
-   Security: path validation, API key validation, symlink prevention

**State Management** (`sync/state_db.py`):

-   SQLite with WAL mode for concurrency
-   Tracks note hashes, card metadata, sync sessions
-   Supports resumable syncs via progress tracking

### Data Flow

1. **Discovery**: `discover_notes()` finds markdown files in vault
2. **Parsing**: `parse_note()` extracts Q&A pairs and metadata
3. **Slug Generation**: `generate_slug()` creates stable IDs with collision resolution
4. **Card Generation**:
    - Agent System: Pre-Validator → Generator → Post-Validator
    - Direct: `APFGenerator` creates cards from Q&A pairs
5. **Anki Sync**: `AnkiClient` communicates with AnkiConnect API
6. **State Update**: `StateDB` persists sync state for incremental/resumable syncs

### Key Design Patterns

**Provider Factory**: `ProviderFactory.create_from_config()` instantiates LLM providers based on `config.llm_provider`

**Transaction System**: `CardTransaction` (`sync/transactions.py`) provides rollback capability for Anki operations

**Content Hashing**: `compute_content_hash()` detects changes for incremental sync

**Deterministic GUIDs**: `deterministic_guid()` ensures stable Anki note IDs across syncs

**Model Configuration Cascade**:

1. Check explicit per-task override (e.g., `generator_model`)
2. Use preset default (`model_preset` → `ModelTask` → model name)
3. Fallback to `default_llm_model`

## Important Implementation Details

### Agent System

The agent system has two implementations:

-   **Legacy** (`agents/orchestrator.py`): Simple sequential pipeline
-   **LangGraph** (`agents/langgraph_orchestrator.py`): State machine with conditional routing, persistence

Enable via config:

```yaml
use_agent_system: true
use_langgraph: true # Use LangGraph orchestrator
use_pydantic_ai: true # Use PydanticAI for structured outputs
```

LangGraph workflow nodes:

-   `pre_validate_node`: Validates note structure
-   `card_splitting_node`: Analyzes if note should be split (optional)
-   `generate_node`: Creates flashcard content
-   `post_validate_node`: Quality validation with auto-fix and retry
-   `context_enrichment_node`: Adds examples/mnemonics (optional)
-   `memorization_quality_node`: Checks SRS effectiveness (optional)

### Model Selection

Models are configured via preset + optional overrides:

```yaml
# Preset approach (recommended)
model_preset: "balanced" # or "cost_effective", "high_quality", "fast"

# Override specific agents
generator_model: "openai/gpt-4-turbo-preview"
pre_validator_model: "qwen/qwen-2.5-14b-instruct"
```

Access in code:

```python
model = config.get_model_for_agent("generator")
model_config = config.get_model_config_for_task("generation")
```

### APF Format

Cards follow APF v2.1 specification:

-   `APFGenerator` (`apf/generator.py`): Converts Q&A pairs to APF cards
-   `validate_apf()` (`apf/linter.py`): Validates against APF spec
-   `validate_card_html()` (`apf/html_validator.py`): HTML syntax validation
-   `map_apf_to_anki_fields()` (`anki/field_mapper.py`): Maps APF to Anki note type

### Sync Modes

**Incremental Sync**: `--incremental` only processes notes not yet in `StateDB`

**Resumable Sync**: Session ID + checkpoint system allows recovering from interruptions

**Dry Run**: `--dry-run` previews all changes without modifying Anki

### Path Security

All path handling uses `utils/path_validator.py`:

-   `validate_vault_path()`: Prevents symlink attacks
-   `validate_source_dir()`: Prevents path traversal (`..`)
-   `validate_db_path()`: Ensures DB is in safe location

### Error Handling

Custom exceptions in `exceptions.py`:

-   `ConfigurationError`: Invalid config with helpful suggestions
-   `ParserError`: Note parsing failures
-   `AnkiConnectError`: Anki communication issues
-   `CardOperationError`: Card transaction failures
-   `PreValidationError`, `PostValidationError`: Agent validation errors

Always log errors with context:

```python
logger.error("operation_failed", note_path=str(path), error=str(e))
```

## Code Style (from .cursorrules)

### Python Standards

-   Python 3.13+ (note: pyproject.toml specifies 3.13, .cursorrules mentions 3.14)
-   Line length: 88 characters (Black default)
-   Type hints required for all public APIs
-   Google-style docstrings
-   No bare `except Exception` - use specific exceptions

### Naming

-   `snake_case` for functions/variables
-   `PascalCase` for classes
-   `UPPER_CASE` for constants

### Database

-   SQLite with WAL mode
-   Parameterized queries only (prevent SQL injection)
-   Index frequently queried columns

### Logging

-   Use `get_logger(__name__)` from `utils/logging.py`
-   Structured logging with context (file, slug, operation)
-   Never log secrets (API keys, tokens)
-   Levels: DEBUG (dev), INFO (ops), ERROR (failures)

### Testing

-   Minimum 90% coverage for new code
-   Test both success and failure cases
-   Use fixtures from `conftest.py`
-   Descriptive test names: `test_feature_with_valid_input`

### Commit Messages

Format: `<type>(<scope>): <subject>`

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example: `feat(parser): add multi-line question support`

## Configuration Files

**config.yaml**: Main service configuration (vault path, LLM settings, agent config)

**.env**: Secrets (API keys) - never commit

**pyproject.toml**: Dependencies, tool config (ruff, mypy, pytest)

**.cursorrules**: Detailed code style and project standards

## Common Patterns

### Adding New LLM Provider

1. Create `providers/new_provider.py` inheriting `BaseLLMProvider`
2. Implement `generate()`, `generate_structured()`, `validate_config()`
3. Add to `ProviderFactory.PROVIDER_MAP` in `factory.py`
4. Add config fields to `Config` class in `config.py`
5. Update validation in `Config.validate()`

### Adding New Agent

1. Create agent in `agents/new_agent.py`
2. Define Pydantic result model in `agents/models.py`
3. Add to LangGraph workflow in `langgraph_orchestrator.py`:
    - Add state field
    - Create node function
    - Add to graph edges
4. Add config flags to `Config` class
5. Update model preset in `models/config.py`

### Running Single Test During Development

```bash
# Run specific test function with output
uv run pytest tests/test_parser.py::test_parse_note_with_qa_pairs -v -s

# Run test file with coverage report
uv run pytest tests/test_parser.py --cov=src/obsidian_anki_sync/obsidian/parser --cov-report=term-missing
```

## Performance Considerations

-   **Batch Operations**: Enable `enable_batch_operations: true` for batch Anki/DB writes
-   **Concurrent Generations**: Set `max_concurrent_generations: 5` for parallel card generation
-   **Connection Pooling**: HTTP clients use connection pooling (see `providers/base.py`)
-   **Caching**: LLM responses cached with `diskcache` (see `sync/engine.py`)
-   **Indexing**: Full index built once per sync (`sync/indexer.py`), use `--no-index` to skip

## Security Requirements

-   Never hardcode secrets
-   Validate all user input
-   Use parameterized SQL queries
-   Validate file paths (prevent traversal)
-   Use HTTPS for external APIs
-   Verify SSL certificates
-   Path symlink prevention via `allow_symlinks=False`
