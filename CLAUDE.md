# CLAUDE.md

Obsidian-to-Anki: sync Obsidian markdown notes to Anki flashcards via LLM-powered card generation.

## Commands

```bash
# Format + lint (REQUIRED before commits)
uv run ruff format . && uv run isort . && uv run ruff check .

# Type check
uv run mypy src/

# Test with coverage
uv run pytest --cov -x

# All quality checks
uv run ruff format . && uv run isort . && uv run ruff check . && uv run mypy src/ && uv run pytest --cov -x
```

After every atomic change, run `uv run ruff check .` and fix issues immediately.

## Source Tree

```
src/obsidian_anki_sync/
  agents/              # LLM agent orchestration (LangGraph, PydanticAI)
    langgraph/         # LangGraph state machine orchestrator
    parser_repair/     # Parser repair agent
    pydantic_ai/       # PydanticAI-based agents
    specialized/       # Domain-specific agents
    post_validation/   # Post-validation and auto-fix
  anki/                # AnkiConnect client and field mapping
  apf/                 # APF card format: generator, linter, HTML validator
  application/         # Application layer: factories, services, use cases
  cli_commands/        # CLI command implementations
  config.py            # Pydantic config with YAML + env var support
  domain/              # Domain entities, interfaces, services
  infrastructure/      # Cache, LLM infrastructure
  models/              # Model presets and configuration
  obsidian/            # Markdown parser, frontmatter, Q&A extraction
  prompts/             # Agent prompt templates
  providers/           # LLM providers: Ollama, LM Studio, OpenRouter
  rag/                 # RAG indexing and retrieval
  sync/                # Sync engine, state DB, transactions, change applier
  utils/               # Logging, path validation, preflight checks
  validation/          # Note validation scripts
  cli.py               # Typer CLI entry point
  exceptions.py        # Custom exception hierarchy
```

## Key Invariants

- All path operations go through `utils/path_validator.py` (symlink prevention, traversal checks)
- State DB uses SQLite WAL mode with parameterized queries only
- LLM providers implement `BaseLLMProvider` interface; created via `ProviderFactory`
- Agent pipeline: Pre-Validator -> Generator -> Post-Validator (optional enhancement agents)
- Cards follow APF v2.1 spec; validated by `apf/linter.py`
- Content hashing (`compute_content_hash()`) drives incremental sync
- Model selection cascade: per-task override -> preset default -> `default_llm_model`

## Deep-Dive Docs

| Topic | File |
|---|---|
| Architecture & data flow | `.claude/docs/ARCHITECTURE.md` |
| Common patterns (add provider, add agent) | `.claude/docs/PATTERNS.md` |
| Code style & conventions | `.claude/docs/STYLE.md` |
| Project setup & usage | `README.md` |
| APF format reference | `.docs/TEMPLATES/` |
| AnkiConnect API | `.docs/reference/ANKICONNECT_API.md` |
