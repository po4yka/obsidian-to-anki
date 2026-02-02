# Common Patterns

## Adding New LLM Provider

1. Create `providers/new_provider.py` inheriting `BaseLLMProvider`
2. Implement `generate()`, `generate_structured()`, `validate_config()`
3. Add to `ProviderFactory.PROVIDER_MAP` in `factory.py`
4. Add config fields to `Config` class in `config.py`
5. Update validation in `Config.validate()`

## Adding New Agent

1. Create agent in `agents/new_agent.py`
2. Define Pydantic result model in `agents/models.py`
3. Add to LangGraph workflow in `langgraph_orchestrator.py`:
   - Add state field
   - Create node function
   - Add to graph edges
4. Add config flags to `Config` class
5. Update model preset in `models/config.py`

## Running Single Test During Development

```bash
# Run specific test function with output
uv run pytest tests/test_parser.py::test_parse_note_with_qa_pairs -v -s

# Run test file with coverage report
uv run pytest tests/test_parser.py --cov=src/obsidian_anki_sync/obsidian/parser --cov-report=term-missing
```

## Path Security

All path handling uses `utils/path_validator.py`:
- `validate_vault_path()`: Prevents symlink attacks
- `validate_source_dir()`: Prevents path traversal (`..`)
- `validate_db_path()`: Ensures DB is in safe location

## Error Handling

Custom exceptions in `exceptions.py`:
- `ConfigurationError`: Invalid config with helpful suggestions
- `ParserError`: Note parsing failures
- `AnkiConnectError`: Anki communication issues
- `CardOperationError`: Card transaction failures
- `PreValidationError`, `PostValidationError`: Agent validation errors

Always log errors with context:
```python
logger.error("operation_failed", note_path=str(path), error=str(e))
```

## Configuration Files

| File | Purpose |
|---|---|
| `config.yaml` | Main service configuration (vault path, LLM settings, agent config) |
| `.env` | Secrets (API keys) -- never commit |
| `pyproject.toml` | Dependencies, tool config (ruff, mypy, pytest) |
| `.cursorrules` | Detailed code style and project standards |
