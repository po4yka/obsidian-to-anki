# Code Style

## Python Standards

- Python 3.13+
- Line length: 88 characters (Black default)
- Type hints required for all public APIs
- Google-style docstrings
- No bare `except Exception` -- use specific exceptions

## Naming

- `snake_case` for functions/variables
- `PascalCase` for classes
- `UPPER_CASE` for constants

## Database

- SQLite with WAL mode
- Parameterized queries only (prevent SQL injection)
- Index frequently queried columns

## Logging

- Use `get_logger(__name__)` from `utils/logging.py`
- Structured logging with context (file, slug, operation)
- Never log secrets (API keys, tokens)
- Levels: DEBUG (dev), INFO (ops), ERROR (failures)

## Testing

- Minimum 90% coverage for new code
- Test both success and failure cases
- Use fixtures from `conftest.py`
- Descriptive test names: `test_feature_with_valid_input`

## Commit Messages

Format: `<type>(<scope>): <subject>`

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example: `feat(parser): add multi-line question support`

## Security

- Never hardcode secrets
- Validate all user input
- Use parameterized SQL queries
- Validate file paths (prevent traversal)
- Use HTTPS for external APIs
- Verify SSL certificates
- Path symlink prevention via `allow_symlinks=False`
