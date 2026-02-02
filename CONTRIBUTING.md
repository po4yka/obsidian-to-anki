# Contributing

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- [pre-commit](https://pre-commit.com/)

## Setup

```bash
uv sync --dev
uv run pre-commit install
```

Or use the Makefile:

```bash
make setup
```

## Development Workflow

1. Create a feature branch from `main`
2. Make changes in small, focused commits
3. Run quality checks before pushing:
   ```bash
   make check  # format + lint + typecheck + test
   ```
4. Open a pull request against `main`

## Code Style

- Line length: 88 characters
- Type hints required for all public APIs
- Google-style docstrings
- No bare `except Exception` -- use specific exceptions
- See `.claude/docs/STYLE.md` for full conventions

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example: `feat(parser): add multi-line question support`

## Testing

```bash
# Run all tests with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_parser.py

# Run specific test
uv run pytest tests/test_parser.py::test_parse_note_with_qa_pairs -v -s

# Skip integration tests
uv run pytest -m "not integration"
```

Minimum 90% coverage for new code. Test both success and failure cases.

## Quality Checks

```bash
make format      # ruff format + isort
make lint        # ruff check
make typecheck   # mypy
make test        # pytest
make check       # all of the above
```
