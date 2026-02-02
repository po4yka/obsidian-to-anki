Run all quality checks: format, lint, typecheck, and test.

```bash
uv run ruff format . && uv run isort . && uv run ruff check . && uv run mypy src/ && uv run pytest --cov -x -q
```

Fix any issues before proceeding. Address them in order: formatting, linting, type errors, then test failures.
