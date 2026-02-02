Format and lint the codebase:

```bash
uv run ruff format . && uv run isort . && uv run ruff check .
```

Fix any linting errors reported by ruff before proceeding.
