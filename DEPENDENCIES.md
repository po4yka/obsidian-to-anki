# Dependencies

This document lists all project dependencies with their locked versions.

## Environment Management

- **Tool**: [uv](https://docs.astral.sh/uv/) v0.8.17
- **Python Version**: 3.14.0rc2
- **Lock File**: `uv.lock` (556 lines, 39 packages total)

## Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| click | 8.3.0 | CLI framework |
| pyyaml | 6.0.3 | YAML parsing for frontmatter |
| httpx | 0.28.1 | HTTP client for AnkiConnect |
| openai | 2.3.0 | OpenRouter LLM integration |
| python-dotenv | 1.1.1 | Environment variable management |
| structlog | 25.4.0 | Structured logging |
| ruamel.yaml | 0.18.15 | Advanced YAML processing |

## Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | 8.4.2 | Testing framework |
| pytest-cov | 7.0.0 | Code coverage |
| pytest-mock | 3.15.1 | Mocking support |
| respx | 0.22.0 | HTTP client mocking |
| black | 25.9.0 | Code formatting |
| ruff | 0.14.0 | Linting |
| mypy | 1.18.2 | Type checking |

## Key Transitive Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pydantic | 2.12.0 | Data validation (via openai) |
| anyio | 4.11.0 | Async I/O (via httpx) |
| certifi | 2025.10.5 | SSL certificates |
| httpcore | 1.0.9 | HTTP protocol (via httpx) |

## Installation

### Using uv (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv sync --all-extras

# Activate environment
source .venv/bin/activate
```

### Using pip

```bash
# Create virtual environment
python3.14 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

## Updating Dependencies

To update all dependencies to their latest compatible versions:

```bash
# Update lock file
uv lock --upgrade

# Sync environment
uv sync --all-extras
```

To update a specific package:

```bash
# Update specific package
uv lock --upgrade-package <package-name>

# Sync environment
uv sync --all-extras
```

## Version Constraints

All dependencies use minimum version constraints (`>=`) to allow for compatible updates while maintaining the lock file for reproducibility:

- **Production dependencies**: Locked to specific versions in `uv.lock`
- **Development dependencies**: Also locked for consistent development environment
- **Python version**: Requires Python >= 3.14

## Notes

- The `uv.lock` file ensures reproducible builds across all environments
- All dependencies are pinned to exact versions in the lock file
- The lock file includes SHA256 hashes for security verification
- Dependencies are automatically resolved for compatibility by uv

