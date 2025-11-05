# Obsidian to Anki APF Sync Service

Synchronize Obsidian Q&A notes to Anki APF cards using LLM-powered generation.

## Features

- Parse Obsidian markdown notes with Q&A pairs
- Generate APF-compliant Anki cards via OpenRouter LLM
- Bidirectional synchronization (create/update/delete/restore)
- Bilingual card support (EN/RU)
- Stable slug-based tracking with collision resolution
- SQLite state management
- Dry-run mode for previewing changes

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management.

### Prerequisites

Install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd obsidian-interview-qa-to-anki
```

2. Create virtual environment and install dependencies:

```bash
uv sync --all-extras
```

This will:
- Create a `.venv` directory with Python 3.11+
- Install all dependencies with locked versions from `uv.lock`
- Install the project in editable mode

3. Activate the virtual environment:

```bash
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

4. (Optional) Enable Git hooks and formatter tooling:

```bash
pre-commit install
```

## Code Style

The project uses Black and isort for formatting and import sorting. Run them with uv:

```bash
uv run black .
uv run isort .
```

These tools are also available via the pre-commit hook chain for consistent formatting on commit.

## Continuous Integration

Every push and pull request runs Black, isort, and pytest in GitHub Actions to keep formatting and tests consistent.

### Alternative: Traditional pip

If you prefer using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

1. Initialize configuration:

```bash
obsidian-anki-sync init
```

2. Edit `.env` with your settings:
   - `VAULT_PATH`: Path to your Obsidian vault
   - `OPENROUTER_API_KEY`: Your OpenRouter API key
   - Other configuration as needed

3. Run synchronization:

```bash
obsidian-anki-sync sync
```

Or preview changes without applying:

```bash
obsidian-anki-sync sync --dry-run
```

Inspect decks/models via diagnostics:

```bash
obsidian-anki-sync decks
obsidian-anki-sync models
obsidian-anki-sync model-fields --model "APF::Simple"
```

Run a sample dry-run on a random subset (default 10 notes):

```bash
obsidian-anki-sync test-run --count 10
```

## Documentation

See `.docs/REQUIREMENTS.md` for detailed requirements and specifications.

## License

MIT
