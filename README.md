# Obsidian to Anki APF Sync Service

Synchronize Obsidian Q&A notes to Anki APF cards using LLM-powered generation.

## Features

- Parse Obsidian markdown notes with Q&A pairs
- Generate APF-compliant Anki cards via OpenRouter LLM **or local multi-agent AI system**
- **NEW: Multi-Agent AI System** - Three-stage validation pipeline using local LLMs (Qwen3)
  - Pre-Validator: Fast structural checks before generation
  - Generator: High-quality card generation with powerful local models
  - Post-Validator: Quality validation with retry and auto-fix
- Bidirectional synchronization (create/update/delete/restore)
- Bilingual card support (EN/RU)
- Stable slug-based tracking with collision resolution
- SQLite state management
- Dry-run mode for previewing changes
- **100% Privacy**: Optional fully local processing with no cloud APIs

## Multi-Agent System (NEW!)

This project now supports a sophisticated multi-agent AI system for generating Anki flashcards using local LLMs running on Apple Silicon.

### Architecture

```
ObsidianNote → Pre-Validator → Generator → Post-Validator → AnkiCard
               (Qwen3-8B)      (Qwen3-32B)  (Qwen3-14B)
```

**Benefits:**
- **15-20% faster**: Pre-validator rejects malformed notes early
- **Higher quality**: Independent validation catches errors before sync
- **Auto-fix**: Automatic retry with corrections
- **Privacy-first**: 100% local processing, no cloud APIs
- **Cost-effective**: No API costs after initial model download

### System Requirements

**Minimum:**
- Mac: M3 or M4 series (any variant)
- RAM: 32GB unified memory
- Storage: 25GB for models

**Recommended:**
- Mac: M4 Max or M4 Ultra
- RAM: 48GB+ unified memory
- Storage: 50GB for models and cache

### Quick Start with Agents

1. **Install Ollama:**
```bash
brew install ollama
ollama serve
```

2. **Download Models:**
```bash
ollama pull qwen3:8b   # Pre-validator (~5GB)
ollama pull qwen3:32b  # Generator (~20GB)
ollama pull qwen3:14b  # Post-validator (~8GB)
```

3. **Enable in Config:**
```yaml
# config.yaml
use_agent_system: true
ollama_base_url: "http://localhost:11434"
```

Or use the example config:
```bash
cp config.agents.example.yaml config.yaml
# Edit config.yaml with your vault path and preferences
```

4. **Run with Agents:**
```bash
obsidian-anki-sync sync --use-agents
```

For detailed configuration options, see `config.agents.example.yaml` and `.docs/AGENT_INTEGRATION_PLAN.md`.

### Agent vs OpenRouter

| Feature | Agent System | OpenRouter |
|---------|-------------|------------|
| **Privacy** | ✅ 100% Local | ❌ Cloud API |
| **Cost** | ✅ Free after download | ❌ Per-token pricing |
| **Quality** | ✅ Three-stage validation | ⚠️ Single-pass |
| **Speed** | ⚠️ First run slower | ✅ Fast API calls |
| **Requirements** | ❌ 32GB+ RAM, M3+ Mac | ✅ Any system |
| **Setup** | ⚠️ Install Ollama + models | ✅ API key only |

**When to use which:**
- **Use Agents** if you have a compatible Mac, value privacy, and process many notes
- **Use OpenRouter** if you need quick setup, have limited hardware, or process notes infrequently

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
