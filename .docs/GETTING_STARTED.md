# Getting Started with Obsidian to Anki Sync

This guide will get you up and running with the Obsidian to Anki sync service in under 15 minutes.

## Prerequisites

-   **Obsidian** with Q&A notes in your vault
-   **Anki** installed on your system
-   **Python 3.11+** (managed via `uv`)
-   **Git** for cloning the repository

## 1. Installation

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/po4yka/obsidian-to-anki.git
cd obsidian-to-anki

# Install dependencies
uv sync --all-extras

# Activate virtual environment
source .venv/bin/activate
```

## 2. Anki Setup

1. **Install Anki**: Download from [apps.ankiweb.net](https://apps.ankiweb.net/)
2. **Install AnkiConnect**:
    - Open Anki → Tools → Add-ons → Get Add-ons
    - Enter code: `2055492159`
    - Restart Anki
3. **Verify**: AnkiConnect runs on `http://localhost:8765`

## 3. Choose Your LLM Provider

### Option A: Ollama (Recommended - Local & Private)

```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve

# Pull required models (in another terminal)
ollama pull qwen3:8b qwen3:14b qwen3:32b
```

**Pros**: Free, private, no API keys, works offline
**Cons**: Requires ~25GB disk space for models

### Option B: OpenAI

```bash
export OPENAI_API_KEY="sk-your-openai-api-key-here"  # pragma: allowlist secret
```

### Option C: Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here"  # pragma: allowlist secret
```

## 4. Configuration

Create a `config.yaml` file in the project root:

```yaml
# Required: Your Obsidian vault location
vault_path: "~/Documents/ObsidianVault"
source_dir: "Notes" # Subdirectory with Q&A notes
anki_deck_name: "My Deck"

# Required: Choose your LLM provider
llm_provider: "ollama" # or "openai", "anthropic", "openrouter", "lm_studio"

# Agent System (recommended for best results)
use_langgraph: true
use_pydantic_ai: true

# Model Configuration (Ollama example)
pre_validator_model: "qwen3:8b"
generator_model: "qwen3:32b"
post_validator_model: "qwen3:14b"
# For OpenAI (uncomment and modify if using OpenAI)
# generator_model: "gpt-4-turbo-preview"
# post_validator_model: "gpt-4"
# pre_validator_model: "gpt-3.5-turbo"
```

**💡 Tip**: Copy from `config.example.yaml` and modify for your setup.

## 5. First Sync

```bash
# Test with a small batch first
obsidian-anki-sync test-run --count 3

# Full synchronization
obsidian-anki-sync sync

# Preview changes without applying
obsidian-anki-sync sync --dry-run
```

## 6. Verify Results

1. **Open Anki** and check your deck
2. **Review cards** - they should have proper formatting and content
3. **Check logs** for any warnings or errors

## Next Steps

🎉 **Congratulations!** Your first sync is complete.

### Learn More

-   **[Configuration Guide](GUIDES/configuration.md)** - Advanced configuration options
-   **[Synchronization Guide](GUIDES/synchronization.md)** - Understanding sync behavior
-   **[Troubleshooting](GUIDES/troubleshooting.md)** - Common issues and solutions
-   **[Architecture Overview](ARCHITECTURE/README.md)** - How the system works

### Common Tasks

-   **Update configuration**: Edit `config.yaml` and re-run sync
-   **Add new notes**: Just add Q&A notes to your Obsidian vault
-   **Monitor sync progress**: Check the logs in `.logs/sync.log`
-   **Backup Anki**: Regular backups before major syncs

## Having Issues?

1. **Check AnkiConnect**: Visit `http://localhost:8765` in your browser
2. **Verify models**: Run `ollama list` (if using Ollama)
3. **Test configuration**: Run `obsidian-anki-sync validate <path-to-note>`
4. **Check logs**: Look in `.logs/` directory for detailed error messages

**Need help?** [Open an issue](https://github.com/po4yka/obsidian-to-anki/issues) or check the [troubleshooting guide](GUIDES/troubleshooting.md).

---

**Estimated setup time**: 10-15 minutes
**Estimated first sync time**: 1-5 minutes (depending on note count)
