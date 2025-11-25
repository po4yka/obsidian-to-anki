# Launch Instructions - Obsidian to Anki APF Sync Service

Complete guide to set up and launch the Obsidian to Anki synchronization service.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Anki Setup](#anki-setup)
4. [LLM Provider Configuration](#llm-provider-configuration)
5. [Application Configuration](#application-configuration)
6. [Launching the Application](#launching-the-application)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

- **Python**: 3.13 or higher
- **Operating System**: Linux, macOS, or Windows
- **Anki**: Latest version (with AnkiConnect addon)
- **Storage**: ~500 MB for dependencies

### Additional Requirements for Local LLM (Ollama)

- **RAM**: 32GB+ recommended (for running multiple Qwen3 models)
- **Storage**: ~25GB for model files
- **CPU/GPU**: Apple Silicon (M3/M4) or NVIDIA GPU recommended

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/po4yka/obsidian-to-anki.git
cd obsidian-to-anki
```

### Step 2: Install uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal or run:

```bash
source ~/.bashrc  # or ~/.zshrc on macOS
```

### Step 3: Install Dependencies

**Option A: Using uv (Recommended)**

```bash
# Install all dependencies including dev tools
uv sync --all-extras

# Activate the virtual environment
source .venv/bin/activate
```

**Option B: Using pip**

```bash
# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Step 4: Verify Installation

```bash
# Check if the CLI is available
obsidian-anki-sync --help
```

You should see the CLI help menu with available commands.

---

## Anki Setup

### Install Anki

1. Download Anki from [https://apps.ankiweb.net/](https://apps.ankiweb.net/)
2. Install and launch Anki

### Install AnkiConnect Addon

AnkiConnect enables communication between this tool and Anki.

1. Open Anki
2. Go to **Tools** → **Add-ons**
3. Click **Get Add-ons...**
4. Enter code: `2055492159`
5. Click **OK**
6. Restart Anki

### Verify AnkiConnect

Ensure Anki is running, then test the connection:

```bash
curl http://localhost:8765
```

You should receive a response like:

```
AnkiConnect v.6
```

---

## LLM Provider Configuration

Choose one of the following LLM providers based on your needs:

### Option 1: OpenRouter (Recommended for Cloud)

OpenRouter provides access to multiple models through a single API.

1. Get an API key from [https://openrouter.ai/](https://openrouter.ai/)
2. Set the environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

To make it permanent, add to `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export OPENROUTER_API_KEY="sk-or-v1-..."' >> ~/.bashrc
source ~/.bashrc
```

### Option 2: Ollama (Local/Privacy-First)

Best for offline processing and privacy.

**Install Ollama:**

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

**Start Ollama server:**

```bash
ollama serve
```

**Pull required models:**

```bash
ollama pull qwen3:8b     # Pre-validator (fast)
ollama pull qwen3:14b    # Post-validator (balanced)
ollama pull qwen3:32b    # Generator (high quality)
```

Note: This will download ~25GB of model files.

**Verify installation:**

```bash
curl http://localhost:11434/api/tags
```

### Option 3: OpenAI

1. Get an API key from [https://platform.openai.com/](https://platform.openai.com/)
2. Set the environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

### Option 4: Anthropic (Claude)

1. Get an API key from [https://console.anthropic.com/](https://console.anthropic.com/)
2. Set the environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Option 5: LM Studio (Local with GUI)

1. Download LM Studio from [https://lmstudio.ai/](https://lmstudio.ai/)
2. Load your preferred model
3. Start the local server (default: `http://localhost:1234`)

---

## Application Configuration

### Create Configuration File

Copy the example configuration:

```bash
cp config.yaml.example config.yaml
```

### Edit Configuration

Open `config.yaml` in your text editor and configure the following sections:

#### Essential Settings

```yaml
# Path to your Obsidian vault
vault_path: "/Users/yourname/Documents/ObsidianVault"

# Subdirectory containing notes to sync
source_dir: "anki"

# Anki deck to sync to
anki_deck_name: "Obsidian::Imported"

# Choose your LLM provider
llm_provider: "openrouter"  # or "ollama", "openai", "anthropic", "lm_studio"

# Set default model (for OpenRouter)
default_llm_model: "openrouter/polaris-alpha"
```

#### For Ollama Users

```yaml
llm_provider: "ollama"
default_llm_model: "qwen3:32b"

# Enable agent system for multi-stage validation
use_langgraph: true
pre_validator_model: "qwen3:8b"
generator_model: "qwen3:32b"
post_validator_model: "qwen3:14b"
```

#### For OpenAI Users

```yaml
llm_provider: "openai"
default_llm_model: "gpt-4o-mini"

# Optional: Use different models for different agents
# generator_model: "gpt-4o"
# pre_validator_model: "gpt-4o-mini"
```

#### Optional: Multiple Source Directories

If your notes are in multiple folders:

```yaml
source_subdirs:
  - "Interviews"
  - "CS/Algorithms"
  - "Projects/Notes"
```

### Full Configuration Reference

See `config.yaml.example` for all available options including:

- Enhancement agents (card splitting, context enrichment, memorization quality)
- LangGraph workflow settings
- LLM parameters (temperature, timeout, max tokens)
- Logging levels
- Database paths

---

## Launching the Application

### Initialize Configuration (Optional)

Generate a fresh config file interactively:

```bash
obsidian-anki-sync init
```

### List Available Anki Decks

Verify connectivity with Anki:

```bash
obsidian-anki-sync decks
```

### Test Run

Process a small sample before full sync:

```bash
# Process 5 random notes in dry-run mode
obsidian-anki-sync test-run --count 5 --dry-run
```

Review the output to ensure cards are generated correctly.

### Dry Run (Preview Changes)

See what will be synced without making changes:

```bash
obsidian-anki-sync sync --dry-run
```

This will:
- Show which notes will be processed
- Display preview of generated cards
- Not create/update/delete any Anki cards

### Full Synchronization

Run the complete sync:

```bash
obsidian-anki-sync sync
```

**With agent system (multi-stage validation):**

```bash
obsidian-anki-sync sync --use-agents
```

### Validate Individual Notes

Check if a note has valid structure:

```bash
obsidian-anki-sync validate path/to/note.md
```

### Export to Anki Package

Export cards to `.apkg` file:

```bash
obsidian-anki-sync export
```

---

## Verification

### Check Sync Status

After running sync, verify in Anki:

1. Open Anki
2. Navigate to your configured deck (e.g., "Obsidian::Imported")
3. Review newly created cards
4. Check that tags are correct
5. Verify card content and formatting

### Check Database

The sync state is stored in SQLite:

```bash
# Default location (relative to vault)
ls -la .obsidian/plugins/obsidian-to-anki/sync.db
```

### View Logs

Logs provide detailed information about the sync process:

```bash
# Increase verbosity for debugging
obsidian-anki-sync sync --dry-run --verbose
```

Or edit `config.yaml`:

```yaml
log_level: "DEBUG"  # Change from INFO to DEBUG
```

---

## Troubleshooting

### Common Issues

#### 1. "Command not found: obsidian-anki-sync"

**Solution:** Ensure the virtual environment is activated:

```bash
source .venv/bin/activate
```

Or use `uv run` directly:

```bash
uv run obsidian-anki-sync --help
```

#### 2. "Cannot connect to Anki"

**Symptoms:** Error connecting to `http://localhost:8765`

**Solutions:**

1. Ensure Anki is running
2. Check AnkiConnect addon is installed:
   - Anki → Tools → Add-ons
   - Look for "AnkiConnect"
3. Restart Anki
4. Verify port:
   ```bash
   curl http://localhost:8765
   ```

#### 3. "Ollama not responding"

**Symptoms:** Timeout or connection errors to Ollama

**Solutions:**

1. Check if Ollama is running:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. Start Ollama server:
   ```bash
   ollama serve
   ```

3. Verify models are downloaded:
   ```bash
   ollama list
   ```

4. Pull missing models:
   ```bash
   ollama pull qwen3:32b
   ```

#### 4. "API Key not found"

**Symptoms:** Environment variable errors for OpenAI/Anthropic/OpenRouter

**Solutions:**

1. Set the API key in your shell:
   ```bash
   export OPENROUTER_API_KEY="your-key-here"
   ```

2. Or add directly to `config.yaml`:
   ```yaml
   openrouter_api_key: "sk-or-v1-..."
   ```

3. Verify the variable is set:
   ```bash
   echo $OPENROUTER_API_KEY
   ```

#### 5. "Import errors" or "Module not found"

**Solution:** Reinstall dependencies:

```bash
# Using uv
uv sync --all-extras

# Using pip
pip install -e ".[dev]"
```

#### 6. "Vault path not found"

**Solution:** Check `config.yaml` and ensure `vault_path` is absolute and correct:

```yaml
vault_path: "/full/path/to/your/vault"  # Not ~/vault
```

Expand `~` manually:

```bash
vault_path: "/Users/yourname/Documents/ObsidianVault"
```

#### 7. Low Performance / Timeouts

**For Ollama users:**

- Ensure sufficient RAM (32GB+ for multi-agent)
- Use smaller models for agents:
  ```yaml
  pre_validator_model: "qwen3:8b"
  ```

- Increase timeout in config:
  ```yaml
  llm_timeout: 900.0  # 15 minutes
  ```

**For Cloud API users:**

- Check network connection
- Verify API rate limits
- Consider upgrading API plan

#### 8. Cards not appearing in Anki

**Solutions:**

1. Check deck name matches:
   ```bash
   obsidian-anki-sync decks
   ```

2. Verify note type exists (should be created automatically)

3. Check for errors in sync output

4. Try manual refresh in Anki: `Ctrl+R` / `Cmd+R`

---

## Development Mode

### Code Formatting

```bash
uv run black . && uv run isort .
```

### Linting

```bash
uv run ruff check .
```

### Type Checking

```bash
uv run mypy src/
```

### Running Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov

# Specific test file
uv run pytest tests/test_parser.py
```

See [TESTING.md](TESTING.md) for detailed testing information.

---

## Next Steps

1. Configure your Obsidian vault with Q&A notes
2. Run a test sync with `--dry-run`
3. Review generated cards in Anki
4. Adjust prompts/configuration as needed
5. Schedule regular syncs (e.g., via cron)

## Support

- **Documentation**: See other `.md` files in repository
- **Issues**: [GitHub Issues](https://github.com/po4yka/obsidian-to-anki/issues)
- **Provider Guides**: Check `.docs/LLM_PROVIDERS.md` (if exists)

---

**Happy syncing!**
