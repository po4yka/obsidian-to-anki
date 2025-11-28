# Obsidian to Anki APF Sync Service

Synchronize Obsidian Q&A notes to Anki flashcards using LLM-powered generation with multi-agent validation.

[![Tests](https://img.shields.io/badge/tests-539%20passed-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-35%25-yellow)]()
[![Python](https://img.shields.io/badge/python-3.13%2B-blue)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

-   **🔄 Bidirectional Sync**: Full synchronization between Obsidian notes and Anki cards
-   **🤖 Multi-Agent LLM System**: Advanced agent orchestration with LangGraph and PydanticAI
-   **🌐 Multi-Provider Support**: Ollama, OpenAI, Anthropic, OpenRouter, LM Studio
-   **🧠 Memory-Enhanced Learning**: Continuous quality improvement through pattern learning
-   **🔒 Privacy-First Architecture**: 100% local processing option with Ollama
-   **🌍 Bilingual Support**: English/Russian content with automatic language detection
-   **📊 Progress Tracking**: Resumable syncs with detailed progress reporting
-   **🛡️ Security First**: Path validation, API key protection, input sanitization
-   **📈 Quality Assurance**: Multi-stage validation with auto-fix capabilities
-   **🔍 RAG Integration**: Retrieval-augmented generation for enhanced context
-   **📋 Batch Operations**: Efficient bulk processing with transaction rollback
-   **🎯 APF v2.1 Compliance**: Strict adherence to Anki Prompts Format specifications
-   **📊 Validation Suite**: Comprehensive note validation and repair tools

## Table of Contents

-   [Features](#features)
-   [Multi-Agent Architecture](#multi-agent-architecture)
-   [System Architecture](#system-architecture)
-   [Data Flow](#data-flow)
-   [Quick Start](#quick-start)
-   [Providers Comparison](#providers-comparison)
-   [System Requirements](#system-requirements)
-   [Commands](#commands)
-   [Development](#development)
-   [Security Features](#security-features)
-   [Troubleshooting](#troubleshooting)
-   [Documentation](#documentation)
-   [Contributing](#contributing)
-   [License](#license)

## Multi-Agent Architecture

```mermaid
flowchart TD
    A[Obsidian Note] --> B{Pre-Validator}
    B -->|Invalid| Z[Skip/Repair]
    B -->|Valid| C[Context Enrichment<br/>Optional]
    C --> D[Card Splitting<br/>Optional]
    D --> E[Generator]
    E --> F{Post-Validator}
    F -->|Valid| G[APF Card]
    F -->|Invalid| H[Auto-Fix & Retry]
    H -->|Success| G
    H -->|Failed| Z

    style B fill:#e1f5fe
    style E fill:#f3e5f5
    style F fill:#e8f5e8
```

**Key Benefits**:

-   **🚀 Performance**: 15-20% faster through early rejection
-   **🎯 Quality**: Multi-stage validation with auto-correction
-   **🔄 Resilience**: Automatic retry with different strategies
-   **🧠 Learning**: Pattern recognition improves over time
-   **🔒 Privacy**: 100% local processing with Ollama

## System Architecture

```mermaid
graph TB
    subgraph "Input Sources"
        A[Obsidian Vault<br/>Markdown Notes]
        B[Anki Database<br/>Existing Cards]
    end

    subgraph "Core Engine"
        C[Sync Engine<br/>Orchestrator]
        D[Parser<br/>Q&A Extraction]
        E[Agent System<br/>LLM Processing]
        F[APF Generator<br/>Card Formatting]
    end

    subgraph "External Services"
        G[LLM Providers<br/>Ollama/OpenAI/Anthropic]
        H[AnkiConnect API<br/>Card Management]
        I[SQLite Database<br/>State & Progress]
    end

    subgraph "Output"
        J[Generated Cards<br/>APF Format]
        K[Sync Reports<br/>Statistics]
    end

    A --> D
    B --> C
    D --> E
    E --> F
    F --> J
    C --> H
    C --> I
    G --> E

    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style J fill:#c8e6c9
    style K fill:#c8e6c9
    style E fill:#fff3cd
    style G fill:#f8d7da
```

## Memory-Enhanced Learning

The system includes advanced memory capabilities that continuously improve card quality:

-   **Pattern Recognition**: Learns successful card generation patterns per topic
-   **User Preferences**: Adapts to individual user preferences for card style and difficulty
-   **Quality Feedback**: Uses memorization quality assessments to optimize future generations
-   **Intelligent Routing**: Automatically selects the best AI agent framework per content

**Memory Benefits**: 20-30% quality improvement, personalized generation, continuous learning.

## Data Flow

```mermaid
sequenceDiagram
    participant O as Obsidian Vault
    participant P as Parser
    participant V as Pre-Validator
    participant G as Generator
    participant PV as Post-Validator
    participant F as APF Formatter
    participant A as AnkiConnect
    participant D as SQLite DB

    O->>P: Scan for .md files
    P->>V: Extract Q&A pairs
    V->>V: Structure validation
    V->>G: Valid notes
    G->>PV: Generate cards
    PV->>PV: Quality check
    PV->>F: Valid cards
    F->>A: Format as APF
    A->>A: Add/Update cards
    A->>D: Store sync state
    D->>D: Track progress

    Note over V,PV: Auto-retry on failure
    Note over G: Memory-enhanced<br/>pattern learning
    Note over A: Transaction rollback<br/>on errors
```

See [Agent System](.docs/ARCHITECTURE/agents.md) for detailed documentation.

## Quick Start

### Prerequisites

-   **Python 3.13+**
-   **Anki** with AnkiConnect plugin
-   **uv** package manager (recommended) or pip

### Setup Flow

```mermaid
flowchart TD
    A[Install Python 3.13+] --> B[Install uv]
    B --> C[Install Anki]
    C --> D[Install AnkiConnect<br/>Addon 2055492159]
    D --> E[Choose LLM Provider]
    E --> F{Ollama/Local}
    E --> G{Cloud API}
    F --> H[Install Ollama<br/>Pull Models]
    G --> I[Get API Key]
    H --> J[Clone Repository]
    I --> J
    J --> K[Setup Environment<br/>uv sync --all-extras]
    K --> L[Create config.yaml]
    L --> M[Test Connection<br/>obsidian-anki-sync check]
    M --> N[First Sync<br/>obsidian-anki-sync sync]

    style A fill:#e3f2fd
    style C fill:#e3f2fd
    style N fill:#c8e6c9
```

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/po4yka/obsidian-to-anki.git
cd obsidian-to-anki
uv sync --all-extras
source .venv/bin/activate

# Alternative with pip
pip install -e .
```

### Anki Setup

1. Install [Anki](https://apps.ankiweb.net/)
2. Install AnkiConnect addon: Tools → Add-ons → Get Add-ons (code: `2055492159`)
3. Restart Anki

### Configuration

#### 1. Choose Your LLM Provider

**Local/Privacy (Recommended)** - Ollama:

```bash
brew install ollama
ollama serve
ollama pull llama3.2:3b qwen2.5:7b qwen2.5:14b
```

**Cloud Providers** - OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
```

**Cloud Providers** - Anthropic:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

#### 2. Create Configuration File

Create `config.yaml` in your project directory:

```yaml
# Vault Configuration
vault_path: "~/Documents/ObsidianVault"
source_dir: "Notes"
anki_deck_name: "My Deck"

# LLM Provider Settings
llm_provider: "ollama" # Options: ollama, openai, anthropic, openrouter, lm_studio

# Agent System Configuration
use_langgraph: true # Use advanced LangGraph orchestration
use_pydantic_ai: true # Use structured AI responses

# Model Configuration (adjust based on your provider)
pre_validator_model: "llama3.2:3b" # Fast, lightweight model
generator_model: "qwen2.5:14b" # Main generation model
post_validator_model: "qwen2.5:7b" # Quality validation

# Optional: Override models for specific providers
# generator_model: "gpt-4-turbo-preview"  # For OpenAI
# generator_model: "claude-3-5-sonnet-20241022"  # For Anthropic
```

### Usage

```bash
# Full synchronization
obsidian-anki-sync sync

# Dry run (preview changes without applying them)
obsidian-anki-sync sync --dry-run

# Incremental sync (only changed notes)
obsidian-anki-sync sync --incremental

# Test run with sample notes
obsidian-anki-sync test-run --count 5

# Validate note structure
obsidian-anki-sync validate "path/to/note.md"
```

## Providers Comparison

```mermaid
graph TD
    subgraph "Local Providers"
        A[Ollama<br/>🐧 Privacy-first<br/>📁 Offline capable<br/>⚡ Fast inference<br/>💾 High RAM usage]
        B[LM Studio<br/>🖥️ GUI interface<br/>🎨 User-friendly<br/>🔧 Local models<br/>⚙️ Easy setup]
    end

    subgraph "Cloud Providers"
        C[OpenAI<br/>🤖 GPT-4 quality<br/>🌐 Always available<br/>💰 Token-based pricing<br/>🔒 Enterprise security]
        D[Anthropic<br/>🧠 Claude excellence<br/>📊 Safety-focused<br/>💎 Premium quality<br/>🔑 API key required]
        E[OpenRouter<br/>🔄 Multi-model access<br/>🛒 Model marketplace<br/>💵 Cost optimization<br/>🎯 Best value]
    end

    A --> F{Choose Based On:}
    B --> F
    C --> F
    D --> F
    E --> F

    F --> G[Privacy Needs<br/>→ Ollama]
    F --> H[GUI Preference<br/>→ LM Studio]
    F --> I[Quality Priority<br/>→ Anthropic]
    F --> J[Cost Optimization<br/>→ OpenRouter]
    F --> K[Simplicity<br/>→ OpenAI]

    style A fill:#e8f5e8
    style B fill:#e3f2fd
    style C fill:#fff3cd
    style D fill:#f8d7da
    style E fill:#e2e3e5
```

## System Requirements

**Agent System (Ollama)**:

-   Mac M3/M4 with 32GB+ RAM
-   25GB storage for models
-   ~600-1200 cards/hour

**Cloud APIs**:

-   Any system with internet
-   No local requirements
-   Cost per token

## Commands

### Command Hierarchy

```mermaid
graph TD
    A[obsidian-anki-sync] --> B[Core Commands]
    A --> C[Configuration]
    A --> D[Anki Management]
    A --> E[Advanced Features]
    A --> F[RAG & Analysis]

    B --> B1[sync<br/>Full sync]
    B --> B2[sync --dry-run<br/>Preview changes]
    B --> B3[sync --incremental<br/>Changed notes only]
    B --> B4[test-run --count N<br/>Test with samples]

    C --> C1[init<br/>Initialize config]
    C --> C2[check<br/>Pre-flight checks]
    C --> C3[validate <file><br/>Validate note]
    C --> C4[lint-note <file><br/>Bilingual check]

    D --> D1[decks<br/>List decks]
    D --> D2[models<br/>List models]
    D --> D3[model-fields <model><br/>Show fields]
    D --> D4[export<br/>Export to .apkg]
    D --> D5[export-deck<br/>Export to YAML/CSV]
    D --> D6[import-deck<br/>Import from YAML/CSV]

    E --> E1[generate <term><br/>Multiple examples]
    E --> E2[process-file <file><br/>Process from file]
    E --> E3[index<br/>Show statistics]
    E --> E4[progress<br/>Sync progress]
    E --> E5[clean-progress<br/>Clean records]
    E --> E6[query <method><br/>API queries]

    F --> F1[rag index<br/>Index for RAG]
    F --> F2[rag search <query><br/>Search content]
    F --> F3[analyze-logs<br/>Log analysis]
    F --> F4[list-problematic-notes<br/>List issues]

    style A fill:#4a90e2,color:#fff
    style B fill:#e8f5e8
    style C fill:#fff3cd
    style D fill:#d1ecf1
    style E fill:#f8d7da
    style F fill:#e2e3e5
```

### Core Commands

```bash
obsidian-anki-sync sync              # Full synchronization
obsidian-anki-sync sync --dry-run    # Preview changes
obsidian-anki-sync sync --incremental # Only process changed notes
obsidian-anki-sync test-run --count 5 # Test with N random notes
```

### Configuration & Setup

```bash
obsidian-anki-sync init              # Initialize configuration
obsidian-anki-sync check             # Pre-flight system checks
obsidian-anki-sync validate <file>   # Validate note structure
obsidian-anki-sync lint-note <file>  # Lint for bilingual completeness
```

### Anki Management

```bash
obsidian-anki-sync decks             # List available decks
obsidian-anki-sync models            # List note models (types)
obsidian-anki-sync model-fields <model> # Show field names for a model
obsidian-anki-sync export            # Export notes to .apkg file
obsidian-anki-sync export-deck       # Export Anki deck to YAML/CSV
obsidian-anki-sync import-deck       # Import cards from YAML/CSV
```

### Advanced Features

```bash
obsidian-anki-sync generate <term>   # Generate multiple card examples
obsidian-anki-sync process-file <file> # Process cards from file
obsidian-anki-sync index             # Show vault and Anki statistics
obsidian-anki-sync progress          # Show recent sync progress
obsidian-anki-sync clean-progress    # Clean up progress records
obsidian-anki-sync query <method>    # Direct AnkiConnect API queries
```

### RAG & Analysis

```bash
obsidian-anki-sync rag index         # Index content for RAG
obsidian-anki-sync rag search <query> # Search indexed content
obsidian-anki-sync analyze-logs      # Analyze log files
obsidian-anki-sync list-problematic-notes # List archived issues
```

## Development

### Code Quality

```bash
# Run all quality checks
uv run ruff format . && uv run isort . && uv run ruff check . && uv run mypy src/

# Or use the built-in format command
obsidian-anki-sync format
```

### Testing

```bash
# Run all tests with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_parser.py

# Run tests with verbose output
uv run pytest -v

# Run only fast tests (skip integration)
uv run pytest -m "not integration"
```

### Development Commands

```bash
# Install in development mode
uv sync --all-extras

# Activate virtual environment
source .venv/bin/activate

# Run pre-commit hooks
uv run pre-commit run --all-files
```

## Security Features

-   **Path Validation**: Prevents `..` traversal, symlink attacks
-   **API Key Validation**: Provider-specific checks at startup
-   **Resource Cleanup**: Proper context managers for DB connections
-   **Specific Exceptions**: No bare `except:` blocks

## Troubleshooting

**Ollama not connecting**:

```bash
curl http://localhost:11434/api/tags
ollama serve
```

**AnkiConnect not responding**:

1. Ensure Anki is running
2. Check addon: Tools → Add-ons
3. Verify port: default `8765`

**Import errors**:

```bash
uv sync --all-extras
```

## Documentation

-   **[📚 Full Documentation](.docs/README.md)** - Complete guides and reference
-   **[🚀 Getting Started](.docs/GETTING_STARTED.md)** - Quick setup guide
-   **[🔧 Configuration](.docs/GUIDES/configuration.md)** - Model and provider setup
-   **[🔄 Synchronization](.docs/GUIDES/synchronization.md)** - Change management guide
-   **[🏗️ Architecture](.docs/ARCHITECTURE/README.md)** - System design overview
-   **[🃏 APF Format](.docs/ARCHITECTURE/apf.md)** - Card format specification
-   **[🤖 Agent System](.docs/ARCHITECTURE/agents.md)** - Multi-agent architecture
-   **[⚙️ LLM Providers](.docs/ARCHITECTURE/providers.md)** - Provider integration
-   **[🔍 Security Best Practices](#security-features)** - Security guidelines

## Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow

```mermaid
flowchart TD
    A[Fork Repository] --> B[Create Feature Branch<br/>feature/description]
    B --> C[Set up Environment<br/>uv sync --all-extras]
    C --> D[Make Changes<br/>Follow coding standards]
    D --> E[Add Tests<br/>Unit + Integration]
    E --> F[Run Quality Checks<br/>format + type-check + test]
    F --> G{Passes All?}
    G -->|No| H[Fix Issues]
    H --> F
    G -->|Yes| I[Commit Changes<br/>feat: description]
    I --> J[Push Branch]
    J --> K[Create Pull Request]
    K --> L[Code Review]
    L --> M{Approved?}
    M -->|No| N[Address Feedback]
    N --> L
    M -->|Yes| O[Merge to Main]

    style A fill:#e3f2fd
    style O fill:#c8e6c9
    style G fill:#fff3cd
    style M fill:#fff3cd
```

### Step-by-Step Guide

1. **Fork and Clone**

    ```bash
    git clone https://github.com/your-username/obsidian-to-anki.git
    cd obsidian-to-anki
    ```

2. **Set up Development Environment**

    ```bash
    uv sync --all-extras
    source .venv/bin/activate
    ```

3. **Create Feature Branch**

    ```bash
    git checkout -b feature/your-feature-name
    # or for bug fixes
    git checkout -b fix/issue-number-description
    ```

4. **Make Changes**

    - Follow the existing code style
    - Add tests for new functionality
    - Update documentation as needed
    - Ensure all tests pass

5. **Quality Checks**

    ```bash
    # Run all quality checks
    obsidian-anki-sync format
    uv run mypy src/
    uv run pytest --cov
    ```

6. **Commit Changes**

    ```bash
    git add .
    git commit -m "feat: add your feature description"
    ```

7. **Submit Pull Request**
    - Push your branch: `git push origin feature/your-feature-name`
    - Create PR with clear description
    - Reference any related issues

### Code Style Guidelines

-   **Python**: Follow PEP 8, use type hints, modern syntax (`list[str]` not `List[str]`)
-   **Commits**: Use conventional commits (`feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`)
-   **Documentation**: Update relevant docs in `.docs/` directory
-   **Tests**: Minimum 90% coverage, test both success and failure cases
-   **Security**: Never commit secrets, validate all inputs, use parameterized queries
