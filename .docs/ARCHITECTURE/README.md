# System Architecture Overview

This directory contains detailed documentation about the Obsidian to Anki sync service architecture, including core components, data flow, and design decisions.

## System Overview

The Obsidian to Anki sync service is a sophisticated system that converts Obsidian Q&A notes into Anki flashcards using LLM-powered generation and validation. The system is built with a clean architecture approach, emphasizing separation of concerns, testability, and maintainability.

### Core Principles

-   **Clean Architecture**: Domain-driven design with clear separation between layers
-   **Multi-Agent System**: Intelligent validation and generation pipeline
-   **Provider Agnostic**: Support for multiple LLM providers
-   **Security First**: Path validation, API key management, input sanitization
-   **Performance Optimized**: Efficient batching, caching, and async processing

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                 CLI Interface                           │ │
│  │  - Command parsing and execution                        │ │
│  │  - Progress reporting and user feedback                 │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Application Layer                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Use Cases & Services                       │ │
│  │  - Sync orchestration                                    │ │
│  │  - Agent coordination                                    │ │
│  │  - Validation workflows                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Domain Layer                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Entities & Business Logic                  │ │
│  │  - Note, Card, SyncState entities                        │ │
│  │  - Business rules and invariants                        │ │
│  │  - Domain services                                       │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Infrastructure Layer                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │          External Systems & Persistence                 │ │
│  │  - AnkiConnect API integration                          │ │
│  │  - LLM provider clients                                 │ │
│  │  - File system operations                               │ │
│  │  - Database (SQLite)                                    │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Synchronization Engine

**Location**: `src/obsidian_anki_sync/sync/`

The heart of the system, responsible for orchestrating the entire sync process:

-   **Note Discovery**: Scanning vault for markdown files
-   **Change Detection**: Identifying modified notes using content hashing
-   **Agent Coordination**: Managing multi-agent validation pipeline
-   **Anki Integration**: Creating/updating/deleting cards via AnkiConnect
-   **State Management**: Tracking sync progress and handling resumability

**Key Classes**:

-   `SyncEngine`: Main orchestrator
-   `NoteScanner`: Vault scanning and processing
-   `StateDB`: SQLite-based state persistence

### 2. Agent System

**Location**: `src/obsidian_anki_sync/agents/`

Intelligent multi-agent pipeline for content validation and generation:

-   **Pre-Validator**: Early rejection of invalid inputs (15-20% faster)
-   **Generator**: Core card creation with LLM assistance
-   **Post-Validator**: Quality assurance and error correction

**Key Technologies**:

-   LangGraph for workflow orchestration
-   PydanticAI for type-safe structured outputs
-   LangChain agents for flexible agent implementations
-   OpenRouter for unified LLM provider access

**[Detailed Documentation](agents.md)**

### 3. LLM Providers

**Location**: `src/obsidian_anki_sync/providers/`

Unified interface to multiple LLM providers with automatic failover:

**Supported Providers**:

-   Ollama (local, privacy-focused)
-   OpenRouter (100+ models, unified API)
-   OpenAI (GPT models, reliability)
-   Anthropic (Claude models, reasoning)
-   LM Studio (GUI management, local)

**Features**:

-   Provider abstraction via `ILLMProvider` interface
-   Automatic model selection and optimization
-   Cost tracking and usage analytics
-   Circuit breaker pattern for reliability

**[Detailed Documentation](providers.md)**

### 4. APF Format System

**Location**: `src/obsidian_anki_sync/apf/`

Proprietary format for generating Anki flashcards optimized for spaced repetition:

**Components**:

-   **Generator**: Converts Q&A pairs to APF HTML
-   **Validator**: Ensures APF compliance
-   **Linter**: Automated format checking
-   **Field Mapper**: Maps APF to Anki note types

**Card Types**:

-   Simple: Question-answer format
-   Missing: Cloze deletion (fill-in-the-blank)
-   Draw: Diagram and sequence recall

**[Detailed Documentation](apf.md)**

## Data Flow

### Sync Process

```
Obsidian Vault → Note Discovery → Change Detection → Validation → Generation → Anki Sync
       ↓              ↓              ↓              ↓            ↓            ↓
   Markdown      File Scanning   Content Hash   Agent Pipeline  APF Cards   API Calls
   Files         & Filtering    Comparison     (Pre→Gen→Post)   Creation    & Updates
```

### Agent Pipeline

```
Input Note → Pre-Validator → Generator → Post-Validator → Output Cards
     ↓            ↓            ↓            ↓
   Validate    Generate     Validate     Finalize
  Structure    Content      Quality      Cards
```

### Provider Flow

```
User Request → Provider Selection → Model Routing → API Call → Response Processing
       ↓             ↓              ↓            ↓            ↓
   Config        Factory       OpenRouter    HTTP Request   Pydantic
   Parsing      Pattern       /Direct API    w/Retry       Validation
```

## Key Design Decisions

### ADR Index

The system follows Architecture Decision Records for major design choices:

-   **[ADR-001](ADR/ADR-001-adopt-clean-architecture.md)**: Clean Architecture adoption
-   **[ADR-002](ADR/ADR-002-dependency-injection-container.md)**: Dependency injection pattern
-   **[ADR-003](ADR/ADR-003-interface-segregation.md)**: Interface segregation principle

### Security Architecture

-   **Path Validation**: Prevents directory traversal attacks
-   **API Key Management**: Environment variables only, no hardcoded secrets
-   **Input Sanitization**: All external inputs validated and sanitized
-   **Symlink Prevention**: Detects and rejects symbolic links

### Performance Optimizations

-   **Batch Processing**: Multiple notes/cards processed together
-   **Async I/O**: Non-blocking API calls and file operations
-   **Caching**: Result caching for expensive LLM calls
-   **Connection Pooling**: Reuse HTTP connections

## Component Relationships

### Dependencies

```
CLI Commands
    ↓
Sync Engine ←→ Agent System
    ↓            ↓
State DB    ←→ LLM Providers
    ↓            ↓
AnkiConnect ←→ APF Generator
```

### Data Persistence

-   **SQLite Database**: Sync state, change history, session tracking
-   **File System**: Configuration files, logs, temporary data
-   **Anki Database**: Final card storage (managed by Anki)

## Development Considerations

### Testing Strategy

-   **Unit Tests**: Individual components and functions
-   **Integration Tests**: Component interactions and API calls
-   **End-to-End Tests**: Complete sync workflows
-   **Mock Providers**: LLM provider simulation for testing

### Error Handling

-   **Graceful Degradation**: System continues with reduced functionality
-   **Retry Logic**: Exponential backoff for transient failures
-   **Circuit Breaker**: Prevents cascade failures
-   **Comprehensive Logging**: Structured logs for debugging

### Monitoring

-   **Performance Metrics**: Response times, throughput, error rates
-   **Usage Analytics**: API costs, model usage patterns
-   **Health Checks**: Provider connectivity and system status
-   **Audit Logs**: Change tracking and security events

## Getting Started with Architecture

### For New Contributors

1. **[Getting Started](../../GETTING_STARTED.md)** - Quick setup guide
2. **[Agent System](agents.md)** - Understanding the multi-agent pipeline
3. **[Providers](providers.md)** - LLM provider integration
4. **[APF Format](apf.md)** - Card generation format

### For Architecture Changes

1. Review existing **[ADRs](ADR/)** for design patterns
2. Create new ADR for significant changes
3. Update component documentation
4. Ensure test coverage for new features

## Related Documentation

-   **[Full Documentation Index](../README.md)** - Complete documentation map
-   **[Configuration Guide](../GUIDES/configuration.md)** - Setup and configuration
-   **[Implementation Notes](../IMPLEMENTATION_NOTES/)** - Technical analysis and optimizations

---

**Last Updated**: November 28, 2025
**Version**: 2.0
