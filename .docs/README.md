# Obsidian to Anki Documentation

Welcome to the comprehensive documentation for the Obsidian to Anki APF Sync Service. This documentation is organized to help you understand, configure, and contribute to the project.

## Quick Start

New to the project? Start here:

-   **[Getting Started](GETTING_STARTED.md)** - Quick setup guide
-   **[README.md](../README.md)** - Project overview and installation

## Documentation Structure

### 🏗️ Architecture

Core system design and implementation details.

-   **[Overview](ARCHITECTURE/README.md)** - System architecture and components
-   **[Agent System](ARCHITECTURE/agents.md)** - Multi-agent validation and generation
-   **[LLM Providers](ARCHITECTURE/providers.md)** - Provider integration and configuration
-   **[Synchronization](ARCHITECTURE/sync.md)** - Bidirectional sync architecture
-   **[APF Format](ARCHITECTURE/apf.md)** - Anki Prompts Format specification

### 📋 Guides

Practical guides for using and configuring the system.

-   **[Configuration](GUIDES/configuration.md)** - Model presets, providers, and settings
-   **[Synchronization](GUIDES/synchronization.md)** - Change management and conflict resolution
-   **[AI Validation](GUIDES/validation.md)** - Automated validation and fixing
-   **[Troubleshooting](GUIDES/troubleshooting.md)** - Common issues and solutions

### 📚 Reference

Detailed specifications and technical reference.

-   **[APF Cards](REFERENCE/apf/)** - Complete APF format documentation
    -   [Card Block Template](REFERENCE/apf/Doc%20A%20—%20Card%20Block%20Template%20&%20Formatting%20Invariants.md)
    -   [Tag Taxonomy](REFERENCE/apf/Doc%20B%20—%20Tag%20Taxonomy.md)
    -   [Examples](REFERENCE/apf/Doc%20C%20—%20Examples.md)
    -   [Linter Rules](<REFERENCE/apf/Doc%20D%20—%20Linter%20Rules%20(Regex%20&%20Policies).md>)
    -   [JSON Mode & Manifest](REFERENCE/apf/Doc%20E%20—%20JSON%20Mode%20&%20Manifest%20Spec.md)
    -   [Draw Diagram DSL](REFERENCE/apf/Doc%20F%20—%20Draw%20Diagram%20DSL.md)
-   **[Templates](REFERENCE/templates/)** - Documentation and code templates
-   **[Interview Questions](REFERENCE/interview_questions/)** - Sample content and templates

### 📝 Architecture Decisions

Historical context and design rationale.

-   **[ADR Directory](ADR/)** - Architecture Decision Records
-   **[README](ADR/README.md)** - ADR process and current decisions

### 🛠️ Implementation Notes

Technical analysis and optimization details.

-   **[Model Optimization](IMPLEMENTATION_NOTES/model_optimization.md)** - Performance improvements
-   **[Performance Analysis](IMPLEMENTATION_NOTES/performance_bottlenecks.md)** - Bottleneck identification

## Key Topics

### For Users

-   **[Configuration](GUIDES/configuration.md)** - Setting up providers and models
-   **[Synchronization](GUIDES/synchronization.md)** - Understanding sync behavior
-   **[Troubleshooting](GUIDES/troubleshooting.md)** - Solving common problems

### For Developers

-   **[Architecture Overview](ARCHITECTURE/README.md)** - System design principles
-   **[Agent System](ARCHITECTURE/agents.md)** - Extending or modifying agents
-   **[APF Format](ARCHITECTURE/apf.md)** - Working with card formats
-   **[Templates](REFERENCE/templates/)** - Creating new documentation

### For Contributors

-   **[ADR Process](ADR/README.md)** - Making architectural decisions
-   **[Templates](REFERENCE/templates/)** - Consistent documentation
-   **[Testing](../TESTING.md)** - Testing guidelines
-   **[Code Style](../CLAUDE.md)** - Development standards

## Navigation Tips

-   **🔍 Search**: Use your editor's search to find specific topics
-   **📖 Cross-references**: Documents link to related content
-   **📋 Checklists**: Look for checklists in guides for step-by-step processes
-   **🔗 API Reference**: Technical details link to code when available

## Contributing to Documentation

-   Use templates from `REFERENCE/templates/` for new documents
-   Follow the ADR process for architectural changes
-   Update this index when adding new sections
-   Test all links when making changes

---

**Last Updated**: November 28, 2025
**Version**: 2.0 (Reorganized Documentation)
