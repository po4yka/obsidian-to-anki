# Claude Code Configuration

This directory contains Claude Code configuration files for this project.

## Files

### SKILLS.md
Comprehensive guide to recommended Claude Skills for this project:
- Python development skills (async patterns, testing, packaging)
- LLM application development skills (LangGraph, prompt engineering, evaluation)
- Installation and usage instructions
- Project-specific skill applications

### settings.local.json
Local permissions configuration for Claude Code:
- Approved bash operations (find)
- Web search permissions
- Approved web fetch domains (anthropic.com, claude.com, github.com, claude-plugins.dev)

## Quick Start

1. Install recommended skills:
```bash
/plugin install python-development@wshobson/claude-code-workflows
/plugin install llm-application-dev@wshobson/claude-code-workflows
```

2. Skills will automatically load when relevant (e.g., when working with async code, tests, or prompts)

3. See `SKILLS.md` for detailed information

## Additional Resources

- Project architecture and commands: `../CLAUDE.md`
- Code style and standards: `../.cursorrules`
- Project documentation: `../README.md`
