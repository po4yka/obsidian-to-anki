# Claude Skills Configuration

This file documents the recommended Claude Skills for this project.

## What are Skills?

Skills are specialized folders containing instructions, scripts, and resources that enhance Claude's performance on specific tasks. They load automatically when relevant and work together composably.

## Recommended Skills for This Project

### Python Development Skills

**Plugin:** `@wshobson/claude-code-workflows/python-development`

Install command:
```bash
/plugin install python-development@wshobson/claude-code-workflows
```

This plugin includes skills for:

#### 1. async-python-patterns
**Why relevant:** This project uses async/await patterns extensively:
- LangGraph orchestrator uses asyncio
- PydanticAI agents run asynchronously
- HTTP clients use async operations

**Usage:** Automatically loads when working with async code, helping with:
- Proper async/await usage
- Concurrent task management
- Event loop handling
- Performance optimization for async operations

#### 2. python-testing-patterns
**Why relevant:** Project has comprehensive test suite (81 tests, 90% coverage requirement):
- Uses pytest with fixtures in conftest.py
- pytest-cov for coverage reporting
- pytest-mock for mocking
- respx for HTTP testing

**Usage:** Automatically loads when writing tests, helping with:
- Test structure and organization
- Fixture design patterns
- Mocking strategies
- Coverage improvement techniques
- Test-driven development (TDD)

#### 3. python-packaging
**Why relevant:** Project uses modern Python packaging:
- pyproject.toml configuration
- uv package manager
- setuptools build system
- Distribution on PyPI potential

**Usage:** Loads when working with package configuration, helping with:
- Dependency management
- Build system configuration
- Version management
- Distribution packaging

### LLM Application Development Skills

**Plugin:** `@wshobson/claude-code-workflows/llm-application-dev`

Install command:
```bash
/plugin install llm-application-dev@wshobson/claude-code-workflows
```

This plugin includes skills for:

#### 4. langchain-architecture
**Why relevant:** Project uses LangGraph for orchestration:
- LangGraph state machine workflow (langgraph_orchestrator.py)
- Conditional routing and retries
- State persistence with checkpoints
- Multi-agent coordination

**Usage:** Loads when working with LangGraph/LangChain code, helping with:
- State machine design patterns
- Graph node implementation
- Checkpoint and persistence strategies
- Workflow optimization

#### 5. llm-evaluation
**Why relevant:** Project validates LLM outputs at multiple stages:
- Pre-validator agent checks structure
- Post-validator agent validates quality
- APF linter validates card format
- HTML validator checks syntax

**Usage:** Loads when working on validation logic, helping with:
- Output validation strategies
- Quality metrics design
- Error detection patterns
- Retry and auto-fix logic

#### 6. prompt-engineering-patterns
**Why relevant:** Project has extensive prompt engineering:
- Agent-specific prompts in agents/*_prompts.py
- Multiple agent types (pre-validator, generator, post-validator)
- Context enrichment and memorization quality agents
- Bilingual (EN/RU) prompt handling

**Usage:** Loads when working with prompts, helping with:
- Prompt structure optimization
- Temperature and parameter tuning
- Few-shot examples design
- Multi-language prompt handling

## Installation

### Quick Install (All Recommended Skills)

```bash
# In Claude Code CLI
/plugin install python-development@wshobson/claude-code-workflows
/plugin install llm-application-dev@wshobson/claude-code-workflows
```

### Manual Installation

Skills can also be manually installed to `~/.claude/skills/` directory:

```bash
mkdir -p ~/.claude/skills
cd ~/.claude/skills
git clone https://github.com/anthropics/skills.git
```

### Verification

After installation, skills will load automatically when relevant. You can verify by:
1. Working on async code → async-python-patterns should activate
2. Writing tests → python-testing-patterns should activate
3. Modifying prompts → prompt-engineering-patterns should activate

Check Claude's reasoning chain to see which skills are active.

## Skill Usage Examples

### When Writing Tests

Skills activate automatically when you work in `tests/` directory:

```python
# python-testing-patterns helps with:
def test_parse_note_with_bilingual_content(tmp_path):
    """Test note parsing with English and Russian Q&A pairs."""
    # Suggests appropriate fixtures
    # Recommends assertion strategies
    # Identifies edge cases to test
```

### When Working with Async Code

Skills activate when editing async functions:

```python
# async-python-patterns helps with:
async def generate_card_async(note: NoteMetadata) -> GeneratedCard:
    # Suggests proper async patterns
    # Identifies potential race conditions
    # Recommends concurrent execution strategies
```

### When Engineering Prompts

Skills activate when editing prompt files:

```python
# prompt-engineering-patterns helps with:
PRE_VALIDATOR_PROMPT = """
You are a pre-validator agent...
"""
# Suggests prompt structure improvements
# Recommends few-shot examples
# Identifies ambiguous instructions
```

### When Working with LangGraph

Skills activate when editing orchestrator code:

```python
# langchain-architecture helps with:
workflow = StateGraph(PipelineState)
workflow.add_node("pre_validate", pre_validate_node)
# Suggests state management patterns
# Recommends error handling strategies
# Identifies optimization opportunities
```

## Additional Skills to Consider

### From Anthropic Official Repository

These skills are pre-included with Claude Code:

- **mcp-builder**: Useful if creating MCP servers for integration
- **webapp-testing**: Relevant for testing web interfaces (if adding web UI)
- **skill-creator**: Helpful for creating custom project-specific skills

### Custom Skills

Consider creating custom skills for:

1. **APF Format Validation**: Specific rules for APF v2.1 compliance
2. **Obsidian Note Parsing**: Domain-specific note structure patterns
3. **Anki Integration**: AnkiConnect API patterns and best practices
4. **Bilingual Content**: EN/RU content handling patterns

Custom skill template location: `~/.claude/skills/template-skill/`

## Best Practices

1. **Trust**: Only install skills from trusted sources (Anthropic official, verified community)
2. **Composability**: Skills work together - multiple skills can activate simultaneously
3. **Context**: Skills load only when relevant, keeping Claude fast
4. **Security**: Skills can execute code - review before installation

## Troubleshooting

### Skills Not Loading

1. Check installation: `/plugin list`
2. Verify permissions in `.claude/settings.local.json`
3. Ensure working in relevant context (e.g., .py files for Python skills)

### Conflicts

If skills conflict, check:
1. Skill versions: `/plugin upgrade`
2. Skill descriptions for overlapping domains
3. Consider disabling specific skills if needed

## Documentation

- Official Guide: https://claude.com/blog/skills
- Skills Repository: https://github.com/anthropics/skills
- Claude Code Docs: https://docs.claude.ai/
- Community Skills: https://github.com/travisvn/awesome-claude-skills

## Project-Specific Notes

Skills enhance work on this project's core components:

- `agents/` - langchain-architecture, prompt-engineering-patterns, llm-evaluation
- `tests/` - python-testing-patterns
- `providers/` - async-python-patterns
- `config.py` - python-packaging
- `sync/engine.py` - async-python-patterns
- Agent prompt files - prompt-engineering-patterns

Skills complement but don't replace the CLAUDE.md file, which provides project-specific architecture and patterns.
