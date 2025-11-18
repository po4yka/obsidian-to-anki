# Skills System

**Purpose**: Dynamic, on-demand loading of specialized prompt instructions for LLM agents.

**Concept**: Inspired by Claude's Skills system - markdown documents containing domain-specific guidance that agents can load just-in-time, avoiding permanent context overhead.

---

## Overview

Skills are markdown documents containing specialized instructions, constraints, and domain knowledge. Agents load Skills dynamically based on task requirements, providing focused guidance without bloating system prompts.

### Benefits

1. **Reduced Context Overhead**: Load only what's needed, when it's needed
2. **Modularity**: Each Skill focuses on one domain
3. **Reusability**: Skills can be shared across agents
4. **Maintainability**: Update Skills independently without touching code
5. **Discoverability**: Clear organization makes it easy to find relevant guidance

---

## Available Skills

### Core Skills

#### `apf_compliance.md`
**Purpose**: Ensure all generated cards strictly follow APF v2.1 specifications.

**When to Use**: Always loaded for card generation tasks.

**Key Topics**:
- Format strictness and structure
- Field rules and requirements
- Tagging taxonomy
- Card type selection
- Quality gates

---

#### `memorization_principles.md`
**Purpose**: Ensure flashcards follow evidence-based spaced repetition principles.

**When to Use**: Load for all card generation and quality validation tasks.

**Key Topics**:
- Atomicity (one concept per card)
- Clear question-answer relationships
- Active recall triggers
- Context sufficiency
- Appropriate difficulty
- No information leakage
- Memorable formatting
- Practical applicability

---

#### `card_splitting.md`
**Purpose**: Determine optimal card generation strategy - single vs. multiple cards.

**When to Use**: Load before card generation to analyze note structure.

**Key Topics**:
- Decision framework for splitting
- When to create single cards
- When to create multiple cards
- Splitting strategies
- Edge cases and quality checks

---

#### `context_enrichment.md`
**Purpose**: Enhance flashcards with examples, mnemonics, and practical context.

**When to Use**: Load during card generation to add enriching context.

**Key Topics**:
- Concrete examples
- Mnemonics and memory aids
- Visual structure
- Related concepts
- Practical tips

---

#### `bilingual_handling.md`
**Purpose**: Handle bilingual content (EN/RU) in Obsidian notes.

**When to Use**: Load when processing notes with multiple languages.

**Key Topics**:
- Language detection
- Card generation strategies
- Tagging for bilingual content
- Content extraction patterns

---

## Usage

### Loading Skills in Code

```python
from pathlib import Path
from obsidian_anki_sync.utils.skill_loader import SkillLoader

# Initialize skill loader
skill_loader = SkillLoader(base_path=Path(".docs/skills"))

# Load specific skills
apf_skill = skill_loader.load("apf_compliance")
memorization_skill = skill_loader.load("memorization_principles")

# Combine skills for agent
system_prompt = f"""
{apf_skill}
{memorization_skill}
"""
```

### Agent Integration

Agents should load Skills based on task requirements:

```python
class GeneratorAgent:
    def __init__(self):
        self.skill_loader = SkillLoader()

        # Load core skills for generation
        self.core_skills = [
            self.skill_loader.load("apf_compliance"),
            self.skill_loader.load("memorization_principles"),
        ]

    def generate_cards(self, note_content, metadata):
        # Load additional skills based on note characteristics
        skills = self.core_skills.copy()

        if self._is_bilingual(note_content):
            skills.append(self.skill_loader.load("bilingual_handling"))

        if self._needs_splitting(note_content):
            skills.append(self.skill_loader.load("card_splitting"))

        # Use combined skills as system prompt
        system_prompt = "\n\n".join(skills)
        # ... generate cards
```

---

## Skill Format Guidelines

### Structure

Each Skill should follow this structure:

```markdown
# Skill Name

**Purpose**: Brief description of what this skill provides.

**When to Use**: When to load this skill.

---

## Core Principle

Main principle or philosophy.

---

## Key Topics

### Topic 1
Explanation with examples.

### Topic 2
More guidance.

---

## Guidelines

Specific actionable guidance.

---

## Common Mistakes to Avoid

❌ What not to do

---

## Best Practices

✓ What to do
```

### Writing Style

- **Clear and actionable**: Provide specific guidance, not vague suggestions
- **Examples**: Include before/after examples showing good vs. bad
- **Right altitude**: Not too low-level (exact hex codes), not too high-level (vague principles)
- **Structured**: Use clear sections and formatting
- **Practical**: Focus on implementable guidance

---

## Creating New Skills

### When to Create a New Skill

Create a new Skill when:
1. You have domain-specific guidance that doesn't fit existing Skills
2. The guidance is reusable across multiple agents
3. The guidance is substantial enough to warrant its own document
4. The guidance benefits from being loaded conditionally

### Steps

1. Create new markdown file in `.docs/skills/`
2. Follow Skill format guidelines
3. Add to this README's "Available Skills" section
4. Update agents to load the new Skill when appropriate
5. Document usage examples

---

## Skill Loading Strategy

### Always Loaded (Core Skills)
- `apf_compliance.md` - Required for all card generation
- `memorization_principles.md` - Required for quality

### Conditionally Loaded
- `card_splitting.md` - When analyzing note structure
- `context_enrichment.md` - When enhancing cards
- `bilingual_handling.md` - When processing multilingual content

### Future Skills
- `tag_taxonomy.md` - Detailed tagging rules
- `code_examples.md` - Code example best practices
- `diagram_generation.md` - Draw card type guidance

---

## Migration from Python Prompts

Existing prompts in Python files can be converted to Skills:

**Before** (Python):
```python
MEMORIZATION_QUALITY_PROMPT = """You are a memorization expert...
"""
```

**After** (Skill):
```markdown
# Memorization Principles Skill

**Purpose**: Ensure flashcards follow evidence-based principles...
```

**Benefits**:
- Easier to read and maintain
- Can be loaded dynamically
- Better organization
- No code changes needed to update prompts

---

## References

- [Claude Skills Blog Post](https://claude.com/blog/improving-frontend-design-through-skills)
- [APF Cards Documentation](.docs/APF%20Cards/)
- [Project Requirements](.docs/REQUIREMENTS.md)

