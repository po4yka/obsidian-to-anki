# Skills System Usage Guide

**Purpose**: Guide for using the Skills system in the obsidian-to-anki project.

---

## Overview

The Skills system provides dynamic, on-demand loading of specialized prompt instructions for LLM agents. This allows agents to load only the guidance they need, reducing context overhead and improving maintainability.

---

## Quick Start

### Basic Usage

```python
from obsidian_anki_sync.utils.skill_loader import SkillLoader

# Initialize loader
skill_loader = SkillLoader()

# Load a single skill
apf_skill = skill_loader.load("apf_compliance")

# Load multiple skills
skills = skill_loader.load_multiple(["apf_compliance", "memorization_principles"])

# Combine skills into one prompt
combined = skill_loader.combine(
    ["apf_compliance", "memorization_principles"],
    separator="\n\n---\n\n"
)
```

---

## Agent Integration

### Example: Generator Agent

The `GeneratorAgent` uses Skills to load core and additional guidance:

```python
class GeneratorAgent:
    def __init__(self):
        self.skill_loader = SkillLoader()

        # Load core skills (always needed)
        core_skills = ["apf_compliance", "memorization_principles"]
        self.system_prompt = self.skill_loader.combine(core_skills)

    def generate_cards(self, note_content, metadata):
        # Load additional skills based on note characteristics
        additional_skills = self._determine_additional_skills(note_content, metadata)

        # Build system prompt with all skills
        system_prompt = self._build_system_prompt(additional_skills)

        # Use system_prompt for LLM generation
        ...
```

### Dynamic Skill Loading

Agents can load Skills conditionally based on task requirements:

```python
def _determine_additional_skills(self, note_content, metadata):
    """Load skills based on note characteristics."""
    additional_skills = []

    # Bilingual content
    if self._is_bilingual(note_content):
        if self.skill_loader.skill_exists("bilingual_handling"):
            additional_skills.append("bilingual_handling")

    # Complex structure
    if self._needs_splitting(note_content):
        if self.skill_loader.skill_exists("card_splitting"):
            additional_skills.append("card_splitting")

    # Always enrich context
    if self.skill_loader.skill_exists("context_enrichment"):
        additional_skills.append("context_enrichment")

    return additional_skills
```

---

## Skill Loading Strategies

### Always Loaded (Core Skills)

These Skills are loaded for all card generation tasks:

- `apf_compliance.md` - Required for APF v2.1 format compliance
- `memorization_principles.md` - Required for quality validation

### Conditionally Loaded

These Skills are loaded based on note characteristics:

- `bilingual_handling.md` - When processing multilingual content
- `card_splitting.md` - When analyzing note structure for splitting
- `context_enrichment.md` - When enhancing cards with examples and context

---

## Skill Loader API

### `SkillLoader(base_path: Path | None = None)`

Initialize the skill loader.

**Parameters**:
- `base_path`: Optional path to skills directory. Defaults to `.docs/skills/`.

### `load(skill_name: str) -> str`

Load a single skill by name.

**Parameters**:
- `skill_name`: Name of skill file (without .md extension)

**Returns**: Skill content as string

**Raises**: `FileNotFoundError` if skill doesn't exist

### `load_multiple(skill_names: list[str]) -> list[str]`

Load multiple skills.

**Parameters**:
- `skill_names`: List of skill names to load

**Returns**: List of skill contents in order

### `combine(skill_names: list[str], separator: str = "\n\n---\n\n") -> str`

Load and combine multiple skills into single string.

**Parameters**:
- `skill_names`: List of skill names to load and combine
- `separator`: String to insert between skills

**Returns**: Combined skill content

### `list_available() -> list[str]`

List all available skills.

**Returns**: List of skill names (without .md extension)

### `skill_exists(skill_name: str) -> bool`

Check if a skill exists.

**Parameters**:
- `skill_name`: Name of skill to check

**Returns**: True if skill exists, False otherwise

---

## Best Practices

### 1. Load Core Skills in `__init__`

Load Skills that are always needed during initialization:

```python
def __init__(self):
    self.skill_loader = SkillLoader()
    self.core_prompt = self.skill_loader.combine(["apf_compliance", "memorization_principles"])
```

### 2. Load Additional Skills Dynamically

Load Skills based on task requirements at runtime:

```python
def process_note(self, note_content):
    skills = ["apf_compliance", "memorization_principles"]

    if self._is_bilingual(note_content):
        skills.append("bilingual_handling")

    prompt = self.skill_loader.combine(skills)
```

### 3. Check Skill Existence Before Loading

Use `skill_exists()` to check before loading:

```python
if self.skill_loader.skill_exists("bilingual_handling"):
    skills.append("bilingual_handling")
```

### 4. Handle Missing Skills Gracefully

Provide fallback behavior when Skills are missing:

```python
try:
    prompt = self.skill_loader.combine(skills)
except FileNotFoundError:
    logger.warning("Skills not found, using fallback")
    prompt = self.fallback_prompt
```

### 5. Use Appropriate Separators

Choose separators that help LLMs distinguish between Skills:

```python
# Clear separation
prompt = skill_loader.combine(skills, separator="\n\n---\n\n")

# Minimal separation
prompt = skill_loader.combine(skills, separator="\n\n")
```

---

## Migration from Python Prompts

### Before (Python Prompts)

```python
MEMORIZATION_QUALITY_PROMPT = """You are a memorization expert...
"""
```

### After (Skills)

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

## Troubleshooting

### Skill Not Found

**Error**: `FileNotFoundError: Skill 'xyz' not found`

**Solution**:
1. Check skill name spelling
2. Verify skill file exists in `.docs/skills/`
3. Ensure file has `.md` extension

### Skills Directory Not Found

**Error**: Warning about skills directory not existing

**Solution**:
1. Verify `.docs/skills/` directory exists
2. Check path resolution (defaults to project root)
3. Provide explicit `base_path` if needed

### Context Window Overflow

**Problem**: Too many Skills loaded, exceeding context limits

**Solution**:
1. Load only necessary Skills
2. Use conditional loading
3. Consider splitting large Skills into smaller ones

---

## Examples

### Example 1: Basic Agent

```python
from obsidian_anki_sync.utils.skill_loader import SkillLoader

class SimpleAgent:
    def __init__(self):
        self.skill_loader = SkillLoader()
        self.prompt = self.skill_loader.combine(["apf_compliance"])

    def generate(self, content):
        # Use self.prompt for generation
        ...
```

### Example 2: Conditional Loading

```python
class SmartAgent:
    def __init__(self):
        self.skill_loader = SkillLoader()
        self.core_skills = ["apf_compliance", "memorization_principles"]

    def process(self, note_content, metadata):
        skills = self.core_skills.copy()

        # Add bilingual skill if needed
        if self._is_bilingual(note_content):
            skills.append("bilingual_handling")

        # Add splitting skill if complex
        if len(note_content.split("Q:")) > 2:
            skills.append("card_splitting")

        prompt = self.skill_loader.combine(skills)
        # Use prompt for generation
        ...
```

### Example 3: Error Handling

```python
class RobustAgent:
    def __init__(self):
        self.skill_loader = SkillLoader()
        self.fallback_prompt = "Generate APF cards."

    def get_prompt(self, skills):
        try:
            return self.skill_loader.combine(skills)
        except FileNotFoundError as e:
            logger.warning(f"Skills not found: {e}, using fallback")
            return self.fallback_prompt
```

---

## References

- [Skills README](README.md) - Overview of available Skills
- [APF Compliance Skill](apf_compliance.md) - APF format requirements
- [Memorization Principles Skill](memorization_principles.md) - Quality guidelines
- [Claude Skills Blog Post](https://claude.com/blog/improving-frontend-design-through-skills)

