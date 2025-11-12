# Multi-Agent System Summary

**Last Updated**: 2025-01-12
**System**: PydanticAI + LangGraph orchestration
**Total Agents**: 7 specialized agents

## Overview

The Obsidian-to-Anki sync system uses a sophisticated multi-agent architecture to generate high-quality Anki flashcards. Each agent is a specialist handling a specific aspect of the card generation pipeline.

## Agent Roster

### 🔍 Core Pipeline Agents

These agents form the main card generation workflow:

#### 1. Pre-Validation Agent
**Status**: ✅ Production
**Purpose**: Validate note structure before generation
**Model**: openrouter/polaris-alpha

**Checks**:
- Frontmatter structure
- Content formatting
- Required fields present

**Documentation**: [LANGGRAPH_PYDANTIC_AI.md](LANGGRAPH_PYDANTIC_AI.md)

---

#### 2. Generator Agent
**Status**: ✅ Production
**Purpose**: Convert Q&A pairs to APF cards
**Model**: openrouter/polaris-alpha

**Generates**:
- APF 2.1 format HTML
- Front/Back/Extra sections
- Proper metadata

**Documentation**: [LANGGRAPH_PYDANTIC_AI.md](LANGGRAPH_PYDANTIC_AI.md)

---

#### 3. Post-Validation Agent
**Status**: ✅ Production
**Purpose**: Validate generated cards for quality
**Model**: openrouter/polaris-alpha

**Checks**:
- Syntax errors
- Factual accuracy
- Template compliance

**Documentation**: [LANGGRAPH_PYDANTIC_AI.md](LANGGRAPH_PYDANTIC_AI.md)

---

### 🎯 Quality Enhancement Agents

These agents improve card quality and effectiveness:

#### 4. Memorization Quality Agent
**Status**: ✅ Production
**Purpose**: Ensure cards follow SRS best practices
**Model**: openrouter/polaris-alpha

**Evaluates**:
- Atomic principle
- Clear Q-A relationship
- Active recall trigger
- Context sufficiency
- Appropriate difficulty
- No information leakage
- Memorable formatting
- Practical applicability

**Documentation**: [MEMORIZATION_QUALITY_AGENT.md](MEMORIZATION_QUALITY_AGENT.md)

---

#### 5. Context Enrichment Agent
**Status**: ✅ Production
**Purpose**: Add examples, mnemonics, and context
**Model**: openrouter/polaris-alpha (temperature=0.3)

**Adds**:
- Concrete examples (code, scenarios)
- Mnemonics and memory aids
- Visual structure (formatting)
- Related concepts
- Practical tips and warnings

**Documentation**: [CONTEXT_ENRICHMENT_AGENT.md](CONTEXT_ENRICHMENT_AGENT.md)

---

### 🔀 Routing & Optimization Agents

These agents handle workflow decisions and optimization:

#### 6. Card Splitting Agent
**Status**: ✅ Production
**Purpose**: Decide if note should generate 1 or N cards
**Model**: openrouter/polaris-alpha

**Strategies**:
- Concept splitting (multiple topics)
- List item splitting (N items → N+1 cards)
- Example splitting (concept + examples)
- Hierarchical splitting (parent + children)
- Step-by-step splitting (process steps)

**Documentation**: [CARD_SPLITTING_AGENT.md](CARD_SPLITTING_AGENT.md)

---

#### 7. Duplicate Detection Agent
**Status**: ✅ Production
**Purpose**: Identify redundant/overlapping cards
**Model**: openrouter/polaris-alpha

**Detects**:
- Exact duplicates (≥95% similar)
- Semantic duplicates (80-94% similar)
- Partial overlap (50-79% similar)
- Unique cards (<50% similar)

**Documentation**: [DUPLICATE_DETECTION_AGENT.md](DUPLICATE_DETECTION_AGENT.md)

---

## Current Workflow (LangGraph)

```
┌──────────────────┐
│  Pre-Validation  │ ✅ Validate structure
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Card Splitting  │ ✅ Determine 1 or N cards
└────────┬─────────┘
         │
┌────────▼─────────┐
│   Generation     │ ✅ Create APF cards
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Post-Validation  │ ✅ Quality check (with retry)
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Context Enrich   │ ✅ Add examples, mnemonics
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Memorization QA  │ ✅ SRS effectiveness check
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Duplicate Check  │ ✅ Compare with existing (optional)
└────────┬─────────┘
         │
┌────────▼─────────┐
│    Complete      │ Ready for Anki
└──────────────────┘
```

## Integration Status

| Agent | Status | In Pipeline | Documented |
|-------|--------|-------------|------------|
| Pre-Validation | ✅ Production | ✅ | ✅ |
| Card Splitting | ✅ Production | ✅ | ✅ |
| Generator | ✅ Production | ✅ | ✅ |
| Post-Validation | ✅ Production | ✅ | ✅ |
| Context Enrichment | ✅ Production | ✅ | ✅ |
| Memorization Quality | ✅ Production | ✅ | ✅ |
| Duplicate Detection | ✅ Production | ✅ (optional) | ✅ |

## Usage Examples

### Standalone Agent Usage

```python
from obsidian_anki_sync.agents.pydantic_ai_agents import (
    ContextEnrichmentAgentAI,
    DuplicateDetectionAgentAI,
    MemorizationQualityAgentAI,
    CardSplittingAgentAI
)
from obsidian_anki_sync.providers.pydantic_ai_models import create_openrouter_model_from_env

# Create model
model = create_openrouter_model_from_env("openrouter/polaris-alpha")

# Initialize agents
enrichment = ContextEnrichmentAgentAI(model, temperature=0.3)
duplicate = DuplicateDetectionAgentAI(model)
memorization = MemorizationQualityAgentAI(model)
splitting = CardSplittingAgentAI(model)

# Use agents
enrichment_result = await enrichment.enrich(card, metadata)
duplicate_result = await duplicate.find_duplicates(card, existing_cards)
quality_result = await memorization.assess([card], metadata)
splitting_result = await splitting.analyze(note_content, metadata, qa_pairs)
```

### LangGraph Integration (Production)

All agents are now integrated into the production LangGraph pipeline. See [LANGGRAPH_INTEGRATION_COMPLETE.md](LANGGRAPH_INTEGRATION_COMPLETE.md) for complete details.

## Key Features

### Type Safety
- All agents use Pydantic models for inputs/outputs
- Structured outputs validated automatically
- Type errors caught at development time

### Error Handling
- Custom exception hierarchy
- Graceful degradation (fallback results)
- Comprehensive logging

### Observability
- Timing for each agent
- Confidence scores
- Detailed reasoning

### Configurability
- Model selection per agent
- Temperature control
- Enable/disable agents

## Benefits Summary

### For Users
✅ **Higher Quality Cards**: Better formatting, examples, context
✅ **Better Learning**: Cards optimized for SRS effectiveness
✅ **Cleaner Decks**: No duplicates or redundancy
✅ **Smarter Splitting**: Optimal card count per note

### For Developers
✅ **Modular Architecture**: Easy to add new agents
✅ **Type Safe**: Pydantic validation throughout
✅ **Well Tested**: Each agent independently testable
✅ **Observable**: Rich logging and metrics

### For the Project
✅ **Production Ready**: Error handling and fallbacks
✅ **Efficient**: Fast, accurate models
✅ **Scalable**: Parallel agent execution possible
✅ **Maintainable**: Clear separation of concerns

## Future Roadmap

### Phase 1: Advanced Features
- [ ] Difficulty Calibration Agent
- [ ] Prerequisite Detection Agent
- [ ] Image Generation Agent
- [ ] Performance Analytics Agent

### Phase 2: Intelligence
- [ ] User feedback loops
- [ ] Learning from review statistics
- [ ] Personalization based on user level
- [ ] Domain-specific specialization

## Support

- **Issues**: https://github.com/po4yka/obsidian-to-anki/issues
- **Documentation**: This directory
- **Code**: `src/obsidian_anki_sync/agents/`

---

**Total Agents**: 7
**Status**: 7/7 in Production (100%)
**Model**: openrouter/polaris-alpha (unified configuration)
