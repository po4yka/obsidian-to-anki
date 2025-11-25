# Multi-Agent System Summary

**Last Updated**: 2025-11-16
**System**: PydanticAI + LangGraph orchestration
**Total Agents**: 7 specialized agents
**Models**: Optimized for 2025 (MiniMax M2, Kimi K2, DeepSeek V3, Qwen 2.5)

## Overview

The Obsidian-to-Anki sync system uses a sophisticated multi-agent architecture to generate high-quality Anki flashcards. Each agent is a specialist handling a specific aspect of the card generation pipeline.

## Agent Roster

### 🔍 Core Pipeline Agents

These agents form the main card generation workflow:

#### 1. Pre-Validation Agent

**Status**: ✅ Production
**Purpose**: Validate note structure before generation
**Model**: qwen/qwen-2.5-32b-instruct (fast, efficient validation)

**Checks**:

-   Frontmatter structure
-   Content formatting
-   Required fields present

**Documentation**: [LANGGRAPH_PYDANTIC_AI.md](LANGGRAPH_PYDANTIC_AI.md)

---

#### 2. Generator Agent

**Status**: ✅ Production
**Purpose**: Convert Q&A pairs to APF cards
**Model**: qwen/qwen-2.5-72b-instruct (powerful content creation)

**Generates**:

-   APF 2.1 format HTML
-   Front/Back/Extra sections
-   Proper metadata

**Documentation**: [LANGGRAPH_PYDANTIC_AI.md](LANGGRAPH_PYDANTIC_AI.md)

---

#### 3. Post-Validation Agent

**Status**: ✅ Production
**Purpose**: Validate generated cards for quality
**Model**: deepseek/deepseek-chat (DeepSeek V3 - strong reasoning)

**Checks**:

-   Syntax errors
-   Factual accuracy
-   Template compliance

**Documentation**: [LANGGRAPH_PYDANTIC_AI.md](LANGGRAPH_PYDANTIC_AI.md)

---

### 🎯 Quality Enhancement Agents

These agents improve card quality and effectiveness:

#### 4. Memorization Quality Agent

**Status**: ✅ Production
**Purpose**: Ensure cards follow SRS best practices
**Model**: moonshotai/kimi-k2 (strong reasoning and analysis)

**Evaluates**:

-   Atomic principle
-   Clear Q-A relationship
-   Active recall trigger
-   Context sufficiency
-   Appropriate difficulty
-   No information leakage
-   Memorable formatting
-   Practical applicability

**Documentation**: [MEMORIZATION_QUALITY_AGENT.md](MEMORIZATION_QUALITY_AGENT.md)

---

#### 5. Context Enrichment Agent

**Status**: ✅ Production
**Purpose**: Add examples, mnemonics, and context
**Model**: minimax/minimax-m2 (excellent for code generation and creative tasks)

**Adds**:

-   Concrete examples (code, scenarios)
-   Mnemonics and memory aids
-   Visual structure (formatting)
-   Related concepts
-   Practical tips and warnings

**Documentation**: [CONTEXT_ENRICHMENT_AGENT.md](CONTEXT_ENRICHMENT_AGENT.md)

---

### 🔀 Routing & Optimization Agents

These agents handle workflow decisions and optimization:

#### 6. Card Splitting Agent (Enhanced)

**Status**: ✅ Production (Enhanced 2025)
**Purpose**: Decide if note should generate 1 or N cards
**Model**: moonshotai/kimi-k2-thinking (advanced reasoning for decision making)

**Strategies**:

-   Concept splitting (multiple topics)
-   List item splitting (N items → N+1 cards)
-   Example splitting (concept + examples)
-   Hierarchical splitting (parent + children)
-   Step-by-step splitting (process steps)
-   **Difficulty-based splitting** (NEW - order by difficulty)
-   **Prerequisite-aware splitting** (NEW - foundational concepts first)
-   **Context-aware splitting** (NEW - group or separate related concepts)

**Enhanced Features**:

-   Confidence scoring (0.0-1.0) for split decisions
-   Fallback strategies for low-confidence cases
-   User preferences (preferred card size, splitting behavior)
-   Safety limits (max cards per note)

**Documentation**: [CARD_SPLITTING_AGENT.md](CARD_SPLITTING_AGENT.md)

---

#### 7. Duplicate Detection Agent

**Status**: ✅ Production
**Purpose**: Identify redundant/overlapping cards
**Model**: qwen/qwen-2.5-32b-instruct (efficient comparison)

**Detects**:

-   Exact duplicates (≥95% similar)
-   Semantic duplicates (80-94% similar)
-   Partial overlap (50-79% similar)
-   Unique cards (<50% similar)

**Documentation**: [DUPLICATE_DETECTION_AGENT.md](DUPLICATE_DETECTION_AGENT.md)

---

## Current Workflow (LangGraph)

```
┌──────────────────────┐
│  Note Correction     │ ⚙️ Optional proactive correction (if enabled)
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  Pre-Validation      │ ✅ Validate structure
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  Card Splitting      │ ✅ Determine 1 or N cards (with confidence & preferences)
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│   Generation         │ ✅ Create APF cards (validates against split plan)
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Post-Validation      │ ✅ Quality check (with retry)
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Context Enrich      │ ✅ Add examples, mnemonics
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Memorization QA      │ ✅ SRS effectiveness check
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Duplicate Check      │ ✅ Compare with existing (optional)
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│    Complete          │ Ready for Anki
└──────────────────────┘
```

## Integration Status

| Agent                | Status                   | In Pipeline   | Documented | Enhanced Features                                      |
| -------------------- | ------------------------ | ------------- | ---------- | ------------------------------------------------------ |
| Pre-Validation       | ✅ Production            | ✅            | ✅         | -                                                      |
| Card Splitting       | ✅ Production (Enhanced) | ✅            | ✅         | Confidence scoring, advanced strategies, preferences   |
| Generator            | ✅ Production            | ✅            | ✅         | Split plan validation                                  |
| Post-Validation      | ✅ Production            | ✅            | ✅         | -                                                      |
| Context Enrichment   | ✅ Production            | ✅            | ✅         | -                                                      |
| Memorization Quality | ✅ Production            | ✅            | ✅         | -                                                      |
| Duplicate Detection  | ✅ Production            | ✅ (optional) | ✅         | -                                                      |
| Parser Repair        | ✅ Production (Enhanced) | ⚙️ Reactive   | ✅         | Error diagnosis, quality scoring, grammar improvements |
| Note Correction      | ⚙️ Optional              | ⚙️ Optional   | ✅         | Proactive correction (placeholder)                     |

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

-   All agents use Pydantic models for inputs/outputs
-   Structured outputs validated automatically
-   Type errors caught at development time

### Error Handling

-   Custom exception hierarchy
-   Graceful degradation (fallback results)
-   Comprehensive logging

### Observability

-   Timing for each agent
-   Confidence scores
-   Detailed reasoning

### Configurability

-   Model selection per agent
-   Temperature control
-   Enable/disable agents

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

-   [ ] Difficulty Calibration Agent
-   [ ] Prerequisite Detection Agent
-   [ ] Image Generation Agent
-   [ ] Performance Analytics Agent

### Phase 2: Intelligence

-   [ ] User feedback loops
-   [ ] Learning from review statistics
-   [ ] Personalization based on user level
-   [ ] Domain-specific specialization

## Support

-   **Issues**: https://github.com/po4yka/obsidian-to-anki/issues
-   **Documentation**: This directory
-   **Code**: `src/obsidian_anki_sync/agents/`

---

**Total Agents**: 7
**Status**: 7/7 in Production (100%)
**Models**: Optimized 2025 configuration with task-specific models:

-   **Qwen 2.5 72B**: Generator (powerful content creation)
-   **Qwen 2.5 32B**: Pre-validator, Duplicate Detection (fast, efficient)
-   **DeepSeek V3**: Post-validator (strong reasoning)
-   **MiniMax M2**: Context Enrichment (excellent for code and creative tasks)
-   **Kimi K2**: Memorization Quality (analytical capabilities)
-   **Kimi K2 Thinking**: Card Splitting (advanced reasoning)
