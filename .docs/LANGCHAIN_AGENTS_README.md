# LangChain Agent System for Obsidian → Anki Pipeline

## Overview

The LangChain Agent System is a sophisticated, multi-agent pipeline for converting Obsidian notes into high-quality Anki flashcards with:

- **Intelligent Mapping**: LLM-powered conversion from note structure to Anki fields
- **Strong Validation**: Schema compliance and semantic QA checks
- **Safe Updates**: Field-level diffing for existing cards
- **Auto-Repair**: Retry logic with feedback-driven improvements
- **Extensibility**: Modular agent architecture

## Architecture

### Components

```
Obsidian Note (Markdown)
    ↓
NoteContext (Parsed)
    ↓
┌───────────────────── Supervisor Orchestrator ─────────────────────┐
│                                                                    │
│  1. Card Mapping Agent (LLM)                                     │
│     └─> Converts NoteContext → ProposedCard                      │
│                                                                    │
│  2. Schema Validation Tool (Non-LLM)                             │
│     └─> Validates ProposedCard against Anki model schema         │
│     └─> Retries mapping if validation fails                      │
│                                                                    │
│  3. QA Agent (LLM)                                               │
│     └─> Semantic correctness checks                              │
│     └─> Pedagogical quality assessment                           │
│                                                                    │
│  4. Style/Hint Agent (LLM, Optional)                             │
│     └─> Refines wording and generates hints                      │
│                                                                    │
│  5. Card Diff Agent (Non-LLM)                                    │
│     └─> Compares with existing cards                             │
│     └─> Determines update safety                                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
    ↓
CardDecision (create/update/skip/manual_review)
    ↓
Anki Sync Layer
    ↓
AnkiConnect
```

### Data Flow

1. **Input**: Obsidian markdown note with YAML front-matter
2. **Parsing**: Existing parser converts to `NoteContext`
3. **Agent Pipeline**: Supervisor coordinates multi-agent processing
4. **Output**: `CardDecision` with proposed card and quality reports
5. **Sync**: Existing sync engine applies changes to Anki

## Key Features

### 1. Card Mapping Agent

- **Purpose**: Intelligently maps note content to Anki card fields
- **Capabilities**:
  - Selects appropriate card type (Basic, Cloze, Custom)
  - Chooses correct Anki model and deck
  - Handles bilingual content
  - Generates relevant tags
  - Provides confidence scores

**Example Mapping**:
```
Input Note:
  Question: What is Big O notation?
  Answer: A mathematical notation describing algorithm complexity...

Output Card:
  Type: Basic
  Model: APF: Simple (3.0.0)
  Deck: Interview::Algorithms
  Fields:
    Front: "What is Big O notation?"
    Back: "A mathematical notation describing algorithm complexity..."
    Extra: "Examples: O(1), O(n), O(log n)"
    Hint: "Think about time/space complexity"
  Tags: [algorithms, complexity, big-o, interview]
```

### 2. Schema Validation Tool

- **Purpose**: Ensure card complies with Anki model schema
- **Checks**:
  - Model exists in Anki
  - Required fields present and non-empty
  - No unknown fields
  - Cloze syntax validity (for Cloze cards)
  - Field length constraints
- **Advantage**: Non-LLM for deterministic validation

### 3. QA Agent

- **Purpose**: Semantic quality assurance
- **Checks**:
  - **Answer Correctness**: Does Back answer Front completely?
  - **No Leakage**: Does Front reveal the answer?
  - **Language Consistency**: Matches declared language?
  - **Completeness**: Critical details present?
  - **Pedagogical Quality**: Clear, standalone, memorable?
- **Output**: QA score (0-1) and issue list

**Example QA Report**:
```json
{
  "qa_score": 0.85,
  "issues": [
    {
      "type": "style_issue",
      "severity": "low",
      "message": "Front field could be more concise",
      "suggested_change": "Shorten to: 'Define Big O notation'"
    }
  ]
}
```

### 4. Style/Hint Agent (Optional)

- **Purpose**: Polish card style and generate hints
- **Actions**:
  - Shorten overly long Front
  - Ensure Back is clear and standalone
  - Generate helpful hints
  - Fix grammar/punctuation

### 5. Card Diff Agent

- **Purpose**: Safe updates of existing cards
- **Capabilities**:
  - Field-by-field comparison
  - Change severity classification (cosmetic/content/structural)
  - Risk level assessment
  - Policy enforcement (allow_content_updates, etc.)

**Example Diff**:
```json
{
  "changes": [
    {
      "field": "Back",
      "old_value": "A notation for complexity",
      "new_value": "A mathematical notation for algorithm complexity",
      "severity": "content",
      "message": "Field 'Back' changed (more detailed)"
    }
  ],
  "should_update": true,
  "reason": "Update approved (1 content change)",
  "risk_level": "medium"
}
```

### 6. Supervisor Orchestrator

- **Purpose**: Coordinate agent pipeline with retry logic
- **Features**:
  - Automatic retries with feedback
  - LLM call limiting
  - Error handling and fallback
  - Decision aggregation

**Retry Example**:
```
Attempt 1: Map → Schema FAIL (missing required field)
Attempt 2: Map with feedback → Schema PASS
QA Check: Score 0.75 (below threshold 0.8)
Attempt 3: Map with QA feedback → QA Score 0.87 → PASS
```

## Configuration

### Enable LangChain Agents

In `config.yaml`:

```yaml
use_langchain_agents: true

langchain_agents:
  # Quality thresholds
  max_mapping_retries: 2
  min_qa_score: 0.8
  allow_auto_fix: true
  max_llm_calls_per_card: 8

  # Update policies
  allow_content_updates: true
  allow_structural_updates: false

  # Optional features
  enable_style_polish: false
  strict_schema_validation: false

  # Schema validation
  max_front_length: 500
  max_back_length: 2000
  max_field_length: 4000
  max_tags: 20

  # Deck settings
  default_deck_prefix: "Interview"

  # Bilingual settings
  bilingual_mode: "front_back"
```

### Model Assignment

The system uses existing LLM configuration:

- **Card Mapping**: `generator_model` (e.g., qwen3:32b)
- **QA Agent**: `post_validator_model` (e.g., qwen3:14b)
- **Style Agent**: `pre_validator_model` (e.g., qwen3:8b)

## Usage

### Basic Usage

```bash
# Enable in config.yaml
use_langchain_agents: true

# Run sync with LangChain agents
obsidian-anki-sync sync
```

### CLI Flags (Planned)

```bash
# Force enable LangChain agents
obsidian-anki-sync sync --use-langchain-agents

# Use custom agent config
obsidian-anki-sync sync --agents-config custom_agents.yaml

# Dry-run mode
obsidian-anki-sync sync --use-langchain-agents --dry-run
```

## Programmatic Usage

```python
from obsidian_anki_sync.agents.langchain.supervisor import LangChainSupervisor, SupervisorConfig
from obsidian_anki_sync.agents.langchain.adapter import AgentSystemAdapter
from obsidian_anki_sync.config import load_config

# Load config
config = load_config()

# Create supervisor
supervisor_config = SupervisorConfig(
    max_mapping_retries=2,
    min_qa_score=0.8,
    allow_auto_fix=True,
)

supervisor = LangChainSupervisor(
    config=config,
    supervisor_config=supervisor_config,
)

# Convert existing models to NoteContext
note_context = AgentSystemAdapter.to_note_context(
    metadata=note_metadata,
    qa_pair=qa_pair,
    vault_root=vault_root,
    config=config,
)

# Process note
decision = supervisor.process_note(note_context)

# Check result
if decision.action == "create":
    print(f"Card approved for creation (QA: {decision.qa_report.qa_score:.2f})")
elif decision.action == "manual_review":
    print(f"Manual review required: {decision.messages}")
```

## Data Models

### NoteContext

Input to the agent system:

```python
@dataclass
class NoteContext:
    slug: str
    note_path: str
    vault_root: str
    frontmatter: NoteContextFrontmatter  # title, lang, topic, difficulty, tags
    sections: NoteContextSections        # question, answer, extra, examples
    existing_anki_note: Optional[ExistingAnkiNote]
    config_profile: Optional[str]
```

### ProposedCard

Output of Card Mapping Agent:

```python
@dataclass
class ProposedCard:
    card_type: CardType  # Basic, Cloze, Custom
    model_name: str
    deck_name: str
    fields: dict[str, str]  # Front, Back, Extra, Hint
    tags: list[str]
    language: Language
    bilingual_mode: BilingualMode
    slug: str
    origin: NoteContextOrigin
    confidence: float
    notes: str
```

### CardDecision

Final output:

```python
@dataclass
class CardDecision:
    action: SyncAction  # create, update, skip, manual_review
    proposed_card: ProposedCard
    qa_report: QAReport
    schema_validation: SchemaValidationResult
    diff: Optional[CardDiffResult]
    messages: list[str]
    slug: str
```

## Comparison with Existing System

| Feature | Existing System | LangChain System |
|---------|----------------|------------------|
| **Mapping** | APF Generator (OpenRouter) | Card Mapping Agent (Local LLM) |
| **Validation** | Post-validation agent | Schema Validator + QA Agent |
| **Retries** | Up to 3 with auto-fix | Configurable with feedback |
| **Updates** | Full card replacement | Field-level diff with policies |
| **Models** | Single model | Multi-model (32b/14b/8b) |
| **Format** | APF HTML | Direct Anki fields |
| **Extensibility** | Monolithic | Modular agents |

## Benefits

1. **Higher Quality**: Multi-stage validation (schema + QA)
2. **Safer Updates**: Field-level diffing prevents accidental overwrites
3. **Better Retry Logic**: Feedback-driven improvements
4. **Local LLMs**: Uses Ollama/LM Studio (no API costs)
5. **Modular**: Easy to add/remove/replace agents
6. **Observable**: Detailed logs and reports for each step
7. **Configurable**: Fine-grained control over behavior

## Limitations

1. **Performance**: More LLM calls = slower (8 calls max per card)
2. **Complexity**: More moving parts than simple APF generation
3. **Integration**: Requires adapter layer to existing sync engine
4. **Testing**: More comprehensive testing needed

## Future Enhancements

1. **Multiple Cards per Note**: Generate separate cards for subquestions
2. **Curriculum Agent**: Prioritize notes by importance
3. **Review Agent**: Periodic re-evaluation of existing cards
4. **Semantic Similarity**: Use embeddings for better QA checks
5. **Graph Awareness**: Leverage note relationships
6. **Bilingual Separate Cards**: Generate two cards from one note
7. **Custom Templates**: User-defined card mapping rules

## Troubleshooting

### Issue: Schema validation always fails

**Solution**: Check that `anki_note_type` in config matches an actual Anki model name. Enable AnkiConnect client for dynamic model fetching.

### Issue: QA score too low

**Solution**: Lower `min_qa_score` threshold or enable `allow_auto_fix` for retry with QA feedback.

### Issue: Too slow

**Solution**: Disable `enable_style_polish` and reduce `max_mapping_retries`. Consider using smaller models for QA (qwen3:8b instead of 14b).

### Issue: Updates not applied

**Solution**: Check `allow_content_updates` and `allow_structural_updates` settings. Review diff results in logs.

## Testing

### Unit Tests

```bash
# Test data models
pytest tests/agents/langchain/test_models.py

# Test schema validator
pytest tests/agents/langchain/test_schema_validator.py

# Test adapter
pytest tests/agents/langchain/test_adapter.py
```

### Integration Tests

```bash
# Test full pipeline with mocked LLMs
pytest tests/agents/langchain/test_supervisor.py
```

### End-to-End Tests

```bash
# Test with real LLMs (slow)
pytest tests/agents/langchain/test_e2e.py --slow
```

## Development

### Adding a New Agent

1. Create agent file in `src/obsidian_anki_sync/agents/langchain/tools/`
2. Implement as LangChain `Tool` or standalone class
3. Add to supervisor orchestration in `supervisor.py`
4. Update configuration schema in `config.yaml`
5. Write unit tests
6. Update documentation

### Debugging

Enable debug logging:

```yaml
log_level: "DEBUG"
```

Check agent logs:

```bash
grep "langchain" .logs/sync.log | tail -50
```

## References

- [LangChain Documentation](https://python.langchain.com/)
- [Anki Model Schema](https://docs.ankiweb.net/templates/fields.html)
- [APF Format Specification](.docs/APF_SPEC.md)
- [Architecture Document](.docs/LANGCHAIN_AGENT_ARCHITECTURE.md)

## Support

For issues or questions:
- File issue: [GitHub Issues](https://github.com/po4yka/obsidian-to-anki/issues)
- Review specification: `.docs/LANGCHAIN_AGENT_ARCHITECTURE.md`
- Check logs: `.logs/sync.log`

---

**Version**: 1.0.0
**Last Updated**: 2025-11-10
**Status**: Implemented (Testing Pending)
