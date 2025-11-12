# LangGraph Pipeline Integration - Complete Guide

**Last Updated**: 2025-01-12
**Status**: ✅ Production Ready
**New Agents**: Card Splitting + Context Enrichment + Memorization Quality + Duplicate Detection

## Overview

The LangGraph pipeline now includes **7 agents** working together to generate high-quality Anki flashcards:

1. **Pre-Validation** - Structure and format checks
2. **Card Splitting** - Determines if note should be split
3. **Generation** - APF card creation
4. **Post-Validation** - Quality and accuracy validation
5. **Context Enrichment** - Add examples and mnemonics
6. **Memorization Quality** - Check SRS effectiveness
7. **Duplicate Detection** ← NEW - Pre-sync duplicate check

## Enhanced Workflow

```
┌──────────────────┐
│  Pre-Validation  │ Validate note structure
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Card Splitting  │ Analyze if note should be split
└────────┬─────────┘
         │
┌────────▼─────────┐
│    Generation    │ Create APF cards
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Post-Validation  │ Check quality (with retry)
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Context Enrich   │ Add examples, mnemonics
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Memorization QA  │ Check SRS effectiveness
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Duplicate Check  │ ← NEW: Compare with existing cards
└────────┬─────────┘
         │
┌────────▼─────────┐
│    Complete      │ Ready for Anki
└──────────────────┘
```

## What's New

### Card Splitting Agent

**Purpose**: Automatically detect when a note should be split into multiple cards

**Analysis performed**:
- Concept complexity (multiple distinct concepts vs single concept)
- Logical grouping opportunities
- Independent testability
- Cognitive load assessment

**When it helps**:
```
BEFORE:
Single note: "Python Data Structures"
→ Generates 1 massive card covering lists, dicts, sets, tuples

AFTER:
Card Splitting detects: "Should split into 4 cards"
→ Generates 4 focused cards:
   1. Lists and indexing
   2. Dictionaries and keys
   3. Sets and uniqueness
   4. Tuples and immutability
```

**Model**: openrouter/polaris-alpha (default)

### Context Enrichment Agent

**Purpose**: Automatically enhance cards with:
- Concrete examples (code snippets, real-world scenarios)
- Mnemonics and memory aids
- Visual structure (bullets, emphasis, formatting)
- Related concepts and comparisons
- Practical tips and common pitfalls

**When it helps**:
```
BEFORE:
Q: What is array destructuring in JavaScript?
A: Syntax to unpack array values into variables.

AFTER:
Q: What is array destructuring in JavaScript?
A: Syntax to unpack array values into variables.

Extra:
Example:
const [first, second] = [10, 20];
// first = 10, second = 20

Common use case:
const [state, setState] = useState(0);
```

**Model**: openrouter/polaris-alpha (default)

### Memorization Quality Agent

**Purpose**: Evaluate cards for spaced repetition effectiveness

**Checks**:
- ✅ Atomic principle (one concept per card)
- ✅ Clear question-answer relationship
- ✅ Active recall trigger (not recognition)
- ✅ Context sufficiency (self-contained)
- ✅ Appropriate difficulty
- ✅ No information leakage
- ✅ Memorable formatting
- ✅ Practical applicability

**Output**:
```json
{
  "is_memorizable": true,
  "memorization_score": 0.87,
  "issues": [],
  "strengths": [
    "Tests single concept (atomic)",
    "Includes concrete example",
    "Well-formatted with structure"
  ],
  "suggested_improvements": []
}
```

**Model**: openrouter/polaris-alpha (default)

### Duplicate Detection Agent

**Purpose**: Pre-sync check to detect duplicate cards before adding to Anki

**Detection performed**:
- Semantic similarity analysis (compares questions and answers)
- Exact duplicate detection (≥95% similarity)
- Semantic duplicate detection (80-94% similarity)
- Partial overlap detection (50-79% similarity)
- Recommendations: delete, merge, keep_both, review_manually

**When it helps**:
```
SCENARIO: You're adding a new card about Python lists

Existing card in Anki:
Q: How do you access the first element of a list in Python?
A: Use index 0: my_list[0]

New card being generated:
Q: What's the syntax to get the first item from a Python list?
A: Access with [0]: list_name[0]

RESULT:
⚠ Duplicate detected (similarity: 0.92)
Recommendation: merge
Better card: existing (more concise)
```

**Requirements**:
- **Must pass existing_cards** to `process_note()` method
- Cards must be from Anki to compare against
- Disabled by default (set `enable_duplicate_detection: true`)

**Model**: openrouter/polaris-alpha (default)

## Configuration

### config.yaml

Add these new settings:

```yaml
# Enable/disable enhancement agents
enable_card_splitting: true
enable_context_enrichment: true
enable_memorization_quality: true
enable_duplicate_detection: false  # NEW - requires existing_cards parameter

# Model selection (optional, defaults to openrouter/polaris-alpha)
card_splitting_model: ""  # Empty = use default_llm_model
context_enrichment_model: ""  # Empty = use default_llm_model
memorization_quality_model: ""  # Empty = use default_llm_model
duplicate_detection_model: ""  # Empty = use default_llm_model
```

### Environment Variables

Required:
```bash
OPENROUTER_API_KEY=your_key_here
```

## Usage

### Basic Usage (Recommended)

The orchestrator automatically reads config values:

```python
from pathlib import Path
from obsidian_anki_sync.config import load_config
from obsidian_anki_sync.agents import LangGraphOrchestrator
from obsidian_anki_sync.obsidian.parser import parse_note

# Load configuration
config = load_config(Path("config.yaml"))

# Create orchestrator (uses config values)
orchestrator = LangGraphOrchestrator(config=config)

# Parse note
note_path = Path("vault/notes/my-note.md")
note_content = note_path.read_text()
metadata, qa_pairs = parse_note(note_content, note_path)

# Optionally fetch existing cards from Anki for duplicate detection
existing_cards = fetch_cards_from_anki()  # Your Anki integration

# Process through pipeline
result = orchestrator.process_note(
    note_content=note_content,
    metadata=metadata,
    qa_pairs=qa_pairs,
    file_path=note_path,
    existing_cards=existing_cards  # Optional: for duplicate detection
)

# Check results
if result.success:
    print(f"✓ Generated {result.generation.total_cards} cards")

    if result.context_enrichment:
        print(f"  Enhanced: {result.context_enrichment.additions_summary}")

    if result.memorization_quality:
        print(f"  Quality score: {result.memorization_quality.memorization_score:.2f}")
        if not result.memorization_quality.is_memorizable:
            print("  ⚠ Quality issues found:")
            for issue in result.memorization_quality.issues:
                print(f"    - {issue['message']}")

    # Check duplicate detection results
    if result.duplicate_detection:
        duplicates = result.duplicate_detection.get("duplicates_found", 0)
        if duplicates > 0:
            print(f"  ⚠ {duplicates} duplicate(s) detected")
            for card_result in result.duplicate_detection["results"]:
                if card_result["result"] and card_result["result"]["is_duplicate"]:
                    print(f"    - {card_result['card_slug']}: {card_result['result']['recommendation']}")
```

### Advanced Usage (Override Config)

You can override config values:

```python
# Disable enrichment for this run
orchestrator = LangGraphOrchestrator(
    config=config,
    enable_context_enrichment=False,  # Override config
    enable_memorization_quality=True,
    max_retries=5
)
```

### Selective Enrichment

Enable/disable agents per run:

```python
# Production: All quality checks
prod_orchestrator = LangGraphOrchestrator(
    config=config,
    enable_context_enrichment=True,
    enable_memorization_quality=True
)

# Fast mode: Skip enrichment for speed
fast_orchestrator = LangGraphOrchestrator(
    config=config,
    enable_context_enrichment=False,
    enable_memorization_quality=False
)
```

## Performance

### Timing Breakdown (typical note)

| Stage | Time | Notes |
|-------|------|-------|
| Pre-Validation | ~0.5s | Fast checks |
| **Card Splitting** | **~0.4s** | **Split analysis** |
| Generation | ~2.0s | Main LLM work |
| Post-Validation | ~0.8s | Quality check |
| **Context Enrichment** | **~1.2s** | **Per-card enhancement** |
| **Memorization Quality** | **~0.6s** | **All cards assessed** |
| **Duplicate Detection** | **~0.5s** | **Per-card comparison** |
| **Total** | **~6.0s** | **For 1 card** |

**Scaling**:
- Card splitting runs once per note
- Enrichment runs per-card (parallelizable)
- Quality assessment runs once for all cards
- Duplicate detection runs per-card against existing set
- ~2.0s overhead per card for all enhancements

## Integration Examples

### Example 1: Production Pipeline

```python
config = load_config()

# Full quality pipeline
orchestrator = LangGraphOrchestrator(config=config)

for note_path in vault.notes:
    result = orchestrator.process_note(...)

    if result.success:
        # Check quality score
        if result.memorization_quality.memorization_score < 0.7:
            logger.warning(f"Low quality: {note_path}")

        # Log enrichments
        if result.context_enrichment and result.context_enrichment.should_enrich:
            logger.info(f"Enhanced: {result.context_enrichment.additions_summary}")

        # Use cards
        sync_to_anki(result.generation.cards)
```

### Example 2: Batch Processing with Progress

```python
from tqdm import tqdm

orchestrator = LangGraphOrchestrator(config=config)
notes = list(vault.notes)

results = []
for note_path in tqdm(notes, desc="Processing"):
    result = orchestrator.process_note(...)
    results.append(result)

# Analyze results
total_enriched = sum(
    1 for r in results
    if r.context_enrichment and r.context_enrichment.should_enrich
)
avg_quality = sum(
    r.memorization_quality.memorization_score
    for r in results
    if r.memorization_quality
) / len(results)

print(f"Enhanced: {total_enriched}/{len(results)} cards")
print(f"Average quality score: {avg_quality:.2f}")
```

### Example 3: Quality Filtering

Only sync high-quality cards:

```python
result = orchestrator.process_note(...)

if result.success and result.memorization_quality:
    if result.memorization_quality.memorization_score >= 0.8:
        sync_to_anki(result.generation.cards)
    else:
        logger.warning(
            f"Quality too low ({result.memorization_quality.memorization_score:.2f})",
            issues=result.memorization_quality.issues
        )
        # Save for manual review
        save_for_review(result)
```

## Troubleshooting

### Issue: Enrichment Taking Too Long

**Solution**: Disable for specific notes
```python
orchestrator = LangGraphOrchestrator(
    config=config,
    enable_context_enrichment=False  # Speed up
)
```

### Issue: Quality Scores Too Strict

**Behavior**: Memorization quality agent is conservative by design

**Options**:
1. Use scores as warnings, not blockers
2. Adjust threshold: `score >= 0.7` instead of `>= 0.8`
3. Disable: `enable_memorization_quality=False`

### Issue: OpenRouter API Errors

**Check**:
```python
import os
print(os.getenv("OPENROUTER_API_KEY"))  # Should not be None
```

**Fix**: Set environment variable or add to `.env` file

### Issue: Cards Not Being Enriched

**Possible causes**:
1. Config has `enable_context_enrichment: false`
2. Cards already comprehensive (agent decides no enrichment needed)
3. Agent encountered error (check logs)

**Check logs**:
```python
import logging
logging.basicConfig(level=logging.INFO)
# Look for "context_enrichment_skipped" or "card_enrichment_failed"
```

## Migration Guide

### From Old Pipeline

If you're using the old 3-agent pipeline:

**Before**:
```python
orchestrator = LangGraphOrchestrator(
    config=config,
    max_retries=3,
    auto_fix_enabled=True,
    strict_mode=True
)
```

**After** (automatically includes new agents):
```python
orchestrator = LangGraphOrchestrator(config=config)
# Now includes card splitting + enrichment + quality by default
# Duplicate detection OFF by default (requires existing_cards)
```

**Opt-out/Opt-in**:
```python
orchestrator = LangGraphOrchestrator(
    config=config,
    enable_card_splitting=False,  # Disable if needed
    enable_context_enrichment=False,
    enable_memorization_quality=False,
    enable_duplicate_detection=True,  # Enable if you have existing cards
)
```

## Benefits Summary

### For Users

✅ **Better Card Structure**: Automatic detection of notes needing multiple cards
✅ **Richer Cards**: Examples, mnemonics, practical tips
✅ **Better Learning**: Higher retention through enriched context
✅ **Quality Assurance**: Cards validated for SRS effectiveness
✅ **Duplicate Prevention**: Avoid adding similar cards to Anki
✅ **Actionable Feedback**: Know which cards need improvement

### For Developers

✅ **Configurable**: Easy to enable/disable agents
✅ **Observable**: Detailed logging and metrics
✅ **Backward Compatible**: Works with existing code
✅ **Type Safe**: Full Pydantic validation

## Related Documentation

- [Context Enrichment Agent](CONTEXT_ENRICHMENT_AGENT.md) - Detailed agent documentation
- [Memorization Quality Agent](MEMORIZATION_QUALITY_AGENT.md) - Quality criteria explained
- [Agent Summary](AGENT_SUMMARY.md) - Complete system overview
- [LangGraph + PydanticAI](LANGGRAPH_PYDANTIC_AI.md) - Original pipeline documentation

## Support

- **Issues**: https://github.com/po4yka/obsidian-to-anki/issues
- **Configuration**: See `config.yaml` examples
- **Code**: `src/obsidian_anki_sync/agents/langgraph_orchestrator.py`

---

**Integration Status**: ✅ Complete
**Production Ready**: ✅ Yes
**Default Behavior**: Both agents **enabled** by default
**Opt-out**: Set config flags to `false`
