# Memorization Quality Agent

**Status**: ✅ Implemented (2025-01-12)
**Agent Type**: PydanticAI-based quality assessment
**Purpose**: Ensure generated Anki cards are effective for spaced repetition learning

## Overview

The Memorization Quality Agent evaluates whether generated flashcards follow best practices for long-term memory retention using spaced repetition systems (SRS) like Anki.

### Why This Matters

Not all flashcards are created equal. Poor card design leads to:

-   ❌ **Failed reviews**: Can't recall the answer
-   ❌ **Wasted time**: Reviewing ineffective cards
-   ❌ **Demotivation**: Frustration with the learning process
-   ❌ **Poor retention**: Information doesn't stick long-term

Good card design results in:

-   ✅ **Effective reviews**: Clear active recall
-   ✅ **Efficient learning**: Time well spent
-   ✅ **High motivation**: Success breeds success
-   ✅ **Long-term retention**: Knowledge that sticks

## Evaluation Criteria

The agent evaluates cards on 8 key criteria:

### 1. Atomic Principle (Single Concept)

**What**: Card tests ONE specific fact or concept
**Why**: Multiple concepts create interference and make review ambiguous
**Example**:

-   ✅ Good: "What is the time complexity of binary search?" → "O(log n)"
-   ❌ Bad: "What are the complexities of bubble sort, merge sort, and quicksort?"

### 2. Clear Question-Answer Relationship

**What**: Question unambiguously leads to one correct answer
**Why**: Ambiguity causes frustration and poor retention
**Example**:

-   ✅ Good: "What does `fetch()` return in JavaScript?" → "A Promise that resolves to a Response"
-   ❌ Bad: "What does it return?" → "A promise"

### 3. Active Recall Trigger

**What**: Question requires retrieving information from memory
**Why**: Active recall strengthens memory more than passive recognition
**Example**:

-   ✅ Good: "What Git command shows commit history?" → "git log"
-   ❌ Bad: "Is `git log` the command for history? (Yes/No)"

### 4. Context Sufficiency

**What**: Card provides enough context to answer without external references
**Why**: Cards should be self-contained for effective review
**Example**:

-   ✅ Good: Includes examples and context in Extra section
-   ❌ Bad: Requires remembering previous cards or external knowledge

### 5. Appropriate Difficulty

**What**: Card is challenging but answerable with study
**Why**: Optimal difficulty maintains engagement and learning
**Example**:

-   ✅ Good: Standard knowledge in the domain
-   ❌ Bad: Trivial ("What letter comes after A?") or impossible

### 6. No Information Leakage

**What**: Front doesn't hint at or reveal the answer
**Why**: Leakage bypasses active recall and weakens learning
**Example**:

-   ✅ Good: "What Python feature creates lists in one line?"
-   ❌ Bad: "What syntax uses `[]` to create sequences?" (hints at lists)

### 7. Memorable Formatting

**What**: Uses examples, mnemonics, or visual structure
**Why**: Structured information is easier to encode and retrieve
**Example**:

-   ✅ Good: Includes code examples, bullet points, visual structure
-   ❌ Bad: Wall of text with no structure

### 8. Practical Applicability

**What**: Tests knowledge useful in real scenarios
**Why**: Practical knowledge is more motivating and retained longer
**Example**:

-   ✅ Good: "What command deploys to production?"
-   ❌ Bad: "Who invented the algorithm in 1952?"

## How It Works

### Architecture

```
Generated Cards
      │
      v
┌─────────────────────────────────────┐
│ Memorization Quality Agent          │
│                                     │
│  1. Parse card content              │
│  2. Evaluate 8 criteria             │
│  3. Score 0.0-1.0                   │
│  4. Identify issues                 │
│  5. Suggest improvements            │
└─────────────────────────────────────┘
      │
      v
MemorizationQualityResult
 - is_memorizable: bool
 - memorization_score: float
 - issues: list[dict]
 - strengths: list[str]
 - suggested_improvements: list[str]
```

### Integration Point

The agent can be added to the LangGraph workflow:

```
Pre-Validation → Generation → Post-Validation → Memorization Quality → Complete
```

**Current Status**: Implemented but **not yet integrated** into LangGraph workflow
**Usage**: Available for standalone assessment or future integration

## Usage

### Programmatic Usage

```python
from obsidian_anki_sync.agents.pydantic import MemorizationQualityAgentAI
from obsidian_anki_sync.providers.pydantic_ai_models import create_openrouter_model_from_env
from obsidian_anki_sync.agents.models import GeneratedCard
from obsidian_anki_sync.models import NoteMetadata

# Create model (cheap validator)
model = create_openrouter_model_from_env("openai/gpt-4o-mini")

# Initialize agent
agent = MemorizationQualityAgentAI(model=model, temperature=0.0)

# Assess cards
result = await agent.assess(
    cards=generated_cards,
    metadata=note_metadata
)

# Check result
if result.is_memorizable:
    print(f"✓ Cards are effective (score: {result.memorization_score:.2f})")
    for strength in result.strengths:
        print(f"  + {strength}")
else:
    print(f"✗ Cards need improvement (score: {result.memorization_score:.2f})")
    for issue in result.issues:
        print(f"  - [{issue['severity']}] {issue['message']}")
    print("\nSuggested improvements:")
    for improvement in result.suggested_improvements:
        print(f"  → {improvement}")
```

### Example Output

```json
{
    "is_memorizable": false,
    "memorization_score": 0.55,
    "issues": [
        {
            "type": "information_leakage",
            "severity": "medium",
            "message": "Question contains '[]' which hints at lists, reducing active recall"
        },
        {
            "type": "weak_recall",
            "severity": "medium",
            "message": "Visual hint makes this recognition rather than recall"
        }
    ],
    "strengths": [
        "Tests single concept (atomic)",
        "Grammatically clear question"
    ],
    "suggested_improvements": [
        "Rephrase to: 'What Python feature allows creating lists in one line?'",
        "Add example in Extra: [x**2 for x in range(5)]"
    ],
    "assessment_time": 0.8
}
```

## Configuration

### Model Selection

Recommended model: **gpt-4o-mini** (fast, cheap, sufficient for assessment)

```yaml
# config.yaml (future integration)
memorization_agent_enabled: true
memorization_agent_model: "openai/gpt-4o-mini"
memorization_min_score: 0.7 # Minimum acceptable score
memorization_block_poor_cards: false # Warn but don't block
```

### Cost

-   **Model**: gpt-4o-mini (~$0.15/1M tokens)
-   **Per card**: ~$0.0001 (very cheap)
-   **100 cards**: ~$0.01

## Common Issues Detected

### Anti-Patterns Flagged

| Issue Type              | Description                   | Severity | Example                      |
| ----------------------- | ----------------------------- | -------- | ---------------------------- |
| **atomic_violation**    | Multiple concepts in one card | High     | "List 5 SOLID principles"    |
| **information_leakage** | Question hints at answer      | Medium   | "What uses `[]` syntax?"     |
| **context_missing**     | Card depends on external info | High     | "What does it return?"       |
| **too_broad**           | Question too vague            | High     | "What language uses braces?" |
| **weak_recall**         | Recognition vs recall         | Medium   | "Is X true? (Yes/No)"        |
| **cognitive_overload**  | Too much to remember          | High     | 10+ items in one card        |
| **low_practical_value** | Trivial knowledge             | Low      | Historical trivia            |
| **vague_answer**        | Answer not specific enough    | Medium   | "Many things"                |

### Best Practices Rewarded

| Strength           | Description          | Example                       |
| ------------------ | -------------------- | ----------------------------- |
| **atomic**         | Single concept       | "What is O(n) complexity?"    |
| **self_contained** | Includes context     | Examples in Extra section     |
| **clear_trigger**  | Unambiguous question | Specific, actionable question |
| **memorable**      | Well-structured      | Bullet points, code examples  |
| **practical**      | Real-world usage     | Common development tasks      |

## Integration Scenarios

### Scenario 1: Standalone Assessment (Current)

```python
# After post-validation, assess cards separately
cards = result.generation.cards
quality = await memorization_agent.assess(cards, metadata)
if not quality.is_memorizable:
    logger.warning("memorization_issues", issues=quality.issues)
```

### Scenario 2: LangGraph Integration (Future)

Add as optional node after post-validation:

```python
def memorization_quality_node(state: PipelineState) -> PipelineState:
    """Assess memorization quality."""
    # Run assessment
    quality_result = run_memorization_assessment(state["generation"]["cards"])

    # Add to state
    state["memorization_quality"] = quality_result.model_dump()

    # Decide: warn or block?
    if quality_result.memorization_score < 0.5:
        state["current_stage"] = "manual_review"  # Block poor cards
    else:
        state["current_stage"] = "complete"  # Allow with warning

    return state
```

### Scenario 3: Batch Analysis

Analyze all existing cards to find improvement opportunities:

```python
# Analyze vault
for note in vault.notes:
    cards = get_cards_for_note(note)
    quality = await agent.assess(cards, note.metadata)

    if quality.memorization_score < 0.6:
        print(f"⚠ {note.title}: score={quality.memorization_score:.2f}")
        for improvement in quality.suggested_improvements:
            print(f"  → {improvement}")
```

## Research Background

The agent's criteria are based on established spaced repetition research:

1. **Atomic Principle**: Wozniak's "20 Rules of Formulating Knowledge"
2. **Active Recall**: Karpicke & Roediger (2008) - Testing effect
3. **Context Independence**: Bjork's desirable difficulties
4. **Information Leakage**: Cue dependency effects
5. **Appropriate Difficulty**: Zone of proximal development (Vygotsky)

## Limitations

1. **No User History**: Can't assess based on individual learner's knowledge level
2. **Static Analysis**: Doesn't track actual review performance
3. **Language Specific**: Optimized for English/Russian tech content
4. **Subjective Criteria**: "Practical applicability" varies by domain

## Future Enhancements

-   [ ] Integration into LangGraph workflow (configurable)
-   [ ] User-specific difficulty assessment
-   [ ] Domain-specific criteria (medical, language learning, etc.)
-   [ ] Batch analysis CLI command
-   [ ] Historical performance tracking
-   [ ] A/B testing of assessment impact

## Support

-   **Issues**: https://github.com/po4yka/obsidian-to-anki/issues
-   **Documentation**: This file
-   **Code**: `src/obsidian_anki_sync/agents/pydantic_ai_agents.py` (line 564+)

---

**Version**: 1.0
**Last Updated**: 2025-01-12
**Status**: Implemented, ready for integration
