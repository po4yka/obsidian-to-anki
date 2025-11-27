# Duplicate Detection Agent

**Status**: ✅ Implemented (2025-01-12)
**Agent Type**: PydanticAI-based semantic similarity analysis
**Purpose**: Identify redundant and overlapping flashcards to maintain a clean deck

## Overview

The Duplicate Detection Agent analyzes flashcards to find:

-   **Exact Duplicates** (≥95% similar): Nearly identical cards
-   **Semantic Duplicates** (80-94% similar): Same concept, different wording
-   **Partial Overlap** (50-79% similar): Related but distinct
-   **Unique** (<50% similar): No significant overlap

This solves a critical problem: **Duplicate cards waste review time and interfere with learning**.

### The Problem

Without duplicate detection:

-   ❌ **Wasted Time**: Reviewing the same concept multiple times
-   ❌ **Confusion**: Slightly different answers for same question
-   ❌ **Deck Bloat**: Growing collection with redundant content
-   ❌ **Inconsistency**: Multiple versions with conflicting information

With duplicate detection:

-   ✅ **Clean Deck**: Only unique cards remain
-   ✅ **Efficient Reviews**: No redundant repetitions
-   ✅ **Consistent Knowledge**: Single source of truth per concept
-   ✅ **Better Organization**: Clear card relationships

## Detection Criteria

### Exact Duplicate (similarity ≥ 0.95)

**Indicators**:

-   Questions are essentially identical
-   Answers convey the same information
-   Both cards test identical knowledge

**Example**:

```
Card A: "What is REST?"
Card B: "What does REST stand for?"
→ EXACT DUPLICATE ✗
```

**Action**: Delete or merge immediately

---

### Semantic Duplicate (similarity 0.80-0.94)

**Indicators**:

-   Same core concept, different phrasing
-   Answers are equivalent but worded differently
-   Testing identical knowledge

**Example**:

```
Card A: "What is the time complexity of binary search?"
Card B: "How efficient is binary search?"
→ SEMANTIC DUPLICATE ✗
```

**Action**: Merge and keep best version

---

### Partial Overlap (similarity 0.50-0.79)

**Indicators**:

-   Related concepts but different aspects
-   Questions test different facets of same topic
-   Cards complement each other

**Example**:

```
Card A: "What does binary search return?"
Card B: "What is the time complexity of binary search?"
→ PARTIAL OVERLAP ⚠️
```

**Action**: Keep both but note relationship

---

### Unique (similarity < 0.50)

**Indicators**:

-   Different concepts
-   No meaningful overlap
-   Independent knowledge

**Example**:

```
Card A: "What is binary search?"
Card B: "What is bubble sort?"
→ UNIQUE ✓
```

**Action**: No action needed

## How It Works

### Architecture

```
┌─────────────────────────────────────────┐
│     Duplicate Detection Agent           │
│                                         │
│  Input:                                 │
│  - New card (question + answer)         │
│  - Existing cards list                  │
│                                         │
│  Analysis:                              │
│  1. Extract Q&A from APF HTML           │
│  2. Compare with each existing card     │
│  3. Compute similarity score (LLM)      │
│  4. Classify duplicate type             │
│  5. Generate recommendation             │
│                                         │
│  Output:                                │
│  - is_duplicate: bool                   │
│  - best_match: DuplicateMatch           │
│  - all_matches: list[DuplicateMatch]    │
│  - recommendation: enum                 │
│  - merge_suggestion: str                │
└─────────────────────────────────────────┘
```

### Comparison Methods

The agent uses **LLM-based semantic comparison** rather than simple string matching:

**Why LLM over embeddings?**

-   Understands paraphrasing and synonyms
-   Recognizes conceptual equivalence
-   Handles code examples and formatting differences
-   Provides explanations for similarity scores

**Similarity Scoring**:

-   0.95-1.00: Trivial differences (punctuation, minor wording)
-   0.80-0.94: Same concept, rephrased
-   0.50-0.79: Related topics, different focus
-   0.00-0.49: Unrelated content

## Usage

### Programmatic Usage

```python
from obsidian_anki_sync.agents.pydantic import DuplicateDetectionAgentAI
from obsidian_anki_sync.providers.pydantic_ai_models import create_openrouter_model_from_env

# Create model (cheap, fast model for comparison)
model = create_openrouter_model_from_env("openai/gpt-4o-mini")

# Initialize agent
agent = DuplicateDetectionAgentAI(model=model, temperature=0.0)

# Check single card against another
is_dup, similarity, reasoning = await agent.check_duplicate(
    new_card=new_card,
    existing_card=existing_card
)

if is_dup:
    print(f"✗ Duplicate detected (similarity: {similarity:.2f})")
    print(f"Reasoning: {reasoning}")
else:
    print("✓ Unique card")

# Find all duplicates in deck
result = await agent.find_duplicates(
    new_card=new_card,
    existing_cards=deck_cards
)

if result.is_duplicate:
    print(f"Found {len(result.all_matches)} potential duplicates")
    print(f"Best match: {result.best_match.card_slug} ({result.best_match.similarity_score:.2f})")
    print(f"Recommendation: {result.recommendation}")
    if result.merge_suggestion:
        print(f"Merge suggestion: {result.merge_suggestion}")
```

### Example Output (Exact Duplicate)

```json
{
    "is_duplicate": true,
    "best_match": {
        "card_slug": "rest-api-definition-en-001",
        "similarity_score": 0.96,
        "duplicate_type": "exact",
        "reasoning": "Both cards ask for REST definition. Questions nearly identical, answers convey same information."
    },
    "all_matches": [
        {
            "card_slug": "rest-api-definition-en-001",
            "similarity_score": 0.96,
            "duplicate_type": "exact"
        }
    ],
    "recommendation": "delete",
    "better_card": "existing",
    "merge_suggestion": "Keep existing card. New card adds no new information.",
    "detection_time": 0.8
}
```

### Example Output (Partial Overlap)

```json
{
    "is_duplicate": false,
    "best_match": {
        "card_slug": "binary-search-complexity-en-001",
        "similarity_score": 0.68,
        "duplicate_type": "partial_overlap",
        "reasoning": "Both about binary search, but test different aspects: return value vs complexity."
    },
    "all_matches": [
        {
            "card_slug": "binary-search-complexity-en-001",
            "similarity_score": 0.68,
            "duplicate_type": "partial_overlap"
        }
    ],
    "recommendation": "keep_both",
    "better_card": null,
    "merge_suggestion": "Keep both. Consider adding relationship note.",
    "detection_time": 0.7
}
```

## Configuration

### Model Selection

Recommended model: **gpt-4o-mini** (fast, cheap, sufficient for comparison)

```yaml
# config.yaml (future integration)
duplicate_detection_enabled: true
duplicate_detection_model: "openai/gpt-4o-mini"
duplicate_threshold: 0.80 # Minimum similarity for duplicate
similarity_threshold_exact: 0.95
similarity_threshold_semantic: 0.80
similarity_threshold_partial: 0.50
```

### Cost

-   **Model**: gpt-4o-mini (~$0.15/1M tokens)
-   **Per comparison**: ~$0.0001
-   **100 cards vs 1000 cards**: ~$10 (100,000 comparisons)

**Optimization**: Use embedding-based pre-filtering to reduce LLM calls:

1. Compute embeddings for all cards (cheap)
2. Filter to top 10 most similar by cosine similarity
3. Use LLM only for detailed comparison of top matches
4. **Cost reduction**: 100x cheaper

## Integration Scenarios

### Scenario 1: Pre-Sync Check (Recommended)

Check new cards before adding to deck:

```python
async def sync_with_duplicate_check(new_cards, existing_cards):
    agent = DuplicateDetectionAgentAI(model)

    for new_card in new_cards:
        result = await agent.find_duplicates(new_card, existing_cards)

        if result.recommendation == "delete":
            logger.info(f"Skipping duplicate: {new_card.slug}")
            continue
        elif result.recommendation == "merge":
            # Merge logic
            merged_card = merge_cards(new_card, existing_cards[result.best_match.card_slug])
            add_to_deck(merged_card)
        else:
            add_to_deck(new_card)
```

### Scenario 2: Batch Deck Cleanup

Clean up existing deck:

```python
async def clean_deck(deck_cards):
    agent = DuplicateDetectionAgentAI(model)
    duplicates = []

    for i, card_a in enumerate(deck_cards):
        for card_b in deck_cards[i+1:]:
            is_dup, score, reason = await agent.check_duplicate(card_a, card_b)
            if score >= 0.80:
                duplicates.append((card_a, card_b, score, reason))

    # Sort by similarity (highest first)
    duplicates.sort(key=lambda x: x[2], reverse=True)

    # Review and merge
    for card_a, card_b, score, reason in duplicates:
        print(f"Duplicate found ({score:.2f}): {card_a.slug} ↔ {card_b.slug}")
        print(f"Reason: {reason}")
        # Manual review or auto-merge
```

### Scenario 3: LangGraph Integration (Future)

Add as optional post-generation node:

```python
def duplicate_detection_node(state: PipelineState) -> PipelineState:
    """Check for duplicates after card generation."""

    # Get existing cards from deck
    existing_cards = load_existing_cards()

    # Check each new card
    duplicate_results = []
    for card in state["generation"]["cards"]:
        result = agent.find_duplicates(card, existing_cards)
        duplicate_results.append(result)

    # Filter out duplicates
    unique_cards = [
        card for card, result in zip(state["generation"]["cards"], duplicate_results)
        if result.recommendation != "delete"
    ]

    state["generation"]["cards"] = unique_cards
    state["duplicate_detection"] = duplicate_results

    return state
```

## Special Cases

### Language Mixing

Cards in different languages are NOT duplicates:

```
Card A (English): "What is recursion?"
Card B (Russian): "Что такое рекурсия?"
→ UNIQUE ✓ (different language learning goals)
```

### Example vs Concept

Concept + example cards are complementary:

```
Card A: "What is polymorphism?"
Card B: "Give an example of polymorphism in Python"
→ UNIQUE ✓ (concept vs application)
```

### Different Difficulty Levels

Same concept at different depths:

```
Card A (Basic): "What is a linked list?"
Card B (Advanced): "Cache performance: linked lists vs arrays?"
→ UNIQUE ✓ (different complexity levels)
```

### Improved Versions

Better card replaces worse one:

```
Old: "What is REST?" → "An architectural style"
New: "What is REST?" → "Representational State Transfer - uses HTTP..."
→ EXACT DUPLICATE but REPLACE with new ✓
```

## Benefits

### For Deck Quality

✅ **Eliminate Redundancy**: Clean, focused deck
✅ **Consistency**: Single source of truth
✅ **Better Organization**: Clear card relationships
✅ **Quality Control**: Keep best versions

### For Learning

✅ **Efficient Reviews**: No wasted time on duplicates
✅ **Clear Knowledge**: No conflicting information
✅ **Better Retention**: Focus on unique concepts
✅ **Reduced Confusion**: Consistent terminology

### For Workflow

✅ **Automatic Detection**: No manual checking
✅ **Actionable Recommendations**: Clear guidance
✅ **Batch Processing**: Analyze entire deck
✅ **Explained Decisions**: Understand similarity

## Limitations

1. **LLM-Based**: Requires API calls (cost and latency)
2. **No Context**: Can't consider user's learning history
3. **Binary Decision**: Can't track "evolving" cards
4. **Language Specific**: Best for English/Russian tech content

## Future Enhancements

-   [ ] Embedding-based pre-filtering for efficiency
-   [ ] Card versioning (track improvements over time)
-   [ ] Cluster analysis (find groups of related cards)
-   [ ] User feedback loop (learn from merge decisions)
-   [ ] Integration with Anki's built-in duplicate detection
-   [ ] Fuzzy matching for typos and formatting differences

## Research Background

Duplicate detection is based on:

1. **Semantic Similarity**: NLP techniques for meaning comparison
2. **Information Retrieval**: Document similarity metrics
3. **SRS Best Practices**: Avoid interference from similar cards
4. **Cognitive Load**: Reduce redundancy for better focus

## Support

-   **Issues**: https://github.com/po4yka/obsidian-to-anki/issues
-   **Documentation**: This file
-   **Code**: `src/obsidian_anki_sync/agents/pydantic_ai_agents.py` (DuplicateDetectionAgentAI)

---

**Version**: 1.0
**Last Updated**: 2025-01-12
**Status**: Implemented, ready for integration
