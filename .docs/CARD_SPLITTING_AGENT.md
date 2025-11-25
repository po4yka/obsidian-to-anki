# Card Splitting Agent

**Status**: ✅ Implemented (2025-01-12)
**Agent Type**: PydanticAI-based routing decision
**Purpose**: Intelligently determine whether to generate one card or multiple cards from a note

## Overview

The Card Splitting Agent analyzes Obsidian notes and makes intelligent routing decisions:

-   **Single Card**: Note contains one atomic concept → Generate 1 card
-   **Multiple Cards**: Note contains multiple concepts → Split into N cards

This solves a critical problem in flashcard generation: **one-size-fits-all doesn't work**.

### The Problem

Without intelligent splitting:

-   ❌ **Too much per card**: "List 5 SOLID principles" → Cognitive overload
-   ❌ **Loss of atomic principle**: Multiple concepts in one card → Poor retention
-   ❌ **Inefficient reviews**: Can't selectively review specific concepts
-   ❌ **All-or-nothing learning**: Must know all concepts or fail the card

With intelligent splitting:

-   ✅ **Atomic cards**: One concept per card → Better retention
-   ✅ **Granular reviews**: Review each concept separately
-   ✅ **Progressive learning**: Master concepts one at a time
-   ✅ **Better scheduling**: Each concept gets optimal spaced repetition

## Decision Criteria

### When to Create SINGLE Card

✅ **Single Atomic Concept**

-   Example: "What is Big O notation?"
-   Rationale: One clear concept, one card

✅ **Simple Q&A**

-   Example: "What does `git log` do?"
-   Rationale: Straightforward question-answer pair

✅ **Tightly Coupled Information**

-   Example: "Pros and cons of microservices"
-   Rationale: Comparison requires both sides together

✅ **Short Content**

-   Q+A < 200 words
-   Rationale: Fits comfortably in one card

### When to Split into MULTIPLE Cards

❌ **Multiple Independent Concepts**

-   Example: "Python Lists AND Dictionaries"
-   Action: Split into 2 cards (one per data structure)

❌ **List of Items** (N ≥ 3)

-   Example: "5 SOLID principles"
-   Action: Overview card + 5 detail cards = 6 total

❌ **Multiple Examples**

-   Example: "SQL Joins" with 3 examples
-   Action: Concept card + 3 example cards = 4 total

❌ **Subquestions**

-   Example: "What is REST?" + "What are REST verbs?"
-   Action: 2 separate cards

❌ **Complex Topic with Parts**

-   Example: "React Hooks" (useState, useEffect, useContext)
-   Action: Overview + 3 hook-specific cards = 4 total

❌ **Hierarchical Content**

-   Example: "React Lifecycle" (mounting/updating/unmounting phases)
-   Action: Phases card + methods per phase = 4+ cards

## Splitting Strategies

### 1. Concept Splitting

**Pattern**: Multiple independent concepts in one note

**Example**:

```
Input: "Python Lists and Dictionaries"
Output: 2 cards
- Card 1: "What are Python lists?"
- Card 2: "What are Python dictionaries?"
```

### 2. List Item Splitting

**Pattern**: List of N items (principles, steps, types)

**Example**:

```
Input: "5 SOLID Principles"
Output: 6 cards
- Card 1: "What are the 5 SOLID principles?" (overview)
- Cards 2-6: One per principle (details)
```

### 3. Example Splitting

**Pattern**: Main concept + multiple examples

**Example**:

```
Input: "SQL Joins with Examples"
Output: 4 cards
- Card 1: "What does SQL JOIN do?" (concept)
- Card 2: "How does INNER JOIN work?"
- Card 3: "How does LEFT JOIN work?"
- Card 4: "How does RIGHT JOIN work?"
```

### 4. Hierarchical Splitting

**Pattern**: Main topic with subtopics

**Example**:

```
Input: "React Component Lifecycle"
Output: 4 cards
- Card 1: "What are the 3 lifecycle phases?" (overview)
- Card 2: "Mounting phase methods"
- Card 3: "Updating phase methods"
- Card 4: "Unmounting phase methods"
```

### 5. Step-by-Step Splitting

**Pattern**: Process with multiple steps

**Example**:

```
Input: "Git Branching Workflow"
Output: 6 cards
- Card 1: "What are the Git branching steps?" (overview)
- Cards 2-6: One per step (details)
```

### 6. Difficulty-Based Splitting (NEW)

**Pattern**: Concepts with varying difficulty levels

**Example**:

```
Input: "JavaScript Promises"
Output: 5 cards (ordered by difficulty)
- Card 1: "How to create a Promise?" (easy)
- Card 2: "How to chain Promises?" (medium)
- Card 3: "What does Promise.all() do?" (medium)
- Card 4: "How to use async/await?" (medium)
- Card 5: "What is Promise.allSettled()?" (hard)
```

### 7. Prerequisite-Aware Splitting (NEW)

**Pattern**: Concepts with dependencies (foundational concepts first)

**Example**:

```
Input: "React State Management"
Output: 5 cards (ordered by prerequisites)
- Card 1: "What is useState hook?" (foundational)
- Card 2: "What is useEffect hook?" (requires useState)
- Card 3: "What is useContext hook?" (requires hooks)
- Card 4: "How to create custom hooks?" (requires useState/useEffect)
- Card 5: "What is Redux?" (advanced, uses hooks)
```

### 8. Context-Aware Splitting (NEW)

**Pattern**: Related concepts that can be grouped or separated

**Example**:

```
Input: "HTTP Status Codes"
Decision: Group by category
Output: 4 cards
- Card 1: "What are HTTP 2xx status codes?" (Success: 200, 201, 204)
- Card 2: "What are HTTP 3xx status codes?" (Redirection: 301, 302, 304)
- Card 3: "What are HTTP 4xx status codes?" (Client error: 400, 401, 404)
- Card 4: "What are HTTP 5xx status codes?" (Server error: 500, 502, 503)
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Card Splitting Agent            │
│                                         │
│  Input:                                 │
│  - Note content                         │
│  - Metadata                             │
│  - Q&A pairs                            │
│                                         │
│  Analysis:                              │
│  1. Count distinct concepts             │
│  2. Identify patterns (list/steps/etc)  │
│  3. Apply decision criteria             │
│  4. Generate split plan                 │
│                                         │
│  Output:                                │
│  - should_split: bool                   │
│  - card_count: int                      │
│  - splitting_strategy: enum             │
│  - split_plan: list[CardPlan]           │
│  - reasoning: str                       │
└─────────────────────────────────────────┘
              │
              ├─ Single Card Route
              │    └─> Generate 1 card
              │
              └─ Multiple Cards Route
                   └─> Generate N cards (per split plan)
```

## Integration with LangGraph

### Current Workflow (No Splitting)

```
Pre-Validation → Generation → Post-Validation → Complete
```

### Enhanced Workflow (With Splitting)

```
                    ┌─────────────────┐
                    │ Pre-Validation  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Card Splitting  │ ◄── NEW AGENT
                    │    Agent        │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     should_split=false          should_split=true
              │                             │
              ▼                             ▼
    ┌─────────────────┐          ┌──────────────────┐
    │   Generation    │          │   Multi-Card     │
    │   (1 card)      │          │   Generation     │
    │                 │          │   (N cards)      │
    └────────┬────────┘          └────────┬─────────┘
             │                            │
             └────────────┬───────────────┘
                          │
                 ┌────────▼─────────┐
                 │ Post-Validation  │
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │    Complete      │
                 └──────────────────┘
```

## Usage

### Programmatic Usage

```python
from obsidian_anki_sync.agents.pydantic_ai_agents import CardSplittingAgentAI
from obsidian_anki_sync.providers.pydantic_ai_models import create_openrouter_model_from_env

# Create model (cheap, fast model)
model = create_openrouter_model_from_env("openai/gpt-4o-mini")

# Initialize agent
agent = CardSplittingAgentAI(model=model, temperature=0.0)

# Analyze note
result = await agent.analyze(
    note_content=content,
    metadata=metadata,
    qa_pairs=qa_pairs
)

# Check decision
if result.should_split:
    print(f"✓ Split into {result.card_count} cards ({result.splitting_strategy})")
    print(f"Reasoning: {result.reasoning}")

    # Use split plan
    for plan in result.split_plan:
        print(f"\nCard {plan.card_number}: {plan.concept}")
        print(f"  Q: {plan.question}")
        print(f"  A: {plan.answer_summary}")
else:
    print("✓ Keep as single card")
```

### Example Output (Split Decision)

```json
{
    "should_split": true,
    "card_count": 6,
    "splitting_strategy": "list",
    "split_plan": [
        {
            "card_number": 1,
            "concept": "HTTP methods overview",
            "question": "What are the 5 main HTTP methods in REST APIs?",
            "answer_summary": "GET, POST, PUT, PATCH, DELETE",
            "rationale": "Overview card for context"
        },
        {
            "card_number": 2,
            "concept": "GET method",
            "question": "What does GET do in REST API?",
            "answer_summary": "Retrieves data from server",
            "rationale": "Individual method card"
        }
        // ... 4 more cards
    ],
    "reasoning": "Note lists 5 HTTP methods. Creating overview + detail cards follows SRS best practices.",
    "confidence": 0.92,
    "fallback_strategy": null,
    "decision_time": 0.8
}
```

### Example Output (No Split)

```json
{
    "should_split": false,
    "card_count": 1,
    "splitting_strategy": "none",
    "split_plan": [
        {
            "card_number": 1,
            "concept": "Binary search complexity",
            "question": "What is the time complexity of binary search?",
            "answer_summary": "O(log n)",
            "rationale": "Single focused question"
        }
    ],
    "reasoning": "Single atomic concept with clear Q&A. No splitting needed.",
    "confidence": 0.95,
    "fallback_strategy": null,
    "decision_time": 0.5
}
```

## Configuration

### Model Selection

Recommended model: **moonshotai/kimi-k2-thinking** (advanced reasoning for decision making)

```yaml
# Card Splitting Configuration
enable_card_splitting: true
card_splitting_model: "moonshotai/kimi-k2-thinking"

# Card Splitting Preferences
card_splitting_preferred_size: "medium" # small, medium, large
card_splitting_prefer_splitting: true # Prefer splitting complex notes
card_splitting_min_confidence: 0.7 # Minimum confidence to apply split
card_splitting_max_cards_per_note: 10 # Safety limit
```

### Preference Options

-   **preferred_size**:

    -   `small`: Encourages more splits (more cards per note)
    -   `medium`: Balanced approach (default)
    -   `large`: Prefers fewer splits (larger cards)

-   **prefer_splitting**:

    -   `true`: Default to splitting when ambiguous (better retention)
    -   `false`: Default to single card when ambiguous

-   **min_confidence**:

    -   Minimum confidence score (0.0-1.0) required to apply split decision
    -   Low confidence decisions will use fallback strategy or default to no split

-   **max_cards_per_note**:
    -   Safety limit to prevent excessive card generation
    -   Split plans exceeding this limit will be truncated

### Cost

-   **Model**: gpt-4o-mini (~$0.15/1M tokens)
-   **Per note**: ~$0.0002 (very cheap)
-   **100 notes**: ~$0.02

## Red Flags for Splitting

The agent automatically detects these patterns:

🚩 **Title contains "and"** → Likely 2+ concepts
🚩 **Answer has list (3+ items)** → Split per item
🚩 **Multiple code examples** → Split per example
🚩 **"Steps to..." or "How to..."** → Split per step
🚩 **"Types of..." or "Kinds of..."** → Split per type
🚩 **Answer > 300 words** → Likely too much for one card
🚩 **Multiple subquestions** → Split per question

## Decision Examples

### Example 1: No Split (Simple Q&A)

**Input**:

```
Title: "Binary Search Complexity"
Q: What is the time complexity of binary search?
A: O(log n) because it halves the search space each iteration.
```

**Decision**: ✅ Single card
**Reason**: One atomic concept, short content

---

### Example 2: Split (List Pattern)

**Input**:

```
Title: "REST API HTTP Methods"
Q: What are the main HTTP methods?
A:
- GET: Retrieve data
- POST: Create resource
- PUT: Update entire resource
- PATCH: Update partial resource
- DELETE: Remove resource
```

**Decision**: ✗ Split into 6 cards
**Strategy**: List item splitting
**Cards**:

1. Overview (what are the 5 methods?)
   2-6. One per method (GET, POST, PUT, PATCH, DELETE)

---

### Example 3: Split (Multiple Concepts)

**Input**:

```
Title: "Python Lists and Dictionaries"
Q: What are lists and dictionaries?
A:
Lists are ordered, mutable sequences.
Dictionaries are key-value mappings.
```

**Decision**: ✗ Split into 2 cards
**Strategy**: Concept splitting
**Cards**:

1. What are Python lists?
2. What are Python dictionaries?

---

### Example 4: No Split (Tightly Coupled)

**Input**:

```
Title: "Microservices Tradeoffs"
Q: What are pros and cons of microservices?
A:
Pros: independent deployment, flexibility, scalability
Cons: complexity, latency, harder debugging
```

**Decision**: ✅ Single card
**Reason**: Comparative content requires both sides for context

---

## Benefits

### For Learning

✅ **Better Retention**: Atomic cards = stronger memories
✅ **Progressive Mastery**: Learn one concept at a time
✅ **Flexible Reviews**: Review specific concepts
✅ **Optimal Scheduling**: Each concept gets individual SRS timing

### For Quality

✅ **Follows SRS Best Practices**: One fact per card
✅ **Reduces Cognitive Load**: Manageable chunks
✅ **Improves Success Rate**: Easier to recall single facts
✅ **Increases Motivation**: More success = more engagement

### For Workflow

✅ **Automatic Decision**: No manual splitting needed
✅ **Consistent Strategy**: Same logic for all notes
✅ **Detailed Plans**: Clear card specifications
✅ **Explained Reasoning**: Understand why split/no split

## Limitations

1. **Content-Based Only**: Can't consider user's existing knowledge
2. **Static Analysis**: Doesn't adapt based on review performance
3. **Language Specific**: Optimized for English/Russian tech content
4. **No User Preferences**: Some users may prefer different splitting

## Enhanced Features (2025)

### Confidence Scoring

The agent now provides confidence scores for split decisions:

-   **High Confidence (0.85-1.0)**: Clear pattern, unambiguous decision
-   **Medium Confidence (0.6-0.84)**: Some ambiguity, but decision is reasonable
-   **Low Confidence (0.0-0.59)**: Unclear, uses fallback strategy

Low confidence decisions (< min_confidence threshold) will:

-   Use fallback strategy if provided
-   Default to no split if fallback unavailable
-   Log warnings for review

### Fallback Strategies

When confidence is low, the agent can specify a fallback strategy:

-   Alternative splitting approach
-   Default to single card
-   Manual review recommendation

### User Preferences

Users can configure splitting behavior:

-   **Preferred card size**: small/medium/large
-   **Prefer splitting**: Encourage or discourage splitting
-   **Confidence threshold**: Minimum confidence to apply split
-   **Safety limits**: Maximum cards per note

## Future Enhancements

-   [ ] Domain-specific splitting strategies (medical, language, etc.)
-   [ ] Learning from review performance (adjust splitting based on success rates)
-   [ ] Interactive mode (suggest split, ask for confirmation)
-   [ ] Batch analysis (analyze entire vault, suggest resplits)
-   [ ] Integration with card merging (inverse operation)

## Research Background

Card splitting decisions are based on:

1. **Atomic Principle**: Wozniak's "20 Rules" (one fact per card)
2. **Cognitive Load Theory**: Miller's 7±2 chunks
3. **Spaced Repetition Research**: Granular items = better scheduling
4. **Progressive Learning**: Build from simple to complex

## Support

-   **Issues**: https://github.com/po4yka/obsidian-to-anki/issues
-   **Documentation**: This file
-   **Code**: `src/obsidian_anki_sync/agents/pydantic_ai_agents.py` (CardSplittingAgentAI)

---

**Version**: 1.0
**Last Updated**: 2025-01-12
**Status**: Implemented, ready for LangGraph integration
