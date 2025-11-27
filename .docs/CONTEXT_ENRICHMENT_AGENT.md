# Context Enrichment Agent

**Status**: ✅ Implemented (2025-01-12)
**Agent Type**: PydanticAI-based content enhancement
**Purpose**: Automatically add examples, mnemonics, and context to improve card memorability

## Overview

The Context Enrichment Agent enhances flashcards by adding:

1. **Concrete Examples**: Code snippets, real-world scenarios
2. **Mnemonics**: Memory aids and mental models
3. **Visual Structure**: Formatting, bullet points, emphasis
4. **Related Concepts**: Context and connections
5. **Practical Tips**: Best practices and common pitfalls

This solves a critical problem: **Bare facts are hard to remember without context**.

### The Problem

Without context enrichment:

-   ❌ **Poor Retention**: Abstract facts don't stick
-   ❌ **No Application**: Can't use knowledge in practice
-   ❌ **Boring Reviews**: Text-heavy cards are demotivating
-   ❌ **Shallow Understanding**: Definition without depth

With context enrichment:

-   ✅ **Better Retention**: Examples make concepts concrete
-   ✅ **Practical Knowledge**: Know how to apply information
-   ✅ **Engaging Reviews**: Rich content maintains interest
-   ✅ **Deep Understanding**: Context builds comprehension

## Enrichment Types

### 1. Concrete Examples

**What**: Real-world usage and code samples

**When to add**:

-   Abstract concepts
-   Programming syntax
-   Algorithms and data structures

**Example**:

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

---

### 2. Mnemonics & Memory Aids

**What**: Acronyms, metaphors, memorable phrases

**When to add**:

-   Lists of items
-   Confusing terminology
-   Complex sequences

**Example**:

```
BEFORE:
Q: What are the HTTP safe methods?
A: GET, HEAD, OPTIONS

AFTER:
Q: What are the HTTP safe methods?
A: GET, HEAD, OPTIONS

Extra:
💡 Mnemonic: "GHO" (like ghost)
These methods are "safe" - they only read data, never modify it.
Think: "ghosts can observe but can't touch"
```

---

### 3. Visual Structure

**What**: Formatting, bullet points, emphasis

**When to add**:

-   List-heavy content
-   Multiple concepts in one answer
-   Important key terms

**Example**:

```
BEFORE:
Q: What are React hooks rules?
A: Only call at top level, only call from React functions.

AFTER:
Q: What are React hooks rules?
A: Two key rules:

1. **Only call at top level**
   - Not inside loops, conditions, or nested functions
   - Ensures consistent hook order

2. **Only call from React functions**
   - React function components
   - Custom hooks (must start with "use")
```

---

### 4. Related Concepts

**What**: Comparisons, alternatives, prerequisites

**When to add**:

-   Concepts with alternatives
-   Terms often confused
-   Part of larger concept

**Example**:

```
BEFORE:
Q: What is Promise.all()?
A: Waits for all promises to resolve.

AFTER:
Q: What is Promise.all()?
A: Waits for all promises to resolve or any to reject.

Extra:
Compare with:
- Promise.race() - first to settle wins
- Promise.allSettled() - waits for all, doesn't short-circuit
- Promise.any() - first to resolve wins

Use when: You need all results before proceeding (e.g., loading multiple API endpoints)
```

---

### 5. Practical Tips

**What**: Best practices, common mistakes, debugging tips

**When to add**:

-   Concepts with common pitfalls
-   Tricky syntax
-   Performance implications

**Example**:

```
BEFORE:
Q: What does useEffect do?
A: Runs side effects in React components.

AFTER:
Q: What does useEffect do?
A: Runs side effects in React components after render.

Extra:
Common use cases:
- API calls
- Subscriptions
- DOM manipulation

⚠️ Common mistake: Forgetting dependency array
// Runs every render ❌
useEffect(() => { fetchData(); });

// Runs once ✓
useEffect(() => { fetchData(); }, []);

// Runs when id changes ✓
useEffect(() => { fetchData(id); }, [id]);
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────┐
│      Context Enrichment Agent           │
│                                         │
│  Input:                                 │
│  - Card (Q, A, Extra)                   │
│  - Note metadata                        │
│                                         │
│  Analysis:                              │
│  1. Evaluate if card needs enrichment   │
│  2. Identify enrichment opportunities   │
│  3. Generate appropriate additions      │
│  4. Maintain atomic principle           │
│  5. Format for readability              │
│                                         │
│  Output:                                │
│  - should_enrich: bool                  │
│  - enriched_card: GeneratedCard         │
│  - additions: list[EnrichmentAddition]  │
│  - rationale: str                       │
└─────────────────────────────────────────┘
```

### Decision Criteria

**✅ DO Enrich When**:

-   Answer is too abstract
-   Concept has common misconceptions
-   Information is list-heavy without structure
-   Concept has practical applications
-   Term has helpful mnemonic
-   Frequently confused with similar concepts

**❌ DON'T Enrich When**:

-   Card is already comprehensive
-   Concept is self-explanatory
-   Would cause information overload
-   Context is domain-specific jargon

## Usage

### Programmatic Usage

```python
from obsidian_anki_sync.agents.pydantic import ContextEnrichmentAgentAI
from obsidian_anki_sync.providers.pydantic_ai_models import create_openrouter_model_from_env

# Create model (slightly creative temperature)
model = create_openrouter_model_from_env("openai/gpt-4o-mini")

# Initialize agent
agent = ContextEnrichmentAgentAI(model=model, temperature=0.3)

# Enrich a card
result = await agent.enrich(
    card=generated_card,
    metadata=note_metadata
)

if result.should_enrich:
    print(f"✓ Card enriched with: {', '.join([a.enrichment_type for a in result.additions])}")
    print(f"Summary: {result.additions_summary}")
    print(f"Rationale: {result.enrichment_rationale}")

    # Use enriched card
    enriched_card = result.enriched_card
else:
    print("Card is already comprehensive - no enrichment needed")
```

### Example Output (Enrichment Applied)

```json
{
    "should_enrich": true,
    "enriched_card": {
        "card_index": 1,
        "slug": "python-list-comprehension-en-001",
        "lang": "en",
        "apf_html": "<div class=\"card\">...</div>",
        "confidence": 0.92
    },
    "additions": [
        {
            "enrichment_type": "example",
            "content": "Traditional approach vs list comprehension syntax with output",
            "rationale": "Concrete code example makes abstract syntax tangible"
        },
        {
            "enrichment_type": "visual",
            "content": "Syntax reference: [expression for item in iterable if condition]",
            "rationale": "Quick reference for future lookups"
        }
    ],
    "additions_summary": "Added code example showing traditional vs comprehension syntax, plus syntax reference",
    "enrichment_rationale": "Concrete code example makes abstract syntax tangible. Comparison shows why it's useful. Syntax reference provides quick lookup.",
    "enrichment_time": 1.2
}
```

### Example Output (No Enrichment Needed)

```json
{
    "should_enrich": false,
    "enriched_card": null,
    "additions": [],
    "additions_summary": "No enrichment needed",
    "enrichment_rationale": "Card is already comprehensive with concrete example, visualization, requirements, and comparison. Adding more would violate atomic principle.",
    "enrichment_time": 0.6
}
```

## Configuration

### Model Selection

Recommended model: **gpt-4o-mini** with temperature=0.3 (slight creativity for examples)

```yaml
# config.yaml (future integration)
context_enrichment_enabled: true
context_enrichment_model: "openai/gpt-4o-mini"
enrichment_temperature: 0.3 # Allow some creativity
enrichment_types_enabled:
    - example
    - mnemonic
    - visual
    - related
    - practical
max_extra_length: 500 # Prevent information overload
```

### Cost

-   **Model**: gpt-4o-mini (~$0.15/1M tokens)
-   **Per card**: ~$0.0003 (slightly more than other agents due to generation)
-   **100 cards**: ~$0.03

## Integration Scenarios

### Scenario 1: Post-Generation Enhancement (Recommended)

Enrich cards immediately after generation:

```python
async def generate_and_enrich(note, metadata):
    # Generate cards
    generation_result = await generator_agent.generate(note, metadata)

    # Enrich each card
    enrichment_agent = ContextEnrichmentAgentAI(model)
    enriched_cards = []

    for card in generation_result.cards:
        enrichment_result = await enrichment_agent.enrich(card, metadata)

        if enrichment_result.should_enrich:
            enriched_cards.append(enrichment_result.enriched_card)
            logger.info(f"Enriched {card.slug}: {enrichment_result.additions_summary}")
        else:
            enriched_cards.append(card)

    return enriched_cards
```

### Scenario 2: Batch Enrichment

Enrich existing deck:

```python
async def enrich_existing_deck(deck_cards, metadata_map):
    agent = ContextEnrichmentAgentAI(model)
    enriched_count = 0

    for card in deck_cards:
        metadata = metadata_map[card.slug]
        result = await agent.enrich(card, metadata)

        if result.should_enrich:
            # Update card in deck
            update_card(result.enriched_card)
            enriched_count += 1
            print(f"✓ {card.slug}: {result.additions_summary}")

    print(f"\nEnriched {enriched_count}/{len(deck_cards)} cards")
```

### Scenario 3: LangGraph Integration (Future)

Add as optional post-validation node:

```python
def context_enrichment_node(state: PipelineState) -> PipelineState:
    """Enrich cards with examples and context."""

    enrichment_agent = create_enrichment_agent()
    enriched_cards = []

    for card in state["generation"]["cards"]:
        result = enrichment_agent.enrich(card, state["metadata"])

        if result.should_enrich:
            enriched_cards.append(result.enriched_card)
        else:
            enriched_cards.append(card)

    state["generation"]["cards"] = enriched_cards
    state["enrichment_results"] = [...]

    return state
```

## Enrichment Examples

### Example 1: Programming Concept

**Input**:

```
Q: What is a Python generator?
A: Function that uses yield to return values lazily.
```

**Enrichment**:

```
Extra added:
Example:
def count_up_to(n):
    i = 0
    while i < n:
        yield i
        i += 1

# Usage
for num in count_up_to(5):
    print(num)  # 0, 1, 2, 3, 4

Benefits:
- Memory efficient (doesn't store all values)
- Infinite sequences possible
- Lazy evaluation

Compare with: list comprehension (eager, stores all values)
```

---

### Example 2: Mnemonic Addition

**Input**:

```
Q: What are the SOLID principles?
A: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion
```

**Enrichment**:

```
Answer restructured:
Five object-oriented design principles:

1. **S**ingle Responsibility - One class, one job
2. **O**pen/Closed - Open for extension, closed for modification
3. **L**iskov Substitution - Subtypes must be substitutable
4. **I**nterface Segregation - Many specific interfaces > one general
5. **D**ependency Inversion - Depend on abstractions, not concretions

Extra added:
💡 Mnemonic: "**SOLID**" - the acronym itself means "well-built"
Think: "A SOLID foundation makes code maintainable"
```

---

### Example 3: Comparison Table

**Input**:

```
Q: Difference between undefined and null in JavaScript?
A: undefined = not assigned, null = explicit absence
```

**Enrichment**:

```
Answer enhanced:
**undefined**: Variable declared but not assigned (implicit)
**null**: Explicit assignment to represent "no value" (intentional)

Extra added:
Comparison:
| undefined | null |
|-----------|------|
| let x; // undefined | let x = null; |
| Function param not passed | Intentional absence |
| Object property doesn't exist | Property set to "nothing" |
| typeof undefined // 'undefined' | typeof null // 'object' (JS quirk!) |

⚠️ Common pitfall:
if (x == null) // true for BOTH undefined and null
if (x === null) // true only for null ✓
```

## Benefits

### For Learning

✅ **Better Retention**: Examples make facts memorable
✅ **Deeper Understanding**: Context builds comprehension
✅ **Practical Skills**: Know how to apply knowledge
✅ **Reduced Confusion**: Comparisons clarify differences

### For Quality

✅ **Comprehensive Cards**: Complete information
✅ **Professional Polish**: Well-formatted and structured
✅ **Consistent Style**: Uniform enrichment across deck
✅ **Best Practices**: Includes warnings and tips

### For Workflow

✅ **Automatic Enhancement**: No manual editing needed
✅ **Smart Decisions**: Only enriches when beneficial
✅ **Preserves Atomicity**: Doesn't overload cards
✅ **Explained Additions**: Clear rationale for changes

## Limitations

1. **Context-Limited**: Can't access external documentation
2. **Domain-Specific**: Best for programming/tech content
3. **No Personalization**: Can't adapt to user's knowledge level
4. **Generation Quality**: Depends on LLM capabilities

## Future Enhancements

-   [ ] Web search integration for accurate examples
-   [ ] Domain-specific enrichment strategies (medical, languages, etc.)
-   [ ] User preferences (prefer mnemonics vs examples)
-   [ ] Interactive enrichment (suggest, ask for approval)
-   [ ] Learning from user feedback (which enrichments are helpful)
-   [ ] Image generation for visual concepts
-   [ ] LaTeX math formatting for mathematical concepts

## Research Background

Context enrichment is based on:

1. **Dual Coding Theory**: Visual + verbal = better memory
2. **Elaborative Rehearsal**: Deep processing aids retention
3. **Chunking**: Structure information in memorable units
4. **Transfer-Appropriate Processing**: Practice aids application
5. **Spaced Repetition Research**: Rich context enhances recall

## Support

-   **Issues**: https://github.com/po4yka/obsidian-to-anki/issues
-   **Documentation**: This file
-   **Code**: `src/obsidian_anki_sync/agents/pydantic_ai_agents.py` (ContextEnrichmentAgentAI)

---

**Version**: 1.0
**Last Updated**: 2025-01-12
**Status**: Implemented, ready for integration
