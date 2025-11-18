# Card Splitting Skill

**Purpose**: Determine optimal card generation strategy - when to create single vs. multiple cards from a note.

**When to Use**: Load before card generation to analyze note structure and decide splitting strategy.

---

## Core Decision Framework

### Single Card vs. Multiple Cards

The fundamental question: **Can this content be learned as one atomic concept, or does it contain multiple independent concepts?**

---

## When to Create SINGLE Card

### ✅ Single Atomic Concept

**Pattern**: Note discusses ONE specific topic or fact.

**Example**:
- "What is Big O notation?"
- "What does `git log` do?"
- "What is the time complexity of binary search?"

**Rationale**: One concept = one card. Atomic principle applies.

---

### ✅ Simple Q&A

**Pattern**: Single question with straightforward answer.

**Example**:
- Q: "What is a closure in JavaScript?"
- A: "Function that retains access to outer scope variables"

**Rationale**: Simple, focused question-answer pair doesn't need splitting.

---

### ✅ Tightly Coupled Information

**Pattern**: Components must be learned together.

**Example**:
- "Pros and cons of approach X" (comparison needs both sides)
- "React component lifecycle: mount → update → unmount" (sequence is one concept)

**Rationale**: Splitting would lose important relationships between parts.

---

### ✅ Short Content

**Pattern**: Question + Answer < 200 words total, no subquestions or examples.

**Example**:
- Brief definition with concise explanation
- Simple fact with minimal context

**Rationale**: Content is naturally atomic and doesn't warrant splitting.

---

## When to Create MULTIPLE Cards

### ❌ Multiple Independent Concepts

**Pattern**: Note covers 2+ distinct topics that can be learned separately.

**Example**:
- "Python Lists AND Dictionaries" → Split into 2 cards
- "React Hooks: useState, useEffect, useContext" → Split into 3 cards

**Rationale**: Each concept deserves its own card for focused learning.

**Action**: Create one card per concept.

---

### ❌ List of Items

**Pattern**: Note contains a list of principles, steps, or items.

**Example**:
- "5 SOLID principles" → Create 5 separate cards (one per principle)
- "Steps to deploy" → One card per step
- "Common HTTP status codes" → One card per status code

**Rationale**: Lists create cognitive overload. Each item should be a separate recall target.

**Action**: Create N+1 cards:
- 1 overview card: "What are the 5 SOLID principles?" → "SRP, OCP, LSP, ISP, DIP"
- N detail cards: One per item

---

### ❌ Multiple Examples

**Pattern**: Main concept + multiple code examples or use cases.

**Example**:
- Main concept: "What is array destructuring?"
- Example 1: Basic destructuring
- Example 2: Nested destructuring
- Example 3: Rest operator with destructuring

**Rationale**: Examples can be learned independently and reinforce the concept.

**Action**:
- 1 concept card: Core definition
- N example cards: One per example (if examples teach distinct patterns)

---

### ❌ Subquestions

**Pattern**: Main question + follow-up questions.

**Example**:
- Main: "What is REST?"
- Follow-up: "What are REST verbs?"
- Follow-up: "What is RESTful design?"

**Rationale**: Each question tests different knowledge and should be separate.

**Action**: Create one card per question.

---

### ❌ Complex Topic with Parts

**Pattern**: Topic has multiple aspects that can be learned independently.

**Example**:
- "React Hooks" → useState, useEffect, useContext (separate cards)
- "Python data structures" → Lists, Tuples, Dictionaries (separate cards)

**Rationale**: Each part is a distinct concept requiring focused study.

**Action**: Split by concept, create one card per part.

---

### ❌ Comparative Content

**Pattern**: "X vs Y" where X and Y are complex concepts.

**Example**:
- "REST vs GraphQL"
- "SQL vs NoSQL"
- "Class components vs Functional components"

**Rationale**: Each concept deserves focused learning, plus comparison adds value.

**Action**:
- Card 1: Concept X
- Card 2: Concept Y
- Card 3: Comparison (if comparison itself is valuable knowledge)

---

## Splitting Strategies

### Strategy 1: Concept Splitting

**Pattern**: Multiple independent concepts in one note.

**Approach**: Create one card per concept.

**Example**:
```
Note: "Python Data Structures"
Content:
- Lists are ordered, mutable sequences
- Tuples are ordered, immutable sequences
- Dictionaries are key-value mappings

Split:
1. "What are Python lists?" → "Ordered, mutable sequences"
2. "What are Python tuples?" → "Ordered, immutable sequences"
3. "What are Python dictionaries?" → "Key-value mappings"
```

---

### Strategy 2: List Item Splitting

**Pattern**: List of N items (principles, steps, examples).

**Approach**: Create N+1 cards (overview + details).

**Example**:
```
Note: "SOLID Principles"
Content:
1. Single Responsibility Principle
2. Open/Closed Principle
3. Liskov Substitution Principle
4. Interface Segregation Principle
5. Dependency Inversion Principle

Split:
1. "What are the 5 SOLID principles?" → "SRP, OCP, LSP, ISP, DIP"
2. "What is Single Responsibility Principle?" → "A class should have one reason to change"
3. "What is Open/Closed Principle?" → "Open for extension, closed for modification"
... (one card per principle)
```

---

### Strategy 3: Example Splitting

**Pattern**: Main concept + multiple examples.

**Approach**: 1 concept card + N example cards (if examples teach distinct patterns).

**Example**:
```
Note: "Array Destructuring in JavaScript"
Content:
- Concept: Unpacking array values into variables
- Example 1: Basic destructuring
- Example 2: Nested destructuring
- Example 3: Rest operator

Split:
1. "What is array destructuring in JavaScript?" → "Syntax to unpack array values into variables"
2. "How do you destructure nested arrays?" → [example code]
3. "How do you use rest operator with destructuring?" → [example code]
```

**Note**: Only split examples if they teach distinct patterns. If examples are variations of the same pattern, keep them in Extra section.

---

### Strategy 4: Question Hierarchy Splitting

**Pattern**: Main question + subquestions.

**Approach**: Create one card per question level.

**Example**:
```
Note: "REST API Basics"
Content:
- What is REST?
- What are REST verbs?
- What is RESTful design?

Split:
1. "What is REST?" → "Representational State Transfer, architectural style for APIs"
2. "What are REST verbs?" → "GET, POST, PUT, DELETE, PATCH"
3. "What is RESTful design?" → "Design following REST principles: stateless, resource-based URLs"
```

---

## Decision Tree

```
Is the note about ONE specific concept?
├─ YES → Single card
└─ NO → Does it contain multiple independent concepts?
    ├─ YES → Split by concept (Strategy 1)
    └─ NO → Is it a list of items?
        ├─ YES → Split by list items (Strategy 2)
        └─ NO → Does it have multiple examples?
            ├─ YES → Split concept + examples (Strategy 3)
            └─ NO → Does it have subquestions?
                ├─ YES → Split by question (Strategy 4)
                └─ NO → Analyze further for other patterns
```

---

## Edge Cases

### When NOT to Split

**Tightly Coupled Sequences**:
- "React component lifecycle: mount → update → unmount" → Single card (sequence is one concept)
- "HTTP request flow: client → server → response" → Single card (flow is one concept)

**Comparisons**:
- "Pros and cons of X" → Single card (comparison is one concept)
- "X vs Y (brief)" → Single card if comparison is concise

**Definitions with Examples**:
- If examples are just illustrations → Keep in Extra section
- If examples teach distinct patterns → Split into separate cards

---

## Quality Checks

After splitting, verify:

1. ✓ Each card tests ONE atomic concept
2. ✓ Cards can be reviewed independently
3. ✓ No information leakage between cards
4. ✓ Each card has clear, unambiguous question
5. ✓ Cards follow memorization principles
6. ✓ Appropriate number of cards (not too many, not too few)

---

## Common Mistakes to Avoid

❌ **Over-splitting**: Creating cards for every minor detail
❌ **Under-splitting**: Keeping multiple concepts in one card
❌ **Inconsistent splitting**: Some notes split, others not, without clear reason
❌ **Losing context**: Splitting removes important relationships
❌ **Creating orphan cards**: Cards that reference each other without context

---

## Best Practices

✓ **Start with atomicity**: One concept per card
✓ **Preserve relationships**: Don't split tightly coupled information
✓ **Use overview cards**: Helpful for lists and hierarchies
✓ **Consider learning order**: Some concepts build on others
✓ **Balance granularity**: Not too fine (overwhelming), not too coarse (ineffective)

