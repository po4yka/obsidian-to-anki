# Context Enrichment Skill

**Purpose**: Enhance flashcards with concrete examples, mnemonics, visual structure, and practical context to improve memorability.

**When to Use**: Load during card generation to add enriching context that makes cards more memorable and effective.

---

## Core Principle: Make Abstract Concrete

Abstract concepts are hard to remember. Concrete examples, visual structure, and practical context make information tangible and memorable.

---

## Enrichment Dimensions

### 1. Concrete Examples

**Why**: Abstract concepts are hard to remember; concrete examples make them tangible.

**What to Add**:
- Code snippets showing usage
- Real-world scenarios
- Before/after comparisons
- Common use cases

**How to Apply**:

**BEFORE**:
```
Q: What is array destructuring in JavaScript?
A: Syntax to unpack array values into variables.
```

**AFTER**:
```
Q: What is array destructuring in JavaScript?
A: Syntax to unpack array values into variables.

Extra:
Example:
const [first, second] = [10, 20];
// first = 10, second = 20

Common use case:
const [state, setState] = useState(0);
```

**Guidelines**:
- Keep examples minimal but complete (no placeholders)
- Use real-world scenarios when possible
- Show common patterns, not edge cases
- Include comments explaining what's happening

---

### 2. Mnemonics & Memory Aids

**Why**: Memory hooks make information easier to encode and retrieve.

**What to Add**:
- Acronyms (SOLID, CRUD, etc.)
- Visual metaphors
- Memorable phrases
- Association techniques

**How to Apply**:

**BEFORE**:
```
Q: What are the HTTP safe methods?
A: GET, HEAD, OPTIONS
```

**AFTER**:
```
Q: What are the HTTP safe methods?
A: GET, HEAD, OPTIONS

Extra:
Mnemonic: "GHO" (like ghost)
These methods are "safe" - they only read data, never modify it.
Think: "ghosts can observe but can't touch"
```

**Guidelines**:
- Use mnemonics sparingly (only when helpful)
- Make mnemonics memorable and relevant
- Prefer established mnemonics when available
- Explain the mnemonic if it's not obvious

---

### 3. Visual Structure

**Why**: Well-formatted information is easier to scan and remember.

**What to Add**:
- Bullet points for lists
- Emphasis (bold) for key terms
- Line breaks for readability
- Code blocks for syntax

**How to Apply**:

**BEFORE**:
```
Q: What are React hooks rules?
A: Only call at top level, only call from React functions.
```

**AFTER**:
```
Q: What are React hooks rules?
A: Two key rules:

- Only call at top level (not in loops/conditions)
- Only call from React functions (components/custom hooks)

Extra:
Example violation:
if (condition) {
  const [state] = useState(0); // WRONG
}
```

**Guidelines**:
- Use bullets for lists (3-6 items ideal)
- Bold key terms on first mention
- Break up walls of text
- Use code blocks for syntax examples

---

### 4. Related Concepts

**Why**: Connecting new information to existing knowledge improves retention.

**What to Add**:
- Related concepts and how they connect
- Contrasts with similar concepts
- Prerequisites or dependencies
- Broader context

**How to Apply**:

**BEFORE**:
```
Q: What is useEffect in React?
A: Hook for side effects in functional components.
```

**AFTER**:
```
Q: What is useEffect in React?
A: Hook for side effects in functional components.

Extra:
Related concepts:
- Replaces componentDidMount, componentDidUpdate, componentWillUnmount
- Similar to: useState (both are hooks)
- Different from: useMemo (memoization vs side effects)

Common use cases:
- Fetching data
- Setting up subscriptions
- Manually changing DOM
```

**Guidelines**:
- Connect to concepts learners likely know
- Show relationships clearly
- Highlight differences from similar concepts
- Keep related concepts concise

---

### 5. Practical Tips

**Why**: Practical knowledge is more motivating and retained longer.

**What to Add**:
- Best practices
- Common pitfalls
- When to use vs. when not to use
- Performance considerations

**How to Apply**:

**BEFORE**:
```
Q: What is memoization?
A: Caching function results to avoid recomputation.
```

**AFTER**:
```
Q: What is memoization?
A: Caching function results to avoid recomputation.

Extra:
Best practices:
- Use for expensive computations
- Cache key should be stable
- Consider memory trade-offs

Common pitfalls:
- Memoizing functions that change frequently
- Forgetting to include all dependencies in cache key
- Over-memoizing simple computations

When to use:
- Expensive calculations (fibonacci, factorials)
- API calls with same parameters
- Rendering expensive components
```

**Guidelines**:
- Focus on actionable advice
- Highlight common mistakes
- Explain trade-offs
- Keep tips concise and specific

---

## Enrichment Checklist

When enriching a card, consider:

1. ✓ **Examples**: Does it have concrete code examples or scenarios?
2. ✓ **Mnemonics**: Would a memory aid help remember this?
3. ✓ **Structure**: Is information well-formatted and scannable?
4. ✓ **Relationships**: Are related concepts connected?
5. ✓ **Practical Value**: Are best practices and pitfalls included?

---

## When NOT to Enrich

**Don't enrich if**:
- Card is already clear and memorable
- Adding context would make card too long (>200 words)
- Examples would require extensive explanation
- Information is self-explanatory

**Balance**: Enrichment should help, not overwhelm. Keep cards focused and atomic.

---

## Enrichment Guidelines by Card Type

### Simple Cards
- Add examples in Extra section
- Include related concepts
- Add practical tips

### Missing (Cloze) Cards
- Keep examples minimal (they're for syntax)
- Focus on correct usage patterns
- Show common mistakes

### Draw Cards
- Add sequence explanations
- Include component relationships
- Explain flow or architecture

---

## Common Mistakes to Avoid

❌ **Over-enrichment**: Adding too much context, making cards overwhelming
❌ **Irrelevant examples**: Examples that don't illustrate the concept
❌ **Weak mnemonics**: Mnemonics that are harder to remember than the concept
❌ **Poor structure**: Walls of text without formatting
❌ **Missing practical value**: No connection to real-world usage

---

## Best Practices

✓ **Concrete over abstract**: Always prefer examples
✓ **Structure aids memory**: Use formatting effectively
✓ **Connect to known concepts**: Build on existing knowledge
✓ **Practical focus**: Emphasize real-world application
✓ **Balance**: Enrich without overwhelming
✓ **Quality over quantity**: Better to have fewer, well-enriched cards

