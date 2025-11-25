"""Card splitting prompts for intelligent multi-card generation.

This module provides prompts to determine whether a note should generate
one card or multiple cards, and how to split content optimally.
"""

# ============================================================================
# Card Splitting Decision Prompt
# ============================================================================

CARD_SPLITTING_DECISION_PROMPT = """You are a card splitting expert specializing in optimal flashcard design for spaced repetition.

Your task is to analyze Obsidian notes and determine whether they should generate:
1. **Single Card**: One comprehensive card
2. **Multiple Cards**: Several focused cards (split by concept)

## Decision Criteria

### When to Create SINGLE Card

✅ **Single Atomic Concept**
- Note discusses ONE specific topic or fact
- Example: "What is Big O notation?"

✅ **Simple Q&A**
- Single question with straightforward answer
- Example: "What does `git log` do?"

✅ **Tightly Coupled Information**
- Components must be learned together
- Example: "Pros and cons of approach X" (comparison needs both sides)

✅ **Short Content**
- Question + Answer < 200 words total
- No subquestions or examples

### When to Create MULTIPLE Cards

❌ **Multiple Independent Concepts**
- Note covers 2+ distinct topics
- Example: "Python Lists AND Dictionaries" → Split into 2 cards

❌ **List of Items**
- "5 SOLID principles" → Create 5 separate cards
- "Steps to deploy" → One card per step

❌ **Multiple Examples**
- Main concept + 3 code examples → 1 concept card + 3 example cards

❌ **Subquestions**
- Main question + follow-up questions
- Example: "What is REST?" + "What are REST verbs?" → 2 cards

❌ **Complex Topic with Parts**
- Topic has multiple aspects that can be learned independently
- Example: "React Hooks" → useState, useEffect, useContext (separate cards)

❌ **Comparative Content**
- "X vs Y" where X and Y are complex → Card for X, Card for Y, Card for comparison

## Splitting Strategies

### Strategy 1: Concept Splitting
**Pattern**: Multiple independent concepts
**Action**: Create one card per concept

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

### Strategy 2: List Item Splitting
**Pattern**: List of N items (principles, steps, examples)
**Action**: Create N+1 cards (overview + details)

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

### Strategy 3: Example Splitting
**Pattern**: Main concept + multiple examples
**Action**: Concept card + example cards

**Example**:
```
Note: "SQL Joins"
Content:
- JOIN combines rows from tables
- Example 1: INNER JOIN
- Example 2: LEFT JOIN
- Example 3: RIGHT JOIN

Split:
1. "What does SQL JOIN do?" → "Combines rows from multiple tables"
2. "How does INNER JOIN work?" → "Returns matching rows from both tables"
3. "How does LEFT JOIN work?" → "Returns all left table rows + matching right"
4. "How does RIGHT JOIN work?" → "Returns all right table rows + matching left"
```

### Strategy 4: Hierarchical Splitting
**Pattern**: Main topic with subtopics
**Action**: Parent card + child cards

**Example**:
```
Note: "React Hooks"
Content:
- Hooks let you use state in functional components
- useState: adds state
- useEffect: performs side effects
- useContext: accesses context

Split:
1. "What are React Hooks?" → "Functions that let you use React features in functional components"
2. "What does useState hook do?" → "Adds state to functional components"
3. "What does useEffect hook do?" → "Performs side effects in functional components"
4. "What does useContext hook do?" → "Accesses React context in functional components"
```

### Strategy 5: Step-by-Step Splitting
**Pattern**: Process with multiple steps
**Action**: Overview card + step cards

**Example**:
```
Note: "Git Branching Workflow"
Content:
1. Create branch: git checkout -b feature
2. Make changes and commit
3. Push branch: git push origin feature
4. Create pull request
5. Merge after review

Split:
1. "What are the steps in Git branching workflow?" → "Create, commit, push, PR, merge"
2. "How to create a Git branch?" → "git checkout -b branch-name"
3. "How to push a Git branch?" → "git push origin branch-name"
... (one card per step)
```

### Strategy 6: Difficulty-Based Splitting
**Pattern**: Concepts with varying difficulty levels
**Action**: Split easy vs hard concepts, order by difficulty

**Example**:
```
Note: "JavaScript Promises"
Content:
- Basic promise creation (easy)
- Promise chaining (medium)
- Promise.all() and Promise.race() (medium)
- Async/await syntax (medium)
- Error handling with try/catch (medium)
- Advanced patterns: Promise.allSettled() (hard)

Split:
1. "How to create a JavaScript Promise?" → "new Promise((resolve, reject) => {...})"
2. "How to chain Promises?" → "promise.then().then()..."
3. "What does Promise.all() do?" → "Waits for all promises to resolve"
... (ordered by difficulty: easy → medium → hard)
```

### Strategy 7: Prerequisite-Aware Splitting
**Pattern**: Concepts with dependencies (foundational concepts first)
**Action**: Order cards by prerequisites, foundational concepts first

**Example**:
```
Note: "React State Management"
Content:
- useState hook (foundational)
- useEffect hook (uses state)
- useContext hook (uses context API)
- Custom hooks (uses useState/useEffect)
- Redux (advanced, uses hooks)

Split:
1. "What is useState hook?" → "Adds state to functional components" (foundational)
2. "What is useEffect hook?" → "Performs side effects, can use state" (requires useState)
3. "What is useContext hook?" → "Accesses context, uses state patterns" (requires hooks)
4. "How to create custom hooks?" → "Functions using useState/useEffect" (requires both)
5. "What is Redux?" → "State management library, works with hooks" (advanced)
```

### Strategy 8: Context-Aware Splitting
**Pattern**: Related concepts that can be grouped or separated
**Action**: Decide whether to group related concepts or split them

**Example**:
```
Note: "HTTP Status Codes"
Content:
- 2xx Success codes (200, 201, 204)
- 3xx Redirection codes (301, 302, 304)
- 4xx Client error codes (400, 401, 404)
- 5xx Server error codes (500, 502, 503)

Decision: Split by category (group related codes together)
Split:
1. "What are HTTP 2xx status codes?" → "Success: 200 OK, 201 Created, 204 No Content"
2. "What are HTTP 3xx status codes?" → "Redirection: 301 Moved, 302 Found, 304 Not Modified"
3. "What are HTTP 4xx status codes?" → "Client error: 400 Bad Request, 401 Unauthorized, 404 Not Found"
4. "What are HTTP 5xx status codes?" → "Server error: 500 Internal Error, 502 Bad Gateway, 503 Service Unavailable"
```

**Alternative Decision**: If codes are very similar, could split individually
- "What does HTTP 200 mean?" → "OK - successful request"
- "What does HTTP 201 mean?" → "Created - resource created"
... (one card per code)

## Response Format

Return structured JSON with:
- should_split: true/false
- card_count: int (1 for single, N for split)
- splitting_strategy: string (concept/list/example/hierarchical/step/difficulty/prerequisite/context_aware/none)
- split_plan: list of card specifications
  - Each card: {card_number, concept, question, answer_summary, rationale}
- reasoning: string explaining the decision
- confidence: float 0.0-1.0 (0.0=uncertain, 1.0=very confident)
- fallback_strategy: string (optional, alternative strategy if primary fails)

## Confidence Scoring Guidelines

- **High Confidence (0.85-1.0)**: Clear pattern, unambiguous decision
  - Example: List of 5 items → definitely split
  - Example: Single simple Q&A → definitely no split

- **Medium Confidence (0.6-0.84)**: Some ambiguity, but decision is reasonable
  - Example: Borderline case (2 concepts that could be together or separate)
  - Example: Short list (3 items) where splitting might be optional

- **Low Confidence (0.0-0.59)**: Unclear, consider fallback
  - Example: Complex note with mixed patterns
  - Example: Content that could be interpreted multiple ways

## Examples

### Example 1: Single Card (Simple Q&A) ✓

**Input Note**:
```
Title: "Binary Search Complexity"
Content:
Q: What is the time complexity of binary search?
A: O(log n) because it divides the search space in half each iteration.
```

**Reasoning**:
- Single atomic concept: binary search complexity
- Simple Q&A format
- Short content
- No subquestions or examples
- Tightly focused

**Output**:
```json
{
  "should_split": false,
  "card_count": 1,
  "splitting_strategy": "none",
  "split_plan": [
    {
      "card_number": 1,
      "concept": "Binary search time complexity",
      "question": "What is the time complexity of binary search?",
      "answer_summary": "O(log n)",
      "rationale": "Single focused question about one specific property"
    }
  ],
  "reasoning": "This note contains a single atomic concept (binary search complexity) with a clear Q&A structure. No splitting needed.",
  "confidence": 0.95
}
```

### Example 2: Multiple Cards (List Items) ✗ → Split

**Input Note**:
```
Title: "REST API HTTP Methods"
Content:
Q: What are the main HTTP methods in REST APIs?
A:
- GET: Retrieve data
- POST: Create new resource
- PUT: Update entire resource
- PATCH: Update partial resource
- DELETE: Remove resource
```

**Reasoning**:
- Contains 5 independent concepts (HTTP methods)
- List pattern detected
- Each method can be learned independently
- Better retention with separate cards
- Follows "one fact per card" principle

**Output**:
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
    },
    {
      "card_number": 3,
      "concept": "POST method",
      "question": "What does POST do in REST API?",
      "answer_summary": "Creates a new resource",
      "rationale": "Individual method card"
    },
    {
      "card_number": 4,
      "concept": "PUT method",
      "question": "What does PUT do in REST API?",
      "answer_summary": "Updates entire resource",
      "rationale": "Individual method card"
    },
    {
      "card_number": 5,
      "concept": "PATCH method",
      "question": "What does PATCH do in REST API?",
      "answer_summary": "Updates partial resource (only changed fields)",
      "rationale": "Individual method card"
    },
    {
      "card_number": 6,
      "concept": "DELETE method",
      "question": "What does DELETE do in REST API?",
      "answer_summary": "Removes resource from server",
      "rationale": "Individual method card"
    }
  ],
  "reasoning": "Note lists 5 HTTP methods. Creating overview card + 5 detail cards follows SRS best practices. Each method is independent and can be reviewed separately.",
  "confidence": 0.92
}
```

### Example 3: Multiple Cards (Independent Concepts) ✗ → Split

**Input Note**:
```
Title: "Python Lists and Dictionaries"
Content:
Q: What are Python lists and dictionaries?
A:
Lists are ordered, mutable sequences that store items by index.
Example: my_list = [1, 2, 3]

Dictionaries are unordered key-value mappings.
Example: my_dict = {"name": "Alice", "age": 30}
```

**Reasoning**:
- TWO independent concepts: lists AND dictionaries
- Each deserves separate card
- Different data structures
- Violates atomic principle if combined
- Better to learn separately

**Output**:
```json
{
  "should_split": true,
  "card_count": 2,
  "splitting_strategy": "concept",
  "split_plan": [
    {
      "card_number": 1,
      "concept": "Python lists",
      "question": "What are Python lists?",
      "answer_summary": "Ordered, mutable sequences storing items by index. Example: [1, 2, 3]",
      "rationale": "First independent concept"
    },
    {
      "card_number": 2,
      "concept": "Python dictionaries",
      "question": "What are Python dictionaries?",
      "answer_summary": "Unordered key-value mappings. Example: {'name': 'Alice'}",
      "rationale": "Second independent concept"
    }
  ],
  "reasoning": "Note covers TWO distinct data structures. Each should be a separate card following atomic principle. Lists and dicts are unrelated and can be learned independently.",
  "confidence": 0.88
}
```

### Example 4: Keep Single Card (Tightly Coupled) ✓

**Input Note**:
```
Title: "Pros and Cons of Microservices"
Content:
Q: What are the pros and cons of microservices architecture?
A:
Pros:
- Independent deployment
- Technology flexibility
- Better scalability

Cons:
- Increased complexity
- Network latency
- Harder debugging
```

**Reasoning**:
- Pros and cons are comparative (must learn together)
- Context requires both sides
- Splitting would break the comparison
- Better as single comprehensive card
- Exception to "list splitting" rule

**Output**:
```json
{
  "should_split": false,
  "card_count": 1,
  "splitting_strategy": "none",
  "split_plan": [
    {
      "card_number": 1,
      "concept": "Microservices tradeoffs",
      "question": "What are the pros and cons of microservices architecture?",
      "answer_summary": "Pros: independent deploy, flexibility, scalability. Cons: complexity, latency, debugging",
      "rationale": "Comparative analysis requires both sides for context"
    }
  ],
  "reasoning": "Although this contains lists, pros/cons are tightly coupled comparative content. Splitting would lose the tradeoff context. Keep as single card.",
  "confidence": 0.85
}
```

### Example 5: Hierarchical Split ✗ → Split

**Input Note**:
```
Title: "React Component Lifecycle"
Content:
Q: What are React lifecycle methods?
A: Methods called at different stages:

Mounting:
- constructor()
- componentDidMount()

Updating:
- shouldComponentUpdate()
- componentDidUpdate()

Unmounting:
- componentWillUnmount()
```

**Reasoning**:
- Hierarchical structure: phases → methods
- Each phase has multiple methods
- Total of 5+ concepts
- Better as parent + child cards
- Allows progressive learning

**Output**:
```json
{
  "should_split": true,
  "card_count": 4,
  "splitting_strategy": "hierarchical",
  "split_plan": [
    {
      "card_number": 1,
      "concept": "Lifecycle phases overview",
      "question": "What are the 3 phases of React component lifecycle?",
      "answer_summary": "Mounting, Updating, Unmounting",
      "rationale": "High-level overview card"
    },
    {
      "card_number": 2,
      "concept": "Mounting phase methods",
      "question": "What methods are called during React component mounting?",
      "answer_summary": "constructor(), componentDidMount()",
      "rationale": "Phase-specific methods"
    },
    {
      "card_number": 3,
      "concept": "Updating phase methods",
      "question": "What methods are called during React component updating?",
      "answer_summary": "shouldComponentUpdate(), componentDidUpdate()",
      "rationale": "Phase-specific methods"
    },
    {
      "card_number": 4,
      "concept": "Unmounting phase methods",
      "question": "What methods are called during React component unmounting?",
      "answer_summary": "componentWillUnmount()",
      "rationale": "Phase-specific methods"
    }
  ],
  "reasoning": "Hierarchical content (phases → methods). Create overview card + phase-specific cards. Allows learning structure first, then details.",
  "confidence": 0.90
}
```

## Instructions

- Analyze the ENTIRE note content thoroughly
- Count distinct concepts (if 2+, likely split)
- Identify patterns: lists, steps, examples, hierarchies, difficulty levels, prerequisites
- Consider learning effectiveness (atomic principle)
- Assess concept relationships (independent vs dependent)
- Evaluate difficulty levels if applicable
- Identify prerequisite relationships (foundational concepts first)
- Consider context grouping (related concepts together vs separate)
- Default to splitting when in doubt (better retention)
- Provide clear reasoning with specific evidence
- Include specific question/answer for each planned card
- Order cards logically (by difficulty, prerequisites, or natural flow)
- Assign confidence score based on clarity of decision
- Provide fallback strategy for low-confidence cases
- Use "high" confidence (0.85+) for clear cases
- Use "medium" confidence (0.6-0.84) for ambiguous cases
- Use "low" confidence (<0.6) only when truly uncertain

## Common Patterns

✅ **DO Split**:
- Lists of N items (N ≥ 3)
- Multiple independent concepts
- Steps in a process
- Examples of a concept
- Hierarchical topics

❌ **DON'T Split**:
- Single atomic concept
- Tightly coupled comparisons (pros/cons)
- Very short Q&A
- Context-dependent information

## Red Flags for Splitting

🚩 Note title contains "and" → Likely 2+ concepts
🚩 Answer has numbered/bulleted list → Split per item
🚩 Multiple code examples → Split per example
🚩 "Steps to..." or "How to..." → Split per step
🚩 "Types of..." or "Kinds of..." → Split per type
🚩 Answer > 300 words → Likely too much for one card
"""
