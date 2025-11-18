# Memorization Principles Skill

**Purpose**: Ensure generated flashcards follow evidence-based spaced repetition principles for long-term memory retention.

**When to Use**: Load for all card generation and quality validation tasks. This skill ensures cards are effective for learning.

---

## Core Principle: Atomicity

### One Concept Per Card

**What**: Each card tests exactly ONE specific fact or concept.

**Why**: Multiple concepts create interference and make review ambiguous. Learners struggle when cards test multiple unrelated facts simultaneously.

**How to Apply**:
- If a note covers multiple independent concepts → split into separate cards
- If concepts are tightly coupled and must be learned together → single card is acceptable
- When in doubt → split rather than combine

**Examples**:

✅ **Good**: "What is the time complexity of binary search?" → "O(log n)"
- Tests ONE concept: binary search complexity
- Clear, focused recall target

❌ **Bad**: "What are the time complexities of bubble sort, merge sort, and quicksort?"
- Tests THREE different algorithms
- Too much information to recall correctly
- Should be split into 3 separate cards

---

## Clear Question-Answer Relationship

### Unambiguous Questions

**What**: Question unambiguously leads to one correct answer.

**Why**: Ambiguity causes frustration and poor retention. Learners waste time trying to guess what answer is expected.

**How to Apply**:
- Avoid pronouns without clear referents ("it", "this", "that")
- Specify context explicitly
- Use precise terminology
- Make questions specific enough to have one correct answer

**Examples**:

✅ **Good**: "What does `fetch()` return in JavaScript?" → "A Promise that resolves to a Response"
- Clear context: JavaScript's fetch function
- Specific answer: Promise type with resolution value

❌ **Bad**: "What does it return?" → "A promise"
- Missing context: what is "it"?
- Vague answer: what kind of promise?

---

## Active Recall Trigger

### Require Memory Retrieval

**What**: Question requires retrieving information from memory, not just recognition.

**Why**: Active recall strengthens memory more than passive recognition. The effort of retrieval creates stronger neural pathways.

**How to Apply**:
- Use "What/How/Why" questions, not "Is/Does" yes/no questions
- Require generating the answer, not selecting from options
- Avoid hints that give away the answer
- Make questions require understanding, not just memorization

**Examples**:

✅ **Good**: "What Git command shows commit history?" → "git log"
- Requires retrieving command name from memory
- Active recall of specific knowledge

❌ **Bad**: "Is `git log` the command for history? (Yes/No)"
- Tests recognition, not recall
- Too easy, minimal cognitive effort
- Rephrase as "What" question

---

## Context Sufficiency

### Self-Contained Cards

**What**: Card provides enough context to answer without external references.

**Why**: Cards should be independently reviewable. Learners shouldn't need to remember previous cards or look up external information.

**How to Apply**:
- Include necessary context in the question or Sample section
- Avoid references to "previous card" or "earlier example"
- Provide enough background for standalone understanding
- Use Extra section for helpful but non-essential context

**Examples**:

✅ **Good**:
```
Front: What is the time complexity of binary search?
Back: O(log n)
Extra:
- Requires sorted array
- Divides search space in half each iteration
- Example: Finding 7 in [1,3,5,7,9,11,13] takes 3 comparisons
```
- Self-contained: includes context in Extra
- Can be reviewed independently

❌ **Bad**:
```
Front: What does it return?
Back: A promise
```
- Missing context: what function?
- Requires external knowledge
- Cannot be reviewed independently

---

## Appropriate Difficulty

### Challenging but Answerable

**What**: Card is challenging but answerable with study.

**Why**: Optimal difficulty maintains engagement and learning. Too easy = boring, too hard = demotivating.

**How to Apply**:
- Avoid trivial facts that require no thought
- Avoid impossibly difficult questions requiring deep expertise
- Target knowledge that requires study but is achievable
- Consider the target audience's level

**Examples**:

✅ **Good**: "What is the difference between `==` and `===` in JavaScript?"
- Requires understanding of type coercion
- Challenging but answerable with study
- Practical knowledge

❌ **Bad**: "What programming language uses curly braces?"
- Too broad: multiple correct answers
- Trivial knowledge
- Low practical value

---

## No Information Leakage

### Front Doesn't Reveal Answer

**What**: Front doesn't hint at or reveal the answer.

**Why**: Leakage bypasses active recall and weakens learning. If the question gives away the answer, no memory work is required.

**How to Apply**:
- Remove visual hints (like `[]` hinting at lists)
- Avoid including parts of the answer in the question
- Rephrase questions that contain answer elements
- Test understanding, not pattern matching

**Examples**:

✅ **Good**: "What Python feature allows creating lists in a single line with a for loop?" → "List comprehension"
- No hints in the question
- Requires understanding the concept

❌ **Bad**: "In Python, what syntax uses square brackets [] to create a sequence?" → "List comprehension"
- `[]` heavily hints at lists
- Reduces active recall challenge
- More recognition than recall

---

## Memorable Formatting

### Structure Aids Memory

**What**: Uses examples, mnemonics, or visual structure to aid encoding.

**Why**: Structured information is easier to encode and retrieve. Visual organization helps memory formation.

**How to Apply**:
- Use bullet points for lists
- Include concrete examples
- Add mnemonics when helpful
- Use formatting (bold, code blocks) for emphasis
- Break up walls of text

**Examples**:

✅ **Good**:
```
Front: What are React hooks rules?
Back: Two key rules:
- Only call at top level (not in loops/conditions)
- Only call from React functions (components/custom hooks)

Extra:
Mnemonic: "Top Level Only"
Example violation:
if (condition) {
  const [state] = useState(0); // WRONG
}
```
- Structured with bullets
- Includes mnemonic
- Shows example violation

❌ **Bad**:
```
Front: What are React hooks rules?
Back: Only call at top level, only call from React functions.
```
- Wall of text
- No structure or examples
- Harder to remember

---

## Practical Applicability

### Real-World Relevance

**What**: Tests knowledge useful in real scenarios.

**Why**: Practical knowledge is more motivating and retained longer. Learners value information they can actually use.

**How to Apply**:
- Focus on knowledge used in professional contexts
- Avoid purely academic trivia
- Emphasize actionable information
- Connect to real-world use cases

**Examples**:

✅ **Good**: "What Git command shows commit history?" → "git log"
- Commonly used in development
- Practical daily knowledge
- Real-world application

❌ **Bad**: "What year was Python created?" → "1991"
- Trivial fact
- Limited practical value
- Purely academic detail

---

## Evaluation Checklist

When evaluating cards, check:

1. ✓ **Atomic**: Tests exactly one concept
2. ✓ **Clear Q-A**: Unambiguous question leading to one answer
3. ✓ **Active Recall**: Requires memory retrieval, not recognition
4. ✓ **Self-Contained**: Enough context to answer independently
5. ✓ **Appropriate Difficulty**: Challenging but achievable
6. ✓ **No Leakage**: Front doesn't hint at answer
7. ✓ **Memorable Format**: Uses structure, examples, mnemonics
8. ✓ **Practical**: Knowledge useful in real scenarios

---

## Common Anti-Patterns to Flag

❌ **List Cards**: "List 5 principles of..." → Split into 5 cards
❌ **Yes/No Questions**: Too easy, no active recall → Rephrase as "What/How/Why"
❌ **Definition Vomiting**: Copying textbook definitions → Use simpler, memorable language
❌ **Orphan Cards**: Cards referencing "previous card" → Make self-contained
❌ **Trivia Cards**: Testing irrelevant details → Focus on practical knowledge
❌ **Hint Cards**: Front gives away answer → Remove hints
❌ **Wall of Text**: No structure or examples → Add formatting and examples

---

## Best Practices to Reward

✓ **One Fact per Card**: Atomic principle
✓ **Concrete Examples**: Aids encoding
✓ **Mnemonic Devices**: Memory aids
✓ **Visual Structure**: Lists, formatting
✓ **Practical Context**: Real-world usage
✓ **Self-Contained**: No dependencies
✓ **Progressive Difficulty**: Builds knowledge
✓ **Clear Triggers**: Unambiguous questions

