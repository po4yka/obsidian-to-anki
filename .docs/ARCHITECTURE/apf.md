# APF (Anki Prompts Format) Reference

APF is the proprietary format for generating flashcards that are optimized for Anki's spaced repetition system. This guide provides a quick reference for working with APF in the Obsidian to Anki sync service.

## Overview

**APF (Anki Prompts Format)** is a structured HTML format for creating flashcards that:

-   Follows evidence-based learning principles (SuperMemo 20 Rules, FSRS)
-   Ensures atomic recall (one concept per card)
-   Provides consistent formatting for optimal Anki performance
-   Supports multiple card types (Simple, Missing/Cloze, Draw)

## Card Types

### Simple Cards

**Purpose:** Question-answer format for single-concept recall

**Use for:** Definitions, API facts, contrasts, code analysis

```html
<!-- Card 1 | slug: kotlin-null-safety | CardType: Simple | Tags: kotlin types null_safety -->
<!-- Title -->
What are Kotlin's null safety operators?

<!-- Key point (code block) -->
<pre><code class="language-kotlin">?.  // Safe call
?:  // Elvis operator
!!  // Not-null assertion</code></pre>

<!-- Key point notes -->
<ul>
    <li>Safe call (?.) returns null instead of throwing NPE</li>
    <li>Elvis (?:) provides default value for null expressions</li>
    <li>Not-null assertion (!!) throws NPE if value is null</li>
</ul>
```

### Missing (Cloze) Cards

**Purpose:** Fill-in-the-blank format for memorizing exact syntax

**Use for:** Keywords, operators, exact code patterns

```html
<!-- Card 2 | slug: kotlin-variables | CardType: Missing | Tags: kotlin basics variables -->
<!-- Title -->
Complete the Kotlin variable declaration syntax.

<!-- Key point (code block with cloze) -->
<pre><code class="language-kotlin">val {{c1::name}}: {{c2::String}} = "{{c3::value}}"</code></pre>

<!-- Key point notes -->
<ul>
    <li>val declares read-only variables</li>
    <li>Type annotation follows colon</li>
    <li>String literals use double quotes</li>
</ul>
```

### Draw Cards

**Purpose:** Diagram or sequence recall

**Use for:** System architecture, execution flow, state transitions

```html
<!-- Card 3 | slug: http-request-flow | CardType: Draw | Tags: networking http architecture -->
<!-- Title -->
Sketch the HTTP request-response flow.

<!-- Key point (image) -->
<img
    src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='400' height='200'>
  <text x='50' y='50'>Client → Server → Response</text>
  <text x='50' y='100'>Request → Process → Send</text>
</svg>"
    alt="HTTP flow diagram"
/>
```

## Card Structure

### Required Elements

Every APF card must include:

-   **Header:** Card number, slug, type, and tags
-   **Title:** Question or prompt (≤80 characters)
-   **Key point:** The answer (code block, image, or bullets)
-   **Key point notes:** 3-6 bullets explaining why/how/when
-   **Manifest:** JSON metadata for processing

### Optional Elements

-   **Subtitle:** Context or category
-   **Syntax:** Key signature or token
-   **Sample:** Supporting code or example
-   **Other notes:** References and assumptions
-   **Markdown:** Original source if needed

## Output Format

### Batch Structure

```html
<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: example-card | CardType: Simple | Tags: kotlin basics -->
...card content...
<!-- Card 2 | slug: another-card | CardType: Missing | Tags: kotlin advanced -->
...card content...
<!-- END_CARDS -->
END_OF_CARDS
```

### Field Formatting Rules

-   **One field comment per line:** `<!-- Title -->`
-   **Content starts next line:** No content on comment line
-   **Blank line separators:** Exactly one blank line between fields
-   **Plain text for titles:** No HTML wrappers for Title/Subtitle
-   **Proper HTML structure:** Valid tags and nesting

## Quality Standards

### Atomicity Principle

-   **One card = One recall**
-   Split multi-concept content into separate cards
-   Avoid "and" in questions (signals multiple facts)

### Active Recall Focus

-   Force retrieval from memory, not recognition
-   Avoid yes/no questions
-   Prefer "What/Why/How" over "Is/Does/Can"

### Technical Requirements

-   **Line length:** ≤88 characters
-   **Code blocks:** Always specify language class
-   **Cloze numbering:** Dense 1,2,3... (no gaps)
-   **No placeholders:** Complete, runnable code
-   **Unique slugs:** Lowercase, dash-separated

## Tags and Taxonomy

### Tag Structure

Tags follow: `language/tool → platform/runtime → domain → subtopic`

**Examples:**

-   `kotlin android coroutines flow`
-   `python linux concurrency asyncio`
-   `javascript browser react hooks`

### Common Language Tags

`kotlin`, `java`, `python`, `javascript`, `typescript`, `swift`, `cpp`, `rust`, `go`

### Common Domain Tags

`coroutines`, `concurrency`, `flow`, `channels`, `ui`, `networking`, `security`, `testing`, `performance`, `algorithms`

## Manifest Format

Each card includes a JSON manifest for processing:

```json
{
    "slug": "kotlin-null-safety",
    "lang": "kotlin",
    "type": "Simple",
    "tags": ["kotlin", "types", "null_safety"],
    "version": "apf-v2.1"
}
```

## Implementation Details

### Card Generation Process

1. **Content Analysis:** Determine optimal card type
2. **Template Selection:** Choose appropriate APF template
3. **HTML Generation:** Create clean, valid HTML
4. **Manifest Creation:** Generate metadata JSON
5. **Validation:** Run APF linter and fix issues

### Language Support

-   **Code blocks:** `language-kotlin`, `language-python`, `language-java`, etc.
-   **Bilingual content:** Separate cards for each language
-   **Consistent slugs:** Same concept, different languages get related slugs

## Validation and Linting

### Automated Checks

-   **Syntax validation:** Proper HTML structure
-   **Content validation:** Atomic concepts, complete answers
-   **Format compliance:** APF specification adherence
-   **Tag consistency:** Taxonomy compliance

### Manual Review Guidelines

-   **Semantic accuracy:** Answer matches question
-   **Learning effectiveness:** Forces appropriate recall
-   **Clarity:** Unambiguous question and answer
-   **Completeness:** All necessary context provided

## Integration with Agents

### Agent Workflow

1. **Pre-validation:** Content quality check
2. **Generation:** Convert Q&A to APF format
3. **Post-validation:** Quality assurance
4. **Linter:** Format and consistency validation

### Configuration

```yaml
apf_config:
    version: "apf-v2.1"
    enable_validation: true
    strict_mode: false
    max_line_length: 88
```

## Common Patterns

### Code Examples

```html
<!-- Sample (code block) -->
<pre><code class="language-kotlin">fun greet(name: String): String {
    return "Hello, $name!"
}</code></pre>
```

### Cloze Deletions

```html
<!-- Key point (code block with cloze) -->
<pre><code class="language-kotlin">fun {{c1::fibonacci}}(n: Int): Int {
    return if (n <= 1) {{c2::n}} else {{c3::fibonacci}}(n - 1) + fibonacci(n - 2)
}</code></pre>
```

### Image References

```html
<!-- Key point (image) -->
<img src="data:image/svg+xml;utf8,<svg>...</svg>" alt="Architecture diagram" />
```

## Related Documentation

-   **[APF Cards Directory](../REFERENCE/apf/)** - Complete APF specification
    -   [Card Block Template](REFERENCE/apf/Doc%20A%20—%20Card%20Block%20Template%20&%20Formatting%20Invariants.md)
    -   [Tag Taxonomy](REFERENCE/apf/Doc%20B%20—%20Tag%20Taxonomy.md)
    -   [Examples](REFERENCE/apf/Doc%20C%20—%20Examples.md)
    -   [Linter Rules](<REFERENCE/apf/Doc%20D%20—%20Linter%20Rules%20(Regex%20&%20Policies).md>)
-   **[Cards Prompt](../REFERENCE/cards-prompt.md)** - Detailed prompt specifications for AI card generation
-   **[Agent System](agents.md)** - How agents generate APF cards

---

**Version**: 2.1
**Last Updated**: November 28, 2025
