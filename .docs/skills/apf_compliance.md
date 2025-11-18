# APF Compliance Skill

**Purpose**: Ensure all generated cards strictly follow APF (Anki Package Format) v2.1 specifications.

**When to Use**: Always loaded for card generation tasks. This is the foundation skill for all card creation.

---

## Core Principles

### Format Strictness
- Cards MUST follow the exact APF v2.1 template structure
- No deviations from specified delimiters and section markers
- All required fields must be present
- Optional sections should be omitted if unused, but section comment lines must remain

### Output Structure (Non-Negotiable)

Every card batch MUST follow this exact sequence:

```
<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->
[card blocks]
<!-- END_CARDS -->
END_OF_CARDS
```

### Card Header Format

**CRITICAL**: Card headers MUST use this exact format with spaces around pipes:

```html
<!-- Card N | slug: <lowercase-dash-slug> | CardType: <Simple|Missing|Draw> | Tags: <tag1> <tag2> <tag3> -->
```

**Requirements**:
- Space before and after each `|` pipe character
- `CardType:` must have capital C and T
- Tags are space-separated (NOT comma-separated)
- Slug must be lowercase with only letters, numbers, and hyphens
- Card number N must match order (1, 2, 3, ...)

---

## Field Rules

### Title
- Maximum ~80 characters
- Must ask ONE question or state ONE prompt
- No duplicates across a batch
- Use authoritative vocabulary from official documentation

### Subtitle (Optional)
- Short context label (e.g., "Coroutines / Flow")
- Use sparingly for disambiguation

### Syntax (Optional)
- One-liner `<code>` spotlighting key token/signature
- Use for highlighting specific syntax elements

### Sample (Optional)
- Only what's necessary to understand the question
- Code: ~12 lines of code, ~88 columns max
- No unused imports
- Keep minimal yet runnable/valid

### Key Point (Required)
- This is the **answer** to the question
- Can be code block, image, or tight bullet list
- For Missing type: place `{{cN::...}}` clozes here
- NEVER cloze comments or whitespace

### Key Point Notes (Required)
- 3-6 bullets, each ~20 words
- Focus on: mechanism, rule, pitfall, edge case
- Avoid anecdotes
- Use structured formatting

### Other Notes (Optional)
- Maximum 2 links to official docs
- One `Assumption:` bullet if ambiguity was resolved
- Keep minimal

### Markdown (Optional)
- Include only if user supplied original markdown
- Preserve original formatting when relevant

---

## Code & Language Rules

- Always set `<pre><code class="language-XYZ">` with proper language tag
- Examples: `language-kotlin`, `language-python`, `language-yaml`
- Keep examples minimal yet runnable/valid
- No extraneous scaffolding
- Linewrap long strings
- Prefer small, focused snippets over ellipses
- For YAML/JSON: use spaces, consistent indentation, valid keys

---

## Tagging Rules

### Tag Structure
- 3-6 tags per card, **snake_case**
- Order: `language/tool -> platform/runtime -> domain -> subtopic`
- Include at least one non-language tag per card

### Closed Vocabulary Seed

**Languages/Tools**: `kotlin`, `java`, `python`, `javascript`, `typescript`, `swift`, `objective_c`, `c`, `cpp`, `rust`, `go`, `dart`, `ruby`, `php`, `csharp`, `sql`, `yaml`, `json`, `bash`, `powershell`, `docker`, `kubernetes`, `terraform`, `ansible`, `gradle`, `maven`, `git`, `regex`

**Platforms/Runtimes**: `android`, `ios`, `kmp`, `jvm`, `nodejs`, `browser`, `linux`, `macos`, `windows`, `serverless`

**Domains**: `coroutines`, `concurrency`, `flow`, `channels`, `lifecycle`, `viewmodel`, `compose`, `ui`, `retrofit`, `okhttp`, `room`, `persistence`, `sql_lite`, `networking`, `http`, `grpc`, `websocket`, `security`, `cryptography`, `jwt`, `oauth2`, `testing`, `unit_testing`, `integration_testing`, `ui_testing`, `property_based_testing`, `logging`, `monitoring`, `observability`, `tracing`, `metrics`, `performance`, `memory`, `gc`, `profiling`, `algorithms`, `data_structures`, `functional_programming`, `oop`, `design_patterns`, `build`, `packaging`

**CI/CD**: `ci_cd`, `github_actions`, `gitlab_ci`, `fastlane`, `jenkins`

### Examples
- `kotlin android coroutines flow`
- `python linux concurrency asyncio`
- `ci_cd github_actions gradle_cache`

---

## Card Type Selection

### Simple Type
**Use for**: Question-answer format for single-concept recall
- Definitions and terminology
- API facts and method signatures
- Contrasts between concepts
- "Predict the output" code analysis
- Small, atomic programming facts

**Format**: Direct question → Direct answer

### Missing Type
**Use for**: Cloze deletion for memorizing exact tokens, flags, or operators
- Memorizing exact syntax (keywords, operators, flags)
- Code spans where specific tokens must be recalled
- Template patterns with fill-in-the-blank

**Rules**:
- Use `{{cN::...}}` notation (c1, c2, c3, etc.)
- Maximum 1-3 clozes per card
- ONLY if clozes are independent
- If clozes are dependent → split into multiple cards
- NEVER cloze comments or whitespace

### Draw Type
**Use for**: Diagram or sequence recall
- System architecture diagrams
- State machine transitions
- Execution flow sequences
- Component relationships

**Format**:
- Answer can be ordered list OR embedded image (SVG/PNG)
- Keep to 5-9 labeled elements for cognitive load management

---

## Quality Gates (Apply Silently)

### Step 1: Atomicity Check
- Does the card test exactly ONE recall?
- If testing multiple concepts → SPLIT into multiple cards
- One question, one answer, one concept

### Step 2: Answerability Check
- Is the question solvable from provided Sample/context?
- Would a learner have sufficient information to answer?
- Is the answer unambiguous?

### Step 3: Specificity Check
- Do nouns/verbs match canonical API/spec names?
- Is terminology from official documentation used?
- No vague or informal language?

### Step 4: FSRS-Friendly Phrasing
- No ambiguity in the question
- Avoid multi-barrel questions (asking two things at once)
- Clear, single-focus recall target

### Step 5: Accessibility Check
- No lines exceeding 88 columns
- Inline tokens use `<code>` tags
- Code is properly formatted and indented

### Step 6: Completeness Check
- NO placeholders (no `...`, `// fill in`, `TBD`)
- NO ellipses indicating omitted code
- All code snippets are minimal but complete

### Step 7: Validation for Missing Type
- Every `{{cN::...}}` contains at least one non-whitespace token
- Cloze numbering is dense (1, 2, 3, ... no gaps like 1, 3, 5)
- NEVER cloze comments or whitespace
- Maximum 1-3 clozes per card

---

## Common Anti-Patterns to Avoid

❌ **Placeholder Code**: Using `...`, `// fill in`, `TBD` → Provide complete, runnable code
❌ **Format Deviations**: Missing delimiters or wrong structure → Follow exact template
❌ **Tag Violations**: Using non-snake_case or orphan tags → Follow taxonomy
❌ **Incomplete Cards**: Missing required fields → Include all mandatory sections
❌ **Overly Long Code**: Exceeding 12 LOC or 88 columns → Keep minimal and focused

---

## Best Practices to Follow

✓ **Strict Format Compliance**: Follow APF v2.1 template exactly
✓ **Complete Code**: All snippets are runnable/valid, no placeholders
✓ **Proper Tagging**: Use closed vocabulary, correct ordering
✓ **Clear Structure**: Proper HTML formatting, readable layout
✓ **Authoritative Language**: Use official documentation terminology
✓ **Self-Contained**: Each card is independently reviewable

