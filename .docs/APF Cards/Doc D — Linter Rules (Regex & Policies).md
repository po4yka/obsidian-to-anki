**Required sentinels (one each):**

* `^<!-- PROMPT_VERSION: apf-v2\\.1 -->$`
* `^<!-- BEGIN_CARDS -->$`
* `^<!-- END_CARDS -->$`
* Final line equals `END_OF_CARDS`

**Card header:**

* `^<!-- Card \\d+ \\| slug: [a-z0-9-]+ \\| CardType: (Simple|Missing|Draw) \\| Tags: ([a-z0-9_]+(?:\\s+[a-z0-9_]+){2,5}) -->$`

**Field headers (exact order) present once per card:**

```
<!-- Title -->
<!-- Subtitle (optional) -->
<!-- Syntax (inline) (optional) -->
<!-- Sample (caption) (optional) -->
<!-- Sample (code block or image) (optional for Missing) -->
<!-- Key point (code block / image) -->
<!-- Key point notes -->
<!-- Other notes (optional) -->
<!-- Markdown (optional) -->
```

**Header line discipline:**

* Reject if any non-whitespace appears on the same line as a field header.
* Enforce one blank line between a field’s last content line and the next header.

**Title/Subtitle checks:**

* Reject if lines immediately after `Title`/`Subtitle` contain `<text>` tags.

**Cloze density (per card):**

* If any `{{c\\d+::` appears, the set of indices must be contiguous from 1..N.

**Tag order:**

* First token must be a language/tool from Doc B. At least one non-language tag must be present.

**Width (advisory):**

* Warn if any line exceeds 88 characters (excluding the SVG data URI).
