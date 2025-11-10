**Purpose:** Prevent field collapse and make parsing deterministic.

**Invariants**

* Field comment on its **own line only** (e.g., `<!-- Title -->`). Content begins on the **next line**.
* Place **exactly one blank line** between the **end of a fields content** and the **next field comment**.
* After `</code></pre>` or `</ul>`: newline  blank line  next field comment.
* Optional fields: print the comment + **one blank line**; no placeholders (`<text>`, `N/A`, ``) unless the field is actually used.
* Titles/Subtitles are **plain text** (no `<text>` wrappers).
* Headers must match **verbatim** (capitalization and spacing).

**Card block template**

```html
<!-- Card N | slug: <lowercase-dash-slug> | CardType: <Simple|Missing|Draw> | Tags: <tag1> <tag2> <tag3> -->

<!-- Title -->
Your title text here

<!-- Subtitle (optional) -->
Your subtitle here

<!-- Syntax (inline) (optional) -->
<code>token_or_call()</code>

<!-- Sample (caption) (optional) -->
Short caption for the sample below

<!-- Sample (code block or image) (optional for Missing) -->
<pre><code class="language-<lang>">code ‰ 12 LOC, ‰ 88 cols</code></pre>

<!-- Key point (code block / image) -->
<pre><code class="language-<lang>">answer as code OR</code></pre>
<!-- OR for Draw: <img src="data:image/svg+xml;utf8,<svg >" alt="diagram"/> -->

<!-- Key point notes -->
<ul>
  <li>36 bullets; mechanism, rule, pitfall.</li>
</ul>

<!-- Other notes (optional) -->
<ul>
  <li>Assumption:  (only if needed)</li>
  <li>Ref: <a href="https://">official docs</a></li>
</ul>

<!-- Markdown (optional) -->

<!-- manifest: {"slug":"<same-as-header>","lang":"<lang>","type":"<Simple|Missing|Draw>","tags":["tag1","tag2"]} -->
```

**Batch skeleton**

```
<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->
... one or more Card blocks ...
<!-- END_CARDS -->
END_OF_CARDS
```
