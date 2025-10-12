**When:** Only if the user prompt contains `mode=json`. Otherwise use HTML blocks.

**JSON array item fields:**

```
{
  "slug": "lowercase-dash-slug",
  "type": "Simple|Missing|Draw",
  "title": "plain text",
  "subtitle": null|"text",
  "syntax_inline": null|"token()",
  "sample_caption": null|"text",
  "sample_code_or_image": null|"...code or data:image/svg+xml,...",
  "sample_lang": null|"kotlin|yaml|...",
  "keypoint_code_or_image": "code or data:image/svg+xml,...",
  "keypoint_lang": null|"kotlin|yaml|svg|...",
  "keypoint_notes": ["…","…"],
  "other_notes": ["Assumption: …","Ref: https://…"],
  "tags": ["kotlin","android","coroutines","flow"]
}
```

**Manifest (HTML mode):** include at the end of each card:

```
<!-- manifest: {"slug":"<same-as-header>","lang":"<lang>","type":"<Simple|Missing|Draw>","tags":["t1","t2"]} -->
```
