# Troubleshooting

Common problems, causes, and fixes. See also: [first-sync.md](first-sync.md),
[validating-notes.md](validating-notes.md), [performance-tuning.md](performance-tuning.md).

## Quick diagnostics

```bash
# Pre-flight check (vault, DB, providers, Anki, disk, memory, latency)
obsidian-anki-sync check
obsidian-anki-sync check --skip-anki   # test LLM only
obsidian-anki-sync check --skip-llm    # test Anki only

# Analyze log files (error counts by level, type, stage, file)
obsidian-anki-sync analyze-logs --days 7

# List notes that failed processing
# Categories: parser_errors, validation_errors, llm_errors,
#             generation_errors, other_errors
obsidian-anki-sync list-problematic-notes --limit 20
obsidian-anki-sync list-problematic-notes --error-type timeout --category llm_errors
obsidian-anki-sync list-problematic-notes --date 2025-01-15
```

## AnkiConnect issues

AnkiConnect (add-on `2055492159`) exposes a local API at `http://127.0.0.1:8765`.

**Connection refused** -- Start Anki, install the AnkiConnect add-on
(Tools > Add-ons), then restart Anki. Test manually:

```bash
curl http://localhost:8765 -X POST -d '{"action":"version","version":6}'
# Expected: {"result":6,"error":null}
```

**Port conflict** -- Check what owns port 8765:

```bash
lsof -i :8765
```

**CORS errors** -- In Anki, open Tools > Add-ons > AnkiConnect > Config and
verify `webCorsOriginList` includes `"*"` or your specific origin.

## LLM provider issues

```bash
obsidian-anki-sync check --skip-anki   # quick LLM-only diagnostic
```

**Ollama** (default: `http://localhost:11434`):

```bash
curl http://localhost:11434/api/tags    # verify running
ollama list                             # check available models
```

If the configured model is missing, run `ollama pull <model>`.

**OpenRouter** (default: `https://openrouter.ai/api/v1`):
Requires an API key. Model format: `provider/model-name`.

```bash
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" | head -c 200
```

**LM Studio** (default: `http://localhost:1234/v1`):
The local server must be running with a model loaded. Check the LM Studio UI.

## Note parsing errors

**Missing frontmatter** -- Required fields: `id`, `title`, `topic`,
`language_tags`, `created`, `updated`.

```yaml
---
id: "abc-123"
title: "My Note"
topic: "math"
language_tags: ["en"]
created: 2025-01-01
updated: 2025-01-01
---
```

**Invalid YAML** -- Common mistakes: backticks in YAML values (wrap in
quotes), unquoted special chars (`:`, `#`, `{`, `}`), tabs instead of spaces.

**Wrong section headers** -- The parser expects `# Question (EN)` (h1) and
`## Answer (EN)` (h2), not the reverse. A `---` separator is required between
the question and answer sections.

**Validate a single note**:

```bash
obsidian-anki-sync validate note path/to/q-my-note.md
obsidian-anki-sync lint-note path/to/q-my-note.md
```

**Tolerant parsing** -- If many notes have minor format issues, add to
`config.yaml`:

```yaml
tolerant_parsing: true
parser_repair_enabled: true
```

Parser repair uses an LLM agent to fix structural problems automatically.

## Sync failures

**No notes discovered**:
- Notes must match the `q-*.md` naming pattern.
- Check `source_dir` in your config.
- Directories prefixed with `c-`, `moc-`, or `template` are excluded.

**Stale progress state**:

```bash
obsidian-anki-sync clean-progress --all-completed
obsidian-anki-sync clean-progress --session <session-id>
```

**Card creation failed**:
1. Confirm Anki is running (see AnkiConnect section above).
2. Verify the note type exists: `obsidian-anki-sync models`
3. Verify the deck exists: `obsidian-anki-sync decks`

**Resume an interrupted sync**:

```bash
obsidian-anki-sync progress                        # list sessions
obsidian-anki-sync sync --resume <session-id>      # resume
```

## Performance issues

See [performance-tuning.md](performance-tuning.md) for detailed advice.

**Slow sync** -- LLM inference dominates. Use the `fast` model preset, lower
`max_concurrent_generations` (default 5, range 1-500), or test with
`obsidian-anki-sync sync --sample 10`.

**High memory** -- Reduce `batch_size` (default 50). Use `--sample N` for
testing.

**File descriptor exhaustion** -- Tune `archiver_batch_size` (default 64)
and `archiver_min_fd_headroom` (default 32) in your config.

## Log analysis

Logs are written to `project_log_dir` (default: `logs/` relative to `data_dir`).

```bash
obsidian-anki-sync analyze-logs --days 7       # summarize recent errors
obsidian-anki-sync sync --verbose              # all messages to terminal
obsidian-anki-sync sync --log-level DEBUG      # maximum detail
```

The `-v` / `--verbose` flag prints all log messages to the terminal in
addition to writing them to the log file.

## Getting help

Open an issue at <https://github.com/po4yka/obsidian-to-anki/issues>.
Include in your report:

1. **Config** -- your `config.yaml` with API keys redacted.
2. **Log output** -- relevant section from `analyze-logs` or `--verbose`.
3. **Note example** -- a sample `q-*.md` file that triggers the problem.
4. **Version** -- output of `obsidian-anki-sync --version`.
