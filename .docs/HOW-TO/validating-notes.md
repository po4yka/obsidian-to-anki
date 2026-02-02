# Validating Notes

This guide covers the three validation tools, vault-wide batch validation,
the validation cache, and AI-powered validation configuration. Always validate
before syncing.

## Recommended workflow

```
validate -> fix -> sync
```

Run validation first, fix any reported issues (manually or with `--fix`), then
sync to Anki. This prevents malformed cards from reaching your collection.

## Single-note validation

Validate one note and optionally auto-fix issues:

```bash
obsidian-anki-sync validate note /path/to/q-my-note.md

# Auto-fix and see details
obsidian-anki-sync validate note /path/to/q-my-note.md --fix --verbose
```

## Vault-wide validation

Validate every discoverable note in the vault:

```bash
obsidian-anki-sync validate all
```

| Flag              | Default         | Effect                                       |
|-------------------|-----------------|----------------------------------------------|
| `--fix`           | disabled        | Auto-fix detected issues                     |
| `--incremental`   | disabled        | Skip unchanged files (uses validation cache) |
| `--parallel`      | disabled        | Process notes in parallel                    |
| `--workers N`     | CPU count, max 8| Number of parallel workers                   |
| `--report path`   | none            | Write results to a report file               |
| `--verbose`       | disabled        | Show detailed output per note                |
| `--no-colors`     | disabled        | Strip ANSI colors from output                |

Typical invocation for large vaults:

```bash
obsidian-anki-sync validate all --incremental --parallel --fix --verbose
```

## Directory validation

Validate all notes inside a specific directory:

```bash
obsidian-anki-sync validate dir /path/to/notes/subtopic --fix --parallel
```

Accepts the same flags as `validate all`: `--fix`, `--incremental`,
`--parallel`, `--workers N`, `--report path`, `--verbose`.

## Bilingual lint

Check bilingual completeness and code-fence formatting:

```bash
obsidian-anki-sync lint-note /path/to/q-my-note.md
```

| Flag                    | Default              | Effect                              |
|-------------------------|----------------------|-------------------------------------|
| `--enforce-bilingual`   | enabled              | Require all declared languages      |
| `--allow-partial`       | disabled             | Allow incomplete language coverage  |
| `--check-code-fences`   | enabled              | Validate code-fence pairing         |
| `--skip-code-fences`    | disabled             | Skip code-fence checks             |

Example with relaxed settings:

```bash
obsidian-anki-sync lint-note /path/to/q-my-note.md --allow-partial --skip-code-fences
```

## Validation cache

When `--incremental` is used, the validator caches results keyed by content
hash and skips files that have not changed since the last run.

```bash
# Show cache hit/miss statistics
obsidian-anki-sync validate stats

# Clear the cache (forces full re-validation on next run)
obsidian-anki-sync validate clear-cache
```

## AI-powered validation config

The sync pipeline includes LLM-backed validation agents. Configure them in
`config.yaml`.

### Agent toggles

```yaml
# Pre-validator agent (runs before card generation)
pre_validation_enabled: true          # default

# Post-validator agent (runs after card generation)
post_validation_max_retries: 3        # default
post_validation_auto_fix: true        # default
post_validation_strict_mode: true     # default
```

### Parser behavior

```yaml
# Tolerant parser: allow partial/incomplete sections
tolerant_parsing: true                # default

# LLM-powered repair of broken note structure
parser_repair_enabled: true           # default
parser_repair_generate_content: true  # default
```

### Bilingual and correction

```yaml
# Require bilingual completeness during validation
enforce_bilingual_validation: false   # default

# AI note correction (rewrites note content)
enable_note_correction: false         # default
```

### Auto-fix

```yaml
# Write fixes back to source markdown files on disk
autofix_write_back: false             # default

# Choose which auto-fix handlers run
autofix_handlers:
  - trailing_whitespace
  - empty_references
  - title_format
  - moc_mismatch
  - section_order
  - missing_related_questions
  - broken_wikilink
  - broken_related_entry
```

Each handler targets one class of issue. Enable only the handlers relevant to
your vault to keep fixes predictable.

## Next steps

- [Writing notes for card generation](writing-notes.md)
- [Running your first sync](first-sync.md)
- [Troubleshooting common issues](troubleshooting.md)
