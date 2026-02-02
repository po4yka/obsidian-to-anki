# First Sync: Getting Notes into Anki

This guide walks through your first sync from Obsidian to Anki, from
pre-flight checks through a full production run.

## Prerequisites

Before starting, make sure you have:

- **config.yaml** configured for your vault and LLM provider
- **Anki** running with **AnkiConnect** installed (add-on `2055492159`)
- **LLM provider** running (Ollama, LM Studio, or OpenRouter)
- At least one **`q-*.md` note** in your vault's source directory

## Step 1: Pre-flight check

Run the built-in diagnostic to verify your environment:

```bash
obsidian-anki-sync check
```

This validates:

| Check             | What it verifies                                         |
|-------------------|----------------------------------------------------------|
| vault_path        | Configured vault directory exists and is readable        |
| source_dir        | Source directory for notes exists inside the vault       |
| db_path           | SQLite state database path is writable                   |
| LLM provider      | Configured LLM provider is reachable and responds        |
| AnkiConnect       | Anki is running and AnkiConnect API is accessible        |
| note_type         | Configured Anki note type exists                         |
| deck_name         | Configured Anki deck exists                              |
| git repo          | Vault is a git repository (for change tracking)          |
| Disk space        | < 0.5 GB free = blocking, < 1 GB free = warning         |
| Memory            | < 4 GB available = blocking                              |
| Network latency   | > 250 ms to LLM provider = blocking                     |

Each check reports one of four severity levels:

- **error** -- always stops execution
- **blocking_warning** -- stops execution in strict mode
- **warning** -- reported but does not stop execution
- **info** -- informational only

Every check prints **PASS**, **WARN**, or **FAIL** with a fix suggestion
when something is wrong.

To skip specific checks when a service is temporarily unavailable:

```bash
# Skip AnkiConnect check (e.g., Anki not installed on CI)
obsidian-anki-sync check --skip-anki

# Skip LLM provider check
obsidian-anki-sync check --skip-llm

# Skip both
obsidian-anki-sync check --skip-anki --skip-llm
```

Fix every **error** and **blocking_warning** before proceeding.

## Step 2: Dry run

Preview what the sync engine would do without touching Anki:

```bash
obsidian-anki-sync sync --dry-run
```

To use a non-default config file:

```bash
obsidian-anki-sync sync --dry-run --config path/to/config.yaml
```

The command displays a live Rich panel showing:

- **Current phase** (indexing, generating, syncing)
- **Session ID** for this run
- **Processed / total items** counter
- **Card deltas** (cards to add, update, delete)
- **Elapsed time**

No changes are applied to Anki during a dry run. Review the card deltas
to confirm the engine found the notes you expect.

## Step 3: Test run

Process a small batch to verify end-to-end card generation:

```bash
# Preview 3 random notes (dry-run mode, the default)
obsidian-anki-sync test-run --count 3

# Actually create cards from 3 random notes
obsidian-anki-sync test-run --count 3 --no-dry-run
```

Options:

| Flag              | Default       | Effect                                    |
|-------------------|---------------|-------------------------------------------|
| `--count N`       | 3             | Number of random notes to process         |
| `--dry-run`       | enabled       | Preview only, no changes to Anki          |
| `--no-dry-run`    | --            | Create cards in Anki for real             |
| `--index`         | disabled      | Re-index the vault before processing      |
| `--no-index`      | enabled       | Skip vault indexing                       |

After running with `--no-dry-run`, open Anki and verify the created cards
look correct before moving on.

## Step 4: Full sync

Run the complete sync:

```bash
obsidian-anki-sync sync
```

For subsequent runs after the initial sync, use incremental mode to process
only new or changed notes (determined by content hash):

```bash
obsidian-anki-sync sync --incremental
```

### Useful flags

```bash
# Limit to N random notes (good for gradual rollout)
obsidian-anki-sync sync --sample 20

# Verbose output
obsidian-anki-sync sync --verbose
# or
obsidian-anki-sync sync -v

# Set log level explicitly
obsidian-anki-sync sync --log-level DEBUG

# Redis-based parallel processing
obsidian-anki-sync sync --use-queue --redis-url redis://localhost:6379
```

### Full reference

```
obsidian-anki-sync sync [OPTIONS]

  --dry-run              Preview changes without applying
  --incremental          Only process new/changed notes (content hash)
  --sample N             Limit to N random notes
  --config PATH          Path to config.yaml
  --use-queue            Enable Redis-based parallel processing
  --redis-url URL        Redis connection URL (requires --use-queue)
  --verbose / -v         Detailed logging output
  --log-level LEVEL      DEBUG | INFO | WARN | ERROR
  --resume SESSION_ID    Resume an interrupted sync session
  --no-resume            Disable automatic resume of incomplete syncs
```

## Understanding the output

During any sync or dry-run, a live Rich panel displays the current phase,
session ID, progress counter, card deltas, and elapsed time. Use
`obsidian-anki-sync progress` for historical session data.

## Resuming interrupted syncs

```bash
obsidian-anki-sync progress                        # list sessions with IDs
obsidian-anki-sync sync --resume <session-id>      # resume a specific session
obsidian-anki-sync sync --no-resume                # disable automatic resume
```

### Cleaning up progress records

```bash
obsidian-anki-sync clean-progress --session <id>   # remove one session
obsidian-anki-sync clean-progress --all-completed  # remove all completed
```

## Next steps

- [Writing notes for card generation](writing-notes.md)
- [Troubleshooting common issues](troubleshooting.md)
- [Backup and restore](backup-restore.md)
