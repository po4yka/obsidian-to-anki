# Incremental Sync Guide

This guide covers the incremental sync mode that allows you to only process new notes, skipping notes that have already been synced.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Use Cases](#use-cases)
- [Performance](#performance)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

Incremental sync mode enables:

- **Faster syncs**: Only process new notes, skip existing ones
- **Efficient workflows**: Perfect for daily additions to large vaults
- **Smart filtering**: Based on database tracking
- **Combined modes**: Works with progress tracking and resume

## Quick Start

### Sync only new notes

```bash
obsidian-anki-sync sync --incremental
```

Output:
```
Incremental mode: Skipping 150 already processed notes

Scanning: Processing 5 new notes
✓ Processed 5 notes
  Created: 15 cards
  Errors: 0
```

### Combine with other options

```bash
# Incremental + Dry run
obsidian-anki-sync sync --incremental --dry-run

# Incremental + Agents
obsidian-anki-sync sync --incremental --use-agents

# Incremental + Resume
obsidian-anki-sync sync --incremental --resume <session-id>
```

## How It Works

### 1. Database Tracking

The system maintains a list of processed notes in the `cards` table:

```sql
SELECT DISTINCT source_path FROM cards
```

This query returns all note paths that have been synced.

### 2. Filtering Logic

During sync initialization:

```python
# Get all notes in vault
note_files = discover_notes(vault_path)  # e.g., 200 notes

# If incremental mode
if incremental:
    processed_paths = db.get_processed_note_paths()  # e.g., 150 notes

    # Filter for new notes only
    note_files = [
        (file_path, rel_path)
        for file_path, rel_path in note_files
        if rel_path not in processed_paths
    ]
    # Result: 50 new notes

logger.info(
    "incremental_mode",
    total_notes=200,
    new_notes=50,
    filtered_out=150
)
```

### 3. Note Identification

A note is considered "processed" if:
- At least one card from that note exists in the database
- The card has a `source_path` matching the note's relative path
- Language doesn't matter (if any language is synced, note is processed)

Example:
```
Note: notes/interview-questions/q-java-hashmap.md
Cards in database:
  - q-java-hashmap-1-en (exists) ✓
  - q-java-hashmap-1-ru (exists) ✓

Result: Note is "processed", will be skipped in incremental mode
```

### 4. Integration with Indexing

When indexing is enabled (default), incremental mode also:
- Updates note index only for new/modified notes
- Checks file modification timestamps
- Skips re-indexing unchanged notes

## Usage

### Basic Incremental Sync

```bash
# Process only new notes
obsidian-anki-sync sync --incremental
```

### View What Will Be Processed

```bash
# Dry run to preview
obsidian-anki-sync sync --incremental --dry-run
```

Output shows:
```
Incremental mode: Skipping 150 already processed notes

=== Sync Plan (Dry Run) ===

[CREATE] q-new-note-1-en
  Reason: New card not in database or Anki

[CREATE] q-new-note-1-ru
  Reason: New card not in database or Anki

=== Summary ===
create: 2
skip: 0
```

### Force Full Sync

```bash
# Process all notes (default behavior)
obsidian-anki-sync sync
```

### Check Processed Notes

```bash
# View index statistics
obsidian-anki-sync index
```

Output:
```
Notes Index:
┌─────────────┬───────┐
│ Total Notes │ 200   │
│ Pending     │ 50    │
│ Completed   │ 150   │
└─────────────┴───────┘
```

## Use Cases

### Use Case 1: Daily Workflow

**Scenario**: You add 5-10 new interview questions daily to a vault of 500+ notes

```bash
# Morning: Add new questions to Obsidian
vim notes/q-new-algorithm.md
vim notes/q-new-design-pattern.md

# Sync: Only process new notes (fast)
obsidian-anki-sync sync --incremental

# Result: ~10 seconds vs 10 minutes for full sync
```

### Use Case 2: Large Vault Migration

**Scenario**: Initial sync of 1000 notes, then daily updates

```bash
# Day 1: Full sync (one-time, slow)
obsidian-anki-sync sync
# Processes: 1000 notes, creates 3000 cards
# Time: ~30 minutes

# Day 2: Add 10 new notes
obsidian-anki-sync sync --incremental
# Processes: 10 notes, creates 30 cards
# Time: ~20 seconds

# Day 3: Add 8 new notes
obsidian-anki-sync sync --incremental
# Processes: 8 notes, creates 24 cards
# Time: ~15 seconds
```

### Use Case 3: Testing New Notes

**Scenario**: Test sync on new notes before full sync

```bash
# Add test note
echo "..." > notes/q-test.md

# Test with incremental mode
obsidian-anki-sync sync --incremental --dry-run

# Verify output, then apply
obsidian-anki-sync sync --incremental
```

### Use Case 4: Selective Re-sync

**Scenario**: Re-sync specific notes after modification

```bash
# Modify existing note
vim notes/q-java-streams.md

# Delete card from database to force re-sync
sqlite3 .sync_state.db "DELETE FROM cards WHERE source_path = 'notes/q-java-streams.md'"

# Incremental sync will process it as "new"
obsidian-anki-sync sync --incremental
```

## Performance

### Speed Comparison

| Vault Size | Full Sync | Incremental (10 new) | Speedup |
|------------|-----------|---------------------|---------|
| 100 notes  | 2 min     | 10 sec              | 12x     |
| 500 notes  | 10 min    | 15 sec              | 40x     |
| 1000 notes | 30 min    | 20 sec              | 90x     |

### Overhead

Incremental mode adds minimal overhead:
- **Database query**: <100ms for DISTINCT query
- **Set operations**: O(n) for filtering, where n = total notes
- **Memory**: ~1KB per 1000 notes for path tracking

### Optimization Tips

1. **Index regularly**: `obsidian-anki-sync sync` (with indexing)
2. **Clean database**: Remove orphaned cards periodically
3. **Use SSD**: Database on SSD improves query speed
4. **Combine modes**: `--incremental` + `--resume` for interrupted syncs

## Best Practices

### 1. Use for Daily Syncs

```bash
# Add to daily workflow/cron
obsidian-anki-sync sync --incremental
```

### 2. Full Sync Periodically

```bash
# Weekly: Full sync to catch any issues
obsidian-anki-sync sync

# Daily: Incremental for speed
obsidian-anki-sync sync --incremental
```

### 3. Verify with Dry Run

```bash
# Check what will be processed
obsidian-anki-sync sync --incremental --dry-run

# If looks good, apply
obsidian-anki-sync sync --incremental
```

### 4. Monitor Processed Notes

```bash
# Check index regularly
obsidian-anki-sync index

# View statistics
obsidian-anki-sync progress
```

### 5. Handle Modified Notes

If you modify an existing note and want to re-sync:

**Option A**: Delete and re-sync
```bash
# Delete old cards
sqlite3 .sync_state.db "DELETE FROM cards WHERE source_path = 'notes/modified.md'"

# Sync incrementally
obsidian-anki-sync sync --incremental
```

**Option B**: Full sync (will update)
```bash
# Full sync detects changes
obsidian-anki-sync sync
```

## Troubleshooting

### New notes not being processed

**Cause**: Note path already in database

**Solution**:
```bash
# Check database
sqlite3 .sync_state.db "SELECT source_path FROM cards WHERE source_path LIKE '%note-name%'"

# If found but shouldn't be there, delete
sqlite3 .sync_state.db "DELETE FROM cards WHERE source_path = 'path/to/note.md'"
```

### Old notes being re-processed

**Cause**: Notes not in database (maybe deleted previously)

**Solution**:
```bash
# Check what's missing
obsidian-anki-sync index

# Run full sync to register all
obsidian-anki-sync sync
```

### Incremental mode too slow

**Cause**: Many new notes, or database query slow

**Solution**:
```bash
# Check number of new notes
obsidian-anki-sync sync --incremental --dry-run | grep "new notes"

# If too many, consider batching:
obsidian-anki-sync sync --incremental --sample 50
```

### Modified notes not updating

**Cause**: Incremental mode skips existing notes

**Solution**:
```bash
# Use full sync for updates
obsidian-anki-sync sync

# Or delete and re-sync specific note
```

## Advanced Usage

### Combine with Sampling

```bash
# Process 20 random new notes
obsidian-anki-sync sync --incremental --sample 20
```

### Scripted Workflow

```bash
#!/bin/bash
# daily-sync.sh

echo "Starting incremental sync..."

# Run incremental sync
obsidian-anki-sync sync --incremental

# Check for errors
if [ $? -ne 0 ]; then
    echo "Sync failed, trying full sync..."
    obsidian-anki-sync sync
fi

echo "Sync complete!"
```

### Monitoring Script

```bash
#!/bin/bash
# check-new-notes.sh

# Count total notes in vault
VAULT_NOTES=$(find /path/to/vault -name "q-*.md" | wc -l)

# Count processed notes
PROCESSED=$(sqlite3 .sync_state.db "SELECT COUNT(DISTINCT source_path) FROM cards")

# Calculate new notes
NEW=$((VAULT_NOTES - PROCESSED))

echo "Total notes: $VAULT_NOTES"
echo "Processed: $PROCESSED"
echo "New: $NEW"

if [ $NEW -gt 0 ]; then
    echo "Run: obsidian-anki-sync sync --incremental"
fi
```

## Examples

### Example 1: First-Time Setup

```bash
# Day 1: Initial full sync
$ obsidian-anki-sync sync
Scanning: 500 notes
Processing: 500 notes
Created: 1500 cards
Time: 15 minutes

# Day 2: Add 5 new notes, incremental sync
$ obsidian-anki-sync sync --incremental
Incremental mode: Skipping 500 already processed notes
Processing: 5 new notes
Created: 15 cards
Time: 15 seconds
```

### Example 2: Interrupted Incremental Sync

```bash
# Start incremental sync
$ obsidian-anki-sync sync --incremental
Processing: 50 new notes
^C  # Interrupted at 25/50

# Resume with incremental mode
$ obsidian-anki-sync sync --incremental --resume abc123
Resuming: 25/50 remaining
✓ Complete!
```

### Example 3: Verification Workflow

```bash
# Check what will be synced
$ obsidian-anki-sync sync --incremental --dry-run
Incremental mode: Skipping 200 already processed notes

[CREATE] q-new-topic-1-en
[CREATE] q-new-topic-1-ru
[CREATE] q-new-topic-2-en

Summary: create: 3

# Looks good, apply
$ obsidian-anki-sync sync --incremental
✓ Created 3 cards
```

## See Also

- [Resumable Sync Guide](RESUMABLE_SYNC.md)
- [Indexing System Guide](INDEXING.md)
- [Main README](../README.md)
