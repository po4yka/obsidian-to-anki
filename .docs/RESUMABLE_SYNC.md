# Resumable Sync Guide

This guide covers the resumable sync feature that allows you to interrupt and resume note-based card creation without losing progress.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Features](#features)
- [Usage](#usage)
- [Progress Tracking](#progress-tracking)
- [CLI Commands](#cli-commands)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

## Overview

The resumable sync feature provides:

- **Progress Tracking**: Real-time tracking of sync progress
- **Graceful Interruption**: Safe handling of Ctrl+C and SIGTERM
- **Automatic Resume**: Detection and resumption of incomplete syncs
- **State Persistence**: Progress saved to SQLite database
- **Per-Note Tracking**: Know exactly which notes have been processed

## Quick Start

### Start a new sync (can be interrupted)

```bash
obsidian-anki-sync sync
```

If interrupted (Ctrl+C), you'll see:
```
Sync interrupted! Progress has been saved (session: a1b2c3d4-...)
Resume with: obsidian-anki-sync sync --resume a1b2c3d4-...
```

### Resume an interrupted sync

```bash
obsidian-anki-sync sync --resume a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

### View sync progress

```bash
obsidian-anki-sync progress
```

## Features

### 1. Progress Tracking

The system tracks:
- **Sync phases**: Initializing, Indexing, Scanning, Generating, Determining Actions, Applying Changes
- **Note processing**: Each note's status (pending, processing, completed, failed)
- **Statistics**: Cards created, updated, deleted, restored, skipped
- **Errors**: Failed notes with error messages
- **Timing**: Start time, update time, completion time

### 2. Graceful Interruption

When you press Ctrl+C:
1. Current operation completes safely
2. Progress is saved to database
3. Session ID is displayed
4. Process exits cleanly (exit code 130)

Supported signals:
- **SIGINT** (Ctrl+C): User interruption
- **SIGTERM**: System termination

### 3. Automatic Resume

When starting a sync:
1. System checks for incomplete syncs
2. If found, prompts: "Resume this sync?"
3. If yes, continues from where it left off
4. Already processed notes are skipped

### 4. State Persistence

Progress is stored in SQLite (`sync_progress` table):
```sql
CREATE TABLE sync_progress (
    session_id TEXT PRIMARY KEY,
    phase TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    total_notes INTEGER,
    notes_processed INTEGER,
    cards_generated INTEGER,
    cards_created INTEGER,
    -- ... more statistics
    note_progress TEXT  -- JSON of per-note progress
)
```

## Usage

### Basic Sync with Progress Tracking

```bash
# Start sync (enabled by default)
obsidian-anki-sync sync

# Output shows progress
Starting new sync session: 12345678-1234-5678-9012-123456789012

Scanning notes... [=====>    ] 50% (5/10)
```

### Disable Progress Tracking

If you don't want progress tracking (not recommended):

```bash
# Progress tracking is always enabled by default
# To disable interruption handling, use --no-resume
obsidian-anki-sync sync --no-resume
```

### Resume Specific Session

```bash
# Resume by session ID
obsidian-anki-sync sync --resume <session-id>

# Example
obsidian-anki-sync sync --resume 12345678-1234-5678-9012-123456789012
```

### View All Progress

```bash
# Show incomplete and recent syncs
obsidian-anki-sync progress
```

Output:
```
Incomplete Syncs:
¬¬¬
‚ Session ID ‚ Phase     ‚ Progress ‚ Updated At  ‚
¼¼¼
‚ 12345...   ‚ scanning  ‚ 25/100   ‚ 2025-01-15  ‚
˜

Recent Syncs:
¬¬¬¬
‚ Session ID ‚ Phase     ‚ Progress ‚ Errors ‚ Started At ‚
¼¼¼¼
‚ 67890...   ‚ completed ‚ 100/100  ‚ 0      ‚ 2025-01-14 ‚
‚ 12345...   ‚ scanning  ‚ 25/100   ‚ 2      ‚ 2025-01-15 ‚
˜
```

### Clean Up Old Progress

```bash
# Delete specific session
obsidian-anki-sync clean-progress --session <session-id>

# Delete all completed sessions
obsidian-anki-sync clean-progress --all-completed
```

## Progress Tracking

### Sync Phases

1. **Initializing**: Setting up sync session
2. **Indexing**: Building vault and Anki card index
3. **Scanning**: Discovering and parsing notes
4. **Generating**: Creating APF cards
5. **Determining Actions**: Deciding create/update/delete
6. **Applying Changes**: Syncing to Anki
7. **Completed**: Successfully finished
8. **Interrupted**: User interrupted (Ctrl+C)
9. **Failed**: Error occurred

### Note Status

- **pending**: Not yet processed
- **processing**: Currently being processed
- **completed**: Successfully processed
- **failed**: Error occurred

### Statistics Tracked

- `notes_processed`: Number of notes completed
- `cards_generated`: Total cards created from notes
- `cards_created`: Cards added to Anki
- `cards_updated`: Cards modified in Anki
- `cards_deleted`: Cards removed from Anki
- `cards_restored`: Cards re-created in Anki
- `cards_skipped`: Cards unchanged
- `errors`: Number of errors encountered

## CLI Commands

### sync

```bash
obsidian-anki-sync sync [OPTIONS]

Options:
  --resume TEXT        Resume interrupted sync by session ID
  --no-resume          Disable automatic resume detection
  --dry-run            Preview changes without applying
  --incremental        Only process new notes
  --use-agents         Use multi-agent system
  --no-index           Skip indexing phase
```

### progress

```bash
obsidian-anki-sync progress

Shows:
  - Incomplete syncs with resume instructions
  - Recent sync history
  - Session IDs, phases, progress, errors
```

### clean-progress

```bash
obsidian-anki-sync clean-progress [OPTIONS]

Options:
  --session TEXT       Delete specific session
  --all-completed      Delete all completed sessions
```

## Technical Details

### Session Management

Each sync gets a unique UUID session ID:
```python
session_id = "12345678-1234-5678-9012-123456789012"
```

Sessions are stored in `sync_progress` table with:
- Full statistics
- Per-note progress (JSON)
- Timestamps
- Current phase

### Signal Handling

Signal handlers are installed at sync start:

```python
def signal_handler(signum, frame):
    logger.warning("sync_interrupted", signal=signal_name)
    save_progress()
    print(f"Resume with: sync --resume {session_id}")
    sys.exit(130)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
```

### Resume Logic

On resume:
1. Load progress from database
2. Check which notes are completed
3. Skip completed notes during scan
4. Continue from last checkpoint

```python
if progress_tracker.is_note_completed(note_path, card_index, lang):
    logger.debug("skipping_completed_note")
    continue
```

### Performance

- Progress saved after each note (immediate commits)
- SQLite with WAL mode for concurrency
- Minimal overhead (<1% for typical syncs)
- Index-based queries for fast lookups

## Troubleshooting

### "No progress found for session"

**Cause**: Session ID doesn't exist or was deleted

**Solution**:
```bash
# View available sessions
obsidian-anki-sync progress

# Use correct session ID
obsidian-anki-sync sync --resume <valid-session-id>
```

### Progress not saving

**Cause**: Database permission issues

**Solution**:
```bash
# Check database file
ls -la .sync_state.db

# Ensure write permissions
chmod 644 .sync_state.db
```

### Sync stuck in interrupted state

**Cause**: Process killed forcefully (SIGKILL)

**Solution**:
```bash
# View stuck sessions
obsidian-anki-sync progress

# Start new sync (will prompt to resume or start fresh)
obsidian-anki-sync sync
```

### Too many incomplete sessions

**Solution**:
```bash
# Clean up old completed sessions
obsidian-anki-sync clean-progress --all-completed

# Or delete specific session
obsidian-anki-sync clean-progress --session <session-id>
```

## Best Practices

1. **Let it complete**: Allow syncs to finish when possible
2. **Clean up**: Periodically clean completed sessions
3. **Check progress**: Use `progress` command to monitor
4. **Incremental**: Use `--incremental` for large vaults
5. **Resume promptly**: Resume interrupted syncs soon to avoid stale state

## Examples

### Example 1: Basic Workflow

```bash
# Start sync
$ obsidian-anki-sync sync
Starting new sync session: abc123...
Scanning: [=====>    ] 50/100 notes
^C  # User presses Ctrl+C

Sync interrupted! Progress saved (session: abc123)
Resume with: obsidian-anki-sync sync --resume abc123

# Later, resume
$ obsidian-anki-sync sync --resume abc123
Resuming sync session: abc123
Scanning: [=========>] 50/100 notes (resuming)
 Sync completed!
```

### Example 2: Auto-Resume

```bash
# Interrupted sync
$ obsidian-anki-sync sync
... interrupted ...

# Start new sync
$ obsidian-anki-sync sync

Found incomplete sync from 2025-01-15
  Session: abc123
  Progress: 50/100 notes
Resume this sync? [Y/n]: y

Resuming sync session: abc123
 Sync completed!
```

### Example 3: Monitoring

```bash
# Check progress while sync runs (in another terminal)
$ obsidian-anki-sync progress

Incomplete Syncs:
¬¬¬
‚ abc123 ‚ scanning ‚ 75/100   ‚ 2 min ago  ‚
˜
```

## See Also

- [Incremental Sync Guide](INCREMENTAL_SYNC.md)
- [Indexing System Guide](INDEXING.md)
- [Main README](../README.md)
