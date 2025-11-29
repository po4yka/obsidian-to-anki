# Advanced Synchronization

Advanced sync features: semantic analysis, conflict resolution, change management, and
resource guard rails for large vaults.

## Problematic Note Archival Guard Rails

The sync engine queues failed notes while generations run in parallel, then archives
them at the end of a batch. Copying thousands of notes in one shot can exhaust the
process file descriptor (FD) limit on macOS and Linux, so the pipeline now enforces
batched archival with live FD monitoring:

-   `archiver_batch_size` (default `64`): maximum number of deferred notes written
    before re-checking system headroom. Increase it for SSD-backed runners with high
    limits; decrease it for constrained CI sandboxes.
-   `archiver_min_fd_headroom` (default `32`): minimum number of free descriptors that
    must remain before another batch runs. When the snapshot falls below this
    threshold the archiver pauses.
-   `archiver_fd_poll_interval` (default `0.05` seconds): backoff interval while we
    wait for other workers to release descriptors.

Headroom data comes from `obsidian_anki_sync.utils.fs_monitor`, which first tries
`psutil.Process().num_fds()` and falls back to `/proc/self/fd` or `/dev/fd`. These
diagnostics are added to structured logs whenever note copying fails so operators
can correlate EMFILE errors with the observed open-count/limit.

```bash
# Example overrides for extremely low ulimit -n environments
ARCHIVER_BATCH_SIZE=16
ARCHIVER_MIN_FD_HEADROOM=8
ARCHIVER_FD_POLL_INTERVAL=0.1
```

If your environment allows only ~128 descriptors, the above tuning keeps backpressure
fair while still draining queued notes deterministically.

## Semantic Diff Analysis

LLM-powered analysis understands change meaning beyond text differences.

**Process**: Text diff → LLM classification → Policy decision → Action

**Output Example**:

```json
{
    "field_analyses": {
        "Front": {
            "change_type": "typo_fix",
            "severity": "cosmetic",
            "recommendation": "approve"
        }
    },
    "overall_assessment": {
        "should_update": true,
        "risk_level": "low"
    }
}
```

## Conflict Detection

**Conflict**: Card modified in both Obsidian and Anki since last sync.

**Detection**:

```python
obsidian_changed = last_obsidian_edit > last_sync_ts
anki_changed = last_anki_edit > last_sync_ts
conflict = obsidian_changed and anki_changed
```

## Conflict Resolution Strategies

| Strategy        | Description          | Use Case                       |
| --------------- | -------------------- | ------------------------------ |
| `obsidian_wins` | Use Obsidian version | Obsidian is source of truth    |
| `anki_wins`     | Keep Anki version    | Preserve manual Anki edits     |
| `manual`        | User decides         | Critical data, review required |
| `merge`         | LLM-assisted merge   | Both versions have value       |
| `newest_wins`   | Use most recent      | Quick resolution               |

## Update Policies

### Field-Specific Policies

```yaml
auto_update_fields: ["Extra", "Hint"]
protected_fields: ["UserNotes"]
review_required_fields: ["Front", "Back"]
```

### Severity-Based Policies

```yaml
auto_approve_severity: ["cosmetic"]
review_required_severity: ["content", "structural"]
block_severity: ["destructive"]
```

### Card Maturity Policies

```yaml
mature_card_threshold: 21 # days
mature_card_policy:
    allow_content_updates: false # Only cosmetic changes
```

## Incremental Updates

### Field-Level Updates

Update only changed fields instead of entire cards:

```python
changes = differ.compare(existing, proposed)
for change in changes:
    if change.approved:
        anki.update_field(note_id, change.field, change.new_value)
```

**Benefits**: Preserves data, reduces conflicts, granular control.

### Delta Updates (Advanced)

Apply minimal diffs instead of full field replacement.

## Change History & Rollback

### Database Schema

```sql
CREATE TABLE change_history (
    id INTEGER PRIMARY KEY,
    note_id INTEGER NOT NULL,
    field_name TEXT NOT NULL,
    old_value TEXT, new_value TEXT,
    change_type TEXT, severity TEXT,
    applied BOOLEAN DEFAULT 0,
    timestamp DATETIME
);
```

### Rollback Commands

```bash
obsidian-anki-sync history --note-id 123    # View changes
obsidian-anki-sync rollback --note-id 123   # Undo last change
obsidian-anki-sync rollback --all --since "1 hour ago"
```

## User Review Interface

### Interactive Mode

```bash
obsidian-anki-sync sync --interactive
# Shows diff and prompts for approval/rejection
```

### Batch Review

```bash
obsidian-anki-sync sync --dry-run --export-review changes.json
# Review changes in external tool, then apply approved ones
```

## Protecting Mature Cards

Mature cards (>21 days) get conservative treatment:

```python
if days_since_creation > 30:
    return change.severity == COSMETIC  # Only safe changes
```

---

**Related**: [Basic Sync](synchronization.md) | [API Reference](../REFERENCE/sync-api.md)
