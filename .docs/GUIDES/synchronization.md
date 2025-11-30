# Synchronization Guide

## Overview

The sync system manages changes between Obsidian notes and Anki cards, balancing accuracy, safety, and user control.

## Change Types & Actions

| Change Type      | Risk   | Auto-Approve | Action               |
| ---------------- | ------ | ------------ | -------------------- |
| Typo fix         | Low    | Yes          | Apply                |
| Clarification    | Low    | Yes          | Apply                |
| Rephrasing       | Medium | No           | Review               |
| Content addition | Medium | No           | Review               |
| Meaning change   | High   | No           | Manual review        |
| Complete rewrite | High   | No           | Manual or new card   |
| Conflict         | High   | No           | Resolution strategy  |

## Conflict Resolution

**Detection**: Card modified in both Obsidian and Anki since last sync.

**Strategies**:
- `obsidian_wins` - Obsidian is source of truth
- `anki_wins` - Preserve manual Anki edits
- `manual` - User decides (default)
- `merge` - LLM-assisted merge
- `newest_wins` - Most recent version

## Configuration

```yaml
# Field policies
auto_update_fields: ["Extra", "Hint"]
protected_fields: ["UserNotes"]
review_required_fields: ["Front", "Back"]

# Severity policies
auto_approve_severity: ["cosmetic"]
block_severity: ["destructive"]

# Mature card protection (>21 days)
mature_card_threshold: 21
mature_card_policy:
    allow_content_updates: false
```

## Resource Limits (Large Vaults)

Batched archival prevents file descriptor exhaustion:

```bash
ARCHIVER_BATCH_SIZE=64          # Notes per batch
ARCHIVER_MIN_FD_HEADROOM=32     # Min free descriptors
ARCHIVER_FD_POLL_INTERVAL=0.05  # Backoff interval (sec)
```

For constrained environments (~128 descriptors):
```bash
ARCHIVER_BATCH_SIZE=16
ARCHIVER_MIN_FD_HEADROOM=8
```

## Commands

```bash
# Preview changes
obsidian-anki-sync sync --dry-run

# Interactive review
obsidian-anki-sync sync --interactive

# View history
obsidian-anki-sync history --note-id 123

# Rollback
obsidian-anki-sync rollback --note-id 123
obsidian-anki-sync rollback --all --since "1 hour ago"
```

## Best Practices

1. **Start conservative** - Begin with `conflict_resolution: "manual"`
2. **Use dry-run** - Always preview before applying
3. **Backup Anki** - Before major sync operations
4. **Monitor patterns** - Review auto-approved changes periodically

---

**Related**: [Configuration](configuration.md) | [Sync API](../reference/sync-api.md)
