# Synchronization API Reference

Technical reference for sync system: commands, database schemas, and troubleshooting.

## Core Configuration

```yaml
vault_path: "~/Documents/ObsidianVault"
source_dir: "Notes"
anki_deck_name: "My Deck"
sync_mode: "incremental" # incremental|full
dry_run: false
max_concurrent_requests: 5
```

See [Configuration Guide](../GUIDES/configuration.md) for complete options.

## Commands

### Sync Commands
```bash
obsidian-anki-sync sync                    # Full sync
obsidian-anki-sync sync --dry-run         # Preview changes
obsidian-anki-sync sync --full-reindex    # Rebuild index
obsidian-anki-sync sync --interactive     # Manual review
```

### Management
```bash
obsidian-anki-sync decks                  # List Anki decks
obsidian-anki-sync validate <file>        # Validate note
obsidian-anki-sync ping                   # Test connectivity
```

### History & Rollback
```bash
obsidian-anki-sync history --note-id 123  # View changes
obsidian-anki-sync rollback --note-id 123 # Undo changes
obsidian-anki-sync rollback --all --since "1 hour ago"
```

## Database Schema

### Core Tables

#### sync_state
```sql
CREATE TABLE sync_state (
    id INTEGER PRIMARY KEY,
    note_path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    anki_note_id INTEGER,
    last_sync_timestamp DATETIME,
    last_obsidian_edit DATETIME,
    last_anki_edit DATETIME,
    sync_status TEXT DEFAULT 'pending'
);
```

#### change_history
```sql
CREATE TABLE change_history (
    id INTEGER PRIMARY KEY,
    note_id INTEGER NOT NULL,
    field_name TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    change_type TEXT,
    severity TEXT,
    applied BOOLEAN DEFAULT 0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## API Classes

### SyncEngine
Main orchestrator for sync operations.

```python
class SyncEngine:
    def sync(self, dry_run=False) -> SyncResult:
        """Execute synchronization."""

    def detect_changes(self) -> List[NoteChange]:
        """Find modified notes."""
```

### ConflictResolver
Handle sync conflicts between systems.

```python
class ConflictResolver:
    def resolve(self, change: NoteChange, anki_card) -> Resolution:
        """Resolve sync conflicts."""
```

## Error Handling

### Exception Types
- `SyncError`: Base sync exception
- `AnkiConnectError`: Anki API failures
- `ConfigurationError`: Invalid config
- `ValidationError`: Content validation failures

### Recovery Strategies
```python
# Retry with backoff
for attempt in range(max_retries):
    try:
        return engine.sync()
    except TransientError:
        sleep(2 ** attempt)
```

## Troubleshooting

### Common Issues

**Sync not detecting changes:**
```bash
# Force full reindex
obsidian-anki-sync sync --full-reindex
```

**Anki connection failed:**
```bash
# Check AnkiConnect
curl http://localhost:8765
# Restart Anki
```

**Memory/performance issues:**
- Reduce `max_concurrent_requests`
- Use `sync_mode: "incremental"`
- Check `batch_size` setting

### Debug Mode
```bash
export LOG_LEVEL=DEBUG
obsidian-anki-sync sync
```

## Performance Optimization

### Batch Processing
```yaml
batch_size: 50              # Cards per batch
max_concurrent_requests: 5  # Parallel operations
```

### Caching
```yaml
enable_result_cache: true
cache_ttl: 3600            # 1 hour
```

### Monitoring
```python
# Get sync metrics
stats = engine.get_statistics()
print(f"Processed: {stats.notes_processed}")
print(f"Success rate: {stats.success_rate}%")
```

---

**Related**: [Configuration](../GUIDES/configuration.md) | [Synchronization Guide](../GUIDES/synchronization.md)