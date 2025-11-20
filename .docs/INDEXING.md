# Indexing System Guide

This guide covers the comprehensive indexing system that catalogs your Obsidian vault and Anki cards, enabling better tracking and synchronization.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Index Structure](#index-structure)
- [Usage](#usage)
- [Use Cases](#use-cases)
- [Maintenance](#maintenance)
- [Troubleshooting](#troubleshooting)

## Overview

The indexing system provides:

- **Vault Catalog**: Complete inventory of all Obsidian notes
- **Card Inventory**: Comprehensive tracking of expected and existing cards
- **Three-Way State**: Track cards across Obsidian, Anki, and sync database
- **Status Tracking**: Know the state of every note and card
- **Performance**: Efficient queries and incremental updates
- **Visibility**: Rich statistics and reporting

## Quick Start

### Run sync with indexing (default)

```bash
obsidian-anki-sync sync
```

Output includes index statistics:
```
Index Statistics:
¬¬
‚ Category ‚ Metric             ‚ Value ‚
¼¼
‚ Notes    ‚ Total              ‚ 250   ‚
‚ Notes    ‚ Pending            ‚ 10    ‚
‚ Notes    ‚ Completed          ‚ 240   ‚
‚ Cards    ‚ Total              ‚ 750   ‚
‚ Cards    ‚ In Obsidian        ‚ 750   ‚
‚ Cards    ‚ In Anki            ‚ 720   ‚
‚ Cards    ‚ In Database        ‚ 720   ‚
˜
```

### View current index

```bash
obsidian-anki-sync index
```

### Skip indexing (faster but less tracking)

```bash
obsidian-anki-sync sync --no-index
```

## How It Works

### Indexing Workflow

```

‚ 1. Vault Indexing                       ‚
‚    - Discover all notes                 ‚
‚    - Parse metadata and Q/A pairs       ‚
‚    - Create note_index entries          ‚
‚    - Create expected card_index entries ‚
˜
              

‚ 2. Database Indexing                    ‚
‚    - Query all cards from sync DB       ‚
‚    - Update card_index with DB info     ‚
‚    - Mark cards as in_database          ‚
˜
              

‚ 3. Anki Indexing                        ‚
‚    - Query Anki deck cards              ‚
‚    - Match to vault via slugs           ‚
‚    - Mark cards as in_anki              ‚
‚    - Identify orphaned cards            ‚
˜
              

‚ 4. Statistics Generation                ‚
‚    - Calculate totals and breakdowns    ‚
‚    - Generate status reports            ‚
‚    - Display to user                    ‚
˜
```

### Three-Way Reconciliation

Each card is tracked in three locations:

| Location | Flag | Meaning |
|----------|------|---------|
| Obsidian | `in_obsidian` | Note exists in vault |
| Anki | `in_anki` | Card exists in Anki deck |
| Database | `in_database` | Card tracked in sync DB |

**Examples:**

```
Card A: [ Obsidian] [ Anki] [ Database]
Status: "synced" - Everything in sync

Card B: [ Obsidian] [— Anki] [— Database]
Status: "expected" - New card to create

Card C: [— Obsidian] [ Anki] [ Database]
Status: "orphaned" - Card in Anki but note deleted

Card D: [ Obsidian] [ Anki] [— Database]
Status: "exists" - Card exists but not tracked
```

## Index Structure

### Note Index Table

```sql
CREATE TABLE note_index (
    source_path TEXT PRIMARY KEY,           -- Relative path to note
    note_id TEXT,                           -- Note ID from frontmatter
    note_title TEXT,                        -- Note title
    topic TEXT,                             -- Note topic
    language_tags TEXT,                     -- Comma-separated languages
    qa_pair_count INTEGER,                  -- Number of Q/A pairs
    file_modified_at TIMESTAMP,             -- File modification time
    last_indexed_at TIMESTAMP,              -- Last indexing time
    last_synced_at TIMESTAMP,               -- Last successful sync
    sync_status TEXT,                       -- pending/processing/completed/failed
    error_message TEXT,                     -- Error if failed
    metadata_json TEXT                      -- Full metadata as JSON
)
```

**Example Entry:**
```json
{
  "source_path": "notes/q-java-hashmap.md",
  "note_id": "java-hashmap-001",
  "note_title": "Java HashMap Internals",
  "topic": "Java Collections",
  "language_tags": "en,ru",
  "qa_pair_count": 3,
  "file_modified_at": "2025-01-15T10:30:00",
  "last_indexed_at": "2025-01-15T11:00:00",
  "last_synced_at": "2025-01-15T11:05:00",
  "sync_status": "completed",
  "error_message": null
}
```

### Card Index Table

```sql
CREATE TABLE card_index (
    id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,              -- Note that card comes from
    card_index INTEGER NOT NULL,            -- Q/A pair index (1-based)
    lang TEXT NOT NULL,                     -- Language code
    slug TEXT UNIQUE,                       -- Unique card slug
    anki_guid INTEGER,                      -- Anki note ID
    note_id TEXT,                           -- Note ID from frontmatter
    note_title TEXT,                        -- Note title
    content_hash TEXT,                      -- Card content hash
    status TEXT,                            -- Card status (see below)
    last_indexed_at TIMESTAMP,              -- Last indexing time
    in_obsidian BOOLEAN,                    -- Exists in vault
    in_anki BOOLEAN,                        -- Exists in Anki
    in_database BOOLEAN,                    -- Tracked in sync DB
    UNIQUE(source_path, card_index, lang)
)
```

**Example Entry:**
```json
{
  "source_path": "notes/q-java-hashmap.md",
  "card_index": 1,
  "lang": "en",
  "slug": "q-java-hashmap-1-en",
  "anki_guid": 1234567890,
  "note_id": "java-hashmap-001",
  "note_title": "Java HashMap Internals",
  "status": "synced",
  "in_obsidian": true,
  "in_anki": true,
  "in_database": true
}
```

### Card Statuses

| Status | Description |
|--------|-------------|
| **expected** | Card should exist (from note) but not created yet |
| **new** | Card exists in Obsidian, ready to create in Anki |
| **exists** | Card found in Anki |
| **modified** | Card in Obsidian differs from Anki version |
| **deleted** | Card removed from Obsidian but still in Anki |
| **orphaned** | Card in Anki but note no longer in vault |
| **synced** | Card fully synchronized across all three locations |

## Usage

### View Index Statistics

```bash
obsidian-anki-sync index
```

Output:
```
Vault & Anki Index:

Notes Index:
¬
‚ Metric          ‚ Count ‚
¼
‚ Total Notes     ‚ 250   ‚
‚   Pending       ‚ 10    ‚
‚   Processing    ‚ 2     ‚
‚   Completed     ‚ 235   ‚
‚   Failed        ‚ 3     ‚
˜

Cards Index:
¬
‚ Metric          ‚ Count ‚
¼
‚ Total Cards     ‚ 750   ‚
‚ In Obsidian     ‚ 750   ‚
‚ In Anki         ‚ 720   ‚
‚ In Database     ‚ 720   ‚
˜

Card Status Breakdown:
  expected: 30
  synced: 690
  orphaned: 30
```

### Sync with Indexing

```bash
# Indexing enabled by default
obsidian-anki-sync sync
```

### Incremental Indexing

```bash
# Only re-index changed files
obsidian-anki-sync sync --incremental
```

Incremental indexing:
- Checks file modification timestamps
- Skips unchanged notes
- Only re-indexes modified or new notes
- Much faster for large vaults

### Skip Indexing

```bash
# Faster but less tracking
obsidian-anki-sync sync --no-index
```

**Use when:**
- Quick sync needed
- Index already up-to-date
- Testing/debugging

**Don't use when:**
- First sync
- After bulk changes
- Troubleshooting sync issues

## Use Cases

### Use Case 1: Identify Orphaned Cards

**Problem**: Cards in Anki but notes deleted from vault

```bash
# View index
$ obsidian-anki-sync index

Card Status Breakdown:
  orphaned: 15

# Query database for details
$ sqlite3 .sync_state.db "
  SELECT source_path, slug, anki_guid
  FROM card_index
  WHERE status = 'orphaned'
"
```

**Solution**:
```bash
# Option A: Delete orphaned cards
# Manually in Anki or via script

# Option B: Restore notes
# Re-create deleted notes in vault
```

### Use Case 2: Find Failed Syncs

**Problem**: Some notes failed to sync

```bash
$ obsidian-anki-sync index

Notes Index:
  Failed: 3

# Query for details
$ sqlite3 .sync_state.db "
  SELECT source_path, error_message
  FROM note_index
  WHERE sync_status = 'failed'
"

notes/q-broken.md | Parse error: Invalid YAML
notes/q-missing.md | File not found
notes/q-invalid.md | Missing Q/A pairs
```

**Solution**:
- Fix errors in notes
- Re-run sync

### Use Case 3: Audit Sync State

**Problem**: Verify sync integrity

```bash
$ obsidian-anki-sync index

Cards Index:
  In Obsidian: 750
  In Anki: 720
  In Database: 720

# 30 cards in Obsidian but not in Anki
# These should be created on next sync
```

### Use Case 4: Performance Monitoring

**Problem**: Track sync performance over time

```bash
# Before sync
$ obsidian-anki-sync index
Cards: 700

# After sync
$ obsidian-anki-sync sync
$ obsidian-anki-sync index
Cards: 750

# 50 new cards created
```

## Maintenance

### Rebuild Index

If index becomes stale or corrupted:

```python
from obsidian_anki_sync.sync.state_db import StateDB

with StateDB(".sync_state.db") as db:
    # Clear old index
    db.clear_index()

# Rebuild with full sync
obsidian-anki-sync sync
```

### Clean Up Old Data

```bash
# Delete completed progress records
obsidian-anki-sync clean-progress --all-completed

# Vacuum database
sqlite3 .sync_state.db "VACUUM"
```

### Update Index Only

To update index without syncing:

```python
from obsidian_anki_sync.config import load_config
from obsidian_anki_sync.sync.state_db import StateDB
from obsidian_anki_sync.sync.indexer import build_full_index
from obsidian_anki_sync.anki.client import AnkiClient

config = load_config()

with StateDB(config.db_path) as db, \
     AnkiClient(config.anki_connect_url) as anki:

    # Build full index
    stats = build_full_index(config, db, anki, incremental=False)
    print(stats)
```

## Troubleshooting

### Index shows incorrect counts

**Cause**: Stale index data

**Solution**:
```python
# Clear and rebuild
db.clear_index()
obsidian-anki-sync sync
```

### Indexing is slow

**Cause**: Many notes, or not using incremental mode

**Solution**:
```bash
# Use incremental indexing
obsidian-anki-sync sync --incremental

# Or skip indexing for quick syncs
obsidian-anki-sync sync --no-index
```

### Orphaned cards not detected

**Cause**: Index not up-to-date

**Solution**:
```bash
# Run full sync with indexing
obsidian-anki-sync sync
```

### Database too large

**Cause**: Many progress records

**Solution**:
```bash
# Clean up old progress
obsidian-anki-sync clean-progress --all-completed

# Vacuum database
sqlite3 .sync_state.db "VACUUM"
```

## Advanced Usage

### Query Index Directly

```bash
# Count notes by status
sqlite3 .sync_state.db "
  SELECT sync_status, COUNT(*)
  FROM note_index
  GROUP BY sync_status
"

# Find cards in Obsidian but not Anki
sqlite3 .sync_state.db "
  SELECT source_path, slug
  FROM card_index
  WHERE in_obsidian = 1 AND in_anki = 0
"

# Find recently modified notes
sqlite3 .sync_state.db "
  SELECT source_path, note_title, file_modified_at
  FROM note_index
  ORDER BY file_modified_at DESC
  LIMIT 10
"
```

### Export Index to CSV

```bash
# Export notes
sqlite3 -header -csv .sync_state.db "
  SELECT * FROM note_index
" > notes_index.csv

# Export cards
sqlite3 -header -csv .sync_state.db "
  SELECT * FROM card_index
" > cards_index.csv
```

### Analyze Index

```python
from obsidian_anki_sync.sync.state_db import StateDB

with StateDB(".sync_state.db") as db:
    stats = db.get_index_statistics()

    print(f"Total notes: {stats['total_notes']}")
    print(f"Total cards: {stats['total_cards']}")
    print(f"Coverage: {stats['cards_in_anki'] / stats['total_cards'] * 100:.1f}%")

    # Note status breakdown
    for status, count in stats['note_status'].items():
        print(f"  {status}: {count}")

    # Card status breakdown
    for status, count in stats['card_status'].items():
        print(f"  {status}: {count}")
```

## Performance

### Indexing Speed

| Operation | 100 notes | 500 notes | 1000 notes |
|-----------|-----------|-----------|------------|
| Full Index | 5 sec | 20 sec | 40 sec |
| Incremental (10 new) | 2 sec | 3 sec | 3 sec |
| Statistics Query | <1 sec | <1 sec | <1 sec |

### Memory Usage

- ~100 bytes per note in memory
- ~200 bytes per card in memory
- Database queries use indexes (fast)

### Optimization

1. **Use Incremental**: Only re-index changed files
2. **Index Selectively**: Skip indexing for quick syncs
3. **Clean Regularly**: Remove old progress records
4. **Use Indexes**: Database has proper indexes

## See Also

- [Resumable Sync Guide](RESUMABLE_SYNC.md)
- [Incremental Sync Guide](INCREMENTAL_SYNC.md)
- [Main README](../README.md)
