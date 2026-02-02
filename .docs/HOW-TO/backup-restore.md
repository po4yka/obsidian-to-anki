# Backup and Restore

How to back up and restore your obsidian-to-anki data so you can recover
from mistakes, migrate machines, or start fresh.

## What to back up

| Item | Default location | Config key |
|------|-----------------|------------|
| Configuration | `config.yaml` | -- |
| Sync state DB | `.sync_state.db` (relative to `data_dir`) | `db_path` |
| RAG index | `.chroma_db/` (relative to `data_dir`) | `rag_db_path` |
| Agent memory | `.agent_memory/` (relative to `data_dir`) | `memory_storage_path` |
| Anki decks/cards | Managed by Anki | -- |

## Back up the sync state database

The sync state DB uses SQLite WAL mode. This means uncommitted data may
live in the `-wal` file alongside the main `.db` file. Two approaches:

### Copy all three files together

Stop any running sync first, then copy the DB and its WAL/SHM sidecars:

```bash
cp .sync_state.db .sync_state.db-wal .sync_state.db-shm /path/to/backup/
```

All three files (`.db`, `.db-wal`, `.db-shm`) must be copied together to
avoid a corrupt backup. The `-wal` and `-shm` files may not always exist;
copy them if they do.

### Use the SQLite backup command

For a guaranteed-consistent snapshot without stopping the sync:

```bash
sqlite3 .sync_state.db ".backup /path/to/backup/backup.db"
```

This produces a single self-contained file.

## Back up Anki decks

### Via Anki GUI

File > Export > select **Anki Deck Package (.apkg)** > choose the deck.

### Via CLI (requires Anki running with AnkiConnect)

```bash
obsidian-anki-sync export-deck "Deck Name" -o backup.yaml -f yaml
```

## Back up RAG index and agent memory

Copy the directories:

```bash
cp -r .chroma_db /path/to/backup/chroma_db
cp -r .agent_memory /path/to/backup/agent_memory
```

## Restore the sync state database

### From a file copy

Copy the backed-up files back into `data_dir`:

```bash
cp /path/to/backup/.sync_state.db .
cp /path/to/backup/.sync_state.db-wal . 2>/dev/null
cp /path/to/backup/.sync_state.db-shm . 2>/dev/null
```

### From a SQLite backup

```bash
sqlite3 .sync_state.db ".restore /path/to/backup/backup.db"
```

## Restore Anki cards

### Via Anki GUI

File > Import > select the `.apkg` file.

### Via CLI (requires Anki running with AnkiConnect)

```bash
obsidian-anki-sync import-deck backup.yaml -d "Deck Name"
```

## Fresh start procedure

Use this when you want to wipe all sync state and rebuild from scratch.
Your Obsidian notes and Anki cards are not deleted.

```bash
# 1. Remove the sync state DB and its WAL/SHM files
rm -f .sync_state.db .sync_state.db-wal .sync_state.db-shm

# 2. Clean completed progress records
obsidian-anki-sync clean-progress --all-completed

# 3. Reset the RAG index
obsidian-anki-sync rag reset --yes

# 4. Run a full sync to rebuild everything
obsidian-anki-sync sync
```

## See also

- [First sync](first-sync.md) -- initial setup and first run
- [Export and import decks](export-import-decks.md) -- moving decks between machines
