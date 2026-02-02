# Export and Import Decks

How to export Anki decks to portable formats, edit them externally, and
import them back -- including standalone `.apkg` export that does not
require a running Anki instance.

## Prerequisites

| Operation              | Anki + AnkiConnect | LLM provider |
|------------------------|--------------------|--------------|
| `export` (.apkg)       | No                 | Yes          |
| `export-deck` (YAML/CSV) | Yes             | No           |
| `import-deck` (YAML/CSV) | Yes             | No           |

## Export to .apkg (standalone)

Generate cards from Obsidian notes using an LLM and package them into an
`.apkg` file. Anki does **not** need to be running.

```bash
# Basic export
obsidian-anki-sync export

# Full options
obsidian-anki-sync export \
  -o output.apkg \
  --deck-name "My Deck" \
  --deck-description "Cards from Obsidian vault" \
  --config path/to/config.yaml

# Export only 5 random notes (useful for testing)
obsidian-anki-sync export --sample 5
```

These values can also be set in `config.yaml`:

```yaml
export_deck_name: "My Deck"
export_deck_description: "Cards from Obsidian vault"
export_output_path: output.apkg
```

## Export a deck to YAML or CSV

Export an existing Anki deck to an editable file. Requires Anki running
with AnkiConnect.

```bash
# Export to YAML (default format)
obsidian-anki-sync export-deck "My Deck"

# Export to CSV
obsidian-anki-sync export-deck "My Deck" -f csv

# Specify output path
obsidian-anki-sync export-deck "My Deck" -o my-deck.yaml

# Exclude note IDs (e.g., for sharing)
obsidian-anki-sync export-deck "My Deck" --no-note-id
```

| Flag                | Default | Description                              |
|---------------------|---------|------------------------------------------|
| `-o PATH`           | auto    | Output file path (auto-generated if omitted) |
| `-f yaml\|csv`      | yaml    | Output format                            |
| `--include-note-id` | true    | Include `noteId` field for update matching |
| `--no-note-id`      | --      | Omit `noteId` from export                |

## Import a deck from YAML or CSV

Import cards from a YAML or CSV file into Anki. Requires Anki running
with AnkiConnect.

```bash
# Basic import (uses metadata from file)
obsidian-anki-sync import-deck cards.yaml

# Override target deck and note type
obsidian-anki-sync import-deck cards.yaml \
  -d "Target Deck" \
  -n "Basic (and reversed card)" \
  -k "Front"
```

| Flag                   | Default        | Description                          |
|------------------------|----------------|--------------------------------------|
| `-d` / `--deck-name`  | from file      | Target deck name                     |
| `-n` / `--note-type`  | auto-detected  | Anki note type (model)               |
| `-k` / `--key-field`  | auto-detected  | Field used to identify existing notes |

## Round-trip editing workflow

Export a deck, edit externally, then import the changes back:

```bash
# 1. Export the deck
obsidian-anki-sync export-deck "My Deck" -o my-deck.yaml

# 2. Edit my-deck.yaml in your editor of choice

# 3. Import the modified file back
obsidian-anki-sync import-deck my-deck.yaml -d "My Deck"
```

Notes with a matching `noteId` are updated in place; new entries are
created as new cards.

## Discovering decks and models

Before exporting or importing, you may need to know which decks and
note types exist in Anki:

```bash
# List all decks
obsidian-anki-sync decks

# List all note models
obsidian-anki-sync models

# Show fields for a specific model
obsidian-anki-sync model-fields --model "Basic (and reversed card)"
```

## See also

- [First sync](first-sync.md) -- initial setup and running your first sync
- [Backup and restore](backup-restore.md) -- protecting your data before bulk operations
