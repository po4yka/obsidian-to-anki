# AnkiConnect API Integration

This document describes how the project integrates with the AnkiConnect API.

## Overview

The project uses AnkiConnect API v6 to communicate with Anki. All API calls are made through the `AnkiClient` class in `src/obsidian_anki_sync/anki/client.py`.

## API Version

- **Version**: 6 (latest stable)
- **Default URL**: `http://127.0.0.1:8765`
- **Protocol**: HTTP POST with JSON payloads

## Request Format

All requests follow this structure:

```python
{
    "action": "actionName",
    "version": 6,
    "params": {
        # action-specific parameters
    }
}
```

## Response Format

All responses follow this structure:

```python
{
    "result": ...,  # Action result (null if error)
    "error": null   # Error message (null if success)
}
```

## Implemented Methods

### Note Operations

#### findNotes
Find notes matching a query.

```python
client.find_notes("deck:MyDeck tag:interview")
# Returns: list[int] - Note IDs
```

**API Action**: `findNotes`
**Parameters**: `{"query": str}`

#### notesInfo
Get detailed information about notes.

```python
client.notes_info([1234567890, 9876543210])
# Returns: list[dict] - Note information
```

**API Action**: `notesInfo`
**Parameters**: `{"notes": list[int]}`

#### addNote
Add a new note to Anki.

```python
client.add_note(
    deck="Interview Questions",
    note_type="APF::Simple",
    fields={
        "Front": "Question",
        "Back": "Answer",
        "Manifest": "{...}"
    },
    tags=["interview", "python"],
    guid="optional-guid"
)
# Returns: int - Note ID
```

**API Action**: `addNote`
**Parameters**:
```python
{
    "note": {
        "deckName": str,
        "modelName": str,
        "fields": dict[str, str],
        "tags": list[str],
        "options": {
            "allowDuplicate": bool
        },
        "guid": str  # optional
    }
}
```

#### updateNoteFields
Update fields of an existing note.

```python
client.update_note_fields(
    note_id=1234567890,
    fields={"Front": "Updated Question"}
)
```

**API Action**: `updateNoteFields`
**Parameters**:
```python
{
    "note": {
        "id": int,
        "fields": dict[str, str]
    }
}
```

#### deleteNotes
Delete notes by ID.

```python
client.delete_notes([1234567890, 9876543210])
```

**API Action**: `deleteNotes`
**Parameters**: `{"notes": list[int]}`

### Tag Operations

#### addTags
Add tags to notes.

```python
client.add_tags(
    note_ids=[1234567890],
    tags="interview python"
)
```

**API Action**: `addTags`
**Parameters**: `{"notes": list[int], "tags": str}`

#### removeTags
Remove tags from notes.

```python
client.remove_tags(
    note_ids=[1234567890],
    tags="old-tag"
)
```

**API Action**: `removeTags`
**Parameters**: `{"notes": list[int], "tags": str}`

#### replaceTags
Replace a tag with another across notes.

```python
client.replace_tags(
    note_ids=[1234567890],
    tag_to_replace="old-tag",
    replace_with="new-tag"
)
```

**API Action**: `replaceTags`
**Parameters**: `{"notes": list[int], "tag_to_replace": str, "replace_with": str}`

### Deck and Model Operations

#### deckNames
Get all deck names.

```python
client.get_deck_names()
# Returns: list[str]
```

**API Action**: `deckNames`
**Parameters**: None

#### modelNames
Get all note type (model) names.

```python
client.get_model_names()
# Returns: list[str]
```

**API Action**: `modelNames`
**Parameters**: None

#### modelFieldNames
Get field names for a note type.

```python
client.get_model_field_names("APF::Simple")
# Returns: list[str]
```

**API Action**: `modelFieldNames`
**Parameters**: `{"modelName": str}`

### Utility Operations

#### canAddNotes
Check if notes can be added (duplicate check).

```python
client.can_add_notes([
    {
        "deckName": "MyDeck",
        "modelName": "Basic",
        "fields": {"Front": "Question", "Back": "Answer"}
    }
])
# Returns: list[bool]
```

**API Action**: `canAddNotes`
**Parameters**: `{"notes": list[dict]}`

#### storeMediaFile
Store a media file in Anki's media folder.

```python
client.store_media_file(
    filename="image.png",
    data="base64_encoded_data"
)
# Returns: str - Stored filename
```

**API Action**: `storeMediaFile`
**Parameters**: `{"filename": str, "data": str}`

#### guiBrowse
Open Anki browser with a search query.

```python
client.gui_browse("deck:MyDeck")
# Returns: list[int] - Note IDs in browser
```

**API Action**: `guiBrowse`
**Parameters**: `{"query": str}`

#### sync
Trigger Anki synchronization.

```python
client.sync()
```

**API Action**: `sync`
**Parameters**: None

## Error Handling

The client implements comprehensive error handling:

```python
try:
    result = client.add_note(...)
except AnkiConnectError as e:
    # Handle AnkiConnect-specific errors
    logger.error("AnkiConnect error", error=str(e))
```

### Error Types

- **Connection Errors**: AnkiConnect not running or network issues
- **HTTP Errors**: Invalid requests or server errors
- **API Errors**: Invalid parameters or Anki-specific errors (e.g., duplicate notes)

## Retry Logic

All API calls are automatically retried on failure:

- **Max Attempts**: 3
- **Initial Delay**: 1.0 seconds
- **Backoff**: Exponential
- **Retryable Errors**: `httpx.HTTPError`, `httpx.TimeoutException`, `AnkiConnectError`

## Connection Pooling

The client uses connection pooling for better performance:

```python
httpx.Limits(
    max_keepalive_connections=5,
    max_connections=10,
    keepalive_expiry=30.0
)
```

## Context Manager Support

The client can be used as a context manager:

```python
with AnkiClient(url="http://127.0.0.1:8765") as client:
    notes = client.find_notes("deck:MyDeck")
    # Connection automatically closed
```

## Configuration

Configure AnkiConnect URL via environment variables:

```bash
# .env
ANKI_CONNECT_URL=http://127.0.0.1:8765
```

Or in code:

```python
from obsidian_anki_sync.config import Config

config = Config()
client = AnkiClient(url=config.anki_connect_url)
```

## Testing

The client is tested with `respx` for HTTP mocking:

```python
import respx
import httpx

@respx.mock
def test_add_note():
    respx.post("http://127.0.0.1:8765").mock(
        return_value=httpx.Response(
            200,
            json={"result": 1234567890, "error": null}
        )
    )

    client = AnkiClient("http://127.0.0.1:8765")
    note_id = client.add_note(...)
    assert note_id == 1234567890
```

## API Compatibility

### Supported AnkiConnect Versions

- **v6**: Fully supported (current)
- **v5**: Not tested
- **v4 and below**: Not supported

### Breaking Changes from Previous Versions

None - this is the initial implementation using v6.

## Common Patterns

### Adding a Note with Duplicate Check

```python
# Check if note can be added
can_add = client.can_add_notes([{
    "deckName": deck,
    "modelName": note_type,
    "fields": fields
}])

if can_add[0]:
    note_id = client.add_note(deck, note_type, fields, tags)
else:
    logger.warning("Note already exists")
```

### Batch Operations

```python
# Find all notes in a deck
note_ids = client.find_notes("deck:MyDeck")

# Get their information
notes_info = client.notes_info(note_ids)

# Update them in batch
for note_info in notes_info:
    client.update_note_fields(
        note_info['noteId'],
        {"Field": "New Value"}
    )
```

### Safe Deletion

```python
# Find notes to delete
note_ids = client.find_notes("tag:to-delete")

if note_ids:
    logger.info(f"Deleting {len(note_ids)} notes")
    client.delete_notes(note_ids)
```

## Troubleshooting

### AnkiConnect Not Running

```
AnkiConnectError: Connection error to AnkiConnect: ...
```

**Solution**: Ensure Anki is running with AnkiConnect addon installed.

### Permission Denied

```
AnkiConnectError: HTTP 403 from AnkiConnect
```

**Solution**: Check AnkiConnect CORS settings in Anki addon configuration.

### Duplicate Note

```
AnkiConnectError: cannot create note because it is a duplicate
```

**Solution**: Use `canAddNotes()` before adding, or set `allowDuplicate: true`.

### Invalid Note Type

```
AnkiConnectError: model was not found: APF::Simple
```

**Solution**: Ensure the note type exists in Anki before adding notes.

## References

- [AnkiConnect Official Documentation](https://foosoft.net/projects/anki-connect/)
- [AnkiConnect GitHub Repository](https://github.com/FooSoft/anki-connect)
- [Anki Manual](https://docs.ankiweb.net/)

## Changelog

### v0.1.0 (2025-10-12)

- Initial implementation with AnkiConnect API v6
- Fixed incorrect `update_note_tags` method (was using non-existent `notesTags` action)
- Replaced with correct `addTags`, `removeTags`, and `replaceTags` methods
- Added `canAddNotes` for duplicate checking
- Added `storeMediaFile` for media support
- Added `guiBrowse` for UI integration
- Implemented retry logic and connection pooling
- Added comprehensive error handling

