# Code Refactoring Guide

This document describes the refactoring work done to improve code maintainability and provides patterns for future refactoring.

## Completed Refactoring

### 1. CLI Command Modularization

**Problem:** `cli.py` was 1006 lines with 12 commands, making it difficult to navigate and maintain.

**Solution:** Created modular command handlers:
- `cli_commands/shared.py` - Shared utilities (config/logger loading, console)
- `cli_commands/sync_handler.py` - Sync command implementation logic
  - `run_sync()` - Main sync execution
  - `_handle_progress_tracking()` - Progress tracking setup
  - `_display_sync_results()` - Results display

**Benefits:**
- Easier to test individual components
- Clear separation of concerns
- Reusable command logic

**Usage Pattern:**
```python
# In cli.py
from .cli_commands.sync_handler import run_sync
from .cli_commands.shared import get_config_and_logger

@app.command()
def sync(...):
    config, logger = get_config_and_logger(config_path, log_level)
    run_sync(config, logger, dry_run, incremental, ...)
```

### 2. Security Improvements

**Added:** `utils/path_validator.py` module with:
- `validate_vault_path()` - Prevents symlink attacks
- `validate_source_dir()` - Path traversal protection
- `validate_note_path()` - Vault boundary enforcement
- `validate_db_path()` - Database path validation
- `sanitize_filename()` - Filename injection prevention

### 3. Enhanced Configuration

**Added support for:**
- OpenAI provider configuration
- Anthropic/Claude provider configuration
- API key validation at startup
- Environment variable support for all API keys

## Future Refactoring Opportunities

### Engine.py Modularization (943 lines)

**Recommended Split:**

1. **`sync/handlers/note_scanner.py`** (~250 lines)
   - Extract `_scan_obsidian_notes()` method
   - Move note discovery and parsing logic
   - Include progress tracking for scanning

2. **`sync/handlers/card_generator.py`** (~200 lines)
   - Extract `_generate_cards_with_agents()` method
   - Extract `_generate_card()` method
   - Move LLM interaction logic

3. **`sync/handlers/change_detector.py`** (~150 lines)
   - Extract `_determine_actions()` method
   - Extract `_fetch_anki_state()` method
   - Move change detection logic

4. **`sync/handlers/change_applier.py`** (~150 lines)
   - Extract `_apply_changes()` method
   - Extract `_create_card()`, `_update_card()`, `_delete_card()` methods
   - Move Anki modification logic

5. **`sync/handlers/display.py`** (~50 lines)
   - Extract `_print_plan()` method
   - Move UI/display logic

**Implementation Pattern:**

```python
# sync/handlers/note_scanner.py
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..engine import SyncEngine

class NoteScanner:
    """Handles note scanning and discovery."""

    def __init__(self, engine: "SyncEngine"):
        self.engine = engine
        self.config = engine.config
        self.db = engine.db
        self.progress = engine.progress

    def scan_notes(self, sample_size: int | None = None,
                   incremental: bool = False) -> dict[str, Card]:
        """Scan vault and generate cards."""
        # Implementation here
        pass

# sync/engine.py
from .handlers import NoteScanner

class SyncEngine:
    def __init__(self, ...):
        # ...
        self.scanner = NoteScanner(self)

    def _scan_obsidian_notes(self, ...):
        """Delegate to scanner."""
        return self.scanner.scan_notes(...)
```

### CLI Command Extraction

**Remaining commands to extract:**

1. **`cli_commands/validate_handler.py`**
   - Extract `validate()` command (~50 lines)

2. **`cli_commands/init_handler.py`**
   - Extract `init()` command (~60 lines)

3. **`cli_commands/anki_handler.py`**
   - Extract `list_decks()` command
   - Extract `list_models()` command
   - Extract `show_model_fields()` command

4. **`cli_commands/export_handler.py`**
   - Extract `export()` command (~180 lines)

5. **`cli_commands/index_handler.py`**
   - Extract `show_index()` command
   - Extract `show_progress()` command
   - Extract `clean_progress()` command

6. **`cli_commands/format_handler.py`**
   - Extract `format()` command

## Refactoring Best Practices

### 1. Maintain Backward Compatibility
- Keep existing public APIs
- Use delegation pattern initially
- Deprecate old methods gradually

### 2. Test Coverage
- Write tests before refactoring
- Ensure tests pass after each step
- Add integration tests

### 3. Incremental Refactoring
- One module at a time
- Commit after each successful refactor
- Keep PRs focused and reviewable

### 4. Documentation
- Update docstrings
- Document new module responsibilities
- Keep this guide up to date

## Module Organization Principles

### Single Responsibility Principle
Each module should have one primary responsibility:
- ✅ Good: `path_validator.py` - validates paths
- ❌ Bad: `utils.py` - does everything

### Clear Dependencies
- Avoid circular dependencies
- Use TYPE_CHECKING for type hints
- Keep dependency graph shallow

### Testability
- Make functions/methods easy to test
- Minimize external dependencies
- Use dependency injection

## Metrics

### Before Refactoring
| File | Lines | Functions | Classes | Complexity |
|------|-------|-----------|---------|------------|
| cli.py | 1,006 | 14 | 0 | High |
| engine.py | 943 | 13 | 1 | Very High |

### After Partial Refactoring
| File | Lines | Status |
|------|-------|--------|
| cli.py | ~1,006 | Original (to be updated) |
| cli_commands/shared.py | 64 | ✅ New |
| cli_commands/sync_handler.py | 196 | ✅ New |
| utils/path_validator.py | 214 | ✅ New |

### Target (Full Refactoring)
| Module | Est. Lines | Priority |
|--------|------------|----------|
| cli.py (main) | ~150 | - |
| sync/engine.py (orchestrator) | ~200 | - |
| sync/handlers/note_scanner.py | ~250 | High |
| sync/handlers/card_generator.py | ~200 | High |
| sync/handlers/change_detector.py | ~150 | Medium |
| sync/handlers/change_applier.py | ~150 | Medium |
| cli_commands/* | ~600 | Medium |

## Next Steps

1. **Immediate:**
   - Update cli.py to use sync_handler
   - Add tests for sync_handler
   - Document new patterns

2. **Short-term (1-2 weeks):**
   - Extract note scanner from engine.py
   - Extract card generator from engine.py
   - Add integration tests

3. **Medium-term (1 month):**
   - Complete engine.py refactoring
   - Extract remaining CLI commands
   - Update documentation

4. **Long-term (3 months):**
   - Consider async/await refactoring
   - Performance optimization
   - Microservices consideration for agents

## Performance Considerations

### Async/Await Refactoring
- **Current:** Synchronous I/O operations
- **Target:** Async HTTP clients for LLM APIs
- **Benefit:** 2-3x faster for parallel card generation
- **Effort:** High (requires updating all providers)

**Recommended approach:**
1. Start with provider interfaces
2. Update AnkiClient to async
3. Update engine orchestration
4. Update CLI to use asyncio.run()

## Questions?

Contact the maintainers or open an issue on GitHub.
