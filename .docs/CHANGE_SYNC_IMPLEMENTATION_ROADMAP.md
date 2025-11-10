# Change Synchronization Implementation Roadmap

## Overview

This document outlines the implementation plan for intelligent change synchronization in the Obsidian → Anki pipeline.

---

## Phase 1: Foundation (Week 1)

### 1.1 Enhanced Data Models

**File**: `src/obsidian_anki_sync/agents/langchain/models.py`

Add new models for change tracking:

```python
class ChangeType(str, Enum):
    """Classification of change types."""
    TYPO_FIX = "typo_fix"
    CLARIFICATION = "clarification"
    CONTENT_ADDITION = "content_addition"
    # ... etc

class ChangeRecord(BaseModel):
    """Record of a single change."""
    timestamp: datetime
    note_id: int
    field: str
    old_value: str
    new_value: str
    change_type: ChangeType
    severity: ChangeSeverity
    approved: bool
    approved_by: str  # "auto", "user", "conflict_resolution"
    rollback_available: bool = True

class ConflictInfo(BaseModel):
    """Information about a detected conflict."""
    detected: bool
    last_sync: Optional[datetime]
    last_obsidian_edit: Optional[datetime]
    last_anki_edit: Optional[datetime]
    resolution_strategy: str
    manual_review_required: bool
```

**Status**: ✅ Models defined (enhanced_card_differ.py has enums)

### 1.2 Change History Database

**File**: `src/obsidian_anki_sync/sync/change_history_db.py`

```python
class ChangeHistoryDB:
    """SQLite database for change tracking."""

    def __init__(self, db_path: str):
        """Initialize change history database."""
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        """Create tables for change tracking."""
        # CREATE TABLE change_history (...)
        # CREATE TABLE sync_timestamps (...)
        # CREATE TABLE conflict_resolutions (...)

    def record_change(self, change: ChangeRecord):
        """Record a change for history/rollback."""

    def get_changes(
        self,
        note_id: Optional[int] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[ChangeRecord]:
        """Get change history."""

    def get_last_sync(self, note_id: int) -> Optional[datetime]:
        """Get last sync timestamp for a note."""

    def rollback_to(self, note_id: int, timestamp: datetime) -> dict:
        """Get old values for rollback."""
```

**Dependencies**: SQLite, existing state_db.py pattern

**Estimated Effort**: 2 days

### 1.3 Timestamp Tracking

**File**: `src/obsidian_anki_sync/sync/state_db.py`

Add timestamp columns:

```sql
ALTER TABLE cards ADD COLUMN last_obsidian_edit DATETIME;
ALTER TABLE cards ADD COLUMN last_anki_edit DATETIME;
ALTER TABLE cards ADD COLUMN last_sync DATETIME;
```

Update methods:

```python
def update_card_timestamps(
    self,
    slug: str,
    last_obsidian_edit: datetime,
    last_sync: datetime,
):
    """Update timestamps after successful sync."""
```

**Estimated Effort**: 1 day

---

## Phase 2: Enhanced Card Differ (Week 2)

### 2.1 Integrate Enhanced Differ

**File**: `src/obsidian_anki_sync/agents/langchain/supervisor.py`

Replace basic CardDifferTool with EnhancedCardDiffer:

```python
from obsidian_anki_sync.agents.langchain.tools.enhanced_card_differ import (
    EnhancedCardDiffer,
    ConflictResolution,
)

class LangChainSupervisor:
    def __init__(self, config, ...):
        # ...

        # Use enhanced differ instead of basic
        self.card_differ = EnhancedCardDiffer(
            llm=self.qa_llm,  # Use same LLM as QA
            allow_content_updates=self.supervisor_config.allow_content_updates,
            conflict_resolution=ConflictResolution(
                config.conflict_resolution or "manual"
            ),
            track_history=True,
        )
```

**Estimated Effort**: 1 day

### 2.2 Add Timestamp Support to Diff

Update `process_note()` to pass timestamps:

```python
def process_note(self, note_context: NoteContext) -> CardDecision:
    # ...

    if note_context.existing_anki_note:
        # Get timestamps from state DB
        timestamps = self.state_db.get_card_timestamps(note_context.slug)

        diff_result = self.card_differ.compare(
            note_context.existing_anki_note,
            proposed_card,
            last_obsidian_sync=timestamps.get("last_sync"),
            last_anki_edit=timestamps.get("last_anki_edit"),
        )
```

**Estimated Effort**: 1 day

### 2.3 Configuration for Differ

**File**: `config.yaml`

Add configuration:

```yaml
langchain_agents:
  # Change synchronization settings
  conflict_resolution: "manual"  # manual, obsidian_wins, anki_wins, merge, newest_wins
  min_semantic_similarity: 0.85  # Min similarity for auto-approval
  track_change_history: true
  max_history_records: 10000

  # Field-specific policies
  auto_update_fields:
    - "Extra"
    - "Hint"

  review_required_fields:
    - "Front"
    - "Back"

  protected_fields:  # Never update
    - "UserNotes"

  # Maturity-based policies
  mature_card_threshold_days: 21
  mature_card_allow_content_updates: false
```

**Estimated Effort**: 0.5 days

---

## Phase 3: Conflict Detection (Week 3)

### 3.1 Anki Edit Timestamp Detection

**File**: `src/obsidian_anki_sync/anki/client.py`

Add method to get note modification time:

```python
def get_note_mod_time(self, note_id: int) -> Optional[datetime]:
    """Get last modification time of a note from Anki.

    Args:
        note_id: Note ID

    Returns:
        Last modification timestamp or None
    """
    note_info = self.notes_info([note_id])
    if note_info:
        mod_timestamp = note_info[0].get("mod", 0)
        # Convert Anki timestamp (seconds since epoch) to datetime
        return datetime.fromtimestamp(mod_timestamp)
    return None
```

**Estimated Effort**: 0.5 days

### 3.2 Integrate with State DB

Update sync flow to track Anki edit times:

```python
# In sync engine, after fetching Anki cards
for anki_note in anki_notes:
    mod_time = anki_client.get_note_mod_time(anki_note["noteId"])
    state_db.update_last_anki_edit(slug, mod_time)
```

**Estimated Effort**: 1 day

### 3.3 Conflict Resolution Handlers

**File**: `src/obsidian_anki_sync/sync/conflict_resolver.py`

```python
class ConflictResolver:
    """Handles conflict resolution strategies."""

    def resolve_conflict(
        self,
        existing: ExistingAnkiNote,
        proposed: ProposedCard,
        strategy: ConflictResolution,
    ) -> tuple[ProposedCard, str]:
        """Resolve a detected conflict.

        Returns:
            (resolved_card, resolution_reason)
        """
        if strategy == ConflictResolution.OBSIDIAN_WINS:
            return proposed, "Obsidian wins policy"

        elif strategy == ConflictResolution.ANKI_WINS:
            # Convert existing back to ProposedCard format
            return self._to_proposed_card(existing), "Anki wins policy"

        elif strategy == ConflictResolution.MERGE:
            return self._merge_cards(existing, proposed), "Merged versions"

        elif strategy == ConflictResolution.NEWEST_WINS:
            # Compare timestamps
            ...

        else:  # MANUAL
            raise ManualReviewRequired("Conflict requires manual resolution")

    def _merge_cards(self, existing, proposed) -> ProposedCard:
        """Use LLM to intelligently merge both versions."""
        # Implementation using LLM
        ...
```

**Estimated Effort**: 2 days

---

## Phase 4: Incremental Updates (Week 4)

### 4.1 Field-Level Update API

**File**: `src/obsidian_anki_sync/anki/client.py`

Enhance to support partial updates:

```python
def update_note_fields_selective(
    self,
    note_id: int,
    field_updates: dict[str, str],
) -> None:
    """Update only specified fields, leave others unchanged.

    Args:
        note_id: Note ID
        field_updates: Dict of {field_name: new_value} for only changed fields
    """
    # Get current fields
    current_note = self.notes_info([note_id])[0]
    current_fields = current_note["fields"]

    # Merge with updates
    updated_fields = {**current_fields, **field_updates}

    # Update
    self.update_note_fields(note_id, updated_fields)
```

**Estimated Effort**: 0.5 days

### 4.2 Apply Only Approved Changes

**File**: `src/obsidian_anki_sync/sync/engine.py`

Modify sync logic to apply incremental updates:

```python
def _apply_card_update(self, decision: CardDecision, note_id: int):
    """Apply only the approved changes from CardDecision."""

    if not decision.diff or not decision.diff.should_update:
        return

    # Collect approved field changes
    field_updates = {}
    for change in decision.diff.changes:
        # Only update approved, non-structural changes
        if (change.field not in ("model_name", "deck_name", "tags")
            and change.severity != ChangeSeverity.STRUCTURAL):
            field_updates[change.field] = change.new_value

    # Apply field updates
    if field_updates:
        self.anki_client.update_note_fields_selective(note_id, field_updates)

    # Handle tags separately (if changed and approved)
    tag_change = next(
        (c for c in decision.diff.changes if c.field == "tags"),
        None
    )
    if tag_change:
        self.anki_client.update_note_tags(note_id, tag_change.new_value)

    # Record in change history
    for field, new_value in field_updates.items():
        self.change_history_db.record_change(...)
```

**Estimated Effort**: 2 days

---

## Phase 5: User Review Interface (Week 5)

### 5.1 CLI Interactive Review

**File**: `src/obsidian_anki_sync/cli.py`

Add `--interactive` flag:

```python
@cli.command()
@click.option("--interactive", is_flag=True, help="Review changes interactively")
def sync(interactive: bool, ...):
    """Run sync with optional interactive review."""

    if interactive:
        # Run in dry-run mode first
        decisions = engine.sync(dry_run=True, ...)

        # Show interactive review UI
        approved_decisions = review_ui.review_changes(decisions)

        # Apply only approved changes
        engine.apply_decisions(approved_decisions)
    else:
        # Normal sync
        engine.sync(...)
```

**Estimated Effort**: 2 days

### 5.2 Review UI Implementation

**File**: `src/obsidian_anki_sync/ui/review_interface.py`

```python
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

class ReviewInterface:
    """Interactive CLI for reviewing card changes."""

    def review_changes(self, decisions: list[CardDecision]) -> list[CardDecision]:
        """Show each change and get user approval."""

        approved = []

        for i, decision in enumerate(decisions, 1):
            if decision.action not in (SyncAction.UPDATE, SyncAction.CREATE):
                continue

            console.print(f"\n[bold]Change {i}/{len(decisions)}[/bold]")

            # Show diff table
            self._show_diff_table(decision)

            # Get user input
            choice = Prompt.ask(
                "Action",
                choices=["a", "r", "s", "q"],
                default="a"
            )

            if choice == "a":  # Approve
                approved.append(decision)
            elif choice == "r":  # Reject
                continue
            elif choice == "s":  # Skip this one
                continue
            elif choice == "q":  # Quit
                break

        return approved

    def _show_diff_table(self, decision: CardDecision):
        """Show side-by-side diff of changes."""
        # Implementation using rich Table
        ...
```

**Estimated Effort**: 3 days

### 5.3 Batch Review Export/Import

**File**: `src/obsidian_anki_sync/cli.py`

```python
@cli.command()
@click.option("--export-review", type=click.Path())
def sync(export_review: Optional[str], ...):
    """Export pending changes for external review."""

    decisions = engine.sync(dry_run=True, ...)

    if export_review:
        review_data = {
            "review_session_id": generate_session_id(),
            "generated_at": datetime.utcnow().isoformat(),
            "pending_changes": [d.model_dump() for d in decisions],
        }

        with open(export_review, "w") as f:
            json.dump(review_data, f, indent=2)

@cli.command()
@click.argument("review_file", type=click.Path(exists=True))
def apply_review(review_file: str):
    """Apply user-reviewed changes from JSON file."""

    with open(review_file) as f:
        review_data = json.load(f)

    # Extract approved changes
    approved = [
        d for d in review_data["pending_changes"]
        if d.get("user_decision") == "approve"
    ]

    # Apply
    engine.apply_decisions(approved)
```

**Estimated Effort**: 2 days

---

## Phase 6: Rollback System (Week 6)

### 6.1 Rollback Command

**File**: `src/obsidian_anki_sync/cli.py`

```python
@cli.command()
@click.option("--note-id", type=int)
@click.option("--to-timestamp", type=str)
@click.option("--last-n", type=int, default=1, help="Rollback last N changes")
def rollback(note_id: int, to_timestamp: Optional[str], last_n: int):
    """Rollback changes to a card."""

    if to_timestamp:
        # Rollback to specific timestamp
        ts = datetime.fromisoformat(to_timestamp)
        changes = change_history_db.rollback_to(note_id, ts)
    else:
        # Rollback last N changes
        changes = change_history_db.get_last_n_changes(note_id, last_n)

    # Apply rollback
    for change in reversed(changes):
        anki_client.update_field(
            note_id=note_id,
            field=change.field,
            value=change.old_value,
        )

    console.print(f"✅ Rolled back {len(changes)} changes")

@cli.command()
@click.option("--note-id", type=int)
@click.option("--limit", type=int, default=20)
def history(note_id: int, limit: int):
    """Show change history for a card."""

    changes = change_history_db.get_changes(note_id=note_id, limit=limit)

    # Display as table
    table = Table(title=f"Change History (Note {note_id})")
    table.add_column("Timestamp")
    table.add_column("Field")
    table.add_column("Change Type")
    table.add_column("Approved By")

    for change in changes:
        table.add_row(
            change.timestamp.strftime("%Y-%m-%d %H:%M"),
            change.field,
            change.change_type.value,
            change.approved_by,
        )

    console.print(table)
```

**Estimated Effort**: 2 days

---

## Phase 7: Testing & Validation (Week 7)

### 7.1 Unit Tests

**Files**: `tests/agents/langchain/test_enhanced_card_differ.py`

```python
def test_typo_fix_detection():
    """Test detection of typo fixes."""
    differ = EnhancedCardDiffer()

    existing = ExistingAnkiNote(...)
    proposed = ProposedCard(...)  # with typo fixed

    result = differ.compare(existing, proposed)

    assert result.should_update == True
    assert result.risk_level == RiskLevel.LOW
    # Should classify as typo_fix

def test_conflict_detection():
    """Test conflict detection."""
    last_sync = datetime(2025, 11, 1)
    last_obsidian_edit = datetime(2025, 11, 5)
    last_anki_edit = datetime(2025, 11, 3)

    conflict = differ._detect_conflict(last_sync, last_obsidian_edit, last_anki_edit)

    assert conflict == True

def test_semantic_analysis_with_mocked_llm():
    """Test semantic analysis with mocked LLM responses."""
    # Mock LLM to return specific analysis
    # Verify correct classification
    ...
```

**Estimated Effort**: 3 days

### 7.2 Integration Tests

Test complete sync flow with changes:

```python
def test_end_to_end_update_flow():
    """Test complete update flow from Obsidian change to Anki update."""

    # 1. Create initial card
    # 2. Simulate Obsidian edit (typo fix)
    # 3. Run sync
    # 4. Verify:
    #    - Change detected
    #    - Classified correctly
    #    - Applied to Anki
    #    - Recorded in history
    ...

def test_conflict_resolution():
    """Test conflict resolution strategies."""
    # Simulate edits in both systems
    # Test each resolution strategy
    ...

def test_rollback():
    """Test rollback functionality."""
    # Make changes
    # Record history
    # Rollback
    # Verify state restored
    ...
```

**Estimated Effort**: 3 days

---

## Phase 8: Documentation & Polish (Week 8)

### 8.1 User Documentation

- Update main README with change sync features
- Add troubleshooting section
- Create video walkthrough

### 8.2 Performance Optimization

- Cache semantic diff results
- Batch LLM calls for multiple cards
- Optimize database queries

### 8.3 Monitoring & Reporting

```bash
obsidian-anki-sync report --changes --last-30-days
```

Output statistics on change patterns, auto-approval rates, etc.

**Estimated Effort**: 5 days

---

## Total Timeline

- **Phase 1**: Foundation (Week 1)
- **Phase 2**: Enhanced Differ (Week 2)
- **Phase 3**: Conflict Detection (Week 3)
- **Phase 4**: Incremental Updates (Week 4)
- **Phase 5**: Review Interface (Week 5)
- **Phase 6**: Rollback (Week 6)
- **Phase 7**: Testing (Week 7)
- **Phase 8**: Polish (Week 8)

**Total**: ~8 weeks for complete implementation

---

## Quick Win Priorities

If implementing incrementally, prioritize in this order:

### Priority 1 (Most Impact)
1. ✅ Enhanced Card Differ with LLM analysis (Already implemented!)
2. Change History Database
3. Incremental field updates

### Priority 2 (Safety)
4. Conflict detection
5. Rollback system
6. Protected fields

### Priority 3 (UX)
7. Interactive review
8. Batch review export/import
9. Change reports

---

## Current Status

✅ **Completed**:
- Enhanced Card Differ implementation
- Data models for change tracking
- Documentation (this guide + CHANGE_SYNCHRONIZATION_GUIDE.md)

🔄 **In Progress**:
- Integration with supervisor

📋 **Remaining**:
- Change history database
- Timestamp tracking
- Conflict resolution
- Review UI
- Rollback system
- Full testing suite

---

**Next Immediate Steps**:

1. Create change_history_db.py
2. Add timestamp columns to state_db
3. Integrate EnhancedCardDiffer with supervisor
4. Test with real-world note updates
5. Iterate based on results
