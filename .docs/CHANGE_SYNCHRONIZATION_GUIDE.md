# Change Synchronization Guide
## Intelligent Updates for Existing Anki Cards

## Overview

Synchronizing changes from Obsidian to existing Anki cards is one of the most critical and challenging aspects of the sync system. This guide covers strategies, best practices, and implementation details for intelligent change management.

---

## The Challenge

When updating existing cards, we must balance:

1. **Accuracy**: Ensuring updates improve card quality
2. **Safety**: Not disrupting spaced repetition learning progress
3. **Consistency**: Keeping Obsidian and Anki in sync
4. **User Control**: Allowing manual review when needed

### Common Problems

❌ **Naive Approach** (Always overwrite):
```python
# BAD: Blindly overwrite existing card
anki.update_card(note_id, new_fields)  # Loses user edits!
```

❌ **Too Conservative** (Never update):
```python
# BAD: Skip all updates
if card_exists:
    return "skip"  # Typos never get fixed!
```

✅ **Intelligent Approach** (Analyze and decide):
```python
# GOOD: Semantic analysis + smart decision
diff = enhanced_differ.compare(existing, proposed)
if diff.risk_level == "low" and diff.should_update:
    anki.update_card(note_id, changes)
```

---

## Change Types

### 1. Typo Fixes (Safe)

**Example:**
```diff
- Front: "What is the recieve buffer?"
+ Front: "What is the receive buffer?"
```

**Characteristics:**
- High text similarity (>95%)
- Spelling/grammar corrections only
- No semantic change

**Action**: ✅ Auto-approve

### 2. Clarifications (Usually Safe)

**Example:**
```diff
  Front: "What is Big O notation?"
- Back: "A way to describe algorithm complexity"
+ Back: "A mathematical notation that describes the limiting behavior
+       of a function when the argument tends toward infinity,
+       commonly used to analyze algorithm complexity"
```

**Characteristics:**
- Original content preserved
- Additional details added
- Core meaning unchanged

**Action**: ✅ Auto-approve (with confidence check)

### 3. Rephrasing (Review Recommended)

**Example:**
```diff
- Front: "What's the difference between TCP and UDP?"
+ Front: "Compare TCP and UDP protocols"
```

**Characteristics:**
- Same question, different wording
- Semantic similarity high (>85%)
- Learning cue changed

**Action**: ⚠️ Review recommended (may disrupt recall)

### 4. Content Addition (Review Recommended)

**Example:**
```diff
  Front: "What is a binary search tree?"
  Back: "A tree where left < parent < right"
+ Extra: "Time complexity: O(log n) average, O(n) worst case
+        Used in: Database indexes, compiler symbol tables"
```

**Characteristics:**
- Significant new information
- Expansion of answer
- Learning effort increased

**Action**: ⚠️ Review (especially for mature cards)

### 5. Meaning Change (Requires Review)

**Example:**
```diff
  Front: "What is the time complexity of quicksort?"
- Back: "O(n log n)"
+ Back: "O(n log n) average case, O(n²) worst case"
```

**Characteristics:**
- Answer fundamentally different
- May indicate error correction
- High learning impact

**Action**: 🛑 Manual review required

### 6. Complete Rewrite (Manual Review)

**Example:**
```diff
- Front: "What is REST?"
- Back: "An API design pattern"
+ Front: "Explain the six constraints of REST architecture"
+ Back: "1. Client-Server, 2. Stateless, 3. Cacheable,
+       4. Layered System, 5. Code on Demand, 6. Uniform Interface"
```

**Characteristics:**
- Low text similarity (<70%)
- Different question or answer
- Essentially a new card

**Action**: 🛑 Manual review or create new card

---

## Semantic Diff Analysis

The **Enhanced Card Differ** uses LLM to understand changes semantically:

### How It Works

```
┌─────────────────────────────────────────┐
│  1. Text-based diff                     │
│     - Compute similarity (SequenceMatcher)
│     - Identify changed fields           │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│  2. LLM Semantic Analysis               │
│     - Classify change type              │
│     - Assess severity                   │
│     - Evaluate learning impact          │
│     - Provide recommendation            │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│  3. Policy-based Decision               │
│     - Check update policies             │
│     - Detect conflicts                  │
│     - Determine risk level              │
│     - Approve/reject/review             │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│  4. Execute or Queue for Review         │
│     - Auto-apply low-risk changes       │
│     - Queue medium-risk for review      │
│     - Block high-risk changes           │
└─────────────────────────────────────────┘
```

### LLM Analysis Output

```json
{
  "field_analyses": {
    "Front": {
      "change_type": "typo_fix",
      "severity": "cosmetic",
      "recommendation": "approve",
      "reasoning": "Fixed spelling: 'recieve' → 'receive'",
      "semantic_similarity": 0.99,
      "preserves_learning": true
    },
    "Back": {
      "change_type": "clarification",
      "severity": "content",
      "recommendation": "approve",
      "reasoning": "Added complexity notation, original answer preserved",
      "semantic_similarity": 0.85,
      "preserves_learning": true
    }
  },
  "overall_assessment": {
    "should_update": true,
    "risk_level": "low",
    "update_reason": "Minor typo fix + helpful clarification",
    "conflict_detected": false,
    "learning_impact": "minimal"
  }
}
```

---

## Conflict Detection

### What is a Conflict?

A conflict occurs when a card has been modified in **both** Obsidian and Anki since the last sync.

```
Timeline:
─────────────────────────────────────────────────>

t0: Last sync (Obsidian ← → Anki match)
    │
    ├─ t1: User edits card in Anki (fixes typo)
    │
    └─ t2: User updates Obsidian note (adds content)
         │
         └─ t3: Sync triggered ⚠️ CONFLICT!
```

### Detecting Conflicts

```python
def detect_conflict(
    last_sync_ts: datetime,
    last_obsidian_edit: datetime,
    last_anki_edit: datetime,
) -> bool:
    """Detect if card was edited in both systems."""
    obsidian_changed = last_obsidian_edit > last_sync_ts
    anki_changed = last_anki_edit > last_sync_ts

    return obsidian_changed and anki_changed
```

### Conflict Resolution Strategies

#### 1. Obsidian Wins (Default for Content)
```yaml
conflict_resolution: "obsidian_wins"
```
- Always use Obsidian version
- Anki edits are overwritten
- **Use case**: When Obsidian is the source of truth

#### 2. Anki Wins (Preserve Manual Edits)
```yaml
conflict_resolution: "anki_wins"
```
- Keep Anki version, skip update
- Obsidian changes are ignored
- **Use case**: When user heavily edits cards in Anki

#### 3. Manual Review (Safest)
```yaml
conflict_resolution: "manual"
```
- Require user to resolve conflict
- Show both versions side-by-side
- **Use case**: Production with critical data

#### 4. Merge (Advanced)
```yaml
conflict_resolution: "merge"
```
- Attempt to merge changes intelligently
- Use LLM to combine both versions
- **Use case**: When both versions add value

#### 5. Newest Wins (Simple)
```yaml
conflict_resolution: "newest_wins"
```
- Use whichever was edited most recently
- Timestamp-based decision
- **Use case**: Quick resolution without analysis

---

## Update Policies

### Field-Specific Policies

```yaml
langchain_agents:
  update_policies:
    # Safe to auto-update
    auto_update_fields:
      - "Extra"     # Additional info, low learning impact
      - "Hint"      # Hints can be improved freely

    # Require review
    review_required_fields:
      - "Front"     # Changes learning cue
      - "Back"      # Changes answer

    # Never update (preserve user edits)
    protected_fields:
      - "UserNotes"  # Custom field for user comments
```

### Severity-Based Policies

```yaml
langchain_agents:
  auto_approve_severity:
    - "cosmetic"  # Typos, formatting

  review_required_severity:
    - "content"   # Content changes
    - "structural"  # Model/deck changes

  block_severity:
    - "destructive"  # Data loss risk
```

### Card Maturity Policies

```yaml
langchain_agents:
  # More conservative with mature cards
  mature_card_threshold: 21  # days

  mature_card_policy:
    allow_content_updates: false  # Only typo fixes
    allow_front_changes: false    # Preserve learning cue
    require_review: true

  # More permissive with new cards
  new_card_policy:
    allow_content_updates: true
    allow_front_changes: true
    require_review: false
```

---

## Incremental Updates

### Field-Level Updates (Recommended)

Instead of replacing entire cards, update only changed fields:

```python
# GOOD: Update only changed fields
changes = differ.compare(existing, proposed)
for change in changes:
    if change.approved:
        anki.update_field(
            note_id=note_id,
            field_name=change.field,
            new_value=change.new_value
        )
```

**Benefits:**
- Preserves unchanged fields
- Reduces conflict risk
- More granular control

### Delta Updates (Advanced)

For text fields, apply delta patches instead of full replacement:

```python
import difflib

def apply_delta_update(old_text: str, new_text: str) -> str:
    """Apply minimal diff instead of full replacement."""
    diff = difflib.unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
    )
    # Apply patch...
    return patched_text
```

---

## Change History & Rollback

### Tracking Changes

```python
class ChangeHistory:
    """Track all card updates for rollback."""

    def record_update(
        self,
        note_id: int,
        field: str,
        old_value: str,
        new_value: str,
        timestamp: datetime,
        approved_by: str,  # "auto" or "user"
    ):
        """Record a change for potential rollback."""
        ...

    def rollback(self, note_id: int, to_timestamp: datetime):
        """Rollback card to previous version."""
        ...
```

### Database Schema

```sql
CREATE TABLE change_history (
    id INTEGER PRIMARY KEY,
    note_id INTEGER NOT NULL,
    field_name TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    change_type TEXT,  -- typo_fix, clarification, etc.
    severity TEXT,     -- cosmetic, content, structural
    approved_by TEXT,  -- auto, user:john, conflict_resolution
    applied BOOLEAN,
    timestamp DATETIME,
    rollback_available BOOLEAN DEFAULT 1
);

CREATE INDEX idx_change_history_note ON change_history(note_id);
CREATE INDEX idx_change_history_ts ON change_history(timestamp);
```

### Rollback Command

```bash
# View change history for a card
obsidian-anki-sync history --note-id 1234567890

# Rollback last change
obsidian-anki-sync rollback --note-id 1234567890

# Rollback to specific timestamp
obsidian-anki-sync rollback --note-id 1234567890 --to "2025-11-10T12:00:00Z"

# Rollback all changes in last hour
obsidian-anki-sync rollback --all --since "1 hour ago"
```

---

## User Review Interface

### CLI Review Mode

```bash
# Preview changes before applying
obsidian-anki-sync sync --dry-run --show-diffs

# Interactive review mode
obsidian-anki-sync sync --interactive

# Output:
# ┌─────────────────────────────────────────────────┐
# │ Card Update Review (1/5)                        │
# ├─────────────────────────────────────────────────┤
# │ Note: binary-search-tree.md                     │
# │ Slug: binary-search-tree-en                     │
# │ Risk: MEDIUM                                    │
# ├─────────────────────────────────────────────────┤
# │ Field: Back                                     │
# │ Type: content_addition                          │
# ├─────────────────────────────────────────────────┤
# │ - Old: "A tree where left < parent < right"    │
# │ + New: "A tree where left < parent < right     │
# │         Time complexity: O(log n) average"      │
# ├─────────────────────────────────────────────────┤
# │ [A]pprove  [R]eject  [S]kip  [Q]uit            │
# └─────────────────────────────────────────────────┘
```

### Batch Review

```bash
# Export pending changes to JSON
obsidian-anki-sync sync --dry-run --export-review review.json

# User reviews in editor or GUI

# Apply approved changes
obsidian-anki-sync apply-review review.json
```

### Review JSON Format

```json
{
  "review_session_id": "2025-11-10-001",
  "generated_at": "2025-11-10T15:30:00Z",
  "pending_changes": [
    {
      "note_id": 1234567890,
      "slug": "binary-search-tree-en",
      "changes": [
        {
          "field": "Back",
          "old_value": "A tree where left < parent < right",
          "new_value": "A tree where left < parent < right\nTime complexity: O(log n) average",
          "change_type": "content_addition",
          "severity": "content",
          "risk_level": "medium",
          "recommendation": "review",
          "reasoning": "Added complexity notation, changes learning scope"
        }
      ],
      "user_decision": null,  // To be filled by user: "approve", "reject", "modify"
      "user_notes": null
    }
  ]
}
```

---

## Best Practices

### 1. Start Conservative

```yaml
# Initial configuration (safe)
langchain_agents:
  allow_content_updates: false  # Only cosmetic changes
  allow_structural_updates: false
  min_qa_score: 0.9  # High quality bar
  conflict_resolution: "manual"
```

Then gradually loosen as you build trust:

```yaml
# After testing (more permissive)
langchain_agents:
  allow_content_updates: true
  min_qa_score: 0.8
  conflict_resolution: "obsidian_wins"
```

### 2. Use Dry-Run Mode Extensively

```bash
# Always test before applying
obsidian-anki-sync sync --dry-run --use-langchain-agents

# Review the changes
cat .logs/sync_preview.log

# Apply if satisfied
obsidian-anki-sync sync --use-langchain-agents
```

### 3. Monitor Change Patterns

```bash
# Generate change report
obsidian-anki-sync report --changes --last-7-days

# Output:
# Change Summary (Last 7 Days)
# ════════════════════════════════════════
# Total Updates:        145
# Auto-Approved:        120 (82.8%)
# Manual Review:         20 (13.8%)
# Rejected:               5 (3.4%)
#
# By Type:
#   typo_fix:            85 (58.6%)
#   clarification:       30 (20.7%)
#   content_addition:    15 (10.3%)
#   rephrasing:          10 (6.9%)
#   meaning_change:       5 (3.4%)
```

### 4. Protect Mature Cards

```python
def should_auto_update(card: Card) -> bool:
    """More conservative with mature cards."""
    days_since_creation = (datetime.now() - card.created_at).days

    if days_since_creation > 30:  # Mature card
        # Only allow cosmetic changes
        return change.severity == ChangeSeverity.COSMETIC
    else:  # Young card
        # Allow content changes
        return change.severity in (ChangeSeverity.COSMETIC, ChangeSeverity.CONTENT)
```

### 5. Review Before Major Changes

```bash
# Before updating Obsidian notes significantly
git commit -m "Before major note restructure"

# Run sync in dry-run
obsidian-anki-sync sync --dry-run

# Review impacts
obsidian-anki-sync report --affected-cards

# If concerned, backup Anki
Tools → Create Backup

# Then proceed
obsidian-anki-sync sync
```

---

## Troubleshooting

### Problem: Too Many Manual Reviews

**Symptoms**: Most updates require manual review, slowing workflow

**Solutions**:
1. Lower `min_qa_score` threshold
2. Enable `allow_content_updates`
3. Use `conflict_resolution: "obsidian_wins"`
4. Improve note quality (fewer ambiguous changes)

### Problem: Unwanted Overwrites

**Symptoms**: Anki edits being lost

**Solutions**:
1. Set `conflict_resolution: "manual"` or `"anki_wins"`
2. Use protected_fields for user-editable fields
3. Implement bidirectional sync (future feature)
4. Review change history and rollback if needed

### Problem: Changes Not Detected

**Symptoms**: Updates in Obsidian not appearing in Anki

**Solutions**:
1. Check sync state DB: `obsidian-anki-sync index`
2. Verify content hash calculation
3. Run full re-index: `obsidian-anki-sync sync --full-reindex`
4. Check logs for skip reasons: `grep "skip" .logs/sync.log`

### Problem: Slow Sync Performance

**Symptoms**: Semantic diff is too slow

**Solutions**:
1. Disable semantic analysis for bulk updates:
   ```yaml
   use_semantic_diff: false  # For initial import
   ```
2. Use faster LLM for diff analysis:
   ```yaml
   diff_model: "gpt-3.5-turbo"  # Instead of GPT-4
   ```
3. Batch diff operations (process multiple cards per LLM call)
4. Cache diff results for unchanged note content

---

## Future Enhancements

### 1. Bidirectional Sync
- Detect changes in Anki
- Sync back to Obsidian
- Three-way merge

### 2. Intelligent Merge
- Use LLM to merge Obsidian + Anki changes
- Preserve best of both versions
- Generate merged card

### 3. A/B Testing
- Create variant cards
- Test which version performs better
- Auto-select winning version

### 4. Learning Impact Prediction
- Predict effect of change on retention
- Use Anki statistics
- Warn if change likely to hurt retention

### 5. Suggested Changes
- LLM suggests improvements to existing cards
- User reviews and approves
- Continuous quality improvement

---

## Summary

### Key Principles

1. **Safety First**: Protect user's learning progress
2. **Transparency**: Show what changed and why
3. **User Control**: Allow override of any decision
4. **Reversibility**: Support rollback of changes
5. **Intelligence**: Use LLM to understand semantic meaning

### Decision Matrix

| Change Type | Auto-Approve? | Risk Level | Action |
|-------------|---------------|------------|--------|
| Typo fix | ✅ Yes | Low | Apply |
| Clarification | ✅ Yes (if high similarity) | Low-Medium | Apply |
| Rephrasing | ⚠️ Review | Medium | Queue for review |
| Content addition | ⚠️ Review | Medium | Queue for review |
| Meaning change | 🛑 No | High | Manual review required |
| Complete rewrite | 🛑 No | High | Manual review or new card |
| Structural change | 🛑 No (unless enabled) | High | Manual review required |
| Conflict detected | 🛑 No (unless policy) | High | Resolution strategy |

---

**Version**: 1.0
**Last Updated**: 2025-11-10
