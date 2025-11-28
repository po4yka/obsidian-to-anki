# Change Synchronization Guide

## Overview

Synchronizing changes from Obsidian to existing Anki cards is one of the most critical and challenging aspects of the sync system. This guide covers strategies for intelligent change management that balances accuracy, safety, consistency, and user control.

## The Challenge

When updating existing cards, we must balance:

1. **Accuracy**: Ensuring updates improve card quality
2. **Safety**: Not disrupting spaced repetition learning progress
3. **Consistency**: Keeping Obsidian and Anki in sync
4. **User Control**: Allowing manual review when needed

### Common Approaches

**Naive Approach** (Always overwrite):

```python
# BAD: Blindly overwrite existing card
anki.update_card(note_id, new_fields)  # Loses user edits!
```

**Too Conservative** (Never update):

```python
# BAD: Skip all updates
if card_exists:
    return "skip"  # Typos never get fixed!
```

**Intelligent Approach** (Analyze and decide):

```python
# GOOD: Semantic analysis + smart decision
diff = enhanced_differ.compare(existing, proposed)
if diff.risk_level == "low" and diff.should_update:
    anki.update_card(note_id, changes)
```

## Change Types

### 1. Typo Fixes (Safe)

**Example:**

```diff
- Front: "What is the recieve buffer?"
+ Front: "What is the receive buffer?"
```

**Characteristics:**

-   High text similarity (>95%)
-   Spelling/grammar corrections only
-   No semantic change

**Action**: Auto-approve

### 2. Clarifications (Usually Safe)

**Example:**

```diff
  Front: "What is Big O notation?"
- Back: "A way to describe algorithm complexity"
+ Back: "A mathematical notation that describes the limiting behavior
        of a function when the argument tends toward infinity,
        commonly used to analyze algorithm complexity"
```

**Characteristics:**

-   Original content preserved
-   Additional details added
-   Core meaning unchanged

**Action**: Auto-approve (with confidence check)

### 3. Rephrasing (Review Recommended)

**Example:**

```diff
- Front: "What's the difference between TCP and UDP?"
+ Front: "Compare TCP and UDP protocols"
```

**Characteristics:**

-   Same question, different wording
-   Semantic similarity high (>85%)
-   Learning cue changed

**Action**: Review recommended (may disrupt recall)

### 4. Content Addition (Review Recommended)

**Example:**

```diff
  Front: "What is a binary search tree?"
  Back: "A tree where left < parent < right"
+ Extra: "Time complexity: O(log n) average, O(n) worst case
         Used in: Database indexes, compiler symbol tables"
```

**Characteristics:**

-   Significant new information
-   Expansion of answer
-   Learning effort increased

**Action**: Review (especially for mature cards)

### 5. Meaning Change (Requires Review)

**Example:**

```diff
  Front: "What is the time complexity of quicksort?"
- Back: "O(n log n)"
+ Back: "O(n log n) average case, O(nÂ²) worst case"
```

**Characteristics:**

-   Answer fundamentally different
-   May indicate error correction
-   High learning impact

**Action**: Manual review required

### 6. Complete Rewrite (Manual Review)

**Example:**

```diff
- Front: "What is REST?"
- Back: "An API design pattern"
+ Front: "Explain the six constraints of REST architecture"
+ Back: "1. Client-Server, 2. Stateless, 3. Cacheable,
        4. Layered System, 5. Code on Demand, 6. Uniform Interface"
```

**Characteristics:**

-   Low text similarity (<70%)
-   Different question or answer
-   Essentially a new card

**Action**: Manual review or create new card

## Decision Matrix

| Change Type       | Auto-Approve?            | Risk Level | Action                    |
| ----------------- | ------------------------ | ---------- | ------------------------- |
| Typo fix          | Yes                      | Low        | Apply                     |
| Clarification     | Yes (if high similarity) | Low-Medium | Apply                     |
| Rephrasing        | Review                   | Medium     | Queue for review          |
| Content addition  | Review                   | Medium     | Queue for review          |
| Meaning change    | No                       | High       | Manual review required    |
| Complete rewrite  | No                       | High       | Manual review or new card |
| Structural change | No (unless enabled)      | High       | Manual review required    |
| Conflict detected | No (unless policy)       | High       | Resolution strategy       |

## Best Practices

### 1. Start Conservative

```yaml
# Initial configuration (safe)
langchain_agents:
    allow_content_updates: false # Only cosmetic changes
    allow_structural_updates: false
    min_qa_score: 0.9 # High quality bar
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
#
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

## Key Principles

1. **Safety First**: Protect user's learning progress
2. **Transparency**: Show what changed and why
3. **User Control**: Allow override of any decision
4. **Reversibility**: Support rollback of changes
5. **Intelligence**: Use semantic analysis to understand changes

## Related Documentation

-   **[Advanced Synchronization](synchronization-advanced.md)** - Semantic diff analysis, conflict resolution, policies
-   **[Synchronization API](../REFERENCE/sync-api.md)** - Technical API reference and configuration
-   **[Configuration](configuration.md)** - Setting up sync preferences

---

**Version**: 1.0
**Last Updated**: November 28, 2025
