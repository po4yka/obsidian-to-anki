# Using RAG (Retrieval-Augmented Generation)

RAG adds vector embeddings for semantic search over your vault. It enables
duplicate detection, context enrichment during card generation, and few-shot
examples drawn from existing cards. ChromaDB is used for vector storage.

## Prerequisites

- A working sync setup (see [first-sync.md](first-sync.md))
- An embedding model accessible through your LLM provider
- Sufficient disk space for the ChromaDB index and 500 MB DiskCache

## Step 1: Enable RAG

Add the following to your `config.yaml`:

```yaml
rag_enabled: true
```

RAG is disabled by default. All other RAG settings have sensible defaults
and are optional.

## Step 2: Configure settings

Full configuration reference with defaults:

```yaml
# Required: enable RAG
rag_enabled: true                              # default: false

# Storage
rag_db_path: ".chroma_db"                      # relative to data_dir

# Embedding model
rag_embedding_model: "openai/text-embedding-3-small"

# Chunking
rag_chunk_size: 1000                           # range: 100-10000
rag_chunk_overlap: 200                         # range: 0-500

# Search
rag_search_k: 5                                # range: 1-20, results returned
rag_similarity_threshold: 0.85                 # range: 0.0-1.0

# Behavior toggles (all default to true)
rag_index_on_sync: true                        # auto-index during sync
rag_context_enrichment: true                   # enrich card generation
rag_duplicate_detection: true                  # detect duplicate cards
rag_few_shot_examples: true                    # provide examples to LLM
```

## Step 3: Build the index

Create vector embeddings for your vault:

```bash
obsidian-anki-sync rag index
```

To rebuild the index from scratch (deletes existing data first):

```bash
obsidian-anki-sync rag index --force
```

Verify the index was created:

```bash
obsidian-anki-sync rag stats
```

The stats command shows total chunks, unique files, embedding model, storage
path, topics, chunk types, and cache statistics.

## Searching the index

Run semantic queries against your indexed vault:

```bash
# Basic search (returns up to 5 results by default)
obsidian-anki-sync rag search "coroutine cancellation"

# Limit results and filter by topic
obsidian-anki-sync rag search "memory management" --limit 10 --topic Android

# Set minimum similarity score (default: 0.3)
obsidian-anki-sync rag search "garbage collection" --min-sim 0.5
```

## Detecting duplicates

Check whether a question already exists in your vault before writing a new note:

```bash
# Default threshold: 0.85
obsidian-anki-sync rag similar "What is a Python decorator?"

# Stricter matching
obsidian-anki-sync rag similar "What is a Python decorator?" --threshold 0.95
```

## RAG during sync

When `rag_enabled: true` and `rag_index_on_sync: true`, every sync
automatically:

1. Indexes new and changed notes into the vector store
2. Enriches card generation with semantic context from related notes
3. Detects potential duplicates before creating new cards
4. Provides few-shot examples to the LLM for better card quality

Run a sync with RAG active:

```bash
obsidian-anki-sync sync
```

Disable individual features without turning off RAG entirely:

```yaml
rag_context_enrichment: false   # skip context enrichment
rag_duplicate_detection: false  # skip duplicate checks
rag_few_shot_examples: false    # skip few-shot examples
rag_index_on_sync: false        # skip auto-indexing (manual only)
```

## Resetting the index

Delete all indexed data and start fresh:

```bash
# Interactive confirmation
obsidian-anki-sync rag reset

# Skip confirmation prompt
obsidian-anki-sync rag reset --yes
```

## Command reference

| Command | Description |
|---------|-------------|
| `rag index [--force]` | Create or update vault embeddings. `--force` rebuilds from scratch. |
| `rag search <query> [--limit N] [--topic <name>] [--min-sim 0.3]` | Semantic search. Default limit 5, min similarity 0.3. |
| `rag similar <question> [--threshold 0.85]` | Duplicate detection. Default threshold 0.85. |
| `rag stats` | Show index statistics: chunks, files, model, storage, cache. |
| `rag reset [--yes]` | Delete all indexed data. `--yes` skips confirmation. |

## Performance notes

RAG operations are backed by a 500 MB DiskCache with LRU eviction. Repeated
queries against the same content are served from cache. For tuning chunk size
and overlap for large vaults, see [performance-tuning.md](performance-tuning.md).

## Next steps

- [First sync walkthrough](first-sync.md)
- [Performance tuning](performance-tuning.md)
