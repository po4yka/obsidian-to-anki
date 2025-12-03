#!/bin/bash
# clean_sync_state.sh - Clean up all sync state for a fresh run
#
# This script cleans:
# 1. Redis: arq job queue, result queues, and any other sync-related keys
# 2. SQLite state database: sync sessions, progress tracking
# 3. Log files (optional)
# 4. Cache directories (optional)
#
# Usage:
#   ./scripts/clean_sync_state.sh          # Clean Redis + DB
#   ./scripts/clean_sync_state.sh --all    # Clean everything including logs and cache
#   ./scripts/clean_sync_state.sh --dry-run # Show what would be cleaned

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
STATE_DB="${STATE_DB:-$PROJECT_DIR/.sync_state.db}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_DIR/.cache}"
CHROMA_DIR="${CHROMA_DIR:-$PROJECT_DIR/.chroma_db}"

# Parse arguments
DRY_RUN=false
CLEAN_ALL=false
CLEAN_LOGS=false
CLEAN_CACHE=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --all)
            CLEAN_ALL=true
            CLEAN_LOGS=true
            CLEAN_CACHE=true
            shift
            ;;
        --logs)
            CLEAN_LOGS=true
            shift
            ;;
        --cache)
            CLEAN_CACHE=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Clean up sync state for a fresh run."
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be cleaned without actually cleaning"
            echo "  --all        Clean everything (Redis, DB, logs, cache)"
            echo "  --logs       Also clean log files"
            echo "  --cache      Also clean cache directories"
            echo "  --force, -f  Skip confirmation prompt"
            echo "  --help, -h   Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  REDIS_URL    Redis connection URL (default: redis://localhost:6379)"
            echo "  STATE_DB     Path to SQLite state database (default: .sync_state.db)"
            echo "  LOG_DIR      Path to log directory (default: ./logs)"
            echo "  CACHE_DIR    Path to cache directory (default: ./.cache)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Header
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Obsidian-Anki Sync Cleanup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if $DRY_RUN; then
    echo -e "${YELLOW}DRY RUN MODE - No changes will be made${NC}"
    echo ""
fi

# Function to run command or show what would be run
run_cmd() {
    if $DRY_RUN; then
        echo -e "  ${YELLOW}Would run:${NC} $*"
    else
        eval "$@"
    fi
}

# Function to check if Redis is available
check_redis() {
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping &> /dev/null; then
            return 0
        fi
    fi
    return 1
}

# Kill running workers
echo -e "${GREEN}[1/6] Stopping running workers...${NC}"
if pgrep -f "arq obsidian_anki_sync.worker" > /dev/null 2>&1; then
    echo "  Found running workers, stopping them..."
    run_cmd "pkill -f 'arq obsidian_anki_sync.worker' || true"
    sleep 2
    # Force kill if still running
    if pgrep -f "arq obsidian_anki_sync.worker" > /dev/null 2>&1; then
        echo "  Force killing remaining workers..."
        run_cmd "pkill -9 -f 'arq obsidian_anki_sync.worker' || true"
    fi
    echo -e "  ${GREEN}Workers stopped${NC}"
else
    echo "  No workers running"
fi

# Clean Redis
echo ""
echo -e "${GREEN}[2/6] Cleaning Redis queues...${NC}"
if check_redis; then
    # Count keys before cleanup
    ARQ_KEYS=$(redis-cli KEYS "arq:*" 2>/dev/null | wc -l | tr -d ' ')
    RESULT_KEYS=$(redis-cli KEYS "obsidian_anki_sync:*" 2>/dev/null | wc -l | tr -d ' ')

    echo "  Found $ARQ_KEYS arq:* keys"
    echo "  Found $RESULT_KEYS obsidian_anki_sync:* keys"

    if [ "$ARQ_KEYS" -gt 0 ] || [ "$RESULT_KEYS" -gt 0 ]; then
        # Delete arq keys (job queue, results, etc.)
        if [ "$ARQ_KEYS" -gt 0 ]; then
            run_cmd "redis-cli KEYS 'arq:*' | xargs -r redis-cli DEL > /dev/null"
            echo -e "  ${GREEN}Deleted arq:* keys${NC}"
        fi

        # Delete obsidian_anki_sync keys (result queues)
        if [ "$RESULT_KEYS" -gt 0 ]; then
            run_cmd "redis-cli KEYS 'obsidian_anki_sync:*' | xargs -r redis-cli DEL > /dev/null"
            echo -e "  ${GREEN}Deleted obsidian_anki_sync:* keys${NC}"
        fi
    else
        echo "  No keys to clean"
    fi
else
    echo -e "  ${YELLOW}Redis not available, skipping${NC}"
fi

# Clean SQLite state database
echo ""
echo -e "${GREEN}[3/6] Cleaning SQLite state database...${NC}"
if [ -f "$STATE_DB" ]; then
    DB_SIZE=$(du -h "$STATE_DB" 2>/dev/null | cut -f1)
    echo "  Found state database: $STATE_DB ($DB_SIZE)"

    # Show session info before cleaning
    if command -v sqlite3 &> /dev/null && ! $DRY_RUN; then
        SESSION_COUNT=$(sqlite3 "$STATE_DB" "SELECT COUNT(*) FROM sync_sessions;" 2>/dev/null || echo "0")
        CARD_COUNT=$(sqlite3 "$STATE_DB" "SELECT COUNT(*) FROM cards;" 2>/dev/null || echo "0")
        echo "  Sessions: $SESSION_COUNT, Cards tracked: $CARD_COUNT"
    fi

    run_cmd "rm -f '$STATE_DB' '$STATE_DB-shm' '$STATE_DB-wal'"
    echo -e "  ${GREEN}Deleted state database${NC}"
else
    echo "  No state database found at $STATE_DB"
fi

# Clean LangGraph checkpoints (if exists)
echo ""
echo -e "${GREEN}[4/6] Cleaning LangGraph checkpoints...${NC}"
CHECKPOINT_DIR="$PROJECT_DIR/.langgraph_checkpoints"
if [ -d "$CHECKPOINT_DIR" ]; then
    CHECKPOINT_SIZE=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
    echo "  Found checkpoint directory: $CHECKPOINT_DIR ($CHECKPOINT_SIZE)"
    run_cmd "rm -rf '$CHECKPOINT_DIR'"
    echo -e "  ${GREEN}Deleted checkpoint directory${NC}"
else
    echo "  No checkpoint directory found"
fi

# Clean ChromaDB (vector store)
if $CLEAN_CACHE || $CLEAN_ALL; then
    echo ""
    echo -e "${GREEN}[5/6] Cleaning ChromaDB vector store...${NC}"
    if [ -d "$CHROMA_DIR" ]; then
        CHROMA_SIZE=$(du -sh "$CHROMA_DIR" 2>/dev/null | cut -f1)
        echo "  Found ChromaDB directory: $CHROMA_DIR ($CHROMA_SIZE)"
        run_cmd "rm -rf '$CHROMA_DIR'"
        echo -e "  ${GREEN}Deleted ChromaDB directory${NC}"
    else
        echo "  No ChromaDB directory found"
    fi

    # Clean diskcache
    DISKCACHE_DIR="$PROJECT_DIR/.diskcache"
    if [ -d "$DISKCACHE_DIR" ]; then
        DISKCACHE_SIZE=$(du -sh "$DISKCACHE_DIR" 2>/dev/null | cut -f1)
        echo "  Found diskcache directory: $DISKCACHE_DIR ($DISKCACHE_SIZE)"
        run_cmd "rm -rf '$DISKCACHE_DIR'"
        echo -e "  ${GREEN}Deleted diskcache directory${NC}"
    fi
else
    echo ""
    echo -e "${GREEN}[5/6] Skipping cache cleanup (use --cache or --all)${NC}"
fi

# Clean logs
if $CLEAN_LOGS || $CLEAN_ALL; then
    echo ""
    echo -e "${GREEN}[6/6] Cleaning log files...${NC}"
    if [ -d "$LOG_DIR" ]; then
        LOG_SIZE=$(du -sh "$LOG_DIR" 2>/dev/null | cut -f1)
        LOG_COUNT=$(find "$LOG_DIR" -type f -name "*.log*" 2>/dev/null | wc -l | tr -d ' ')
        echo "  Found log directory: $LOG_DIR ($LOG_SIZE, $LOG_COUNT files)"
        run_cmd "rm -f '$LOG_DIR'/*.log*"
        echo -e "  ${GREEN}Deleted log files${NC}"
    else
        echo "  No log directory found"
    fi
else
    echo ""
    echo -e "${GREEN}[6/6] Skipping log cleanup (use --logs or --all)${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
if $DRY_RUN; then
    echo -e "${YELLOW}DRY RUN COMPLETE - No changes were made${NC}"
    echo ""
    echo "Run without --dry-run to apply changes:"
    echo "  $0"
else
    echo -e "${GREEN}CLEANUP COMPLETE${NC}"
fi
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Start workers:  ./scripts/launch_workers.sh 10"
echo "  2. Run sync:       obsidian-anki-sync sync --use-queue --use-langgraph"
echo ""
