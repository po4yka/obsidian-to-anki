#!/bin/bash
# clean_sync_state.sh - Comprehensive cleanup script for fresh sync starts
#
# This script performs a complete cleanup of all sync-related state:
# 1. Kill running workers (multiple patterns)
# 2. Redis: Complete FLUSHDB or selective key cleanup
# 3. SQLite state database: sync sessions, progress tracking
# 4. LangGraph checkpoints and memory storage
# 5. Vector stores: ChromaDB, diskcache
# 6. Cache directories: .cache, .agent_memory, .debug_artifacts
# 7. Log directories: logs, validation_logs
# 8. Problematic notes and temporary files
#
# Usage:
#   ./scripts/clean_sync_state.sh              # Clean Redis + DB + essential dirs
#   ./scripts/clean_sync_state.sh --all        # Clean everything including logs and cache
#   ./scripts/clean_sync_state.sh --flushdb    # Use FLUSHDB instead of selective Redis cleanup
#   ./scripts/clean_sync_state.sh --dry-run    # Show what would be cleaned
#   ./scripts/clean_sync_state.sh --nuclear    # Complete nuclear option (everything + force)

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
AGENT_MEMORY="${AGENT_MEMORY:-$PROJECT_DIR/.agent_memory}"
DEBUG_ARTIFACTS="${DEBUG_ARTIFACTS:-$PROJECT_DIR/.debug_artifacts}"
VALIDATION_LOGS="${VALIDATION_LOGS:-$PROJECT_DIR/validation_logs}"
PROBLEMATIC_NOTES="${PROBLEMATIC_NOTES:-$PROJECT_DIR/problematic_notes}"

# Parse arguments
DRY_RUN=false
CLEAN_ALL=false
CLEAN_LOGS=false
CLEAN_CACHE=false
FORCE=false
FLUSHDB=false
NUCLEAR=false

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
        --flushdb)
            FLUSHDB=true
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
        --nuclear)
            NUCLEAR=true
            CLEAN_ALL=true
            CLEAN_LOGS=true
            CLEAN_CACHE=true
            FLUSHDB=true
            FORCE=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Comprehensive cleanup script for fresh sync starts."
            echo ""
            echo "Options:"
            echo "  --dry-run       Show what would be cleaned without actually cleaning"
            echo "  --all           Clean everything (Redis, DB, logs, cache, agent memory)"
            echo "  --flushdb       Use FLUSHDB instead of selective Redis key cleanup"
            echo "  --logs          Also clean log files and validation logs"
            echo "  --cache         Also clean cache directories and debug artifacts"
            echo "  --nuclear       Complete nuclear option (everything + force + FLUSHDB)"
            echo "  --force, -f     Skip confirmation prompt"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Default behavior: Clean Redis keys, DB, and essential directories"
            echo "With --all: Includes logs and cache cleanup"
            echo "With --nuclear: Maximum cleanup (equivalent to your manual commands)"
            echo ""
            echo "Environment variables:"
            echo "  REDIS_URL       Redis connection URL (default: redis://localhost:6379)"
            echo "  STATE_DB        Path to SQLite state database (default: .sync_state.db)"
            echo "  LOG_DIR         Path to log directory (default: ./logs)"
            echo "  CACHE_DIR       Path to cache directory (default: ./.cache)"
            echo "  AGENT_MEMORY    Path to agent memory directory (default: ./.agent_memory)"
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
elif $NUCLEAR && ! $FORCE; then
    echo -e "${RED}NUCLEAR OPTION SELECTED${NC}"
    echo "This will delete ALL data including:"
    echo "  - All Redis data (FLUSHDB)"
    echo "  - All databases and state files"
    echo "  - All cache, memory, and debug artifacts"
    echo "  - All logs and temporary files"
    echo ""
    read -p "Are you sure you want to proceed? (type 'NUCLEAR' to confirm): " CONFIRM
    if [ "$CONFIRM" != "NUCLEAR" ]; then
        echo -e "${RED}Nuclear cleanup cancelled${NC}"
        exit 1
    fi
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

# Kill running workers (multiple patterns)
echo -e "${GREEN}[1/8] Stopping running workers...${NC}"
WORKERS_FOUND=false

# Check for various worker patterns
if pgrep -f "arq obsidian_anki_sync.worker" > /dev/null 2>&1 || \
   pgrep -f "arq obsidianankisync.worker" > /dev/null 2>&1 || \
   pgrep -f "obsidian_anki_sync.worker" > /dev/null 2>&1 || \
   pgrep -f "obsidianankisync.worker" > /dev/null 2>&1; then
    WORKERS_FOUND=true
    echo "  Found running workers, stopping them..."

    # Try graceful shutdown first
    run_cmd "pkill -TERM -f 'arq obsidian_anki_sync.worker' || true"
    run_cmd "pkill -TERM -f 'arq obsidianankisync.worker' || true"
    run_cmd "pkill -TERM -f 'obsidian_anki_sync.worker' || true"
    run_cmd "pkill -TERM -f 'obsidianankisync.worker' || true"

    # Wait a bit for graceful shutdown
    sleep 3

    # Force kill any remaining workers
    if pgrep -f "arq obsidian_anki_sync.worker" > /dev/null 2>&1 || \
       pgrep -f "arq obsidianankisync.worker" > /dev/null 2>&1 || \
       pgrep -f "obsidian_anki_sync.worker" > /dev/null 2>&1 || \
       pgrep -f "obsidianankisync.worker" > /dev/null 2>&1; then
        echo "  Force killing remaining workers..."
        run_cmd "pkill -9 -f 'arq obsidian_anki_sync.worker' || true"
        run_cmd "pkill -9 -f 'arq obsidianankisync.worker' || true"
        run_cmd "pkill -9 -f 'obsidian_anki_sync.worker' || true"
        run_cmd "pkill -9 -f 'obsidianankisync.worker' || true"
    fi
    echo -e "  ${GREEN}Workers stopped${NC}"
else
    echo "  No workers running"
fi

# Clean Redis
echo ""
if $FLUSHDB; then
    echo -e "${GREEN}[2/8] Flushing Redis database...${NC}"
    if check_redis; then
        # Count all keys before FLUSHDB
        TOTAL_KEYS=$(redis-cli DBSIZE 2>/dev/null || echo "0")
        echo "  Database has $TOTAL_KEYS keys"

        if [ "$TOTAL_KEYS" -gt 0 ]; then
            run_cmd "redis-cli FLUSHDB > /dev/null"
            echo -e "  ${GREEN}Flushed Redis database${NC}"
        else
            echo "  Database already empty"
        fi
    else
        echo -e "  ${YELLOW}Redis not available, skipping${NC}"
    fi
else
    echo -e "${GREEN}[2/8] Cleaning Redis queues...${NC}"
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
fi

# Clean SQLite state database
echo ""
echo -e "${GREEN}[3/8] Cleaning SQLite state database...${NC}"
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
echo -e "${GREEN}[4/8] Cleaning LangGraph checkpoints...${NC}"
CHECKPOINT_DIR="$PROJECT_DIR/.langgraph_checkpoints"
if [ -d "$CHECKPOINT_DIR" ]; then
    CHECKPOINT_SIZE=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
    echo "  Found checkpoint directory: $CHECKPOINT_DIR ($CHECKPOINT_SIZE)"
    run_cmd "rm -rf '$CHECKPOINT_DIR'"
    echo -e "  ${GREEN}Deleted checkpoint directory${NC}"
else
    echo "  No checkpoint directory found"
fi

# Clean agent memory
echo ""
echo -e "${GREEN}[5/8] Cleaning agent memory...${NC}"
if [ -d "$AGENT_MEMORY" ]; then
    AGENT_SIZE=$(du -sh "$AGENT_MEMORY" 2>/dev/null | cut -f1)
    echo "  Found agent memory directory: $AGENT_MEMORY ($AGENT_SIZE)"
    run_cmd "rm -rf '$AGENT_MEMORY'"
    echo -e "  ${GREEN}Deleted agent memory directory${NC}"
else
    echo "  No agent memory directory found"
fi

# Clean debug artifacts and validation logs
echo ""
echo -e "${GREEN}[6/8] Cleaning debug artifacts and validation logs...${NC}"
DEBUG_CLEANED=false

if [ -d "$DEBUG_ARTIFACTS" ]; then
    DEBUG_SIZE=$(du -sh "$DEBUG_ARTIFACTS" 2>/dev/null | cut -f1)
    echo "  Found debug artifacts directory: $DEBUG_ARTIFACTS ($DEBUG_SIZE)"
    run_cmd "rm -rf '$DEBUG_ARTIFACTS'"
    echo -e "  ${GREEN}Deleted debug artifacts directory${NC}"
    DEBUG_CLEANED=true
fi

if [ -d "$VALIDATION_LOGS" ]; then
    VALIDATION_SIZE=$(du -sh "$VALIDATION_LOGS" 2>/dev/null | cut -f1)
    VALIDATION_COUNT=$(find "$VALIDATION_LOGS" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  Found validation logs directory: $VALIDATION_LOGS ($VALIDATION_SIZE, $VALIDATION_COUNT files)"
    run_cmd "rm -rf '$VALIDATION_LOGS'"
    echo -e "  ${GREEN}Deleted validation logs directory${NC}"
    DEBUG_CLEANED=true
fi

if [ -d "$PROBLEMATIC_NOTES" ]; then
    PROBLEMATIC_SIZE=$(du -sh "$PROBLEMATIC_NOTES" 2>/dev/null | cut -f1)
    PROBLEMATIC_COUNT=$(find "$PROBLEMATIC_NOTES" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  Found problematic notes directory: $PROBLEMATIC_NOTES ($PROBLEMATIC_SIZE, $PROBLEMATIC_COUNT files)"
    run_cmd "rm -rf '$PROBLEMATIC_NOTES'"
    echo -e "  ${GREEN}Deleted problematic notes directory${NC}"
    DEBUG_CLEANED=true
fi

if ! $DEBUG_CLEANED; then
    echo "  No debug artifacts, validation logs, or problematic notes found"
fi

# Clean vector stores and cache
if $CLEAN_CACHE || $CLEAN_ALL || $NUCLEAR; then
    echo ""
    echo -e "${GREEN}[7/8] Cleaning vector stores and cache...${NC}"

    # Clean ChromaDB
    if [ -d "$CHROMA_DIR" ]; then
        CHROMA_SIZE=$(du -sh "$CHROMA_DIR" 2>/dev/null | cut -f1)
        echo "  Found ChromaDB directory: $CHROMA_DIR ($CHROMA_SIZE)"
        run_cmd "rm -rf '$CHROMA_DIR'"
        echo -e "  ${GREEN}Deleted ChromaDB directory${NC}"
    fi

    # Clean diskcache
    DISKCACHE_DIR="$PROJECT_DIR/.diskcache"
    if [ -d "$DISKCACHE_DIR" ]; then
        DISKCACHE_SIZE=$(du -sh "$DISKCACHE_DIR" 2>/dev/null | cut -f1)
        echo "  Found diskcache directory: $DISKCACHE_DIR ($DISKCACHE_SIZE)"
        run_cmd "rm -rf '$DISKCACHE_DIR'"
        echo -e "  ${GREEN}Deleted diskcache directory${NC}"
    fi

    # Clean general cache directory
    if [ -d "$CACHE_DIR" ]; then
        CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
        echo "  Found cache directory: $CACHE_DIR ($CACHE_SIZE)"
        run_cmd "rm -rf '$CACHE_DIR'"
        echo -e "  ${GREEN}Deleted cache directory${NC}"
    fi
else
    echo ""
    echo -e "${GREEN}[7/8] Skipping vector stores and cache cleanup (use --cache, --all, or --nuclear)${NC}"
fi

# Clean logs
if $CLEAN_LOGS || $CLEAN_ALL || $NUCLEAR; then
    echo ""
    echo -e "${GREEN}[8/8] Cleaning log files...${NC}"
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
    echo -e "${GREEN}[8/8] Skipping log cleanup (use --logs, --all, or --nuclear)${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
if $DRY_RUN; then
    echo -e "${YELLOW}DRY RUN COMPLETE - No changes were made${NC}"
    echo ""
    echo "Run without --dry-run to apply changes:"
    if $NUCLEAR; then
        echo "  $0 --nuclear"
    elif $FLUSHDB; then
        echo "  $0 --flushdb --all"
    else
        echo "  $0 --all"
    fi
else
    if $NUCLEAR; then
        echo -e "${RED}NUCLEAR CLEANUP COMPLETE - All data wiped${NC}"
    else
        echo -e "${GREEN}CLEANUP COMPLETE${NC}"
    fi
fi
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Next steps:"
if $NUCLEAR; then
    echo "  1. Reinitialize: obsidian-anki-sync init"
    echo "  2. Start workers: ./scripts/launch_workers.sh 10"
    echo "  3. Run sync:      obsidian-anki-sync sync --use-queue --use-langgraph"
else
    echo "  1. Start workers: ./scripts/launch_workers.sh 10"
    echo "  2. Run sync:      obsidian-anki-sync sync --use-queue --use-langgraph"
fi
echo ""
