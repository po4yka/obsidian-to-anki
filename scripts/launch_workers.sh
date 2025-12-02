#!/bin/bash
# Launch multiple ARQ workers for parallel card generation
#
# Usage:
#   ./scripts/launch_workers.sh [NUM_WORKERS] [MAX_JOBS_PER_WORKER]
#
# Examples:
#   ./scripts/launch_workers.sh           # 5 workers, 100 jobs each (default)
#   ./scripts/launch_workers.sh 10        # 10 workers, 100 jobs each
#   ./scripts/launch_workers.sh 3 50      # 3 workers, 50 jobs each

set -e

NUM_WORKERS=${1:-5}
MAX_JOBS=${2:-50}
LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "Starting $NUM_WORKERS workers with $MAX_JOBS concurrent jobs each..."
echo "Total parallel capacity: $((NUM_WORKERS * MAX_JOBS)) jobs"
echo ""

# Array to store PIDs
PIDS=()

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping workers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait
    echo "All workers stopped."
}

# Set trap for cleanup on exit
trap cleanup EXIT INT TERM

# Launch workers
for i in $(seq 1 $NUM_WORKERS); do
    echo "Starting worker $i..."
    LOG_LEVEL=$LOG_LEVEL MAX_CONCURRENT_GENERATIONS=$MAX_JOBS \
        uv run arq obsidian_anki_sync.worker.WorkerSettings &
    PIDS+=($!)
    sleep 0.5  # Small delay between launches
done

echo ""
echo "All $NUM_WORKERS workers started. Press Ctrl+C to stop."
echo ""

# Wait for all workers
wait
