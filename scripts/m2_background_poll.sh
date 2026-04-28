#!/usr/bin/env bash
# Background polling watcher for M2 plot factory + K-vote extender.
# Runs ~5h (20 cycles × 15 min), writes timestamped findings to LOG.
# Exits early if EXIT_FLAG file is touched (so the agent can stop it).
#
# Usage (background): bash scripts/m2_background_poll.sh
# Stop:               touch /tmp/m2_poll_stop
# Read log:           tail -f /tmp/m2_background_poll.log

set -u
REPO_ROOT="C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ML-TRM"
LOG="/tmp/m2_background_poll.log"
EXIT_FLAG="/tmp/m2_poll_stop"
TRIGGER_FLAG="/tmp/m2_poll_trigger"
MAX_CYCLES="${MAX_CYCLES:-20}"
SLEEP_SEC="${SLEEP_SEC:-900}"

cd "$REPO_ROOT" || { echo "cannot cd to $REPO_ROOT" > "$LOG"; exit 1; }

{
    echo "=== M2 background polling started @ $(date '+%Y-%m-%d %H:%M:%S BST') ==="
    echo "    cycles: $MAX_CYCLES; per-cycle sleep: ${SLEEP_SEC}s"
    echo "    repo: $REPO_ROOT"
    echo "    stop with: touch $EXIT_FLAG"
} > "$LOG"

rm -f "$TRIGGER_FLAG"

for i in $(seq 1 "$MAX_CYCLES"); do
    if [ -f "$EXIT_FLAG" ]; then
        echo "" >> "$LOG"
        echo "=== STOP flag at $EXIT_FLAG detected; exiting after cycle $((i-1)) ===" >> "$LOG"
        rm -f "$EXIT_FLAG"
        break
    fi
    {
        echo ""
        echo "=== bg cycle $i / $MAX_CYCLES @ $(date '+%H:%M:%S') ==="
        git fetch origin --quiet 2>&1 | tail -3
        NEW=$(git log --oneline HEAD..origin/main 2>/dev/null | head -5)
        if [ -n "$NEW" ]; then
            echo "[NEW COMMITS]"
            echo "$NEW"
        fi
        if [ -f results/summary_fixed.csv ]; then
            echo "[TRIGGER] LOCAL results/summary_fixed.csv ($(wc -l < results/summary_fixed.csv) lines)"
            touch "$TRIGGER_FLAG"
        elif git show origin/main:results/summary_fixed.csv > /dev/null 2>&1; then
            echo "[TRIGGER] REMOTE results/summary_fixed.csv (not yet pulled)"
            touch "$TRIGGER_FLAG"
        else
            echo "[ ] no summary_fixed.csv"
        fi
        LATEST_REMOTE=$(git show origin/main:findings.md 2>/dev/null \
            | grep -oE '^### 5\.[0-9]+' | tail -1 | tr -d ' ')
        echo "[ ] remote findings.md latest §5.x: $LATEST_REMOTE"
        if ls -d "C:/ml-trm-work/novelty-qwen-sudoku-fixb-seed0" >/dev/null 2>&1; then
            echo "[TRIGGER] qwen-fixb dir present"
            touch "$TRIGGER_FLAG"
        else
            echo "[ ] no qwen-fixb dir"
        fi
        if ls results/novelty/k_vote_results-*m5*.csv >/dev/null 2>&1; then
            echo "[TRIGGER] M5 -m5 K-vote files present"
            touch "$TRIGGER_FLAG"
        else
            echo "[ ] no -m5 K-vote files"
        fi
        if [ -d results/eval_fixed ]; then
            echo "[TRIGGER] results/eval_fixed dir present"
            touch "$TRIGGER_FLAG"
        else
            echo "[ ] no results/eval_fixed"
        fi
    } >> "$LOG" 2>&1
    sleep "$SLEEP_SEC"
done

echo "" >> "$LOG"
echo "=== M2 background polling exited @ $(date '+%Y-%m-%d %H:%M:%S BST') ===" >> "$LOG"
