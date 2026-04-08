#!/bin/bash
# Hourly checkpoint push for remote training monitoring.
# Pushes training logs AND latest checkpoint to GitHub every hour.
# Usage: bash scripts/hourly_checkpoint_push.sh

INTERVAL=3600  # 1 hour

cd "$(dirname "$0")/.."
echo "=== Hourly checkpoint push started ==="
echo "Interval: ${INTERVAL}s (1 hour)"
echo "Press Ctrl+C to stop"

while true; do
    sleep $INTERVAL

    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M')] Pushing checkpoint..."

    # Force-add logs (gitignored by default)
    git add -f experiments/*_train_log.csv 2>/dev/null
    git add -f experiments/*_results.json 2>/dev/null
    git add -f experiments/*_emissions.csv 2>/dev/null

    # Force-add latest checkpoint (gitignored, ~25MB each)
    for dir in models/sudoku models/maze models/llm; do
        [ -d "$dir" ] || continue
        [ -f "$dir/latest.pt" ] && git add -f "$dir/latest.pt"
        [ -f "$dir/best.pt" ] && git add -f "$dir/best.pt"
    done

    # Check if there's anything to commit
    if git diff --cached --quiet; then
        echo "  No changes to push"
        continue
    fi

    # Get stats from log for commit message
    SUMMARY=""
    for log in experiments/*_train_log.csv; do
        [ -f "$log" ] || continue
        NAME=$(basename "$log" _train_log.csv)
        LAST=$(tail -1 "$log" 2>/dev/null)
        if [ -n "$LAST" ] && [ "$LAST" != "epoch,ce_loss,q_mean,steps_taken,val_cell_acc,val_puzzle_acc,best_puzzle_acc,elapsed_min" ]; then
            EPOCH=$(echo "$LAST" | cut -d',' -f1)
            CE=$(echo "$LAST" | cut -d',' -f2)
            VAL=$(echo "$LAST" | cut -d',' -f6)
            SUMMARY="$SUMMARY $NAME:ep${EPOCH},ce=${CE},val=${VAL}"
        fi
    done

    git commit -m "checkpoint:${SUMMARY:-" hourly save"}"
    git pull --rebase 2>/dev/null

    if git push; then
        echo "  Pushed successfully"
    else
        echo "  Push failed -- will retry next hour"
    fi
done
