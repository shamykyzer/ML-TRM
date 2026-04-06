#!/bin/bash
# Auto-push training logs every hour for remote monitoring.
# Usage: bash scripts/auto_push.sh
# Run this in a SEPARATE terminal alongside training.
# Safe for multiple machines -- uses git pull --rebase before push.

INTERVAL=3600  # seconds (1 hour)

echo "=== Auto-push started (every ${INTERVAL}s) ==="
echo "Press Ctrl+C to stop"

while true; do
    sleep $INTERVAL

    # Stage only log files (not large checkpoint .pt files)
    git add experiments/*_train_log.csv 2>/dev/null
    git add experiments/*_results.json 2>/dev/null
    git add experiments/*_emissions.csv 2>/dev/null

    # Check if there's anything to commit
    if git diff --cached --quiet; then
        echo "[$(date '+%H:%M')] No new log changes to push"
        continue
    fi

    # Get latest stats from logs for commit message
    SUMMARY=""
    for log in experiments/*_train_log.csv; do
        [ -f "$log" ] || continue
        NAME=$(basename "$log" _train_log.csv)
        LAST=$(tail -1 "$log" 2>/dev/null)
        if [ -n "$LAST" ]; then
            SUMMARY="$SUMMARY $NAME:[$(echo $LAST | cut -d',' -f1-4)]"
        fi
    done

    git commit -m "training progress:${SUMMARY:-" update"}"

    # Pull with rebase to handle pushes from other machines
    git pull --rebase 2>/dev/null

    if git push; then
        echo "[$(date '+%H:%M')] Pushed training logs"
    else
        echo "[$(date '+%H:%M')] Push failed -- will retry next cycle"
    fi
done
