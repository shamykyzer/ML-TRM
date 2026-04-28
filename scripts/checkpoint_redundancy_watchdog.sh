#!/usr/bin/env bash
# Contract A — 30-minute checkpoint redundancy snapshot watchdog.
#
# Runs as a separate process from the trainer so a trainer crash cannot take
# the watchdog down. Snapshots the live run-dir contents every 30 minutes
# (1800 s) into /c/ml-trm-work/checkpoints to use/machine{N}/ with a
# timestamped {run_name}__{YYYY-MM-DDTHHMM}__{filename} naming convention so
# older snapshots are not overwritten.
#
# Usage:
#   bash scripts/checkpoint_redundancy_watchdog.sh <machine_n> <run_dir> <run_name>
#
# Example (M5, this machine):
#   bash scripts/checkpoint_redundancy_watchdog.sh 5 \
#     "/c/ml-trm-work/llm-qwen-sudoku-seed0-fixb" \
#     "llm-qwen-sudoku-seed0-fixb"
#
# Stop with `kill <watchdog_pid>` after the trainer exits and you've taken
# one final snapshot manually.

set -u
MACHINE_N="${1:?machine number required, e.g. 5}"
RUN_DIR="${2:?run dir required, e.g. /c/ml-trm-work/llm-qwen-sudoku-seed0-fixb}"
RUN_NAME="${3:?run name required, e.g. llm-qwen-sudoku-seed0-fixb}"

DEST="/c/ml-trm-work/checkpoints to use/machine${MACHINE_N}"
mkdir -p "$DEST"

echo "[redundancy] watchdog started: machine=$MACHINE_N run=$RUN_NAME run_dir=$RUN_DIR dest=$DEST"

while true; do
  ts=$(date +%Y-%m-%dT%H%M)
  copied=0
  for src in "$RUN_DIR"/*.pt "$RUN_DIR"/*.csv "$RUN_DIR"/*.json; do
    [ -e "$src" ] || continue
    base=$(basename "$src")
    if cp -p "$src" "$DEST/${RUN_NAME}__${ts}__${base}" 2>/dev/null; then
      copied=$((copied + 1))
    fi
  done
  echo "[redundancy] $(date -Is) snapshot: $copied files -> $DEST/${RUN_NAME}__${ts}__*"
  sleep 1800
done
