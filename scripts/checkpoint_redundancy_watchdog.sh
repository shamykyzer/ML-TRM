#!/usr/bin/env bash
# Contract A — 30-minute checkpoint redundancy save (sprint 2026-04-28 → 2026-05-01)
#
# Usage:
#   bash scripts/checkpoint_redundancy_watchdog.sh <machine_n> <run_dir> <run_name>
#
# Example:
#   bash scripts/checkpoint_redundancy_watchdog.sh 3 \
#       "C:/ml-trm-work/llama-sudoku-seed0-fixb" \
#       "llama-sudoku-seed0-fixb"
#
# Runs as a separate background process from the trainer so a trainer crash
# cannot take it down. Snapshots every .pt / .csv / .json in <run_dir> into
# C:/ml-trm-work/checkpoints to use/machine{N}/ every 30 minutes, with an
# ISO-ish timestamp suffix (no overwrites of older snapshots).
#
# Naming pattern (Contract A §A.3):
#   {run_name}__{YYYY-MM-DDTHHMM}__{original_filename}
# Double underscore __ is the aggregator's split separator.
set -u

MACHINE_N="${1:?machine number required, e.g. 3}"
RUN_DIR="${2:?run dir required, e.g. C:/ml-trm-work/llama-sudoku-seed0-fixb}"
RUN_NAME="${3:?run name required, e.g. llama-sudoku-seed0-fixb}"

DEST="C:/ml-trm-work/checkpoints to use/machine${MACHINE_N}"
mkdir -p "$DEST"

echo "[redundancy] starting watchdog: machine=$MACHINE_N run_dir=$RUN_DIR run_name=$RUN_NAME"
echo "[redundancy] dest=$DEST  interval=1800s"

while true; do
  ts=$(date +%Y-%m-%dT%H%M)
  for src in "$RUN_DIR"/*.pt "$RUN_DIR"/*.csv "$RUN_DIR"/*.json; do
    [ -e "$src" ] || continue
    base=$(basename "$src")
    cp -p "$src" "$DEST/${RUN_NAME}__${ts}__${base}" 2>/dev/null || true
  done
  echo "[redundancy] $(date -Is) snapshot to $DEST"
  sleep 1800
done
