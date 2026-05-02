#!/usr/bin/env bash
# Contract A redundancy watchdog — every 30 min copies an in-flight
# training run's latest checkpoint + train log + emissions CSV to a
# fleet-uniform location so a crash costs at most ~30 min of compute.
#
# Usage:
#   bash scripts/checkpoint_redundancy_watchdog.sh <machine_n> <run_dir> <run_name>
#
# Example:
#   bash scripts/checkpoint_redundancy_watchdog.sh 1 \
#       "C:/ml-trm-work/llm-smollm-sudoku-seed0" \
#       "llm-smollm-sudoku-seed0"
#
# Runs as a separate background process from the trainer; a trainer
# crash cannot take it down. Stop with: kill <watchdog_pid>.
set -u

MACHINE_N="${1:?machine number required, e.g. 1}"
RUN_DIR="${2:?run dir required, e.g. C:/ml-trm-work/llm-smollm-sudoku-seed0}"
RUN_NAME="${3:?run name required, e.g. llm-smollm-sudoku-seed0}"
DEST="C:/ml-trm-work/checkpoints to use/machine${MACHINE_N}"
mkdir -p "$DEST"

echo "[redundancy] watchdog start: machine${MACHINE_N} <- ${RUN_NAME} (${RUN_DIR})"
echo "[redundancy] dest: $DEST"
echo "[redundancy] interval: 1800s (30 min)"

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
  echo "[redundancy] $(date -Is) snapshot to $DEST (${copied} files)"
  sleep 1800
done
