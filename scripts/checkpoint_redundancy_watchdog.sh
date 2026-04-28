#!/usr/bin/env bash
# Contract A.4 — 30-min checkpoint redundancy watchdog.
# Runs as a separate background process from the trainer so a trainer crash
# cannot take it down. Snapshots <run-dir>/*.{pt,csv,json} into
# C:/ml-trm-work/checkpoints to use/machine{N}/ with timestamped filenames.
#
# Usage:
#   bash scripts/checkpoint_redundancy_watchdog.sh <machine_n> <run_dir> <run_name>
# Example:
#   bash scripts/checkpoint_redundancy_watchdog.sh 6 \
#     "C:/ml-trm-work/llm-qwen-sudoku-seed1-fixb" \
#     "llm-qwen-sudoku-seed1-fixb"
#
# Naming convention in the redundancy folder (Contract A.3):
#   {run_name}__{YYYY-MM-DDTHHMM}__{original_filename}
# Double underscore "__" is the separator the aggregator splits on.

set -u
MACHINE_N="${1:?machine number required, e.g. 6}"
RUN_DIR="${2:?run dir required, e.g. C:/ml-trm-work/llm-qwen-sudoku-seed1-fixb}"
RUN_NAME="${3:?run name required, e.g. llm-qwen-sudoku-seed1-fixb}"
DEST="C:/ml-trm-work/checkpoints to use/machine ${MACHINE_N}"
mkdir -p "$DEST"

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
