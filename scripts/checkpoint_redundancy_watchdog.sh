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

# Canonical destination per the per-machine sprint convention:
# `C:/ml-trm-work/checkpoints to use/Machine N/snapshots/` (capital M, space).
# Snapshots land in a `snapshots/` sub-folder so the parent stays browseable
# alongside live run dirs (llm-qwen-sudoku-seed0-fixb/, …) without the
# timestamped snapshot files cluttering the top level.
DEST="/c/ml-trm-work/checkpoints to use/Machine ${MACHINE_N}/snapshots"
mkdir -p "$DEST"

echo "[redundancy] watchdog started: machine=$MACHINE_N run=$RUN_NAME run_dir=$RUN_DIR dest=$DEST"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

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

  # Drive sync hook — fails quietly if no CLI backend (rclone/gdrive) is
  # installed. Once the user runs `winget install Rclone.Rclone` + `rclone
  # config`, every snapshot cycle automatically mirrors the parent
  # `Machine N/` folder to the configured Drive folder. Failure is
  # non-fatal — the local snapshot already succeeded by this point, the
  # next cycle retries the sync, and the local watchdog doesn't depend
  # on cloud reachability.
  bash "$REPO_ROOT/scripts/sync_machine5_to_drive.sh" \
      >> /tmp/m5_drive_sync.log 2>&1 || true

  sleep 1800
done
