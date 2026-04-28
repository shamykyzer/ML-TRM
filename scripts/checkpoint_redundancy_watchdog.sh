#!/usr/bin/env bash
# Apr-26 sprint Contract A — 30-min checkpoint redundancy save.
#
# Usage:
#   bash scripts/checkpoint_redundancy_watchdog.sh <machine_n> <run_dir> <run_name>
#
# Example (M1 launching SmolLM Sudoku via the prebuilt start.py preflight):
#   # 1. Make sure venv + wandb + HF checkpoints are ready (idempotent).
#   python start.py status
#   # If status shows a stage as not-ready:
#   python start.py            # runs the next missing setup stage; rerun until status is green
#
#   # 2. Start the redundancy watchdog (separate background process).
#   mkdir -p "C:/ml-trm-work/checkpoints to use/machine1"
#   bash scripts/checkpoint_redundancy_watchdog.sh 1 \
#       "C:/ml-trm-work/llm-smollm-sudoku-seed0" \
#       "llm-smollm-sudoku-seed0" > /tmp/watchdog-machine1.log 2>&1 &
#   WATCHDOG_PID=$!
#
#   # 3. Launch the trainer through start.py so the per-run preflight
#   #    (git pull --ff-only, kill stale, back up best.pt) and wandb env
#   #    bootstrap happen the same way on every machine.
#   python start.py llm-smollm-sudoku 0
#
#   # 4. Final snapshot + watchdog teardown.
#   ts=$(date +%Y-%m-%dT%H%M)
#   for src in C:/ml-trm-work/llm-smollm-sudoku-seed0/*.{pt,csv,json}; do
#     [ -e "$src" ] && cp -p "$src" \
#       "C:/ml-trm-work/checkpoints to use/machine1/llm-smollm-sudoku-seed0__${ts}__$(basename "$src")"
#   done
#   kill "$WATCHDOG_PID" 2>/dev/null || true
#
# Snapshot naming: {run_name}__{YYYY-MM-DDTHHMM}__{original_filename}
# Tag for findings.md §5: "redundancy snapshot machine${MACHINE_N}"
# Always launch trainers via `python start.py …` (Contract preamble) so the
# venv, wandb auth, requirements hash, and HF-checkpoint sanity checks fire
# uniformly across the 6-rig fleet.

set -u

MACHINE_N="${1:?machine number required, e.g. 1}"
RUN_DIR="${2:?run dir required, e.g. C:/ml-trm-work/llm-smollm-sudoku-seed0}"
RUN_NAME="${3:?run name required, e.g. llm-smollm-sudoku-seed0}"
DEST="C:/ml-trm-work/checkpoints to use/machine${MACHINE_N}"

mkdir -p "$DEST"

while true; do
    ts=$(date +%Y-%m-%dT%H%M)
    for src in "$RUN_DIR"/*.pt "$RUN_DIR"/*.csv "$RUN_DIR"/*.json; do
        [ -e "$src" ] || continue
        base=$(basename "$src")
        cp -p "$src" "$DEST/${RUN_NAME}__${ts}__${base}" 2>/dev/null || true
    done
    echo "[redundancy] $(date -Is) snapshot machine${MACHINE_N} -> $DEST"
    sleep 1800
done
