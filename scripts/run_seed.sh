#!/usr/bin/env bash
# =============================================================================
# run_seed.sh — launch one seed of one task, with OneDrive-safe local paths.
#
# Usage:
#   scripts/run_seed.sh <task> <seed> [--dry-run]
#
# Arguments:
#   task   : sudoku-mlp | sudoku-att | maze | llm-sudoku
#   seed   : non-negative integer (machines 1..6 -> seeds 0..5)
#   --dry-run : override epochs to 5 for a pipeline smoke test
#
# Environment:
#   TRM_WORK_DIR : root for all per-seed output dirs (checkpoints + experiments).
#                  MUST be a local non-OneDrive path on the machine where this
#                  script runs. Defaults to $HOME/ml-trm-work (or %USERPROFILE%
#                  on Git Bash for Windows). Each seed run gets a subdir
#                  <TRM_WORK_DIR>/<task>-seed<N>/ which is set as both
#                  TRM_CHECKPOINT_DIR and TRM_EXPERIMENT_DIR for main.py.
#
#   TRM_PYTHON   : path to venv python. Defaults to the standard venv locations.
#
# Why: every machine in the 6-box fleet has the repo synced via OneDrive. That
# means code + data + HF checkpoints are already consistent across machines
# (good), but writing training outputs inside the sync folder would corrupt
# checkpoints during OneDrive uploads and create rename conflicts when two
# machines touch the same file. Every run must route its outputs to a local
# path — which is exactly what this script enforces.
#
# Example:
#   # machine 1 (seed 0)
#   scripts/run_seed.sh sudoku-mlp 0
#   # machine 4 (seed 3) — fine-tune maze from HF init
#   scripts/run_seed.sh maze 3
#   # machine 6 — LLM baseline on sudoku (one seed is enough to close the
#   #             proposal's three-way comparison gap)
#   scripts/run_seed.sh llm-sudoku 0
# =============================================================================
set -euo pipefail

TASK="${1:-}"
SEED="${2:-}"
DRY_RUN=0
if [[ "${3:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

if [[ -z "$TASK" || -z "$SEED" ]]; then
  cat <<'USAGE' >&2
Usage: scripts/run_seed.sh <task> <seed> [--dry-run]
  task  : sudoku-mlp | sudoku-att | maze | llm-sudoku
  seed  : non-negative integer
  --dry-run : short 5-epoch run for pipeline validation

Examples:
  scripts/run_seed.sh sudoku-mlp 0
  scripts/run_seed.sh maze 3
  scripts/run_seed.sh sudoku-mlp 0 --dry-run
USAGE
  exit 2
fi

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "ERROR: seed must be a non-negative integer, got: $SEED" >&2
  exit 2
fi

# Resolve repo root (script lives in scripts/, so parent is root).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Pick the venv python. The defaults mirror start.py's VENV_DIR logic so the
# two stay in sync.
if [[ -n "${TRM_PYTHON:-}" ]]; then
  PY="$TRM_PYTHON"
elif [[ -x "$HOME/.venvs/ml-trm/Scripts/python.exe" ]]; then
  PY="$HOME/.venvs/ml-trm/Scripts/python.exe"  # Windows default
elif [[ -x "$HOME/.venvs/ml-trm/bin/python" ]]; then
  PY="$HOME/.venvs/ml-trm/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PY=".venv/Scripts/python.exe"
else
  PY="python"
fi

# TRM_WORK_DIR default: local, not OneDrive. On Git Bash for Windows,
# $HOME is usually /c/Users/<user>, which is outside OneDrive so we can
# safely put a work dir there.
WORK_DIR="${TRM_WORK_DIR:-$HOME/ml-trm-work}"

# Hard check: refuse to run if the work dir resolves inside a OneDrive path,
# even if the user explicitly set TRM_WORK_DIR to such a location. This is
# the single most dangerous mistake in the 6-machine setup, and silently
# letting it through would be worse than making the user fix it.
case "$(echo "$WORK_DIR" | tr '[:upper:]' '[:lower:]')" in
  *onedrive*)
    echo "ERROR: TRM_WORK_DIR='$WORK_DIR' looks like a OneDrive path." >&2
    echo "       Parallel training on shared OneDrive will corrupt checkpoints." >&2
    echo "       Pick a local path (e.g. C:/ml-trm-work on Windows," >&2
    echo "       /home/\$USER/ml-trm-work on Linux) and re-run." >&2
    exit 3
    ;;
esac

# Per-seed output dir. Both checkpoint_dir and experiment_dir are routed here
# via env vars — main.py -> config.py reads these and overrides the YAML.
TASK_DIR="${WORK_DIR}/${TASK}-seed${SEED}"
mkdir -p "$TASK_DIR"
export TRM_CHECKPOINT_DIR="$TASK_DIR"
export TRM_EXPERIMENT_DIR="$TASK_DIR"

# Dispatch on task. CONFIG and INIT_WEIGHTS come from the task label. The
# checkpoint paths match the remapped files produced by scripts/remap_*.py.
case "$TASK" in
  sudoku-mlp)
    CONFIG="configs/trm_official_sudoku_mlp.yaml"
    INIT="hf_checkpoints/Sudoku-Extreme-mlp/remapped_for_local.pt"
    ;;
  sudoku-att)
    CONFIG="configs/trm_official_sudoku.yaml"
    INIT="hf_checkpoints/Sudoku-Extreme-att/remapped_for_local.pt"
    ;;
  maze)
    CONFIG="configs/trm_official_maze.yaml"
    INIT="hf_checkpoints/Maze-Hard/remapped_for_local.pt"
    ;;
  llm-sudoku)
    # LLM dispatch in main.py currently only supports the sudoku data loader,
    # so llm-maze is not yet wired. Using Qwen2.5-0.5B as the representative
    # baseline — all four llm_*.yaml configs work here, swap as needed.
    CONFIG="configs/llm_qwen.yaml"
    INIT=""  # LLMs load their own HF weights via transformers
    ;;
  *)
    echo "ERROR: unknown task '$TASK'." >&2
    echo "       Valid: sudoku-mlp | sudoku-att | maze | llm-sudoku" >&2
    exit 2
    ;;
esac

# Build the argv for main.py. --seed is always explicit so it shows up in
# shell history AND in the wandb run name (the trainer mirrors cfg.seed into
# the run name suffix).
ARGS=(
  main.py
  --mode train
  --config "$CONFIG"
  --seed "$SEED"
)
if [[ -n "$INIT" ]]; then
  ARGS+=(--init-weights "$INIT")
fi
if [[ $DRY_RUN -eq 1 ]]; then
  # main.py --epochs 5 overrides training.epochs from the YAML. Enough to
  # validate: config loads, dataset loads, model forward pass works, wandb
  # logs a train/val row, a checkpoint gets written, emissions.csv appends.
  ARGS+=(--epochs 5)
fi

echo "================================================================"
echo "  task           : $TASK"
echo "  seed           : $SEED"
echo "  config         : $CONFIG"
echo "  init_weights   : ${INIT:-<none, random init>}"
echo "  TRM_CHECKPOINT_DIR : $TRM_CHECKPOINT_DIR"
echo "  TRM_EXPERIMENT_DIR : $TRM_EXPERIMENT_DIR"
echo "  python         : $PY"
echo "  dry-run        : $( [[ $DRY_RUN -eq 1 ]] && echo 'YES (5 epochs)' || echo 'no' )"
echo "================================================================"
echo

exec "$PY" "${ARGS[@]}"
