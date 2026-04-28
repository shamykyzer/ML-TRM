#!/usr/bin/env bash
# Launch Llama Sudoku Fix-B retrain (Step B, M3 sprint 2026-04-28).
#
# Wires together:
#   * Contract A redundancy watchdog (every 30 min snapshot to
#     "C:/ml-trm-work/checkpoints to use/machine3/")
#   * Contract B §B.5 pre-launch sanity check (asserts maze configs have
#     mask_non_path:false; here, sudoku, just confirms config loads)
#   * start.py for wandb auth + requirements check (per user instruction)
#   * Direct trainer_llm launch with the Fix-B config (TASK_DISPATCH does
#     NOT yet have an `llm-llama-sudoku-fixb` task; we launch by config
#     path to avoid mutating a shared file mid-sprint)
#
# Usage:
#   bash scripts/launch_step_b_fixb.sh
#
# Re-run safe: if a previous attempt left WATCHDOG_PID in /tmp, the script
# kills it before starting a new watchdog.
#
# Approx wall-clock: 5-7 h on RTX 5070 (Llama-3.2-1B LoRA r=8, 30 epochs,
# batch_size=2 grad_accum=8, sudoku-extreme-1k subset).
set -euo pipefail

REPO="C:/dev/ML-TRM"
PY="/c/Users/amm-alshamy/.venvs/ml-trm/Scripts/python.exe"
CONFIG="configs/llm_sudoku_fixb.yaml"
RUN_NAME="llama-sudoku-seed0-fixb"
RUN_DIR="C:/ml-trm-work/${RUN_NAME}"
MACHINE_N=3

cd "$REPO"

# ---- Step 1: Contract B §B.5 prelaunch sanity ----
echo "[step-b] §B.5 prelaunch sanity check on $CONFIG"
"$PY" scripts/contract_b_realism_check.py prelaunch "$CONFIG"

# ---- Step 2: start.py preflight (wandb auth, requirements) ----
# `start.py status` runs the bootstrap stages without launching a trainer.
echo "[step-b] start.py status (wandb + requirements preflight)"
"$PY" start.py status

# ---- Step 3: kill any prior watchdog for this machine ----
if [ -f /tmp/watchdog-machine${MACHINE_N}.pid ]; then
  prior=$(cat /tmp/watchdog-machine${MACHINE_N}.pid)
  if kill -0 "$prior" 2>/dev/null; then
    echo "[step-b] killing prior watchdog PID $prior"
    kill "$prior" 2>/dev/null || true
  fi
  rm -f /tmp/watchdog-machine${MACHINE_N}.pid
fi

# ---- Step 4: launch Contract A watchdog in background ----
mkdir -p "C:/ml-trm-work/checkpoints to use/machine${MACHINE_N}"
mkdir -p "$RUN_DIR"
nohup bash scripts/checkpoint_redundancy_watchdog.sh \
  "$MACHINE_N" "$RUN_DIR" "$RUN_NAME" \
  > /tmp/watchdog-machine${MACHINE_N}.log 2>&1 &
WATCHDOG_PID=$!
echo "$WATCHDOG_PID" > /tmp/watchdog-machine${MACHINE_N}.pid
echo "[step-b] watchdog PID $WATCHDOG_PID -> /tmp/watchdog-machine${MACHINE_N}.log"

# ---- Step 5: launch trainer ----
# We call trainer_llm.LLMTrainer directly because TASK_DISPATCH has no
# `llm-llama-sudoku-fixb` entry yet. start.py preflight has already run
# (step 2) so wandb is authed and requirements are present.
echo "[step-b] starting LLMTrainer with $CONFIG"
"$PY" -c "
from src.utils.config import load_config
from src.training.trainer_llm import LLMTrainer
cfg = load_config('$CONFIG')
print('[trainer] loaded', cfg.model.llm_name, 'on', cfg.data.dataset,
      'subsample=', cfg.data.subsample_size, 'epochs=', cfg.training.epochs)
trainer = LLMTrainer(cfg)
trainer.train()
print('[trainer] done')
"
TRAIN_RC=$?

# ---- Step 6: final snapshot then stop watchdog ----
ts=$(date +%Y-%m-%dT%H%M)
DEST="C:/ml-trm-work/checkpoints to use/machine${MACHINE_N}"
for src in "$RUN_DIR"/*.pt "$RUN_DIR"/*.csv "$RUN_DIR"/*.json; do
  [ -e "$src" ] || continue
  base=$(basename "$src")
  cp -p "$src" "$DEST/${RUN_NAME}__${ts}__${base}"
done
echo "[step-b] final snapshot to $DEST (ts=$ts)"

if [ -f /tmp/watchdog-machine${MACHINE_N}.pid ]; then
  kill "$(cat /tmp/watchdog-machine${MACHINE_N}.pid)" 2>/dev/null || true
  rm -f /tmp/watchdog-machine${MACHINE_N}.pid
fi
echo "[step-b] watchdog stopped; trainer exit code = $TRAIN_RC"

# ---- Step 7: Contract B §B.7 viability gate ----
LOG="$RUN_DIR/llama_3.2_1b_sudoku_train_log.csv"
if [ -f "$LOG" ]; then
  echo "[step-b] §B.7 viability gate"
  "$PY" scripts/contract_b_realism_check.py monitor \
    "$LOG" --task sudoku --family llm
fi

echo "[step-b] done — review findings.md §5 to log result"
exit "$TRAIN_RC"
