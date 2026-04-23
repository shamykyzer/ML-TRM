#!/usr/bin/env bash
# Fine-tune queue: waits for GPU to free, then runs maze seed 4, then sudoku seed 4.
# Each run logs to results/finetune_logs/<task>_seed<seed>.log.
# Safe to launch in background — progress is in the log files + wandb.

set -u

REPO="/c/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ML-TRM"
PY="/c/Users/amm-alshamy/.venvs/ml-trm/Scripts/python.exe"
LOGDIR="$REPO/results/finetune_logs"
SEED=4

mkdir -p "$LOGDIR"
cd "$REPO"

echo "[$(date)] queue started, seed=$SEED" | tee -a "$LOGDIR/queue.log"

# --- Wait for GPU to free ---
# "Free" = utilization < 20% AND < 3 GB in use, sustained across 3 checks (~90s).
wait_gpu_free() {
  local consecutive=0
  local needed=3
  while true; do
    local line
    line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    local util="${line%%,*}"
    local mem="${line##*, }"
    util="${util// /}"
    mem="${mem// /}"
    if [ "${util:-100}" -lt 20 ] && [ "${mem:-99999}" -lt 3000 ]; then
      consecutive=$((consecutive + 1))
      echo "[$(date)] GPU free check $consecutive/$needed (util=${util}%, mem=${mem}MiB)" | tee -a "$LOGDIR/queue.log"
      if [ "$consecutive" -ge "$needed" ]; then
        break
      fi
    else
      if [ "$consecutive" -gt 0 ]; then
        echo "[$(date)] GPU busy again (util=${util}%, mem=${mem}MiB), resetting" | tee -a "$LOGDIR/queue.log"
      fi
      consecutive=0
    fi
    sleep 30
  done
  echo "[$(date)] GPU is free, starting queue" | tee -a "$LOGDIR/queue.log"
}

wait_gpu_free

# --- Maze seed 4 ---
echo "[$(date)] starting maze seed $SEED finetune" | tee -a "$LOGDIR/queue.log"
"$PY" main.py \
  --mode train \
  --config configs/trm_official_maze_finetune.yaml \
  --seed "$SEED" \
  --init-weights "hf_checkpoints/Maze-Hard/remapped_for_local.pt" \
  > "$LOGDIR/maze_seed${SEED}.log" 2>&1
maze_rc=$?
echo "[$(date)] maze seed $SEED finished with rc=$maze_rc" | tee -a "$LOGDIR/queue.log"

# --- Sudoku seed 4 (regardless of maze rc — independent experiment) ---
echo "[$(date)] starting sudoku-mlp seed $SEED finetune" | tee -a "$LOGDIR/queue.log"
"$PY" main.py \
  --mode train \
  --config configs/trm_official_sudoku_mlp_finetune.yaml \
  --seed "$SEED" \
  --init-weights "hf_checkpoints/Sudoku-Extreme-mlp/remapped_for_local.pt" \
  > "$LOGDIR/sudoku_seed${SEED}.log" 2>&1
sudoku_rc=$?
echo "[$(date)] sudoku seed $SEED finished with rc=$sudoku_rc" | tee -a "$LOGDIR/queue.log"

echo "[$(date)] queue complete (maze rc=$maze_rc, sudoku rc=$sudoku_rc)" | tee -a "$LOGDIR/queue.log"
