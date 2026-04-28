#!/usr/bin/env bash
# M6 launcher for Qwen Sudoku Fix-B seeds 1 and 2 (Contract A + B compliant).
# Each seed runs sequentially; seed 2 only starts if seed 1 returns 0.
#
# Watchdog (Contract A.4) is forked per seed before the trainer starts and
# killed after the trainer exits. Each seed's run dir is on a non-OneDrive
# path (C:/ml-trm-work/...), so OneDrive cloud-eviction can't take down
# checkpoints mid-run.
#
# Usage:
#   bash scripts/launch_m6_qwen_fixb.sh                # both seeds
#   bash scripts/launch_m6_qwen_fixb.sh seed1          # seed 1 only
#   bash scripts/launch_m6_qwen_fixb.sh seed2          # seed 2 only
#
# Logs:
#   /tmp/m6-watchdog-seed1.log
#   /tmp/m6-watchdog-seed2.log
#   /tmp/m6-train-seed1.log
#   /tmp/m6-train-seed2.log

set -u
WHICH="${1:-both}"
PYTHON=".venv/Scripts/python.exe"
WATCHDOG="scripts/checkpoint_redundancy_watchdog.sh"

run_seed() {
    local seed="$1"
    local cfg="configs/llm_qwen_sudoku_fixb_seed${seed}.yaml"
    local run_dir="C:/ml-trm-work/llm-qwen-sudoku-seed${seed}-fixb"
    local run_name="llm-qwen-sudoku-seed${seed}-fixb"
    # M4 README headline-format prefix: <row>_<arch>_<task>_<note>.pt
    # 04 = "Qwen-0.5B Sudoku LoRA" row from machine 6/README.md.
    # 04a/04b = seed-1/seed-2 variants (mirrors how 02b is a variant of 02_).
    local m4_letter
    if [ "$seed" -eq 1 ]; then m4_letter="a"; else m4_letter="b"; fi
    local m4_prefix="04${m4_letter}_Qwen-0.5B_Sudoku_LoRA-FixB-seed${seed}"
    local watchdog_log="/tmp/m6-watchdog-seed${seed}.log"
    local train_log="/tmp/m6-train-seed${seed}.log"
    local dest="C:/ml-trm-work/checkpoints to use/machine 6"

    mkdir -p "$run_dir"
    mkdir -p "$dest"

    echo "[$(date -Is)] [seed${seed}] starting watchdog -> $watchdog_log"
    bash "$WATCHDOG" 6 "$run_dir" "$run_name" > "$watchdog_log" 2>&1 &
    local watchdog_pid=$!
    echo "[$(date -Is)] [seed${seed}] watchdog pid=$watchdog_pid"

    echo "[$(date -Is)] [seed${seed}] starting trainer -> $train_log"
    "$PYTHON" main.py --mode train --config "$cfg" --seed "$seed" \
        > "$train_log" 2>&1
    local rc=$?
    echo "[$(date -Is)] [seed${seed}] trainer exit code=$rc"

    # Final timestamped snapshot (Contract A.3) — kept for cross-machine
    # aggregator compatibility.
    local ts
    ts=$(date +%Y-%m-%dT%H%M)
    for src in "$run_dir"/*.pt "$run_dir"/*.csv "$run_dir"/*.json; do
        [ -e "$src" ] || continue
        cp -p "$src" "$dest/${run_name}__${ts}__$(basename "$src")" 2>/dev/null || true
    done

    # Canonical M4-format copies for the report headline tables.
    # Only fire on rc=0 — failed runs shouldn't claim a row in the headline.
    if [ "$rc" -eq 0 ]; then
        local latest_pt="$run_dir/qwen2.5_0.5b_sudoku_latest.pt"
        local train_csv="$run_dir/qwen2.5_0.5b_sudoku_train_log.csv"
        local emissions_csv="$run_dir/emissions.csv"
        local results_json="$run_dir/qwen2.5_0.5b_sudoku_training_results.json"
        [ -e "$latest_pt" ] && cp -p "$latest_pt" "$dest/${m4_prefix}-ep30.pt"
        [ -e "$train_csv" ] && cp -p "$train_csv" "$dest/${m4_prefix}-train_log.csv"
        [ -e "$emissions_csv" ] && cp -p "$emissions_csv" "$dest/${m4_prefix}-emissions.csv"
        [ -e "$results_json" ] && cp -p "$results_json" "$dest/${m4_prefix}-training_results.json"
        echo "[$(date -Is)] [seed${seed}] M4-format canonical copies written: ${m4_prefix}-*"
    else
        echo "[$(date -Is)] [seed${seed}] rc=$rc — skipping M4 canonical copy (suspect run)"
    fi

    echo "[$(date -Is)] [seed${seed}] final snapshot done"
    kill "$watchdog_pid" 2>/dev/null || true
    return $rc
}

case "$WHICH" in
    seed1) run_seed 1 ;;
    seed2) run_seed 2 ;;
    both)
        run_seed 1
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "[$(date -Is)] seed 1 failed (rc=$rc); skipping seed 2 per Contract B"
            exit $rc
        fi
        run_seed 2
        ;;
    *)
        echo "Usage: $0 [seed1|seed2|both]" >&2
        exit 2
        ;;
esac
