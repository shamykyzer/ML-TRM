#!/usr/bin/env bash
# Manual fallback for the M4-format canonical copy block in
# scripts/launch_m6_qwen_fixb.sh, which silently no-op'd on seed 1
# (see findings.md §5.17). Run this after a Qwen Sudoku Fix-B seed
# trainer exits with rc=0 to produce the M4 headline-format files
# in C:/ml-trm-work/checkpoints to use/machine 6/.
#
# Usage:
#   bash scripts/m4_canonical_copy_seed.sh 1     # produces 04a_* files
#   bash scripts/m4_canonical_copy_seed.sh 2     # produces 04b_* files

set -eu
SEED="${1:?seed number required, e.g. 1 or 2}"

case "$SEED" in
    1) LETTER="a" ;;
    2) LETTER="b" ;;
    *) echo "ERROR: SEED must be 1 or 2 (got: $SEED)" >&2; exit 2 ;;
esac

SRC="C:/ml-trm-work/llm-qwen-sudoku-seed${SEED}-fixb"
DEST="C:/ml-trm-work/checkpoints to use/machine 6"
PFX="04${LETTER}_Qwen-0.5B_Sudoku_LoRA-FixB-seed${SEED}"

if [ ! -d "$SRC" ]; then
    echo "ERROR: source dir $SRC does not exist" >&2
    exit 1
fi
if [ ! -d "$DEST" ]; then
    echo "ERROR: destination dir $DEST does not exist" >&2
    exit 1
fi

# Source filenames are produced by trainer_llm.py; verified empirically
# on seed 1's run dir.
LATEST_PT="$SRC/qwen2.5_0.5b_sudoku_latest.pt"
TRAIN_CSV="$SRC/qwen2.5_0.5b_sudoku_train_log.csv"
EMISSIONS_CSV="$SRC/emissions.csv"
RESULTS_JSON="$SRC/qwen2.5_0.5b_sudoku_training_results.json"

copy_one() {
    local src="$1"
    local dst="$2"
    if [ ! -e "$src" ]; then
        echo "  skip (missing): $src" >&2
        return 0
    fi
    cp -pv "$src" "$dst"
}

echo "[m4-canonical-copy] seed $SEED -> $DEST"
copy_one "$LATEST_PT"     "$DEST/${PFX}-ep30.pt"
copy_one "$TRAIN_CSV"     "$DEST/${PFX}-train_log.csv"
copy_one "$EMISSIONS_CSV" "$DEST/${PFX}-emissions.csv"
copy_one "$RESULTS_JSON"  "$DEST/${PFX}-training_results.json"
echo "[m4-canonical-copy] done — verify with: ls -la \"$DEST/${PFX}-\"*"
