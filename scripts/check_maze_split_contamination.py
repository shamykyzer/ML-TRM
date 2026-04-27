"""Check whether maze train/test splits share base puzzles.

Hashes each input grid (shape: 900 ints per row) and counts overlaps
between train and test of a given dataset directory. Run on both
``data/maze-30x30-hard-1k`` (non-augmented, expected clean) and
``data/maze-30x30-hard-1k-aug`` (augmented, possibly contaminated)
and print the result.

Decision rule for the sprint:
    overlap == 0  -> augmented split is clean; only the eval-mask
                     fix is required (no retrain needed)
    overlap > 0   -> contaminated; switch all subsequent LLM maze
                     retraining/eval to ``maze-30x30-hard-1k``

Usage:
    python scripts/check_maze_split_contamination.py
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hash_rows(arr: np.ndarray) -> set[str]:
    """Return a set of sha256 hex digests, one per row."""
    return {hashlib.sha256(row.tobytes()).hexdigest()[:16] for row in arr}


def check(data_dir: str) -> dict:
    train_path = os.path.join(data_dir, "train", "all__inputs.npy")
    test_path = os.path.join(data_dir, "test", "all__inputs.npy")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return {"data_dir": data_dir, "error": "missing all__inputs.npy"}

    train = np.load(train_path)
    test = np.load(test_path)
    train_h = hash_rows(train)
    test_h = hash_rows(test)
    overlap = train_h & test_h
    return {
        "data_dir": data_dir,
        "train_rows": int(train.shape[0]),
        "test_rows": int(test.shape[0]),
        "train_unique": len(train_h),
        "test_unique": len(test_h),
        "overlap_count": len(overlap),
        "verdict": "CLEAN" if not overlap else "CONTAMINATED",
    }


def main() -> int:
    targets = [
        os.path.join(REPO_ROOT, "data", "maze-30x30-hard-1k-aug"),
        os.path.join(REPO_ROOT, "data", "maze-30x30-hard-1k"),
    ]
    for d in targets:
        result = check(d)
        print(f"\n=== {os.path.basename(d)} ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
