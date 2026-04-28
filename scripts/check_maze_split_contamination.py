"""Maze train/test split contamination diagnostic.

Compares puzzle identifier sets between the train and test splits of both
maze datasets used in this project — `data/maze-30x30-hard-1k` (clean) and
`data/maze-30x30-hard-1k-aug` (augmented). Reports overlap counts so the
team can decide whether the augmented dataset's "test" split is a leakage
of training puzzles (rotations / reflections of training mazes scored as
held-out test) — which would inflate any Maze accuracy number, not just
the LLM ones blocked by the `mask_non_path: true` bug.

Output: writes a self-contained text report to
`results/diagnostics/maze_dataset_overlap.txt` AND prints to stdout.

Pure numpy, no torch, no GPU. Read-only on the .npy files. Exits 0 on
success regardless of what it finds — a contamination finding is data
not failure.

Usage:
    python scripts/check_maze_split_contamination.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
DATASETS = ("maze-30x30-hard-1k", "maze-30x30-hard-1k-aug")
SPLITS = ("train", "test")
OUT_PATH = os.path.join(REPO_ROOT, "results", "diagnostics", "maze_dataset_overlap.txt")


def load_ids(dataset: str, split: str) -> np.ndarray:
    """Load the puzzle_identifiers.npy for one (dataset, split) pair.

    Returns an empty array if the file isn't present (so the report can
    still describe what's missing rather than crashing).
    """
    path = os.path.join(DATA_DIR, dataset, split, "all__puzzle_identifiers.npy")
    if not os.path.exists(path):
        return np.array([], dtype=np.int64)
    return np.load(path)


def describe_dataset(dataset: str) -> str:
    """Return a multi-line report block for one dataset."""
    lines = [f"## {dataset}"]
    train_ids = load_ids(dataset, "train")
    test_ids = load_ids(dataset, "test")

    if train_ids.size == 0 and test_ids.size == 0:
        lines.append("  (dataset not present on disk)")
        return "\n".join(lines)

    lines.append(f"  train rows           : {train_ids.size}")
    lines.append(f"  train unique ids     : {np.unique(train_ids).size}")
    lines.append(f"  test  rows           : {test_ids.size}")
    lines.append(f"  test  unique ids     : {np.unique(test_ids).size}")

    train_set = set(train_ids.tolist()) if train_ids.size else set()
    test_set = set(test_ids.tolist()) if test_ids.size else set()
    overlap = train_set & test_set

    lines.append(f"  ids in BOTH splits   : {len(overlap)}")

    if overlap:
        sample = sorted(overlap)[:10]
        lines.append(f"  overlap sample (≤10) : {sample}")
        contam_pct = 100.0 * len(overlap) / max(len(test_set), 1)
        lines.append(
            f"  CONTAMINATION FLAG   : {contam_pct:.1f}% of test ids appear in train — "
            f"reported accuracy on this dataset is likely inflated by memorisation"
        )
    else:
        lines.append("  CLEAN                : no test id appears in train (no overlap)")

    return "\n".join(lines)


def main() -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = [
        "Maze dataset train/test split overlap diagnostic",
        f"Generated: {now}",
        f"Repo: {REPO_ROOT}",
        "",
        "Compares puzzle_identifiers.npy between train and test splits for both",
        "the clean and augmented maze datasets. Overlap > 0 means augmented",
        "rotations of training puzzles appear in the test split, which inflates",
        "any Maze accuracy claim under exact-match grading.",
        "",
    ]
    body = [describe_dataset(ds) for ds in DATASETS]
    report = "\n".join(header + body)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(report)
    print(f"\nWritten to: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
