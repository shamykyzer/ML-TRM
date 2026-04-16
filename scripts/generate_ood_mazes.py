"""Generate an out-of-distribution maze test set for the halt-sweep thesis chapter.

HRM's `build_maze_dataset.py` is just an HF downloader, not a generator — the
original sapientinc mazes are a fixed 1000+1000 set with no public generation
script. To test whether best.pt's one-pass solving generalizes beyond the
sapientinc distribution, we need 30x30 mazes from a *different* generator.

This script produces 1000 held-out mazes via a recursive-backtracker algorithm
on a 30x30 grid:

  - Perfect maze (tree topology, exactly one path between any two cells)
  - Structurally different from sapientinc's denser-wall distribution
  - Same 30x30 grid, same {#, ' ', S, G, o} charset, same seq_len=900
  - Same rating range (path length in [100, 160]) as sapientinc's "hard" band

Output: data/maze-30x30-hard-1k-ood/test/ in our expected .npy format, ready
for eval with `scripts/eval_hf_checkpoints.py` or `scripts/halt_sweep.py`
after pointing the data_dir at this new directory.

Usage:
    python scripts/generate_ood_mazes.py

Adjust COUNT, SEED, PATH_LEN_RANGE near the top to change the distribution.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import deque
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# Tunables
SIZE = 30
COUNT = 1000
SEED = 1337  # must differ from sapientinc's unknown seed; 1337 chosen arbitrarily
PATH_LEN_RANGE = (100, 160)  # match sapientinc's "hard" band (observed 110-154)
OUTPUT_DIR = "data/maze-30x30-hard-1k-ood"

# Match CHARSET from data/build_maze_dataset.py so tokens line up 1:1
CHARSET = "# SGo"  # token indices 1..5 after the +1 offset; 0 is pad


def _carve_maze(rng: random.Random) -> np.ndarray:
    """Return a SIZE x SIZE char grid of walls/passages using recursive backtracker.

    Conventions:
      - Start with all walls ('#')
      - Carve passages on every-other-cell (cells at (2i+1, 2j+1) become open)
      - Connect adjacent open cells by knocking the wall between them
      - Result has a tree topology (perfect maze: unique path between any two cells)
    """
    grid = np.full((SIZE, SIZE), '#', dtype='<U1')

    # Stack-based DFS over the (SIZE//2)*(SIZE//2) implicit cell grid
    # where cell (i, j) corresponds to maze position (2i+1, 2j+1).
    start_i = rng.randrange(SIZE // 2)
    start_j = rng.randrange(SIZE // 2)
    grid[2 * start_i + 1, 2 * start_j + 1] = ' '

    stack = [(start_i, start_j)]
    visited = {(start_i, start_j)}

    while stack:
        i, j = stack[-1]
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < SIZE // 2 and 0 <= nj < SIZE // 2 and (ni, nj) not in visited:
                neighbors.append((ni, nj, di, dj))
        if not neighbors:
            stack.pop()
            continue
        ni, nj, di, dj = rng.choice(neighbors)
        # Knock down wall between (i, j) and (ni, nj)
        grid[2 * i + 1 + di, 2 * j + 1 + dj] = ' '
        grid[2 * ni + 1, 2 * nj + 1] = ' '
        visited.add((ni, nj))
        stack.append((ni, nj))

    return grid


def _bfs_path(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> Optional[list[tuple[int, int]]]:
    """Return the shortest open-cell path from start to goal, or None if disconnected."""
    if grid[start] == '#' or grid[goal] == '#':
        return None
    q: deque[tuple[int, int]] = deque([start])
    prev: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            path = []
            node: Optional[tuple[int, int]] = cur
            while node is not None:
                path.append(node)
                node = prev[node]
            path.reverse()
            return path
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = cur[0] + di, cur[1] + dj
            if 0 <= ni < SIZE and 0 <= nj < SIZE and grid[ni, nj] != '#' and (ni, nj) not in prev:
                prev[(ni, nj)] = cur
                q.append((ni, nj))
    return None


def _one_maze(rng: random.Random) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
    """Generate one maze + solution pair. Returns (inp_grid, label_grid, path_len) or None.

    inp_grid: walls/open/S/G (no 'o')
    label_grid: same but path marked with 'o' at intermediate cells
    path_len: number of cells in the path including endpoints
    """
    grid = _carve_maze(rng)

    # Collect all open cells; pick a pair that's in the "hard" path-length range.
    open_cells = [(i, j) for i in range(SIZE) for j in range(SIZE) if grid[i, j] == ' ']
    if len(open_cells) < 2:
        return None

    # Try a handful of start/goal pairs; accept the first one whose shortest path
    # falls inside PATH_LEN_RANGE. Cap at 50 tries so we don't get stuck.
    for _ in range(50):
        s = rng.choice(open_cells)
        g = rng.choice(open_cells)
        if s == g:
            continue
        path = _bfs_path(grid, s, g)
        if path is None:
            continue
        if PATH_LEN_RANGE[0] <= len(path) <= PATH_LEN_RANGE[1]:
            inp = grid.copy()
            inp[s] = 'S'
            inp[g] = 'G'
            label = inp.copy()
            for (i, j) in path[1:-1]:
                label[i, j] = 'o'
            return inp, label, len(path)
    return None


def main() -> int:
    rng = random.Random(SEED)

    # char -> token id matching data/build_maze_dataset.py
    char2id = {c: i + 1 for i, c in enumerate(CHARSET)}  # '#'=1, ' '=2, 'S'=3, 'G'=4, 'o'=5

    inputs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    path_lens: list[int] = []

    attempts = 0
    while len(inputs_list) < COUNT:
        attempts += 1
        if attempts > COUNT * 20:
            print(f"gave up after {attempts} attempts with {len(inputs_list)} mazes", file=sys.stderr)
            return 1
        result = _one_maze(rng)
        if result is None:
            continue
        inp, lab, plen = result
        inp_tok = np.vectorize(char2id.get)(inp).astype(np.uint8).reshape(-1)
        lab_tok = np.vectorize(char2id.get)(lab).astype(np.uint8).reshape(-1)
        inputs_list.append(inp_tok)
        labels_list.append(lab_tok)
        path_lens.append(plen)
        if len(inputs_list) % 100 == 0:
            print(f"  generated {len(inputs_list)}/{COUNT} (attempts={attempts})")

    inputs = np.vstack(inputs_list)
    labels = np.vstack(labels_list)
    group_indices = np.arange(COUNT + 1, dtype=np.int32)
    puzzle_indices = np.arange(COUNT + 1, dtype=np.int32)
    puzzle_identifiers = np.zeros(COUNT, dtype=np.int32)

    save_dir = os.path.join(OUTPUT_DIR, "test")
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "all__inputs.npy"), inputs)
    np.save(os.path.join(save_dir, "all__labels.npy"), labels)
    np.save(os.path.join(save_dir, "all__group_indices.npy"), group_indices)
    np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), puzzle_indices)
    np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)

    metadata = {
        "seq_len": SIZE * SIZE,
        "vocab_size": len(CHARSET) + 1,
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
        "total_groups": COUNT,
        "mean_puzzle_examples": 1,
        "total_puzzles": COUNT,
        "sets": ["all"],
    }
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f)

    # Also create an empty train/ dir so MazeDataset can load "test" without asking
    # for a train split. The eval pipeline only touches the "test" subdir anyway.
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    print()
    print(f"Wrote {COUNT} OOD mazes to {OUTPUT_DIR}/test/")
    print(f"Path length stats: min={min(path_lens)} max={max(path_lens)} mean={np.mean(path_lens):.1f}")
    print(f"Total attempts: {attempts} (acceptance rate {COUNT/attempts:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
