"""Reusable encoders, decoders, and correctness checkers for Sudoku + Maze.

This module is the single source of truth for the token schema used by the
TRM models. The inline encoding in `data/build_sudoku_dataset.py` and
`data/build_maze_dataset.py` is equivalent — these helpers exist so that
report code, evaluation scripts, and failure-inspection renderers can all
share one implementation instead of re-deriving the schema from the build
scripts.

Token schemas (authoritative):
    Sudoku: 0 = pad/ignore, 1 = blank, 2..10 = digits 1..9
            — matches data/build_sudoku_dataset.py:_seq_to_numpy (+1 shift)

    Maze:   0 = pad, 1 = '#', 2 = ' ', 3 = 'S', 4 = 'G', 5 = 'o'
            — matches CHARSET = "# SGo" in data/build_maze_dataset.py

Validity checkers are STRICT: `is_valid_sudoku_solution` only accepts
a fully-solved 9x9 board, and `is_valid_maze_path` only accepts mazes
with exactly one 'S', one 'G', and a 4-connected chain of 'o' cells
linking them. These exist so the evaluation pipeline can distinguish
"cell-accuracy says 99% correct" from "the produced board is actually
a valid solution the rules accept" — a distinction the argmax-based
metrics in src/evaluation/metrics.py cannot make.
"""
from __future__ import annotations

from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants — these names are the public schema. If they change,
# tests/test_encoding.py fails loudly, which is intentional.
# ---------------------------------------------------------------------------

# Sudoku
SUDOKU_SEQ_LEN = 81
SUDOKU_BLANK_TOKEN = 1
SUDOKU_DIGIT_OFFSET = 1  # stored = raw + 1 (raw 0 is blank)

# Maze — matches CHARSET order in data/build_maze_dataset.py
MAZE_CHARSET = "# SGo"  # index 0..4  -> token 1..5
MAZE_CHAR_TO_TOKEN = {ch: i + 1 for i, ch in enumerate(MAZE_CHARSET)}
MAZE_TOKEN_TO_CHAR = {v: k for k, v in MAZE_CHAR_TO_TOKEN.items()}
MAZE_TOKEN_WALL = MAZE_CHAR_TO_TOKEN["#"]   # 1
MAZE_TOKEN_OPEN = MAZE_CHAR_TO_TOKEN[" "]   # 2
MAZE_TOKEN_START = MAZE_CHAR_TO_TOKEN["S"]  # 3
MAZE_TOKEN_GOAL = MAZE_CHAR_TO_TOKEN["G"]   # 4
MAZE_TOKEN_PATH = MAZE_CHAR_TO_TOKEN["o"]   # 5


# ---------------------------------------------------------------------------
# Sudoku
# ---------------------------------------------------------------------------

def encode_sudoku(board: str) -> np.ndarray:
    """Encode an 81-character Sudoku string into the stored token form.

    Parameters
    ----------
    board : str
        81 characters in row-major order. Each character is one of:
        '.' or '0'  -> blank (stored token 1)
        '1'..'9'    -> digits 1..9 (stored tokens 2..10)

    Returns
    -------
    np.ndarray
        Shape (81,), dtype int64, values in {1, 2, ..., 10}.

    Raises
    ------
    ValueError
        If `board` is not exactly 81 characters, or contains any character
        outside the accepted set.
    """
    if len(board) != SUDOKU_SEQ_LEN:
        raise ValueError(
            f"Sudoku board must be {SUDOKU_SEQ_LEN} characters, got {len(board)}"
        )

    tokens = np.empty(SUDOKU_SEQ_LEN, dtype=np.int64)
    for i, ch in enumerate(board):
        if ch in (".", "0"):
            tokens[i] = SUDOKU_BLANK_TOKEN  # stored 1
        elif "1" <= ch <= "9":
            # raw digit d in 1..9 -> stored token d + 1 in 2..10
            tokens[i] = int(ch) + SUDOKU_DIGIT_OFFSET
        else:
            raise ValueError(
                f"Invalid Sudoku character {ch!r} at position {i}; "
                f"expected '.', '0', or '1'..'9'"
            )
    return tokens


def decode_sudoku(tokens: np.ndarray) -> str:
    """Decode stored Sudoku tokens back to an 81-character string.

    Blanks (stored token 1) are canonicalised to '.', matching the
    Sudoku-Extreme CSV convention in data/build_sudoku_dataset.py.
    """
    arr = np.asarray(tokens).reshape(-1)
    if arr.size != SUDOKU_SEQ_LEN:
        raise ValueError(
            f"Sudoku tokens must have length {SUDOKU_SEQ_LEN}, got {arr.size}"
        )

    chars: list[str] = []
    for t in arr.tolist():
        if t == SUDOKU_BLANK_TOKEN:
            chars.append(".")
        elif 2 <= t <= 10:
            # stored token 2..10 -> raw digit 1..9
            chars.append(str(t - SUDOKU_DIGIT_OFFSET))
        else:
            raise ValueError(
                f"Invalid Sudoku token {t}; expected 1..10"
            )
    return "".join(chars)


def is_valid_sudoku_solution(tokens: np.ndarray) -> bool:
    """Return True iff `tokens` is a complete, rule-valid 9x9 Sudoku.

    Checks (in order):
    1. Shape reshapes to (9, 9).
    2. No blanks remain (stored token 1 is forbidden).
    3. Every row contains each of the raw digits 1..9 exactly once.
    4. Every column contains each digit exactly once.
    5. Every 3x3 box contains each digit exactly once.

    Any failure short-circuits to False.
    """
    arr = np.asarray(tokens).reshape(-1)
    if arr.size != SUDOKU_SEQ_LEN:
        return False

    # Incomplete board: has blanks
    if (arr == SUDOKU_BLANK_TOKEN).any():
        return False

    # Values out of range would be a bug in the caller, not a "solution" — reject
    if ((arr < 2) | (arr > 10)).any():
        return False

    # Convert stored 2..10 to raw 1..9 for rule-checking
    raw = (arr - SUDOKU_DIGIT_OFFSET).reshape(9, 9)
    expected = set(range(1, 10))

    # Rows
    for r in range(9):
        if set(raw[r].tolist()) != expected:
            return False

    # Columns
    for c in range(9):
        if set(raw[:, c].tolist()) != expected:
            return False

    # 3x3 boxes
    for br in range(3):
        for bc in range(3):
            box = raw[br * 3:(br + 1) * 3, bc * 3:(bc + 1) * 3]
            if set(box.flatten().tolist()) != expected:
                return False

    return True


# ---------------------------------------------------------------------------
# Maze
# ---------------------------------------------------------------------------

def encode_maze(maze: str) -> np.ndarray:
    """Encode a multi-line maze string into the flattened token form.

    Rows are separated by '\\n'. Every character must be in MAZE_CHARSET
    ("# SGo"). The returned array is row-major flattened.

    Parameters
    ----------
    maze : str
        E.g. "#####\\n#So##\\n##o##\\n##oG#\\n#####" for a 5x5 maze.

    Returns
    -------
    np.ndarray
        Shape (H*W,), dtype int64, values in {1, 2, 3, 4, 5}.

    Raises
    ------
    ValueError
        If `maze` is empty, rows have inconsistent widths, or any
        character is not in MAZE_CHARSET.
    """
    rows = maze.split("\n")
    if not rows:
        raise ValueError("Maze must be non-empty")

    width = len(rows[0])
    if width == 0:
        raise ValueError("Maze rows must be non-empty")
    for r_idx, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"Maze row {r_idx} has width {len(row)}, expected {width}"
            )

    flat: list[int] = []
    for r_idx, row in enumerate(rows):
        for c_idx, ch in enumerate(row):
            if ch not in MAZE_CHAR_TO_TOKEN:
                raise ValueError(
                    f"Invalid maze character {ch!r} at ({r_idx},{c_idx}); "
                    f"expected one of {list(MAZE_CHARSET)!r}"
                )
            flat.append(MAZE_CHAR_TO_TOKEN[ch])

    return np.array(flat, dtype=np.int64)


def decode_maze(tokens: np.ndarray, n_rows: int) -> str:
    """Decode flat maze tokens back to a newline-separated string.

    Parameters
    ----------
    tokens : np.ndarray
        Shape (n_rows * n_cols,) with values in {1, 2, 3, 4, 5}.
    n_rows : int
        Number of rows in the maze (needed because tokens are flat).

    Returns
    -------
    str
        Multi-line string with '\\n' row separators, no trailing newline.
    """
    arr = np.asarray(tokens).reshape(-1)
    if arr.size % n_rows != 0:
        raise ValueError(
            f"Token length {arr.size} is not divisible by n_rows={n_rows}"
        )
    n_cols = arr.size // n_rows
    grid = arr.reshape(n_rows, n_cols)

    lines = []
    for r in range(n_rows):
        chars = []
        for c in range(n_cols):
            t = int(grid[r, c])
            if t not in MAZE_TOKEN_TO_CHAR:
                raise ValueError(
                    f"Invalid maze token {t} at ({r},{c}); expected 1..5"
                )
            chars.append(MAZE_TOKEN_TO_CHAR[t])
        lines.append("".join(chars))
    return "\n".join(lines)


def is_valid_maze_path(
    tokens: np.ndarray,
    grid_shape: tuple[int, int],
) -> bool:
    """Return True iff the 'o' chain forms a 4-connected path S -> G.

    Algorithm:
    1. Reshape tokens to `grid_shape`.
    2. Require exactly one 'S' cell and exactly one 'G' cell.
    3. BFS from S, enqueueing 4-connected neighbours whose token is
       'o' (path) or 'G' (goal). Walls ('#'), open space (' '), and
       out-of-range coordinates are rejected.
    4. Return True when G is popped; False if the queue drains first.

    Note: this does NOT require the path to be a shortest path — a
    longer but valid chain is still accepted. It DOES require the 'o'
    marker: empty space alone is not enough, so the model must mark
    the path explicitly.
    """
    arr = np.asarray(tokens).reshape(-1)
    H, W = grid_shape
    if arr.size != H * W:
        return False
    grid = arr.reshape(H, W)

    starts = np.argwhere(grid == MAZE_TOKEN_START)
    goals = np.argwhere(grid == MAZE_TOKEN_GOAL)
    if len(starts) != 1 or len(goals) != 1:
        return False

    start = (int(starts[0][0]), int(starts[0][1]))
    goal = (int(goals[0][0]), int(goals[0][1]))

    passable = {MAZE_TOKEN_PATH, MAZE_TOKEN_GOAL}
    visited: set[tuple[int, int]] = {start}
    q: deque[tuple[int, int]] = deque([start])

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                if int(grid[nr, nc]) in passable:
                    visited.add((nr, nc))
                    q.append((nr, nc))

    return False
