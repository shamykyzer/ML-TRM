"""Round-trip + validity tests for src.data.encoding.

Runs with either pytest OR plain `python tests/test_encoding.py` — the
bottom of the file contains a tiny runner so no new dev dependency is
required. The functions themselves use plain `assert`, so pytest picks
them up as test_* discovery targets without any extra boilerplate.

Token schema this file pins down (authoritative):
  Sudoku: 0=pad (unused by encoder), 1=blank, 2..10 = digits 1..9
          — matches data/build_sudoku_dataset.py (raw 0-9 with a +1 shift)
  Maze:   0=pad, 1='#', 2=' ', 3='S', 4='G', 5='o'
          — matches CHARSET in data/build_maze_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow `python tests/test_encoding.py` to import src.* without installing
# the package — the repo root is the parent of this tests/ folder.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.encoding import (  # noqa: E402  (sys.path hack above)
    decode_maze,
    decode_sudoku,
    encode_maze,
    encode_sudoku,
    is_valid_maze_path,
    is_valid_sudoku_solution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A real, valid, completed 9x9 Sudoku solution, written as an 81-char string
# in row-major order. Used as the ground truth for every validity test below.
VALID_SOLUTION = (
    "534678912"
    "672195348"
    "198342567"
    "859761423"
    "426853791"
    "713924856"
    "961537284"
    "287419635"
    "345286179"
)

# The matching starting puzzle (53 blanks) — the standard Wikipedia example.
STARTING_PUZZLE = (
    "53..7...."
    "6..195..."
    ".98....6."
    "8...6...3"
    "4..8.3..1"
    "7...2...6"
    ".6....28."
    "...419..5"
    "....8..79"
)

# A 5x5 maze with a valid S -> o -> o -> o -> G path (4-connected).
VALID_MAZE = (
    "#####\n"
    "#So##\n"
    "##o##\n"
    "##oG#\n"
    "#####"
)


# ---------------------------------------------------------------------------
# Sudoku encode / decode round-trip
# ---------------------------------------------------------------------------

def test_encode_sudoku_maps_dot_to_blank_token_one():
    """'.' should encode to stored token 1 (the paper-faithful blank code)."""
    tokens = encode_sudoku("." * 81)
    assert tokens.shape == (81,), f"expected (81,), got {tokens.shape}"
    assert (tokens == 1).all(), "all dots must become stored blank token 1"


def test_encode_sudoku_accepts_zero_as_blank():
    """'0' is also a blank — the reference CSV uses '.', but humans use '0'."""
    tokens = encode_sudoku("0" * 81)
    assert (tokens == 1).all()


def test_encode_sudoku_maps_digits_to_2_through_10():
    """'1'..'9' must become stored tokens 2..10 (not 1..9)."""
    tokens = encode_sudoku("123456789" * 9)
    expected = np.tile(np.arange(2, 11), 9)
    assert np.array_equal(tokens, expected), (
        f"digit encoding mismatch: first row expected {expected[:9]}, "
        f"got {tokens[:9]}"
    )


def test_encode_sudoku_rejects_wrong_length():
    try:
        encode_sudoku("1" * 80)
    except ValueError:
        return
    raise AssertionError("expected ValueError for 80-char input")


def test_encode_sudoku_rejects_invalid_character():
    try:
        encode_sudoku("A" + "1" * 80)
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid character 'A'")


def test_sudoku_roundtrip_complete_solution():
    """encode -> decode must be the identity on a complete solution."""
    tokens = encode_sudoku(VALID_SOLUTION)
    recovered = decode_sudoku(tokens)
    assert recovered == VALID_SOLUTION, (
        f"complete-solution round-trip failed:\n"
        f"  original:  {VALID_SOLUTION}\n"
        f"  recovered: {recovered}"
    )


def test_sudoku_roundtrip_puzzle_with_blanks():
    """encode -> decode canonicalises blanks to '.' (the paper convention)."""
    tokens = encode_sudoku(STARTING_PUZZLE)
    recovered = decode_sudoku(tokens)
    expected = STARTING_PUZZLE.replace("0", ".")
    assert recovered == expected, (
        f"blank round-trip failed:\n"
        f"  original:  {STARTING_PUZZLE}\n"
        f"  expected:  {expected}\n"
        f"  recovered: {recovered}"
    )


# ---------------------------------------------------------------------------
# Sudoku validity
# ---------------------------------------------------------------------------

def test_is_valid_sudoku_accepts_full_valid_solution():
    tokens = encode_sudoku(VALID_SOLUTION)
    assert is_valid_sudoku_solution(tokens) is True


def test_is_valid_sudoku_rejects_incomplete_board():
    """A board with blanks cannot be a valid solution."""
    tokens = encode_sudoku(STARTING_PUZZLE)
    assert is_valid_sudoku_solution(tokens) is False


def test_is_valid_sudoku_rejects_row_duplicate():
    """Corrupt row 0 so cell (0,0) equals cell (0,1) — same row has two 3s."""
    corrupted = list(VALID_SOLUTION)
    # VALID_SOLUTION row 0 is "534678912"; force (0,0) = (0,1) = '3'
    corrupted[0] = corrupted[1]  # now row 0 starts "33..."
    tokens = encode_sudoku("".join(corrupted))
    assert is_valid_sudoku_solution(tokens) is False


def test_is_valid_sudoku_rejects_column_duplicate():
    """Force col 0 to have two 5s (rows 0 and 1 both '5')."""
    corrupted = list(VALID_SOLUTION)
    # VALID_SOLUTION[0] = '5' (row 0 col 0), VALID_SOLUTION[9] = '6' (row 1 col 0)
    corrupted[9] = corrupted[0]  # row 1 col 0 becomes '5'
    tokens = encode_sudoku("".join(corrupted))
    assert is_valid_sudoku_solution(tokens) is False


# ---------------------------------------------------------------------------
# Maze encode / decode
# ---------------------------------------------------------------------------

def test_encode_maze_shape_matches_grid():
    tokens = encode_maze(VALID_MAZE)
    assert tokens.shape == (25,), f"5x5 maze should flatten to 25, got {tokens.shape}"


def test_encode_maze_assigns_correct_tokens():
    """Spot-check every CHARSET entry at a known position."""
    tokens = encode_maze(VALID_MAZE)
    # Row 1 of VALID_MAZE = "#So##", flat indices 5..9
    assert tokens[5] == 1, f"'#' should be 1, got {tokens[5]}"
    assert tokens[6] == 3, f"'S' should be 3, got {tokens[6]}"
    assert tokens[7] == 5, f"'o' should be 5, got {tokens[7]}"
    assert tokens[8] == 1
    assert tokens[9] == 1
    # Row 3 of VALID_MAZE = "##oG#", flat indices 15..19
    assert tokens[15] == 1
    assert tokens[16] == 1
    assert tokens[17] == 5
    assert tokens[18] == 4, f"'G' should be 4, got {tokens[18]}"


def test_encode_maze_rejects_unknown_character():
    try:
        encode_maze("#####\n#S@G#\n#####\n#####\n#####")  # '@' is not in CHARSET
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown maze character '@'")


def test_maze_roundtrip():
    tokens = encode_maze(VALID_MAZE)
    recovered = decode_maze(tokens, n_rows=5)
    assert recovered == VALID_MAZE, (
        f"maze round-trip failed:\n"
        f"  original:\n{VALID_MAZE}\n"
        f"  recovered:\n{recovered}"
    )


# ---------------------------------------------------------------------------
# Maze path validity
# ---------------------------------------------------------------------------

def test_is_valid_maze_path_accepts_connected_S_to_G():
    tokens = encode_maze(VALID_MAZE)
    assert is_valid_maze_path(tokens, grid_shape=(5, 5)) is True


def test_is_valid_maze_path_rejects_disconnected_islands():
    """S is walled off from the 'o' chain that reaches G."""
    disconnected = (
        "#####\n"
        "#S  #\n"  # S at (1,1), no 'o' neighbour
        "#####\n"  # impassable wall row
        "#ooG#\n"
        "#####"
    )
    tokens = encode_maze(disconnected)
    assert is_valid_maze_path(tokens, grid_shape=(5, 5)) is False


def test_is_valid_maze_path_rejects_missing_goal():
    """If the model forgot to place 'G', the maze is not solved."""
    no_goal = (
        "#####\n"
        "#So #\n"
        "##o #\n"
        "# o #\n"
        "#####"
    )
    tokens = encode_maze(no_goal)
    assert is_valid_maze_path(tokens, grid_shape=(5, 5)) is False


def test_is_valid_maze_path_rejects_missing_start():
    no_start = (
        "#####\n"
        "# o #\n"
        "##o #\n"
        "# oG#\n"
        "#####"
    )
    tokens = encode_maze(no_start)
    assert is_valid_maze_path(tokens, grid_shape=(5, 5)) is False


# ---------------------------------------------------------------------------
# Stdlib runner — lets `python tests/test_encoding.py` work without pytest
# ---------------------------------------------------------------------------

def _run_all():
    """Discover + run every top-level test_* function and report results."""
    module = sys.modules[__name__]
    names = sorted(n for n in dir(module) if n.startswith("test_"))
    failures: list[tuple[str, BaseException]] = []
    for name in names:
        fn = getattr(module, name)
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL {name}: {exc}")
        else:
            print(f"PASS {name}")
    print()
    print(f"{len(names) - len(failures)}/{len(names)} tests passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(_run_all())
