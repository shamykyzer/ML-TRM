"""Tests for src.evaluation.inspection — renderers + failure dumper.

Same runner pattern as tests/test_encoding.py so both work with pytest
or plain `python tests/test_inspection.py`.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.encoding import encode_maze, encode_sudoku  # noqa: E402
from src.evaluation.inspection import (  # noqa: E402
    inspect_failures,
    render_maze,
    render_sudoku_board,
)


# ---------------------------------------------------------------------------
# Fixtures — reuse the same valid boards/mazes as tests/test_encoding.py
# ---------------------------------------------------------------------------

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

VALID_MAZE = (
    "#####\n"
    "#So##\n"
    "##o##\n"
    "##oG#\n"
    "#####"
)


# ---------------------------------------------------------------------------
# render_sudoku_board
# ---------------------------------------------------------------------------

def test_render_sudoku_board_includes_title():
    tokens = encode_sudoku(VALID_SOLUTION)
    out = render_sudoku_board(tokens, title="Test Board")
    assert "Test Board" in out, "title should appear in rendered output"


def test_render_sudoku_board_renders_blanks_as_dots():
    tokens = encode_sudoku(STARTING_PUZZLE)
    out = render_sudoku_board(tokens, title="")
    # STARTING_PUZZLE has 53 blanks — the rendered board should contain dots
    assert "." in out


def test_render_sudoku_board_renders_digits():
    tokens = encode_sudoku(VALID_SOLUTION)
    out = render_sudoku_board(tokens, title="")
    # Every digit 1..9 should appear at least once in a complete solution
    for d in "123456789":
        assert d in out, f"digit {d} missing from full-solution render"


def test_render_sudoku_board_has_nine_content_rows():
    """The 9x9 grid should produce exactly 9 rows of digit/dot cells.

    Box-divider lines may inflate the line count, so we grep for lines
    that start with '|' or contain at least three cell glyphs.
    """
    tokens = encode_sudoku(VALID_SOLUTION)
    out = render_sudoku_board(tokens, title="")
    # A row should contain at least 9 of the cell glyphs {digits, '.'}
    cell_rows = 0
    for line in out.split("\n"):
        glyphs = [ch for ch in line if ch.isdigit() or ch == "."]
        if len(glyphs) >= 9:
            cell_rows += 1
    assert cell_rows == 9, f"expected 9 cell rows, got {cell_rows}\n{out}"


# ---------------------------------------------------------------------------
# render_maze
# ---------------------------------------------------------------------------

def test_render_maze_recovers_original():
    tokens = encode_maze(VALID_MAZE)
    out = render_maze(tokens, grid_shape=(5, 5), title="")
    # Every line of VALID_MAZE should appear in the rendered output
    for line in VALID_MAZE.split("\n"):
        assert line in out, f"maze line {line!r} missing from render"


def test_render_maze_includes_title():
    tokens = encode_maze(VALID_MAZE)
    out = render_maze(tokens, grid_shape=(5, 5), title="Maze 42")
    assert "Maze 42" in out


# ---------------------------------------------------------------------------
# inspect_failures — tests use numpy arrays (no torch dependency)
# ---------------------------------------------------------------------------

def _make_sudoku_example(puzzle: str, solution: str):
    """Return (input_tokens, label_tokens_masked, solution_tokens).

    Matches the dataset convention: labels are masked where input == label
    (so the loss only applies to blank cells).
    """
    inp = encode_sudoku(puzzle)
    sol = encode_sudoku(solution)
    # Mask pre-filled positions
    lab = sol.copy()
    lab[inp == sol] = 0
    return inp, lab, sol


def test_inspect_failures_writes_only_failures():
    inp, lab, sol = _make_sudoku_example(STARTING_PUZZLE, VALID_SOLUTION)
    # 2-example batch: #0 correct, #1 wrong
    inputs = np.stack([inp, inp])
    labels = np.stack([lab, lab])
    correct_pred = sol.copy()
    wrong_pred = sol.copy()
    # Corrupt position 2 — it's '.' (blank) in STARTING_PUZZLE, so lab[2] != 0
    # and the failure is actually graded. Position 0 ('5', pre-filled) would
    # be masked out and the "wrong" pred would silently pass.
    wrong_pred[2] = (wrong_pred[2] % 9) + 2
    preds = np.stack([correct_pred, wrong_pred])

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "failures.txt")
        n_written = inspect_failures(
            inputs, preds, labels,
            task_type="sudoku",
            n_samples=10,
            out_path=out_path,
        )
        assert n_written == 1, f"expected 1 failure, got {n_written}"
        body = Path(out_path).read_text()
        # The failing example is index 1
        assert "1" in body  # index appears somewhere
        assert "INPUT" in body or "Input" in body or "input" in body


def test_inspect_failures_returns_zero_when_all_correct():
    inp, lab, sol = _make_sudoku_example(STARTING_PUZZLE, VALID_SOLUTION)
    inputs = np.stack([inp, inp, inp])
    labels = np.stack([lab, lab, lab])
    preds = np.stack([sol, sol, sol])

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "failures.txt")
        n_written = inspect_failures(
            inputs, preds, labels,
            task_type="sudoku",
            n_samples=5,
            out_path=out_path,
        )
        assert n_written == 0


def test_inspect_failures_respects_n_samples_limit():
    """Given 5 failures but n_samples=2, only 2 should be written."""
    inp, lab, sol = _make_sudoku_example(STARTING_PUZZLE, VALID_SOLUTION)
    wrong = sol.copy()
    wrong[2] = (wrong[2] % 9) + 2  # position 2 is blank in the puzzle
    inputs = np.stack([inp] * 5)
    labels = np.stack([lab] * 5)
    preds = np.stack([wrong] * 5)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "failures.txt")
        n_written = inspect_failures(
            inputs, preds, labels,
            task_type="sudoku",
            n_samples=2,
            out_path=out_path,
        )
        assert n_written == 2


def test_inspect_failures_creates_parent_directory():
    """The caller shouldn't have to pre-create results/figures/ etc."""
    inp, lab, sol = _make_sudoku_example(STARTING_PUZZLE, VALID_SOLUTION)
    wrong = sol.copy()
    wrong[2] = (wrong[2] % 9) + 2  # position 2 is blank in the puzzle
    inputs = np.stack([inp])
    labels = np.stack([lab])
    preds = np.stack([wrong])

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "nested", "subdir", "failures.txt")
        n_written = inspect_failures(
            inputs, preds, labels,
            task_type="sudoku",
            n_samples=1,
            out_path=out_path,
        )
        assert n_written == 1
        assert Path(out_path).exists()


def test_inspect_failures_maze_task():
    """Maze task uses grid_shape; verify it runs end-to-end."""
    correct_tokens = encode_maze(VALID_MAZE)
    # Flip one cell (that's not S or G) to create a failure
    wrong_tokens = correct_tokens.copy()
    wrong_tokens[0] = 2  # was wall '#' (token 1), now ' ' (token 2)

    inputs = np.stack([correct_tokens])
    labels = np.stack([correct_tokens])
    preds = np.stack([wrong_tokens])

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "maze_failures.txt")
        n_written = inspect_failures(
            inputs, preds, labels,
            task_type="maze",
            grid_shape=(5, 5),
            n_samples=5,
            out_path=out_path,
        )
        assert n_written == 1


# ---------------------------------------------------------------------------
# Stdlib runner
# ---------------------------------------------------------------------------

def _run_all():
    module = sys.modules[__name__]
    names = sorted(n for n in dir(module) if n.startswith("test_"))
    failures = []
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
