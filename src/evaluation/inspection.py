"""Human-readable renderers for Sudoku/Maze model outputs + failure dumper.

The coursework report needs "example puzzle visualizations — input /
prediction / ground truth" (see plan.md Phase 2 figure 5). The existing
eval pipeline only computes argmax accuracy; this module adds the
side-by-side rendering used to hand-pick figures and to sanity-check
qualitative failure modes.

Design: the renderers are pure string helpers, and `inspect_failures`
operates on already-computed tensors (inputs, preds, labels) so tests
don't need a live TRM model. A thesis script runs the model once,
then passes the results to `inspect_failures` as numpy or torch arrays.

All functions accept numpy arrays directly; if torch is available and
the caller passes torch tensors, they're detached and moved to CPU
first (handled by `_to_numpy_1d`).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.data.encoding import decode_maze, decode_sudoku


# ---------------------------------------------------------------------------
# Torch-optional: accept tensors if torch is installed, else fail explicitly
# ---------------------------------------------------------------------------

def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch.Tensor -> np.ndarray, else pass through."""
    if isinstance(x, np.ndarray):
        return x
    # Duck-type torch without importing it at module load
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Sudoku renderer
# ---------------------------------------------------------------------------

_SUDOKU_HORIZONTAL_RULE = "+-------+-------+-------+"


def render_sudoku_board(tokens, title: str = "") -> str:
    """Render a 9x9 Sudoku board with 3x3 box dividers and an optional title.

    Parameters
    ----------
    tokens : array-like of shape (81,)
        Stored Sudoku tokens (1 = blank, 2..10 = digits 1..9).
    title : str
        Optional header line. Empty string → no header.

    Returns
    -------
    str
        Multi-line string like::

            Example 3
            +-------+-------+-------+
            | 5 3 . | . 7 . | . . . |
            | 6 . . | 1 9 5 | . . . |
            ...
            +-------+-------+-------+
    """
    arr = _to_numpy(tokens).reshape(-1)
    decoded = decode_sudoku(arr)  # 81-char string, blanks = '.'

    lines = []
    if title:
        lines.append(title)
    lines.append(_SUDOKU_HORIZONTAL_RULE)
    for r in range(9):
        row_chars = decoded[r * 9:(r + 1) * 9]
        parts = ["|"]
        for c, ch in enumerate(row_chars):
            parts.append(ch)
            if (c + 1) % 3 == 0:
                parts.append("|")
        lines.append(" ".join(parts))
        if (r + 1) % 3 == 0:
            lines.append(_SUDOKU_HORIZONTAL_RULE)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Maze renderer
# ---------------------------------------------------------------------------

def render_maze(tokens, grid_shape: tuple[int, int], title: str = "") -> str:
    """Render a maze as a multi-line ASCII grid with an optional title.

    Parameters
    ----------
    tokens : array-like of shape (H*W,)
        Stored maze tokens (1='#', 2=' ', 3='S', 4='G', 5='o').
    grid_shape : tuple[int, int]
        (H, W) — needed because tokens are flat.
    title : str
        Optional header line.

    Returns
    -------
    str
        Multi-line string, title first (if any), then H lines of glyphs.
    """
    arr = _to_numpy(tokens).reshape(-1)
    H, _W = grid_shape
    body = decode_maze(arr, n_rows=H)
    if title:
        return f"{title}\n{body}"
    return body


# ---------------------------------------------------------------------------
# Failure dumper
# ---------------------------------------------------------------------------

def inspect_failures(
    inputs,
    preds,
    labels,
    task_type: str,
    *,
    grid_shape: tuple[int, int] | None = None,
    n_samples: int = 10,
    out_path: str = "results/failure_inspection.txt",
) -> int:
    """Find failing examples and write their input/pred/truth renders to a file.

    A puzzle counts as a "failure" when any non-ignored cell in the label
    disagrees with the prediction. The label-masking convention matches
    the datasets: labels[i] == 0 is the ignore sentinel (pre-filled cells
    for sudoku; walls/S/G/empty for maze).

    Parameters
    ----------
    inputs, preds, labels : array-like
        Shape (B, seq_len). Numpy arrays or torch tensors.
    task_type : str
        "sudoku" or "maze". Picks which renderer to use.
    grid_shape : tuple[int, int] | None
        Required for task_type="maze". Ignored for sudoku.
    n_samples : int
        Maximum number of failures to dump.
    out_path : str
        Where to write the text file. Parent directories are created.

    Returns
    -------
    int
        Number of failing examples actually written (0..n_samples).
    """
    if task_type not in ("sudoku", "maze"):
        raise ValueError(f"task_type must be 'sudoku' or 'maze', got {task_type!r}")
    if task_type == "maze" and grid_shape is None:
        raise ValueError("grid_shape is required when task_type='maze'")

    inp_np = _to_numpy(inputs)
    pred_np = _to_numpy(preds)
    lab_np = _to_numpy(labels)

    if inp_np.shape != pred_np.shape or inp_np.shape != lab_np.shape:
        raise ValueError(
            f"shape mismatch: inputs {inp_np.shape}, preds {pred_np.shape}, "
            f"labels {lab_np.shape}"
        )

    B = inp_np.shape[0]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(B):
            if written >= n_samples:
                break

            inp_i = inp_np[i]
            pred_i = pred_np[i]
            lab_i = lab_np[i]
            ignore_mask = lab_i == 0
            # A cell is wrong if it's not ignored AND pred != label.
            # A puzzle is a failure if any non-ignored cell is wrong.
            cell_correct = (pred_i == lab_i) | ignore_mask
            if cell_correct.all():
                continue  # This example is a PASS, skip it

            # Reconstruct the un-masked truth for rendering. Label is 0 on
            # positions the model didn't have to predict (pre-filled cells);
            # on those positions, the "truth" is whatever was in the input.
            truth_i = np.where(ignore_mask, inp_i, lab_i)

            f.write(f"\n{'=' * 60}\n")
            f.write(f"FAIL  example index {i}  ({task_type})\n")
            f.write(f"{'=' * 60}\n")

            if task_type == "sudoku":
                f.write(render_sudoku_board(inp_i, title="INPUT"))
                f.write("\n")
                f.write(render_sudoku_board(pred_i, title="PREDICTION"))
                f.write("\n")
                f.write(render_sudoku_board(truth_i, title="TRUTH"))
                f.write("\n")
            else:  # maze
                f.write(render_maze(inp_i, grid_shape, title="INPUT"))  # type: ignore[arg-type]
                f.write("\n\n")
                f.write(render_maze(pred_i, grid_shape, title="PREDICTION"))  # type: ignore[arg-type]
                f.write("\n\n")
                f.write(render_maze(truth_i, grid_shape, title="TRUTH"))  # type: ignore[arg-type]
                f.write("\n")

            written += 1

    return written
