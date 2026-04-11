"""End-to-end smoke test for datasets, encoding schema, and TRM forward pass.

Run this before launching multi-day training runs to catch schema mismatches,
bad checkpoints, or NaN-producing weights early. The script is intentionally
tiny and CPU-only — it does NOT instantiate ``TRMOfficial`` (which requires
CUDA + bf16) and does NOT touch ``experiments/`` or the real checkpoint dir,
so it is safe to run at any time, including during a live training job.

What this checks
----------------
1. ``get_sudoku_loaders`` / ``get_maze_loaders`` can load a 4-example batch
   from the processed ``.npy`` files under ``data/``.
2. The token histograms fall inside the expected schema ranges
   (sudoku: 0..10, maze: 0..5) — catches a bad preprocessing run where
   an unexpected token leaked through.
3. One sudoku example renders cleanly via
   ``src.evaluation.inspection.render_sudoku_board``.
4. A tiny ``TRMSudoku`` (``d_model=64, ff_hidden=128``) can run
   ``embedding → block → output_head`` on the batch without producing
   any NaN / Inf values.

Exits 0 on success, non-zero on any failure.

Usage
-----
    python scripts/sanity_check.py
    python scripts/sanity_check.py --sudoku-data data/sudoku-extreme-full
    python scripts/sanity_check.py --skip-maze    # e.g. if maze data missing
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Token schema expectations from src/data/encoding.py
SUDOKU_TOKEN_RANGE = (0, 10)   # inclusive [pad, digit-9]
MAZE_TOKEN_RANGE = (0, 5)      # inclusive [pad, 'o']


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

class CheckResult:
    """Accumulator for pass/fail counts + a last-line-of-defence error dump."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def passed_(self, msg: str) -> None:
        self.passed += 1
        print(f"  [PASS] {msg}")

    def failed_(self, msg: str, exc: BaseException | None = None) -> None:
        self.failed += 1
        print(f"  [FAIL] {msg}")
        if exc is not None:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            for line in tb.rstrip().splitlines():
                print(f"         {line}")


def _check_token_range(
    label: str,
    tokens: np.ndarray,
    low: int,
    high: int,
    result: CheckResult,
) -> None:
    """Histogram the tokens and verify every value falls in ``[low, high]``."""
    values, counts = np.unique(tokens, return_counts=True)
    hist = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(values, counts))
    print(f"    {label} token histogram: {hist}")
    out_of_range = [int(v) for v in values if v < low or v > high]
    if out_of_range:
        result.failed_(
            f"{label} has out-of-range tokens {out_of_range} "
            f"(expected [{low}, {high}])"
        )
    else:
        result.passed_(
            f"{label} tokens within expected range [{low}, {high}]"
        )


# ---------------------------------------------------------------------------
# Dataset checks
# ---------------------------------------------------------------------------

def check_sudoku(data_dir: str, result: CheckResult) -> None:
    print(f"\n[1] Sudoku dataset  ({data_dir})")
    if not os.path.isdir(data_dir):
        result.failed_(f"data dir not found: {data_dir}")
        return

    try:
        from src.data.sudoku_dataset import get_sudoku_loaders
        train_loader, _ = get_sudoku_loaders(
            data_dir, batch_size=4, num_workers=0
        )
        batch = next(iter(train_loader))
    except Exception as exc:  # noqa: BLE001
        result.failed_("failed to load a 4-example batch", exc)
        return

    inputs, labels = batch[0], batch[1]
    print(f"    inputs.shape={tuple(inputs.shape)}  "
          f"dtype={inputs.dtype}")
    result.passed_(f"loaded 4-example batch shape {tuple(inputs.shape)}")

    if inputs.shape[1] != 81:
        result.failed_(f"expected seq_len 81, got {inputs.shape[1]}")
    else:
        result.passed_("seq_len = 81")

    _check_token_range("inputs", inputs.numpy(), *SUDOKU_TOKEN_RANGE, result=result)
    _check_token_range("labels", labels.numpy(), *SUDOKU_TOKEN_RANGE, result=result)

    # Pretty-print one example to verify the renderer works end-to-end.
    try:
        from src.evaluation.inspection import render_sudoku_board
        board = render_sudoku_board(inputs[0], title="Sudoku — example 0 (input)")
        print("\n" + "\n".join("    " + line for line in board.splitlines()))
        result.passed_("render_sudoku_board rendered one example")
    except Exception as exc:  # noqa: BLE001
        result.failed_("render_sudoku_board raised", exc)
        return

    # Tiny TRM forward pass — just to confirm the weights don't NaN out.
    try:
        import torch
        from src.models.trm_sudoku import TRMSudoku
        torch.manual_seed(0)
        model = TRMSudoku(
            vocab_size=11, seq_len=81, d_model=64, ff_hidden=128,
            num_classes=11,
        )
        model.eval()
        with torch.no_grad():
            x = model.embedding(inputs)           # [B, L, D]
            x = x + model.y_init + model.z_init   # mirror init path
            x = model.block(x)                    # one shared-block pass
            logits = model.output_head(x)         # [B, L, num_classes]
    except Exception as exc:  # noqa: BLE001
        result.failed_("TRMSudoku forward pass raised", exc)
        return

    if not torch.isfinite(logits).all():
        n_bad = int((~torch.isfinite(logits)).sum().item())
        result.failed_(f"TRMSudoku forward produced {n_bad} non-finite logits")
    else:
        result.passed_(
            f"TRMSudoku forward OK  (logits shape {tuple(logits.shape)}, "
            f"finite, {sum(p.numel() for p in model.parameters()):,} params)"
        )


def check_maze(data_dir: str, result: CheckResult) -> None:
    print(f"\n[2] Maze dataset  ({data_dir})")
    if not os.path.isdir(data_dir):
        result.failed_(f"data dir not found: {data_dir}")
        return

    try:
        from src.data.maze_dataset import get_maze_loaders
        train_loader, _ = get_maze_loaders(
            data_dir, batch_size=4, num_workers=0
        )
        batch = next(iter(train_loader))
    except Exception as exc:  # noqa: BLE001
        result.failed_("failed to load a 4-example batch", exc)
        return

    inputs, labels = batch[0], batch[1]
    print(f"    inputs.shape={tuple(inputs.shape)}  "
          f"dtype={inputs.dtype}")
    result.passed_(f"loaded 4-example batch shape {tuple(inputs.shape)}")

    # Infer grid_shape for the renderer (maze is square in our pipeline).
    seq_len = int(inputs.shape[1])
    side = int(round(seq_len ** 0.5))
    if side * side != seq_len:
        result.failed_(
            f"maze seq_len {seq_len} is not a square — cannot infer grid shape"
        )
        return
    result.passed_(f"inferred grid shape {side}x{side}  (seq_len={seq_len})")

    _check_token_range("inputs", inputs.numpy(), *MAZE_TOKEN_RANGE, result=result)
    _check_token_range("labels", labels.numpy(), *MAZE_TOKEN_RANGE, result=result)

    # Render one maze — 30x30 is too big for stdout but we still want to
    # confirm the function doesn't crash.
    try:
        from src.evaluation.inspection import render_maze
        rendered = render_maze(
            inputs[0], grid_shape=(side, side),
            title=f"Maze — example 0 ({side}x{side})",
        )
        # Print only the first 6 lines so the log stays readable.
        head = "\n".join(rendered.splitlines()[:6])
        print("\n" + "\n".join("    " + line for line in head.splitlines()))
        if len(rendered.splitlines()) > 6:
            print(f"    ... ({len(rendered.splitlines()) - 6} more rows)")
        result.passed_("render_maze rendered one example")
    except Exception as exc:  # noqa: BLE001
        result.failed_("render_maze raised", exc)
        return

    # Tiny TRM forward pass for maze.
    try:
        import torch
        from src.models.trm_sudoku import TRMMaze
        torch.manual_seed(0)
        model = TRMMaze(
            vocab_size=6, seq_len=seq_len, d_model=64, ff_hidden=128,
            num_classes=6, n_heads=4,
        )
        model.eval()
        with torch.no_grad():
            x = model.embedding(inputs)
            x = x + model.y_init + model.z_init
            x = model.block(x)
            logits = model.output_head(x)
    except Exception as exc:  # noqa: BLE001
        result.failed_("TRMMaze forward pass raised", exc)
        return

    if not torch.isfinite(logits).all():
        n_bad = int((~torch.isfinite(logits)).sum().item())
        result.failed_(f"TRMMaze forward produced {n_bad} non-finite logits")
    else:
        result.passed_(
            f"TRMMaze forward OK  (logits shape {tuple(logits.shape)}, "
            f"finite, {sum(p.numel() for p in model.parameters()):,} params)"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sudoku-data",
        default="data/sudoku-extreme-full",
        help="Processed sudoku data dir. (default: data/sudoku-extreme-full)",
    )
    parser.add_argument(
        "--maze-data",
        default="data/maze-30x30-hard-1k",
        help="Processed maze data dir. (default: data/maze-30x30-hard-1k)",
    )
    parser.add_argument("--skip-sudoku", action="store_true")
    parser.add_argument("--skip-maze", action="store_true")
    args = parser.parse_args(argv)

    print("=" * 64)
    print("TRM sanity check — datasets, schema, tiny forward pass")
    print("=" * 64)

    result = CheckResult()

    if not args.skip_sudoku:
        check_sudoku(args.sudoku_data, result)
    else:
        print("\n[1] Sudoku dataset  (skipped)")

    if not args.skip_maze:
        check_maze(args.maze_data, result)
    else:
        print("\n[2] Maze dataset  (skipped)")

    total = result.passed + result.failed
    print()
    print("=" * 64)
    print(f"{result.passed}/{total} checks passed")
    print("=" * 64)
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
