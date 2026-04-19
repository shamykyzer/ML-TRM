"""Plot the sudoku-att rise-and-collapse story figure for the report.

The from-scratch TRM attention variant peaked at puzzle_acc 0.1833 around
epoch 100 then regressed to 0.0 by epoch 500 while training metrics kept
improving. That's a classic overfitting / EMA-vs-raw divergence signature
worth a dedicated figure in the Discussion section.

Output: results/figures/sudoku_att_rise_and_collapse.png
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TRAIN_LOG = "experiments/sudoku-att/trm_official_sudoku_train_log.csv"
OUT_PATH = "results/figures/sudoku_att_rise_and_collapse.png"


def _to_float(s) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            epoch = _to_float(r.get("epoch"))
            if epoch is None:
                continue
            rows.append({
                "epoch": int(epoch),
                "lm_loss": _to_float(r.get("lm_loss")),
                "train_acc": _to_float(r.get("accuracy")),
                "train_exact": _to_float(r.get("exact_accuracy")),
                "val_cell": _to_float(r.get("val_cell_acc")),
                "val_puzzle": _to_float(r.get("val_puzzle_acc")),
                "best_puzzle": _to_float(r.get("best_puzzle_acc")),
            })
    # Sort by epoch — resume rows can arrive non-monotonically.
    rows.sort(key=lambda r: r["epoch"])
    return rows


def main() -> int:
    if not os.path.isfile(TRAIN_LOG):
        print(f"ERROR: train log not found at {TRAIN_LOG}", file=sys.stderr)
        return 1

    rows = _load_rows(TRAIN_LOG)
    if not rows:
        print("ERROR: no rows parsed", file=sys.stderr)
        return 1

    epochs = [r["epoch"] for r in rows]
    val_puzzle = [r["val_puzzle"] if r["val_puzzle"] is not None else float("nan") for r in rows]
    val_cell = [r["val_cell"] if r["val_cell"] is not None else float("nan") for r in rows]
    train_acc = [r["train_acc"] if r["train_acc"] is not None else float("nan") for r in rows]
    train_exact = [r["train_exact"] if r["train_exact"] is not None else float("nan") for r in rows]

    peak_epoch = max(
        (r for r in rows if r["val_puzzle"] is not None),
        key=lambda r: r["val_puzzle"],
    )
    peak_x = peak_epoch["epoch"]
    peak_y = peak_epoch["val_puzzle"]
    final_row = rows[-1]

    fig, (ax_val, ax_train) = plt.subplots(
        2, 1, figsize=(9, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )

    # Top: validation — the story panel
    ax_val.plot(epochs, val_puzzle, label="val puzzle_acc", linewidth=2.0, color="#c0392b")
    ax_val.plot(epochs, val_cell, label="val cell_acc", linewidth=1.8, color="#2980b9", alpha=0.9)
    ax_val.axvline(peak_x, color="#2c3e50", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_val.annotate(
        f"peak puzzle_acc = {peak_y:.4f}\nat epoch {peak_x}",
        xy=(peak_x, peak_y),
        xytext=(peak_x + 40, peak_y + 0.18),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#2c3e50"),
    )
    ax_val.annotate(
        f"collapsed to {final_row['val_puzzle']:.4f}\nby epoch {final_row['epoch']}",
        xy=(final_row["epoch"], final_row["val_puzzle"] or 0.0),
        xytext=(final_row["epoch"] - 200, 0.35),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#c0392b"),
    )
    ax_val.set_ylabel("Validation accuracy")
    ax_val.set_ylim(-0.02, 1.02)
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_val.set_title(
        "TRM sudoku-att (from scratch): validation rise-and-collapse",
        fontsize=11,
    )

    # Bottom: training — the counter-panel. Training metrics keep improving.
    ax_train.plot(epochs, train_acc, label="train accuracy (cell)", linewidth=1.5, color="#16a085")
    ax_train.plot(epochs, train_exact, label="train exact_accuracy (puzzle)", linewidth=1.5, color="#27ae60")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Training accuracy")
    ax_train.set_ylim(-0.02, 1.02)
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] wrote {OUT_PATH}")
    print(f"[plot] rows used: {len(rows)} (epochs {epochs[0]}..{epochs[-1]})")
    print(f"[plot] peak val_puzzle = {peak_y:.4f} at epoch {peak_x}")
    print(f"[plot] final val_puzzle = {final_row['val_puzzle']:.4f} at epoch {final_row['epoch']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
