"""Plot the sudoku-mlp peak-and-overfit story figure for the report.

HF-init TRM-MLP was fine-tuned on consumer GPU; val_puzzle_acc climbed smoothly
from ~0.63 (epoch 50) to a peak 0.7456 at epoch 900, then decayed gracefully
back to 0.5948 by epoch 2245. Training was stopped when the overfitting trend
became unambiguous. This figure shows both phases so the reader can see why
the peak — not the final — is the reported number.

Output: results/figures/sudoku_mlp_peak_and_overfit.png
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


TRAIN_LOG = "C:/ml-trm-work/sudoku-mlp-seed0/trm_official_sudoku_train_log.csv"
OUT_PATH = "results/figures/sudoku_mlp_peak_and_overfit.png"


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
    rows.sort(key=lambda r: r["epoch"])
    return rows


def main() -> int:
    if not os.path.isfile(TRAIN_LOG):
        print(f"ERROR: train log not found at {TRAIN_LOG}", file=sys.stderr)
        return 1

    rows = _load_rows(TRAIN_LOG)
    rows = [r for r in rows if r["val_puzzle"] is not None]
    if not rows:
        print("ERROR: no rows with val_puzzle_acc", file=sys.stderr)
        return 1

    epochs = [r["epoch"] for r in rows]
    val_puzzle = [r["val_puzzle"] for r in rows]
    val_cell = [r["val_cell"] for r in rows]
    train_acc = [r["train_acc"] for r in rows]
    train_exact = [r["train_exact"] for r in rows]

    peak = max(rows, key=lambda r: r["val_puzzle"])
    peak_x, peak_y = peak["epoch"], peak["val_puzzle"]
    final = rows[-1]
    delta = peak_y - final["val_puzzle"]

    fig, (ax_val, ax_train) = plt.subplots(
        2, 1, figsize=(9.2, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )

    # Top: validation — rise to peak, then graceful decline
    ax_val.plot(epochs, val_puzzle, label="val puzzle_acc", linewidth=2.0, color="#c0392b")
    ax_val.plot(epochs, val_cell, label="val cell_acc", linewidth=1.8, color="#2980b9", alpha=0.9)
    ax_val.axvline(peak_x, color="#2c3e50", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_val.axhline(peak_y, color="#c0392b", linestyle=":", linewidth=0.7, alpha=0.5)
    ax_val.annotate(
        f"peak puzzle_acc = {peak_y:.4f}\nat epoch {peak_x}\n(stopping point)",
        xy=(peak_x, peak_y),
        xytext=(peak_x + 300, peak_y + 0.10),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#2c3e50"),
    )
    ax_val.annotate(
        f"overfit: {final['val_puzzle']:.4f}\nat epoch {final['epoch']}\n(Δ −{delta:.4f})",
        xy=(final["epoch"], final["val_puzzle"]),
        xytext=(final["epoch"] - 700, final["val_puzzle"] - 0.18),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#c0392b"),
    )
    ax_val.set_ylabel("Validation accuracy")
    ax_val.set_ylim(0.0, 1.0)
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax_val.set_title(
        "TRM sudoku-mlp (HF-init + fine-tune): peak at epoch "
        f"{peak_x}, overfit thereafter",
        fontsize=11,
    )

    # Bottom: training — keeps improving, the classic overfit signature
    ax_train.plot(epochs, train_acc, label="train accuracy (cell)", linewidth=1.5, color="#16a085")
    ax_train.plot(epochs, train_exact, label="train exact_accuracy (puzzle)", linewidth=1.5, color="#27ae60")
    ax_train.axvline(peak_x, color="#2c3e50", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Training accuracy")
    ax_train.set_ylim(0.0, 1.02)
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] wrote {OUT_PATH}")
    print(f"[plot] rows used: {len(rows)} (epochs {epochs[0]}..{epochs[-1]})")
    print(f"[plot] peak val_puzzle = {peak_y:.4f} at epoch {peak_x}")
    print(f"[plot] final val_puzzle = {final['val_puzzle']:.4f} at epoch {final['epoch']}")
    print(f"[plot] peak-to-final drop: {delta:.4f} ({delta * 100:.2f} pp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
