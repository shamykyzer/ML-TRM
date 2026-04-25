"""Plot the seed-4 fine-tune-failure story for run dz3tkge9.

Unlike the seed-0 sudoku-mlp run (slow rise to peak 0.7456 at epoch 900,
then graceful overfit decay) which is plotted by plot_sudoku_mlp_overfit.py,
seed-4 was initialised from the released HF checkpoint and tells a different
story:

    epoch 10  -> val_puzzle 0.8484 (= EMA still ~equal to loaded init)
    epoch 50  -> val_puzzle 0.6302 (catastrophic drop right after warmup)
    epoch 490 -> val_puzzle 0.7277 (recovered partially, never beat init)

Full diagnosis: analysis_run_dz3tkge9.md.
Output: results/figures/sudoku_mlp_finetune_failure.png
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

TRAIN_LOG = "C:/ml-trm-work/sudoku-mlp-seed4/trm_official_sudoku_train_log.csv"
OUT_PATH = "results/figures/sudoku_mlp_finetune_failure.png"
INIT_PUZZLE_ACC = 0.8484  # released HF checkpoint, observed at epoch 10


def _to_float(s):
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
                "avg_steps": _to_float(r.get("avg_steps")),
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
    train_exact = [r["train_exact"] for r in rows]
    train_acc = [r["train_acc"] for r in rows]
    avg_steps = [r["avg_steps"] for r in rows]

    crash_idx = min(range(len(rows)), key=lambda i: rows[i]["val_puzzle"])
    crash = rows[crash_idx]
    final = rows[-1]
    delta_to_init = INIT_PUZZLE_ACC - final["val_puzzle"]

    fig, (ax_val, ax_train, ax_steps) = plt.subplots(
        3, 1, figsize=(9.6, 9.0), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.1, 0.8]},
    )

    # Top: validation accuracy + horizontal line at the init's level.
    ax_val.axhline(
        INIT_PUZZLE_ACC, color="#2c3e50", linestyle="--", linewidth=1.3,
        alpha=0.85, label=f"loaded HF init = {INIT_PUZZLE_ACC:.4f}",
    )
    ax_val.plot(epochs, val_puzzle, label="val puzzle_acc", linewidth=2.0, color="#c0392b")
    ax_val.plot(epochs, val_cell, label="val cell_acc", linewidth=1.8, color="#2980b9", alpha=0.9)

    ax_val.annotate(
        f"post-warmup crash\n{crash['val_puzzle']:.4f} @ epoch {crash['epoch']}",
        xy=(crash["epoch"], crash["val_puzzle"]),
        xytext=(crash["epoch"] + 60, crash["val_puzzle"] - 0.05),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#c0392b"),
    )
    ax_val.annotate(
        f"final {final['val_puzzle']:.4f} @ ep {final['epoch']}\n"
        f"({delta_to_init * 100:.2f} pp below init)",
        xy=(final["epoch"], final["val_puzzle"]),
        xytext=(final["epoch"] - 230, final["val_puzzle"] - 0.16),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#c0392b"),
    )
    ax_val.set_ylabel("Validation accuracy")
    ax_val.set_ylim(0.55, 0.95)
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax_val.set_title(
        "TRM sudoku-mlp seed-4 fine-tune (run dz3tkge9): "
        "released checkpoint regressed by paper-faithful FT recipe",
        fontsize=11,
    )

    # Middle: training accuracy keeps climbing — classic overfit signature.
    ax_train.plot(epochs, train_acc, label="train cell_acc", linewidth=1.5, color="#16a085")
    ax_train.plot(epochs, train_exact, label="train puzzle_acc (exact)", linewidth=1.5, color="#27ae60")
    ax_train.set_ylabel("Training accuracy")
    ax_train.set_ylim(0.86, 1.0)
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Bottom: avg ACT steps collapses 4.2 -> 2.2, the halt head learns to halt
    # earlier on train than on test (halt_exploration_prob=0.1 perturbing it).
    ax_steps.plot(epochs, avg_steps, label="avg ACT halt steps (train)",
                  linewidth=1.5, color="#8e44ad")
    ax_steps.set_ylabel("ACT steps")
    ax_steps.set_xlabel("Epoch")
    ax_steps.set_ylim(0.0, 5.0)
    ax_steps.grid(True, alpha=0.3)
    ax_steps.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out_abs = REPO_ROOT / OUT_PATH
    out_abs.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_abs), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] wrote {out_abs}")
    print(f"[plot] rows used: {len(rows)} (epochs {epochs[0]}..{epochs[-1]})")
    print(f"[plot] crash trough: {crash['val_puzzle']:.4f} @ ep {crash['epoch']}")
    print(f"[plot] final val_puzzle: {final['val_puzzle']:.4f} @ ep {final['epoch']}")
    print(f"[plot] gap to init: {delta_to_init * 100:.2f} pp ({INIT_PUZZLE_ACC} -> {final['val_puzzle']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
