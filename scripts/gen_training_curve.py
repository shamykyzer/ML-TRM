"""Regenerate paper/fig_training_curve.pdf showing all 3 TRM-MLP seeds.

Reads per-epoch train logs extracted from machine-4 archives:
  /tmp/trm_logs/seed{0,1,2}_trm_official_sudoku_train_log.csv

Single-panel figure: val_puzzle_acc (solid) and train exact_accuracy (dashed)
vs epoch for all 3 seeds, with peak markers and a shaded band over 1-500.

Output: /home/kaizer/ML-TRM/paper/fig_training_curve.pdf
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = Path("/tmp/trm_logs")
OUT = Path("/home/kaizer/ML-TRM/paper/fig_training_curve.pdf")

SEED_COLORS = {0: "#c0392b", 1: "#2980b9", 2: "#27ae60"}
SEED_PEAK_EPOCH = {0: 900, 1: 500, 2: 700}


def _f(s: str) -> float | None:
    if s in ("", None):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_seed(seed: int) -> tuple[list[int], list[float], list[float], list[int], list[float]]:
    val_eps, val_puzzle, val_cell = [], [], []
    train_eps, train_exact = [], []
    with open(LOG_DIR / f"seed{seed}_trm_official_sudoku_train_log.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            ep = _f(r["epoch"])
            vp = _f(r["val_puzzle_acc"])
            vc = _f(r["val_cell_acc"])
            ex = _f(r["exact_accuracy"])
            if ep is None:
                continue
            if vp is not None:
                val_eps.append(int(ep))
                val_puzzle.append(vp)
                val_cell.append(vc if vc is not None else float("nan"))
            if ex is not None:
                train_eps.append(int(ep))
                train_exact.append(ex)
    return val_eps, val_puzzle, val_cell, train_eps, train_exact


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.4))

    ax.axvspan(0, 500, color="#ecf0f1", alpha=0.55, zorder=0)
    ax.text(250, 0.93, "early rise\n(1–500)", fontsize=6,
            color="#34495e", ha="center", va="top", alpha=0.85)

    peaks = {}
    for seed in (0, 1, 2):
        val_eps, val_puzzle, val_cell, train_eps, train_exact = load_seed(seed)
        target_ep = SEED_PEAK_EPOCH[seed]
        peak_i = min(range(len(val_eps)), key=lambda i: abs(val_eps[i] - target_ep))
        peaks[seed] = (val_eps[peak_i], val_puzzle[peak_i], val_cell[peak_i])
        col = SEED_COLORS[seed]

        ax.plot(val_eps, val_puzzle, color=col, linewidth=1.0, alpha=0.85,
                label=f"seed {seed} val (peak {val_puzzle[peak_i]:.2%} @ ep {val_eps[peak_i]})")
        ax.scatter([val_eps[peak_i]], [val_puzzle[peak_i]], color=col,
                   edgecolor="black", linewidth=0.5, s=28, zorder=5)
        ax.plot(train_eps, train_exact, color=col,
                linewidth=0.6, alpha=0.45, linestyle="--")

    ax.plot([], [], color="gray", linewidth=0.6, linestyle="--", alpha=0.55,
            label="train (all seeds)")

    ax.axhline(0.7462, color="#7f8c8d", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.text(2270, 0.755, "mean 74.62%", fontsize=5.5, color="#7f8c8d",
            ha="right", va="bottom")

    ax.set_xlabel("Epoch", fontsize=7)
    ax.set_ylabel("Puzzle accuracy", fontsize=7)
    ax.set_xlim(0, 2300)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.tick_params(axis="both", labelsize=6)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=5.5, framealpha=0.92, handlelength=1.2)
    ax.set_title("TRM-MLP fine-tune: full trajectory (3 seeds)", fontsize=7.5)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUT}")
    for s, (ep, p, c) in peaks.items():
        print(f"  seed {s}: peak puzzle {p:.4f}, cell {c:.4f} at epoch {ep}")


if __name__ == "__main__":
    main()
