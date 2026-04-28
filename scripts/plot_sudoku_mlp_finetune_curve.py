"""Figure 4 — Sudoku-MLP fine-tune curve from the best wandb-aggregated run.

Source: results/history_sudoku-mlp_best.csv (per-step history of the
best sudoku-mlp seed, captured by scripts/aggregate_wandb_runs.py).
At present that file holds run 8hncpi2x (seed 2, peak val/puzzle_acc
0.7486 @ epoch 700, decay to 0.5722 @ epoch 2200).

The curve is framed against the HF released-checkpoint eval (0.8474):
fine-tuning rises to a peak that never beats the loaded init, then
decays. Same overfit signature as plot_sudoku_mlp_overfit.py (which uses
seed-0's per-epoch train log) but driven from the wandb-aggregated CSV
so the report figure stays in lockstep with the multi-machine seed
table in findings.md §5.12.

Output: results/figures/sudoku_mlp_finetune_curve.png
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


HISTORY_CSV = REPO_ROOT / "results" / "history_sudoku-mlp_best.csv"
OUT_PATH = REPO_ROOT / "results" / "figures" / "sudoku_mlp_finetune_curve.png"
HF_INIT_PUZZLE_ACC = 0.8474  # results/hf_eval_sudoku_mlp.json — the released checkpoint eval


def _to_float(s):
    if s in (None, ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_eval_rows(path: Path) -> list[dict]:
    """csv.DictReader handles the embedded JSON-with-commas in the
    halt_steps_hist columns; naive splitting breaks them.
    """
    rows: list[dict] = []
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            epoch = _to_float(r.get("epoch"))
            puzzle = _to_float(r.get("val/puzzle_acc"))
            if epoch is None or puzzle is None:
                continue
            rows.append({
                "epoch": int(epoch),
                "val_puzzle": puzzle,
                "val_cell": _to_float(r.get("val/cell_acc")),
            })
    rows.sort(key=lambda r: r["epoch"])
    return rows


def main() -> int:
    if not HISTORY_CSV.is_file():
        print(f"ERROR: history CSV not found at {HISTORY_CSV}", file=sys.stderr)
        return 1

    rows = _load_eval_rows(HISTORY_CSV)
    if not rows:
        print("ERROR: no eval rows in history CSV", file=sys.stderr)
        return 1

    epochs = [r["epoch"] for r in rows]
    val_puzzle = [r["val_puzzle"] for r in rows]
    val_cell = [r["val_cell"] for r in rows if r["val_cell"] is not None]
    cell_epochs = [r["epoch"] for r in rows if r["val_cell"] is not None]

    peak = max(rows, key=lambda r: r["val_puzzle"])
    peak_x, peak_y = peak["epoch"], peak["val_puzzle"]
    final = rows[-1]
    drop_from_peak = peak_y - final["val_puzzle"]
    gap_to_init = HF_INIT_PUZZLE_ACC - peak_y

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=150)

    # HF released-checkpoint eval — the ceiling fine-tuning never recovers.
    ax.axhline(
        HF_INIT_PUZZLE_ACC, color="#2c3e50", linestyle="--", linewidth=1.3,
        alpha=0.85, label=f"HF released-checkpoint eval = {HF_INIT_PUZZLE_ACC:.4f}",
    )

    ax.plot(
        epochs, val_puzzle, marker="o", markersize=4,
        linewidth=1.8, color="#c0392b", label="val puzzle_acc",
    )
    if val_cell:
        ax.plot(
            cell_epochs, val_cell, marker="s", markersize=3,
            linewidth=1.4, color="#2980b9", alpha=0.85, label="val cell_acc",
        )

    # Peak marker.
    ax.axvline(peak_x, color="#7f8c8d", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.scatter([peak_x], [peak_y], s=70, facecolor="white",
               edgecolor="#c0392b", linewidth=1.6, zorder=5)
    ax.annotate(
        f"peak puzzle_acc = {peak_y:.4f}\n"
        f"@ epoch {peak_x}\n"
        f"({gap_to_init * 100:.2f} pp below HF init)",
        xy=(peak_x, peak_y),
        xytext=(peak_x + 250, peak_y + 0.06),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#2c3e50"),
    )

    # Final-eval annotation showing the decay tail.
    ax.scatter([final["epoch"]], [final["val_puzzle"]], s=60,
               facecolor="white", edgecolor="#c0392b", linewidth=1.4, zorder=5)
    ax.annotate(
        f"final {final['val_puzzle']:.4f}\n"
        f"@ epoch {final['epoch']}\n"
        f"(Δ −{drop_from_peak * 100:.2f} pp from peak)",
        xy=(final["epoch"], final["val_puzzle"]),
        xytext=(final["epoch"] - 600, final["val_puzzle"] - 0.16),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#c0392b"),
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title(
        "TRM-MLP Sudoku fine-tune from HF init: "
        f"peak {peak_y:.4f} @ epoch {peak_x}, decay thereafter",
        fontsize=11,
    )

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PATH), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] wrote {OUT_PATH}")
    print(f"[plot] eval rows used: {len(rows)} (epochs {epochs[0]}..{epochs[-1]})")
    print(f"[plot] HF-init reference: {HF_INIT_PUZZLE_ACC:.4f}")
    print(f"[plot] peak val_puzzle: {peak_y:.4f} @ ep {peak_x} "
          f"({gap_to_init * 100:.2f} pp below init)")
    print(f"[plot] final val_puzzle: {final['val_puzzle']:.4f} @ ep {final['epoch']} "
          f"(-{drop_from_peak * 100:.2f} pp from peak)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
