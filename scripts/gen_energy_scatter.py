"""Regenerate fig_energy_scatter.pdf — training energy vs cell accuracy (Sudoku-Extreme).

Run from repo root:
    python scripts/gen_energy_scatter.py

Outputs:
    paper/fig_energy_scatter.pdf
    (and syncs to Windows path if run from WSL)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns
    sns.set_theme(context="paper", style="whitegrid")
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent

# All values verified against summary_fixed.csv and individual emissions CSVs.
# energy_kwh = training energy (kWh); TRM-MLP HF uses eval inference cost since
# it requires no fine-tuning on our hardware.
POINTS = [
    # label                  energy_kwh  cell_acc  marker  color      ms
    ("GPT-2 Small",          0.2570,     0.1318,   "o",   "#4C72B0",  7),
    ("SmolLM2-360M",         0.3043,     0.1411,   "o",   "#DD8452",  7),
    ("Llama-3.2-1B",         0.5813,     0.1974,   "o",   "#C44E52",  7),
    ("Qwen2.5-0.5B",         0.8960,     0.1907,   "o",   "#55A868",  7),
    ("Distill-GPT-2",        0.01078,    0.2580,   "s",   "#937860",  9),
    ("Distill-Qwen",         0.00902,    0.3587,   "s",   "#8172B3",  9),
    ("TRM-MLP (HF)",         0.48,       0.9155,   "*",   "#DA8BC3", 13),
    ("TRM-MLP (fine-tuned)", 5.45,        0.8601,   "^",   "#666666",  8),
]


LABEL_OFFSETS = {
    "GPT-2 Small":          (0, -0.04),
    "SmolLM2-360M":         (0, 0.03),
    "Llama-3.2-1B":         (0, 0.03),
    "Qwen2.5-0.5B":         (0, -0.04),
    "Distill-GPT-2":        (0, -0.04),
    "Distill-Qwen":         (0, 0.03),
    "TRM-MLP (HF)":         (0, 0.03),
    "TRM-MLP (fine-tuned)": (0, -0.04),
}


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.35, 2.9))

    for label, energy, cell_acc, marker, color, ms in POINTS:
        ax.scatter(
            energy, cell_acc,
            marker=marker, color=color,
            s=ms ** 2,
            edgecolors="black", linewidths=0.45,
            zorder=3,
        )
        dx, dy = LABEL_OFFSETS[label]
        ax.annotate(
            label, (energy, cell_acc),
            textcoords="offset points" if dx != 0 else "data",
            xytext=(energy, cell_acc + dy) if dx == 0 else (dx, dy),
            fontsize=5, ha="center", va="center", color=color, alpha=0.9,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training energy (kWh, log scale)", fontsize=8)
    ax.set_ylabel("Cell accuracy", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.4)
    return fig


def main() -> None:
    out_wsl = REPO_ROOT / "paper" / "fig_energy_scatter.pdf"
    fig = build_figure()
    fig.savefig(str(out_wsl), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_wsl}")

    win_path = Path("/mnt/c/Users/adsha/Downloads/ML final/fig_energy_scatter.pdf")
    if win_path.parent.exists():
        shutil.copy2(str(out_wsl), str(win_path))
        print(f"Synced → {win_path}")


if __name__ == "__main__":
    main()
