"""Matplotlib figures for the thesis report.

The report needs the 5 figures listed in ``plan.md`` §"Figures to Generate":

1. ``accuracy_vs_epoch``      — val puzzle accuracy over training (one line per run)
2. ``model_accuracy_bars``    — best val puzzle accuracy across models
3. ``carbon_footprint_bars``  — training CO₂ kg across models
4. ``params_vs_accuracy``     — scatter of param count (log) vs accuracy
5. ``act_convergence``        — average ACT halting steps over training

Architecture
------------
Every plot function takes already-parsed data (list-of-dict rows from
``csv.DictReader`` for per-run train logs + summary rows from
``aggregate_experiments``) plus an output path. No CSV parsing or file
discovery happens inside this module — ``scripts/plot_results.py`` is the
CLI wrapper that loads the files and calls these functions.

The Agg backend is forced at module import time so the functions work
headless (CI, thesis report generation, no ``$DISPLAY``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib

# Force non-interactive backend BEFORE pyplot is imported anywhere else.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# Project-wide seaborn theme — applied once on import.
sns.set_theme(context="paper", style="whitegrid")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STEP_COLUMNS = ("avg_steps", "steps_taken")


def _to_float(s: Any) -> float | None:
    """Parse a CSV cell as float. None for empty / non-numeric.

    Duplicated from ``aggregate._to_float`` to keep this module independent
    of the aggregator — plots may be called on hand-built rows in notebooks.
    """
    if s is None:
        return None
    if not isinstance(s, str):
        try:
            return float(s)
        except (TypeError, ValueError):
            return None
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _ensure_parent_dir(path: str) -> None:
    parent = Path(path).parent
    if str(parent) and parent != Path(""):
        parent.mkdir(parents=True, exist_ok=True)


def _filter_epoch_series(
    rows: list[dict],
    y_columns: str | tuple[str, ...],
) -> tuple[list[int], list[float]]:
    """Return ``(epochs, ys)`` for rows with non-empty epoch AND y values.

    ``y_columns`` may be a single column name or a tuple of candidates; the
    first non-empty value wins. Output is sorted by epoch so out-of-order
    rows (from crash-recovery resumes) render as a monotone curve.
    """
    candidates = (y_columns,) if isinstance(y_columns, str) else tuple(y_columns)
    pairs: list[tuple[int, float]] = []
    for row in rows:
        ep = _to_float(row.get("epoch"))
        if ep is None:
            continue
        y = None
        for col in candidates:
            y = _to_float(row.get(col))
            if y is not None:
                break
        if y is None:
            continue
        pairs.append((int(ep), y))
    pairs.sort(key=lambda p: p[0])
    if not pairs:
        return [], []
    epochs, ys = zip(*pairs)
    return list(epochs), list(ys)


# ---------------------------------------------------------------------------
# Figure 1 — Validation accuracy over training
# ---------------------------------------------------------------------------

def plot_accuracy_vs_epoch(
    train_logs: dict[str, list[dict]],
    out_path: str,
    metric: str = "val_puzzle_acc",
    title: str = "Validation puzzle accuracy over training",
) -> None:
    """Render one line per run showing ``metric`` vs epoch.

    Parameters
    ----------
    train_logs : dict[str, list[dict]]
        ``task_name -> rows`` (rows = list of dicts from ``csv.DictReader``).
    out_path : str
        Destination PNG path. Parent dirs are created.
    metric : str
        CSV column to plot on the y axis. Default ``val_puzzle_acc``.
    title : str
        Figure title.

    Raises
    ------
    ValueError
        If no run has any non-empty ``metric`` values — an empty figure
        would hide the real problem.
    """
    _ensure_parent_dir(out_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = 0
    for task, rows in train_logs.items():
        xs, ys = _filter_epoch_series(rows, metric)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", label=task, linewidth=1.5, markersize=4)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError(
            f"plot_accuracy_vs_epoch: no runs had any non-empty {metric!r} values"
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Best val accuracy across models
# ---------------------------------------------------------------------------

def plot_model_accuracy_bars(
    summary_rows: list[dict],
    out_path: str,
    metric: str = "best_val_puzzle_acc",
    title: str = "Best validation puzzle accuracy by model",
) -> None:
    """Horizontal bar chart of ``metric`` across all summary rows, sorted asc."""
    _ensure_parent_dir(out_path)

    pairs: list[tuple[str, float]] = []
    for r in summary_rows:
        val = _to_float(r.get(metric))
        if val is not None:
            pairs.append((str(r.get("task", "?")), val))
    if not pairs:
        raise ValueError(
            f"plot_model_accuracy_bars: no rows with {metric!r}"
        )

    pairs.sort(key=lambda p: p[1])  # ascending → best bar at top
    tasks, values = zip(*pairs)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.45 * len(pairs) + 1)))
    ax.barh(
        list(tasks),
        list(values),
        color=sns.color_palette("viridis", len(pairs)),
    )
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.set_xlim(0, 1)
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Training carbon footprint
# ---------------------------------------------------------------------------

def plot_carbon_footprint_bars(
    summary_rows: list[dict],
    out_path: str,
    title: str = "Training carbon footprint",
) -> None:
    """Horizontal bar chart of ``train_co2_kg`` across all summary rows."""
    _ensure_parent_dir(out_path)

    pairs: list[tuple[str, float]] = []
    for r in summary_rows:
        co2 = _to_float(r.get("train_co2_kg"))
        if co2 is not None:
            pairs.append((str(r.get("task", "?")), co2))
    if not pairs:
        raise ValueError(
            "plot_carbon_footprint_bars: no rows with train_co2_kg"
        )

    pairs.sort(key=lambda p: p[1])
    tasks, values = zip(*pairs)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.45 * len(pairs) + 1)))
    ax.barh(
        list(tasks),
        list(values),
        color=sns.color_palette("rocket_r", len(pairs)),
    )
    ax.set_xlabel(r"Training $\mathrm{CO_2}$ (kg)")
    ax.set_title(title)
    v_max = max(values) if values else 0.0
    for i, v in enumerate(values):
        offset = (v_max * 0.01) if v_max > 0 else 0.01
        ax.text(v + offset, i, f"{v:.3f} kg", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Parameter efficiency (params vs accuracy)
# ---------------------------------------------------------------------------

def plot_params_vs_accuracy(
    summary_rows: list[dict],
    param_counts: dict[str, int],
    out_path: str,
    metric: str = "best_val_puzzle_acc",
    title: str = "Parameter efficiency — accuracy vs param count",
) -> None:
    """Scatter plot: log-scale param count vs ``metric``.

    Rows whose ``task`` is not in ``param_counts`` are silently skipped so
    the caller can pass the full summary without pre-filtering.
    """
    _ensure_parent_dir(out_path)

    xs: list[int] = []
    ys: list[float] = []
    labels: list[str] = []
    for r in summary_rows:
        task = str(r.get("task", ""))
        if task not in param_counts:
            continue
        val = _to_float(r.get(metric))
        if val is None:
            continue
        xs.append(param_counts[task])
        ys.append(val)
        labels.append(task)

    if not xs:
        raise ValueError(
            "plot_params_vs_accuracy: no summary rows matched param_counts"
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = sns.color_palette("viridis", len(xs))
    ax.scatter(
        xs, ys,
        s=80, alpha=0.85,
        edgecolor="black", linewidth=0.5,
        color=colors,
    )
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(
            lbl, (x, y),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — ACT convergence over training
# ---------------------------------------------------------------------------

def plot_act_convergence(
    train_logs: dict[str, list[dict]],
    out_path: str,
    title: str = "ACT halting steps over training",
) -> None:
    """Line plot of per-epoch average ACT steps.

    Both ``avg_steps`` (official schema) and ``steps_taken`` (legacy schema)
    are recognised so pre-refactor runs can still be plotted alongside new
    runs.

    Raises
    ------
    ValueError
        If no run has any recognisable step column.
    """
    _ensure_parent_dir(out_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = 0
    for task, rows in train_logs.items():
        xs, ys = _filter_epoch_series(rows, _STEP_COLUMNS)
        if not xs:
            continue
        ax.plot(xs, ys, marker="s", label=task, linewidth=1.5, markersize=4)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError(
            "plot_act_convergence: no runs had avg_steps or steps_taken data"
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average ACT steps")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
