"""Tests for src.evaluation.plots — matplotlib figures for the report.

All figure tests verify (file exists) + (size >= 1 KB) + (PNG magic bytes).
We don't introspect plot contents; the goal is a smoke test that catches
API regressions + renders-without-crashing failures. The CLI wrapper
(`scripts/plot_results.py`) runs on real data for the "does it look right"
check.

Runs with both pytest and `python tests/test_plots.py`.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Force Agg backend before any pyplot import chain gets triggered.
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.plots import (  # noqa: E402
    plot_accuracy_vs_epoch,
    plot_act_convergence,
    plot_carbon_footprint_bars,
    plot_model_accuracy_bars,
    plot_params_vs_accuracy,
)


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_valid_png(path: str, min_bytes: int = 1000) -> None:
    assert os.path.isfile(path), f"file not created: {path}"
    size = os.path.getsize(path)
    assert size >= min_bytes, f"file too small ({size} bytes): {path}"
    with open(path, "rb") as f:
        header = f.read(len(PNG_MAGIC))
    assert header == PNG_MAGIC, f"not a PNG: header={header!r}"


# ---------------------------------------------------------------------------
# Fixtures — row-dict shape matches csv.DictReader output
# ---------------------------------------------------------------------------

OFFICIAL_ROWS = [
    {"epoch": "5", "lm_loss": "29.5", "val_cell_acc": "",
     "val_puzzle_acc": "", "avg_steps": "16.0", "elapsed_min": "4.4"},
    {"epoch": "50", "lm_loss": "11.8", "val_cell_acc": "0.53",
     "val_puzzle_acc": "0.12", "avg_steps": "11.0", "elapsed_min": "92.6"},
    {"epoch": "100", "lm_loss": "9.2", "val_cell_acc": "0.91",
     "val_puzzle_acc": "0.84", "avg_steps": "8.8", "elapsed_min": "200.0"},
]

LEGACY_ROWS = [
    {"epoch": "10", "ce_loss": "5.12", "val_cell_acc": "0.5",
     "val_puzzle_acc": "0.1", "steps_taken": "16.0", "elapsed_min": "10.0"},
    {"epoch": "20", "ce_loss": "4.01", "val_cell_acc": "0.7",
     "val_puzzle_acc": "0.3", "steps_taken": "14.0", "elapsed_min": "20.5"},
    {"epoch": "30", "ce_loss": "3.50", "val_cell_acc": "0.85",
     "val_puzzle_acc": "0.65", "steps_taken": "12.0", "elapsed_min": "31.2"},
]

SUMMARY_ROWS = [
    {"task": "trm-official", "best_val_puzzle_acc": 0.85,
     "best_val_cell_acc": 0.91, "train_energy_kwh": 6.78,
     "train_co2_kg": 1.61, "final_epoch": 500, "train_time_min": 976.4,
     "avg_act_steps": 8.8},
    {"task": "gpt2", "best_val_puzzle_acc": 0.12,
     "best_val_cell_acc": 0.35, "train_energy_kwh": 4.20,
     "train_co2_kg": 0.98, "final_epoch": 50, "train_time_min": 300.0,
     "avg_act_steps": 0.0},
    {"task": "smollm2", "best_val_puzzle_acc": 0.05,
     "best_val_cell_acc": 0.21, "train_energy_kwh": 3.10,
     "train_co2_kg": 0.71, "final_epoch": 50, "train_time_min": 250.0,
     "avg_act_steps": 0.0},
]

PARAM_COUNTS = {
    "trm-official": 7_000_000,
    "gpt2": 124_000_000,
    "smollm2": 135_000_000,
}


# ---------------------------------------------------------------------------
# plot_accuracy_vs_epoch
# ---------------------------------------------------------------------------

def test_plot_accuracy_vs_epoch_creates_file():
    train_logs = {"trm-official": OFFICIAL_ROWS, "legacy-sudoku": LEGACY_ROWS}
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "accuracy_vs_epoch.png")
        plot_accuracy_vs_epoch(train_logs, out)
        _assert_valid_png(out)


def test_plot_accuracy_vs_epoch_handles_empty_val_fields():
    """Early-epoch rows with empty val_puzzle_acc must be filtered, not
    rendered as zero."""
    train_logs = {"trm-official": OFFICIAL_ROWS}
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "accuracy_vs_epoch.png")
        plot_accuracy_vs_epoch(train_logs, out)
        _assert_valid_png(out)


def test_plot_accuracy_vs_epoch_rejects_empty_input():
    """If no run has usable val data, raise ValueError — a silent empty
    figure would be harder to debug than an explicit error."""
    empty = {"task_a": [{"epoch": "1", "val_puzzle_acc": ""}]}
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "accuracy_vs_epoch.png")
        raised = False
        try:
            plot_accuracy_vs_epoch(empty, out)
        except ValueError:
            raised = True
        assert raised, "expected ValueError for empty input"


# ---------------------------------------------------------------------------
# plot_model_accuracy_bars
# ---------------------------------------------------------------------------

def test_plot_model_accuracy_bars_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "model_accuracy_bars.png")
        plot_model_accuracy_bars(SUMMARY_ROWS, out)
        _assert_valid_png(out)


# ---------------------------------------------------------------------------
# plot_carbon_footprint_bars
# ---------------------------------------------------------------------------

def test_plot_carbon_footprint_bars_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "carbon_footprint_bars.png")
        plot_carbon_footprint_bars(SUMMARY_ROWS, out)
        _assert_valid_png(out)


# ---------------------------------------------------------------------------
# plot_params_vs_accuracy
# ---------------------------------------------------------------------------

def test_plot_params_vs_accuracy_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "params_vs_accuracy.png")
        plot_params_vs_accuracy(SUMMARY_ROWS, PARAM_COUNTS, out)
        _assert_valid_png(out)


def test_plot_params_vs_accuracy_skips_rows_without_param_count():
    """Rows whose task name is not in param_counts should be silently skipped
    so the caller can pass the full summary without a manual filter pass."""
    rows = [
        *SUMMARY_ROWS,
        {"task": "unknown-model", "best_val_puzzle_acc": 0.0,
         "train_energy_kwh": 0.0, "train_co2_kg": 0.0, "final_epoch": 0,
         "train_time_min": 0.0},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "params_vs_accuracy.png")
        plot_params_vs_accuracy(rows, PARAM_COUNTS, out)
        _assert_valid_png(out)


# ---------------------------------------------------------------------------
# plot_act_convergence
# ---------------------------------------------------------------------------

def test_plot_act_convergence_creates_file():
    """TRM runs have avg_steps; verify the curve renders."""
    train_logs = {"trm-official": OFFICIAL_ROWS}
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "act_convergence.png")
        plot_act_convergence(train_logs, out)
        _assert_valid_png(out)


def test_plot_act_convergence_accepts_legacy_steps_taken_column():
    """Legacy runs log `steps_taken` instead of `avg_steps` — both must work
    so pre-refactor data can still be plotted against the new runs."""
    train_logs = {"legacy": LEGACY_ROWS}
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "act_convergence.png")
        plot_act_convergence(train_logs, out)
        _assert_valid_png(out)


# ---------------------------------------------------------------------------
# Parent directory creation
# ---------------------------------------------------------------------------

def test_plot_creates_parent_directory():
    """Callers shouldn't have to pre-create results/figures/ etc."""
    train_logs = {"trm-official": OFFICIAL_ROWS}
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "nested", "deep", "figure.png")
        plot_accuracy_vs_epoch(train_logs, out)
        _assert_valid_png(out)


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
