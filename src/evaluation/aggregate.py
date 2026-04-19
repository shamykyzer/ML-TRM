"""Aggregate train_log.csv + emissions.csv from experiments/ into a summary.

This module is the source of truth for parsing the two CSV schemas the project
produces:

1. **Official TRM train log** — emitted by ``src/training/trainer_trm.py`` after
   the Sanjin2024 reproduction refactor. Columns: ``epoch, lm_loss, q_halt_loss,
   q_continue_loss, accuracy, exact_accuracy, q_halt_accuracy, avg_steps,
   val_cell_acc, val_puzzle_acc, best_puzzle_acc, elapsed_min``.

2. **Legacy train log** — older runs kept around for carbon/wall-time
   comparison. Columns: ``epoch, ce_loss, q_mean, steps_taken, val_cell_acc,
   val_puzzle_acc, best_puzzle_acc, elapsed_min``.

3. **CodeCarbon emissions.csv** — cumulative per ``run_id``. Crash recovery
   (ACT divergence, SLURM preemption) creates new ``run_id``s, so the full
   total is ``sum_over_runs(max_per_run(duration/energy/emissions))``.

4. **Optional eval_override.json** — when present in a run directory, its
   fields override metrics parsed from the train log. Useful for runs whose
   train log recorded a buggy metric (e.g. pre-Fix-B Qwen sudoku which logged
   val_puzzle_acc=0 due to an eval off-by-one, but whose post-fix re-eval
   found cell_acc≈0.19). Schema: any of ``val_puzzle_acc``, ``val_cell_acc``,
   ``eval_source`` (free-form label).

The parsing code uses only stdlib (``csv`` module) — no torch, no HF datasets,
no wandb — so the thesis report pipeline runs on minimal/CPU environments.
Out-of-order rows from crash-recovery resumes are handled by taking max-
per-metric rather than last-row, except for ``final_train_loss`` which is
specifically the loss at the row with the largest epoch number.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

# Test set sizes per task family, used to compute CO2/energy-per-correct-puzzle
# from puzzle_acc. Source: proposal §3 and Jolicoeur-Martineau (2025).
# Keep this in sync with data/build_*_dataset.py.
TEST_SET_SIZES: dict[str, int] = {
    "sudoku": 423_000,
    "maze": 1_000,
}

# ---------------------------------------------------------------------------
# Column candidates — schemas differ; we auto-detect per file
# ---------------------------------------------------------------------------

# Loss column candidates. Official uses lm_loss; legacy used ce_loss.
_LOSS_COLS = ("lm_loss", "ce_loss", "loss")

# ACT step column candidates.
_STEP_COLS = ("avg_steps", "steps_taken")


def _to_float(s: Any) -> float | None:
    """Parse a CSV cell as float. Returns None for empty / non-numeric values.

    Using None (not 0.0) is essential: an empty ``val_puzzle_acc`` on early
    epochs must NOT bias ``best_val_puzzle_acc = max(...)`` to zero.
    """
    if s is None:
        return None
    if not isinstance(s, str):
        try:
            return float(s)
        except (TypeError, ValueError):
            return None
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(s: Any) -> int | None:
    f = _to_float(s)
    if f is None:
        return None
    return int(f)


def _first_present(row: dict, candidates: tuple[str, ...]) -> str | None:
    """Return the first header from ``candidates`` present in ``row``.

    Used to pick between official / legacy column names. Header-based (not
    value-based) so a blank cell in the first row does not cause a mis-select.
    """
    for k in candidates:
        if k in row:
            return k
    return None


# ---------------------------------------------------------------------------
# parse_train_log
# ---------------------------------------------------------------------------

def parse_train_log(path: str) -> dict | None:
    """Parse a train_log CSV and return summary metrics.

    Returns
    -------
    dict with keys:
        best_val_puzzle_acc       : float  (max across rows, ignores blanks)
        best_val_cell_acc         : float  (max across rows, ignores blanks)
        peak_epoch                : int    (earliest epoch at best_val_puzzle_acc)
        train_time_min_at_peak    : float  (elapsed_min at peak_epoch — useful
                                           when overfitting made the final
                                           val_acc worse than the peak)
        final_train_loss          : float  (loss at the row with the MAX epoch)
        final_epoch               : int    (max epoch in the file)
        train_time_min            : float  (max elapsed_min — out-of-order safe)
        avg_act_steps             : float  (mean of the per-row step column)

    Returns ``None`` if the file does not exist or contains no data rows.
    """
    if not os.path.isfile(path):
        return None

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        return None

    loss_col = _first_present(rows[0], _LOSS_COLS)
    step_col = _first_present(rows[0], _STEP_COLS)

    best_val_puzzle = 0.0
    best_val_cell = 0.0
    peak_epoch = -1
    train_time_min_at_peak = 0.0
    max_epoch = -1
    final_loss: float | None = None
    max_elapsed = 0.0
    steps_sum = 0.0
    steps_count = 0

    for row in rows:
        epoch = _to_int(row.get("epoch"))
        if epoch is None:
            continue
        elapsed = _to_float(row.get("elapsed_min"))

        # best_val_* — ignore missing/empty entries (don't treat as zero).
        # Track peak epoch + elapsed time so overfitting runs don't misreport.
        vp = _to_float(row.get("val_puzzle_acc"))
        if vp is not None and vp > best_val_puzzle:
            best_val_puzzle = vp
            peak_epoch = epoch
            if elapsed is not None:
                train_time_min_at_peak = elapsed

        vc = _to_float(row.get("val_cell_acc"))
        if vc is not None and vc > best_val_cell:
            best_val_cell = vc

        # final_* come from the row with the maximum epoch — this handles
        # crash-recovery resumes where rows may appear non-monotonically.
        if epoch > max_epoch:
            max_epoch = epoch
            if loss_col is not None:
                final_loss = _to_float(row.get(loss_col))

        # Wall time is max across rows (also out-of-order safe).
        if elapsed is not None and elapsed > max_elapsed:
            max_elapsed = elapsed

        if step_col is not None:
            st = _to_float(row.get(step_col))
            if st is not None:
                steps_sum += st
                steps_count += 1

    if max_epoch < 0:
        return None

    return {
        "best_val_puzzle_acc": best_val_puzzle,
        "best_val_cell_acc": best_val_cell,
        "peak_epoch": peak_epoch if peak_epoch >= 0 else max_epoch,
        "train_time_min_at_peak": train_time_min_at_peak,
        "final_train_loss": final_loss if final_loss is not None else 0.0,
        "final_epoch": max_epoch,
        "train_time_min": max_elapsed,
        "avg_act_steps": (steps_sum / steps_count) if steps_count else 0.0,
    }


# ---------------------------------------------------------------------------
# parse_emissions
# ---------------------------------------------------------------------------

def parse_emissions(path: str) -> dict:
    """Parse a CodeCarbon emissions.csv and return cumulative totals.

    CodeCarbon writes monotonically-increasing counters per ``run_id``.
    Training restarts (ACT crash recovery / SLURM preemption) create new
    run_ids, so the full total is
    ``sum_over_runs(max_per_run(duration/energy/emissions))``.

    Returns
    -------
    dict with keys ``total_duration_s``, ``total_energy_kwh``, ``total_co2_kg``,
    or an empty dict if the file does not exist.
    """
    if not os.path.isfile(path):
        return {}

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        return {}

    # Group by run_id and take max per group. Since CodeCarbon counters are
    # monotonic within a run, max == final row for that run_id.
    per_run: dict[str, dict[str, float]] = {}
    for row in rows:
        rid = (row.get("run_id") or "UNKNOWN").strip() or "UNKNOWN"
        cur = per_run.setdefault(
            rid, {"duration": 0.0, "energy": 0.0, "co2": 0.0}
        )

        dur = _to_float(row.get("duration"))
        if dur is not None and dur > cur["duration"]:
            cur["duration"] = dur

        energy = _to_float(row.get("energy_consumed"))
        if energy is not None and energy > cur["energy"]:
            cur["energy"] = energy

        co2 = _to_float(row.get("emissions"))
        if co2 is not None and co2 > cur["co2"]:
            cur["co2"] = co2

    return {
        "total_duration_s": sum(r["duration"] for r in per_run.values()),
        "total_energy_kwh": sum(r["energy"] for r in per_run.values()),
        "total_co2_kg": sum(r["co2"] for r in per_run.values()),
    }


# ---------------------------------------------------------------------------
# aggregate_experiments
# ---------------------------------------------------------------------------

def _find_train_log(exp_dir: Path) -> Path | None:
    """Locate the train_log CSV inside an experiment directory.

    Prefers ``*_train_log.csv``; falls back to any file containing
    ``train_log`` in its name.
    """
    matches = sorted(exp_dir.glob("*_train_log.csv"))
    if matches:
        return matches[0]
    matches = sorted(exp_dir.glob("*train_log*.csv"))
    if matches:
        return matches[0]
    return None


def _read_eval_override(exp_dir: Path) -> dict:
    """Read optional eval_override.json — overrides for val_* metrics.

    Non-fatal: returns empty dict if the file is absent or malformed.
    """
    path = exp_dir / "eval_override.json"
    if not path.is_file():
        return {}
    try:
        with open(path) as fh:
            return json.load(fh) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def aggregate_experiments(root: str) -> list[dict]:
    """Walk ``root`` for experiment subdirectories, one summary row per dir.

    A subdirectory qualifies when a ``*_train_log.csv`` file is present.
    The emissions CSV is optional — when missing, the emissions fields are 0.
    Subdirectories are visited in sorted order for deterministic output.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []

    rows: list[dict] = []
    for sub in sorted(root_path.iterdir()):
        if not sub.is_dir():
            continue
        train_log = _find_train_log(sub)
        if train_log is None:
            continue

        metrics = parse_train_log(str(train_log))
        if metrics is None:
            continue

        emissions = parse_emissions(str(sub / "emissions.csv"))
        override = _read_eval_override(sub)

        row: dict[str, Any] = {"task": sub.name}
        row.update(metrics)
        # Apply eval_override.json on top of parsed train_log metrics. A
        # post-hoc re-evaluation (e.g. after a metric bug fix) is more
        # trustworthy than the logged value.
        if "val_puzzle_acc" in override:
            row["best_val_puzzle_acc"] = float(override["val_puzzle_acc"])
        if "val_cell_acc" in override:
            row["best_val_cell_acc"] = float(override["val_cell_acc"])
        if "eval_source" in override:
            row["eval_source"] = str(override["eval_source"])

        row["train_duration_s"] = emissions.get("total_duration_s", 0.0)
        row["train_energy_kwh"] = emissions.get("total_energy_kwh", 0.0)
        row["train_co2_kg"] = emissions.get("total_co2_kg", 0.0)
        rows.append(row)

    return rows


def aggregate_all_experiments(roots: list[str]) -> list[dict]:
    """Walk multiple roots and concatenate rows. Preserves root order.

    When the same task name appears in two roots, both rows are kept — the
    caller can dedupe if needed. Duplicate task names inside a single root
    cannot happen because dir listings are unique.
    """
    out: list[dict] = []
    for root in roots:
        out.extend(aggregate_experiments(root))
    return out


def _task_family(task_name: str) -> str:
    """Classify a task name as 'sudoku' or 'maze' based on a keyword search."""
    return "maze" if "maze" in task_name.lower() else "sudoku"


def attach_efficiency_metrics(row: dict, test_set_sizes: dict[str, int] | None = None) -> dict:
    """Add correct_puzzles, co2_per_correct_puzzle, kwh_per_correct_puzzle.

    Uses ``best_val_puzzle_acc * TEST_SET_SIZE[family]`` for ``correct_puzzles``.
    For zero-correct rows (all LLM baselines expected to score 0%), the
    per-correct columns are left blank rather than ``inf`` so CSV consumers
    don't have to special-case infinity.
    """
    sizes = test_set_sizes if test_set_sizes is not None else TEST_SET_SIZES
    family = _task_family(row.get("task", ""))
    test_size = sizes.get(family, 1)

    puzzle_acc = float(row.get("best_val_puzzle_acc") or 0.0)
    correct = puzzle_acc * test_size
    row["correct_puzzles"] = correct

    co2 = float(row.get("train_co2_kg") or 0.0)
    kwh = float(row.get("train_energy_kwh") or 0.0)
    if correct > 0:
        row["co2_per_correct_puzzle"] = co2 / correct
        row["kwh_per_correct_puzzle"] = kwh / correct
    else:
        row["co2_per_correct_puzzle"] = ""
        row["kwh_per_correct_puzzle"] = ""
    return row


# ---------------------------------------------------------------------------
# write_summary_csv
# ---------------------------------------------------------------------------

def write_summary_csv(rows: list[dict], path: str) -> None:
    """Write ``rows`` to ``path`` as a CSV with a header row.

    The header is the union of keys across all rows, preserving insertion
    order from the first row and appending keys unique to later rows.
    The parent directory is created if it does not already exist.
    """
    parent = Path(path).parent
    if str(parent) and parent != Path(""):
        parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(path, "w", newline="") as fh:
            fh.write("")
        return

    header: list[str] = list(rows[0].keys())
    seen = set(header)
    for r in rows[1:]:
        for k in r.keys():
            if k not in seen:
                header.append(k)
                seen.add(k)

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
