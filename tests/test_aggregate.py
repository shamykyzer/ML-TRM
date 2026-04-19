"""Tests for src.evaluation.aggregate — parse train logs + emissions CSVs."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.aggregate import (  # noqa: E402
    aggregate_experiments,
    parse_emissions,
    parse_train_log,
    write_summary_csv,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic CSV strings matching the two real schemas
# ---------------------------------------------------------------------------

OFFICIAL_TRAIN_LOG = """epoch,lm_loss,q_halt_loss,q_continue_loss,accuracy,exact_accuracy,q_halt_accuracy,avg_steps,val_cell_acc,val_puzzle_acc,best_puzzle_acc,elapsed_min
5,29.5474,0.0000,0.0000,0.3404,0.0000,1.0000,16.0,,,0.0000,4.4
10,20.2280,0.1461,0.0000,0.6945,0.0071,0.9929,16.0,,,0.0000,8.8
50,11.8804,0.8125,0.0000,0.8552,0.4516,0.9223,11.0,0.5301,0.1290,0.1290,92.6
100,9.2806,0.4186,0.0000,0.8886,0.6266,0.9649,8.8,0.9100,0.8474,0.8474,200.0
"""

LEGACY_TRAIN_LOG = """epoch,ce_loss,q_mean,steps_taken,val_cell_acc,val_puzzle_acc,best_puzzle_acc,elapsed_min
10,5.12,0.15,16.0,0.5,0.1,0.1,10.0
20,4.01,0.25,14.0,0.7,0.3,0.3,20.5
30,3.50,0.45,12.0,0.85,0.65,0.65,31.2
"""

OUT_OF_ORDER_TRAIN_LOG = """epoch,lm_loss,q_halt_loss,q_continue_loss,accuracy,exact_accuracy,q_halt_accuracy,avg_steps,val_cell_acc,val_puzzle_acc,best_puzzle_acc,elapsed_min
5,29.5,0.0,0.0,0.3,0.0,1.0,16.0,,,0.0,4.4
10,20.2,0.1,0.0,0.7,0.0,0.99,16.0,,,0.0,8.8
350,4.6,0.14,1.98,0.95,0.87,0.99,5.2,0.91,0.85,0.85,659.8
30,14.7,0.99,0.0,0.84,0.22,0.89,13.6,,,0.0,47.9
"""

# LLM trainer CSV schema (trainer_llm.py) — columns differ from the
# official/legacy TRM schemas above. Used for the non-learning diagnostic
# tests added with loss_delta_pct support.
LLM_TRAIN_LOG_WITH_BASELINE = """epoch,loss,val_loss,val_puzzle_acc,val_cell_acc,elapsed_min
0,,2.5000,0.0000,0.1900,0.0
10,2.4800,2.4900,0.0000,0.1895,15.3
20,2.4700,2.4700,0.0000,0.1896,30.5
30,2.4600,2.4500,0.0000,0.1905,45.8
"""

LLM_TRAIN_LOG_WITHOUT_BASELINE = """epoch,loss,val_loss,val_puzzle_acc,val_cell_acc,elapsed_min
10,2.4800,2.4900,0.0000,0.1895,15.3
20,2.4700,2.4700,0.0000,0.1896,30.5
30,2.4600,2.4500,0.0000,0.1905,45.8
"""

LLM_TRAIN_LOG_NEGATIVE_DELTA = """epoch,loss,val_loss,val_puzzle_acc,val_cell_acc,elapsed_min
0,,2.0000,0.0000,0.1900,0.0
10,2.1000,2.1000,0.0000,0.1895,15.3
20,2.1500,2.1500,0.0000,0.1896,30.5
30,2.2000,2.2000,0.0000,0.1905,45.8
"""

EMISSIONS_SINGLE_RUN = """timestamp,project_name,run_id,experiment_id,duration,emissions,emissions_rate,cpu_power,gpu_power,ram_power,cpu_energy,gpu_energy,ram_energy,energy_consumed
2026-04-11T07:53:48,trm_train,RUN_A,EXP_1,50.55,0.00067,1.3e-05,7.5,175.6,20.0,0.0001,0.0024,0.00027,0.00283
2026-04-11T07:54:35,trm_train,RUN_A,EXP_1,98.30,0.00137,1.4e-05,7.1,185.1,20.0,0.00019,0.00505,0.00052,0.00576
"""

EMISSIONS_MULTI_RUN = """timestamp,project_name,run_id,experiment_id,duration,emissions,emissions_rate,cpu_power,gpu_power,ram_power,cpu_energy,gpu_energy,ram_energy,energy_consumed
2026-04-11T07:53:48,trm_train,RUN_A,EXP_1,50.0,0.001,1.0e-05,7.5,175,20,0.0001,0.0024,0.00027,0.002
2026-04-11T07:54:35,trm_train,RUN_A,EXP_1,100.0,0.002,1.0e-05,7.1,185,20,0.00019,0.00505,0.00052,0.004
2026-04-11T08:00:00,trm_train,RUN_B,EXP_1,200.0,0.005,1.0e-05,7.1,185,20,0.0003,0.01,0.001,0.010
"""


# ---------------------------------------------------------------------------
# parse_train_log
# ---------------------------------------------------------------------------

def test_parse_train_log_official_schema():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(OFFICIAL_TRAIN_LOG)
        path = f.name
    try:
        m = parse_train_log(path)
        assert m is not None
        assert m["best_val_puzzle_acc"] == 0.8474, f"got {m['best_val_puzzle_acc']}"
        assert m["best_val_cell_acc"] == 0.9100
        assert m["train_time_min"] == 200.0
        # final_train_loss is the loss of the max-epoch row
        assert m["final_train_loss"] == 9.2806
        assert m["final_epoch"] == 100
    finally:
        os.unlink(path)


def test_parse_train_log_legacy_schema():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(LEGACY_TRAIN_LOG)
        path = f.name
    try:
        m = parse_train_log(path)
        assert m is not None
        assert m["best_val_puzzle_acc"] == 0.65
        assert m["train_time_min"] == 31.2
        assert m["final_train_loss"] == 3.50
    finally:
        os.unlink(path)


def test_parse_train_log_handles_empty_val_columns():
    """Early epochs have empty val_cell_acc / val_puzzle_acc fields — must
    not crash and must not count them as 0."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(OFFICIAL_TRAIN_LOG)
        path = f.name
    try:
        m = parse_train_log(path)
        # Empty val fields in early epochs must NOT make max = 0
        assert m["best_val_puzzle_acc"] > 0.5
    finally:
        os.unlink(path)


def test_parse_train_log_handles_out_of_order_epochs():
    """Crash-recovery resume: epochs may appear non-monotonically. The
    aggregator must pick the global max of each metric regardless of
    row order."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(OUT_OF_ORDER_TRAIN_LOG)
        path = f.name
    try:
        m = parse_train_log(path)
        assert m["final_epoch"] == 350  # max, not last-row
        assert m["best_val_puzzle_acc"] == 0.85
        # train_time_min is max(elapsed_min), not last-row
        assert m["train_time_min"] == 659.8
        # final_train_loss is from the row with max epoch (350)
        assert m["final_train_loss"] == 4.6
    finally:
        os.unlink(path)


def test_parse_train_log_missing_file_returns_none():
    assert parse_train_log("/nonexistent/path.csv") is None


def test_parse_train_log_empty_file_returns_none():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("epoch,ce_loss\n")  # header only, no data
        path = f.name
    try:
        m = parse_train_log(path)
        assert m is None
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# parse_emissions
# ---------------------------------------------------------------------------

def test_parse_emissions_single_run_takes_last_row():
    """CodeCarbon writes cumulative totals — take the max per run_id."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(EMISSIONS_SINGLE_RUN)
        path = f.name
    try:
        e = parse_emissions(path)
        assert abs(e["total_duration_s"] - 98.30) < 1e-6
        assert abs(e["total_energy_kwh"] - 0.00576) < 1e-9
        assert abs(e["total_co2_kg"] - 0.00137) < 1e-9
    finally:
        os.unlink(path)


def test_parse_emissions_sums_across_runs():
    """Restarts create new run_ids. Total = sum over all runs."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(EMISSIONS_MULTI_RUN)
        path = f.name
    try:
        e = parse_emissions(path)
        # RUN_A max: duration=100, energy=0.004, co2=0.002
        # RUN_B max: duration=200, energy=0.010, co2=0.005
        assert abs(e["total_duration_s"] - 300.0) < 1e-6
        assert abs(e["total_energy_kwh"] - 0.014) < 1e-9
        assert abs(e["total_co2_kg"] - 0.007) < 1e-9
    finally:
        os.unlink(path)


def test_parse_emissions_missing_file_returns_empty():
    e = parse_emissions("/nonexistent.csv")
    assert e == {}


# ---------------------------------------------------------------------------
# aggregate_experiments integration
# ---------------------------------------------------------------------------

def test_aggregate_experiments_end_to_end():
    """Mini experiments/ directory: two tasks, each with train_log + emissions.
    Verify the output summary rows match the inputs."""
    with tempfile.TemporaryDirectory() as root:
        # Task A
        a = Path(root) / "task_a"
        a.mkdir()
        (a / "task_a_train_log.csv").write_text(OFFICIAL_TRAIN_LOG)
        (a / "emissions.csv").write_text(EMISSIONS_SINGLE_RUN)
        # Task B
        b = Path(root) / "task_b"
        b.mkdir()
        (b / "trm_train_log.csv").write_text(LEGACY_TRAIN_LOG)
        (b / "emissions.csv").write_text(EMISSIONS_MULTI_RUN)

        rows = aggregate_experiments(root)
        assert len(rows) == 2, f"expected 2 experiments, got {len(rows)}"
        by_task = {r["task"]: r for r in rows}
        assert "task_a" in by_task
        assert "task_b" in by_task
        assert by_task["task_a"]["best_val_puzzle_acc"] == 0.8474
        assert by_task["task_b"]["best_val_puzzle_acc"] == 0.65
        assert abs(by_task["task_a"]["train_energy_kwh"] - 0.00576) < 1e-9
        assert abs(by_task["task_b"]["train_energy_kwh"] - 0.014) < 1e-9


def test_write_summary_csv_creates_valid_csv():
    """write_summary_csv writes a readable CSV with one header + rows."""
    import csv
    rows = [
        {"task": "task_a", "best_val_puzzle_acc": 0.85, "train_energy_kwh": 0.01},
        {"task": "task_b", "best_val_puzzle_acc": 0.65, "train_energy_kwh": 0.02},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        write_summary_csv(rows, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            out = list(reader)
        assert len(out) == 2
        assert out[0]["task"] == "task_a"
        assert float(out[0]["best_val_puzzle_acc"]) == 0.85
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# loss_delta_pct — non-learning diagnostic column (LLM trainer only)
# ---------------------------------------------------------------------------

def test_loss_delta_pct_computed_when_epoch_zero_row_present(tmp_path):
    """A train log containing an epoch=0 baseline row yields a numeric
    loss_delta_pct = (initial - final) / initial * 100."""
    log_path = tmp_path / "test_train_log.csv"
    log_path.write_text(LLM_TRAIN_LOG_WITH_BASELINE)

    result = parse_train_log(str(log_path))

    assert result is not None
    assert result["initial_val_loss"] == 2.5
    assert result["final_val_loss"] == 2.45
    # (2.5 - 2.45) / 2.5 * 100 = 2.0
    assert abs(result["loss_delta_pct"] - 2.0) < 1e-6


def test_loss_delta_pct_blank_when_no_epoch_zero_row(tmp_path):
    """A train log without an epoch=0 baseline row yields loss_delta_pct=""
    and both initial/final val_loss fields blank — matches pre-diagnostics
    LLM runs and all TRM runs."""
    log_path = tmp_path / "test_train_log.csv"
    log_path.write_text(LLM_TRAIN_LOG_WITHOUT_BASELINE)

    result = parse_train_log(str(log_path))

    assert result is not None
    assert result["initial_val_loss"] == ""
    # final_val_loss IS populated (the epoch=30 row has val_loss=2.45)
    # but loss_delta_pct is blank because initial is missing.
    assert result["final_val_loss"] == 2.45
    assert result["loss_delta_pct"] == ""


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
