#!/usr/bin/env python
"""Contract B — Metric realism monitoring (sprint 2026-04-28 → 2026-05-01).

Two modes:

  1. Pre-launch (§B.5): assert config invariants before the first epoch.
       python scripts/contract_b_realism_check.py prelaunch <config.yaml>

  2. Mid-run / post-run (§B.6 / §B.7): scan a train_log CSV for red flags.
       python scripts/contract_b_realism_check.py monitor <train_log.csv> \
           --task {sudoku,maze} --family {trm,llm,distill}

Exit codes:
  0  — no flags fired (green)
  2  — at least one red flag fired (caller should stop the run, log to
       findings.md §5 with the literal string "metric realism violation")
  3  — input file unreadable / config schema violation

Calibration anchor (§B.9): post-fix Maze re-evals on M1 produced
puzzle_acc=0/1000, cell_acc=~0.125 for both Qwen-Maze and Distill-Qwen-Maze.
That is the green-flag shape; anything meaningfully above puzzle_acc=0 or
cell_acc=0.20 on a Maze re-eval is suspect.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Iterable

EXIT_OK = 0
EXIT_RED = 2
EXIT_BAD_INPUT = 3

# §B.2 expected ranges (val_puzzle_acc, val_cell_acc)
EXPECTED = {
    ("trm", "sudoku", "hf"):       {"puzzle": (0.83, 0.87),  "cell": (0.90, 0.93)},
    ("trm", "sudoku", "scratch"):  {"puzzle": (0.72, 0.76),  "cell": (0.83, 0.88)},
    ("trm", "maze",   "hf"):       {"puzzle": (0.78, 0.81),  "cell": (0.99, 1.00)},
    ("llm", "sudoku", None):       {"puzzle": (0.00, 0.00),  "cell": (0.05, 0.25)},
    ("llm", "maze",   None):       {"puzzle": (0.00, 0.00),  "cell": (0.05, 0.25)},
    ("distill", "sudoku", None):   {"puzzle": (0.00, 0.00),  "cell": (0.05, 0.40)},
    ("distill", "maze",   None):   {"puzzle": (0.00, 0.00),  "cell": (0.05, 0.25)},
}


def prelaunch(config_path: str) -> int:
    """§B.5: assert maze configs have mask_non_path: false."""
    if not os.path.exists(config_path):
        print(f"[B.5] FAIL: config not found: {config_path}")
        return EXIT_BAD_INPUT
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils.config import load_config

    cfg = load_config(config_path)
    if cfg.data.dataset not in {"sudoku", "maze"}:
        print(f"[B.5] FAIL: unknown dataset {cfg.data.dataset!r}")
        return EXIT_BAD_INPUT
    if cfg.data.dataset == "maze":
        if getattr(cfg.data, "mask_non_path", True) is not False:
            print(
                "[B.5] FAIL: maze training MUST set data.mask_non_path: false "
                "(currently {})".format(getattr(cfg.data, "mask_non_path", "<missing>"))
            )
            return EXIT_RED
    print(
        "[B.5] OK: dataset={} mask_non_path={} subsample_size={}".format(
            cfg.data.dataset,
            getattr(cfg.data, "mask_non_path", "n/a"),
            getattr(cfg.data, "subsample_size", "n/a"),
        )
    )
    return EXIT_OK


def _read_log(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k, v in list(row.items()):
                if v in ("", None):
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            rows.append(row)
    return rows


def _flag(name: str, msg: str) -> None:
    print(f"[B.3] RED FLAG [{name}] {msg}")


def monitor(log_path: str, task: str, family: str) -> int:
    """§B.6/§B.7: scan train_log.csv for §B.3 red flags."""
    if not os.path.exists(log_path):
        print(f"[monitor] FAIL: log not found: {log_path}")
        return EXIT_BAD_INPUT

    rows = _read_log(log_path)
    if not rows:
        print("[monitor] FAIL: log has no rows")
        return EXIT_BAD_INPUT

    fired = 0

    # Resolve column names — train log has val_puzzle_acc / val_cell_acc /
    # train_loss / lm_loss; train_log.csv shape varies slightly by trainer.
    def col(row: dict, *names: str) -> float | None:
        for n in names:
            if n in row and row[n] is not None:
                return row[n]
        return None

    # Walk rows
    train_losses: list[float | None] = []
    val_cells: list[float | None] = []
    val_puzzles: list[float | None] = []
    avg_steps: list[float | None] = []

    for r in rows:
        train_losses.append(col(r, "train_loss", "lm_loss", "loss"))
        val_puzzles.append(col(r, "val_puzzle_acc", "val_exact_accuracy", "exact_accuracy"))
        val_cells.append(col(r, "val_cell_acc", "val_accuracy", "accuracy"))
        avg_steps.append(col(r, "avg_steps", "avg_act_steps"))

    last_puz = next((v for v in reversed(val_puzzles) if v is not None), None)
    last_cell = next((v for v in reversed(val_cells) if v is not None), None)
    last_loss = next((v for v in reversed(train_losses) if v is not None), None)

    # §B.3: NaN / Inf loss
    for i, v in enumerate(train_losses):
        if v is not None and (v != v or v in (float("inf"), float("-inf"))):
            _flag("loss_nan_inf", f"row {i} train_loss={v}")
            fired += 1
            break

    # §B.3: LLM/distill puzzle_acc >= 0.99 (mask bug)
    if family in {"llm", "distill"}:
        for i, v in enumerate(val_puzzles):
            if v is not None and v >= 0.99:
                _flag(
                    "llm_puzzle_acc_saturated",
                    f"row {i} val_puzzle_acc={v} >= 0.99 — mask_non_path likely true (bug). "
                    f"Apply mask_non_path: false and re-eval.",
                )
                fired += 1
                break

    # §B.3: LLM/distill puzzle_acc > 0.05 sustained ≥ 2 evals (contamination/overfit)
    if family in {"llm", "distill"}:
        sustained = 0
        for v in val_puzzles:
            if v is not None and v > 0.05:
                sustained += 1
                if sustained >= 2:
                    _flag(
                        "llm_puzzle_acc_above_5pct",
                        f"val_puzzle_acc > 0.05 for ≥ 2 evals (last={last_puz}). "
                        f"Possible mask bug, contamination, or tiny-split overfit.",
                    )
                    fired += 1
                    break
            else:
                sustained = 0

    # §B.3: cell_acc flat at chance ≥ 10 epochs (LLM Sudoku ≤ 9.5%, LLM Maze ≤ 17%)
    if family in {"llm", "distill"} and len(val_cells) >= 10:
        ceiling = 0.095 if task == "sudoku" else 0.17
        recent = [v for v in val_cells[-10:] if v is not None]
        if recent and max(recent) <= ceiling:
            _flag(
                "cell_acc_flat_at_chance",
                f"val_cell_acc max over last 10 evals = {max(recent):.4f} ≤ "
                f"{ceiling:.3f} ({task} chance). Loss-mask / optimizer / "
                f"tokenizer issue likely.",
            )
            fired += 1

    # §B.3: training loss flat ≥ 5 epochs (delta < 1e-3)
    losses_clean = [v for v in train_losses if v is not None]
    if len(losses_clean) >= 5:
        if abs(losses_clean[-1] - losses_clean[-5]) < 1e-3:
            _flag(
                "loss_flat_5_epochs",
                f"|train_loss[t] - train_loss[t-5]| = "
                f"{abs(losses_clean[-1] - losses_clean[-5]):.6f} < 1e-3. "
                f"LR / grad-flow / grad-accum issue likely.",
            )
            fired += 1

    # §B.3: cell_acc decreasing monotonically ≥ 3 evals
    cells_clean = [v for v in val_cells if v is not None]
    if len(cells_clean) >= 3:
        last3 = cells_clean[-3:]
        if last3[0] > last3[1] > last3[2]:
            _flag(
                "cell_acc_monotone_decreasing",
                f"val_cell_acc last 3 = {last3} (overfit suspected).",
            )
            fired += 1

    # §B.3: TRM avg_act_steps misbehavior
    if family == "trm":
        steps_clean = [v for v in avg_steps if v is not None]
        if len(steps_clean) >= 10 and all(s > 14 for s in steps_clean[-10:]):
            _flag(
                "trm_avg_steps_high",
                f"avg_act_steps > 14 across last 10 epochs (halt head not "
                f"learning to halt — Q-loss config or weight regression).",
            )
            fired += 1
        if steps_clean and steps_clean[0] < 2:
            _flag(
                "trm_avg_steps_collapsed",
                f"avg_act_steps < 2 from epoch 0 ({steps_clean[0]}) — halt "
                f"head over-confident, q_loss_weight regression.",
            )
            fired += 1

    # Summary line
    print(
        f"[monitor] task={task} family={family} rows={len(rows)} "
        f"last_puzzle_acc={last_puz} last_cell_acc={last_cell} "
        f"last_train_loss={last_loss} flags_fired={fired}"
    )

    # §B.7: viability gate (only meaningful at end of run)
    if fired == 0 and len(rows) >= 5:
        spec = EXPECTED.get((family, task, None))
        if family == "trm" and task == "sudoku":
            spec = EXPECTED[("trm", "sudoku", "scratch")]
        if family == "trm" and task == "maze":
            spec = EXPECTED[("trm", "maze", "hf")]
        if spec and last_puz is not None and last_cell is not None:
            puz_lo, puz_hi = spec["puzzle"]
            cell_lo, cell_hi = spec["cell"]
            puzzle_ok = puz_lo <= last_puz <= puz_hi
            cell_ok = cell_lo <= last_cell <= cell_hi
            print(
                f"[B.7] viability gate: "
                f"puzzle {last_puz:.4f} in [{puz_lo}, {puz_hi}]? {puzzle_ok}; "
                f"cell {last_cell:.4f} in [{cell_lo}, {cell_hi}]? {cell_ok}"
            )
            if puzzle_ok and cell_ok:
                print("[B.7] viability gate passed")

    return EXIT_RED if fired else EXIT_OK


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("prelaunch", help="§B.5 pre-launch sanity check")
    pl.add_argument("config")
    mo = sub.add_parser("monitor", help="§B.6/§B.7 train-log realism scan")
    mo.add_argument("train_log")
    mo.add_argument("--task", choices=["sudoku", "maze"], required=True)
    mo.add_argument(
        "--family", choices=["trm", "llm", "distill"], required=True,
    )
    args = p.parse_args(list(argv) if argv else None)
    if args.cmd == "prelaunch":
        return prelaunch(args.config)
    return monitor(args.train_log, args.task, args.family)


if __name__ == "__main__":
    sys.exit(main())
