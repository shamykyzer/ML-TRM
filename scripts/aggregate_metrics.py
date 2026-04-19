"""CLI wrapper around src.evaluation.aggregate — build results/summary.csv.

Parses every ``experiments/<task>/*_train_log.csv`` + ``emissions.csv`` in one
pass and writes a single ``results/summary.csv`` row per task. This is the
ingest step feeding ``scripts/plot_results.py`` and the thesis report tables.

Usage
-----
    # Default: read experiments/, write results/summary.csv, print a table
    python scripts/aggregate_metrics.py

    # Custom paths
    python scripts/aggregate_metrics.py \\
        --experiments-root experiments \\
        --out-csv results/summary.csv

    # CI / smoke mode — skip the table print
    python scripts/aggregate_metrics.py --quiet

The heavy lifting lives in ``src/evaluation/aggregate.py`` so it is unit-tested
(see ``tests/test_aggregate.py``). This script is intentionally thin: parse
args, call the library, print a human-readable summary.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.aggregate import (  # noqa: E402
    _find_train_log,
    aggregate_all_experiments,
    aggregate_experiments,
    attach_efficiency_metrics,
    parse_train_log,
    write_summary_csv,
)


def _list_skipped_dirs(root: str, included_tasks: set[str]) -> list[str]:
    """Find experiment subdirs that exist but were not included in the summary.

    A dir is considered "skipped" if it contains no train_log CSV, or if its
    train_log is empty/malformed (parse_train_log returns None). Called after
    aggregate_experiments so the CLI can warn the user — otherwise a typo in
    a subdir or an empty train log silently disappears from the summary.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    skipped: list[str] = []
    for sub in sorted(root_path.iterdir()):
        if not sub.is_dir() or sub.name in included_tasks:
            continue
        tl = _find_train_log(sub)
        if tl is None:
            skipped.append(f"{sub.name} (no *_train_log.csv)")
        elif parse_train_log(str(tl)) is None:
            skipped.append(f"{sub.name} (empty/invalid train_log)")
    return skipped


def _format_row(row: dict) -> str:
    """Format one summary row for human consumption on stdout."""
    co2_per = row.get("co2_per_correct_puzzle", "")
    co2_per_str = f"{co2_per * 1000:.2f}g" if isinstance(co2_per, (int, float)) and co2_per else "inf   "
    peak = row.get("peak_epoch", row.get("final_epoch", 0))
    final = row.get("final_epoch", 0)
    epoch_str = f"peak@{peak}/final@{final}" if peak != final else f"epoch={final}"
    return (
        f"  {row['task']:<28} "
        f"puzzle={row.get('best_val_puzzle_acc', 0):.4f}  "
        f"cell={row.get('best_val_cell_acc', 0):.4f}  "
        f"{epoch_str:<22} "
        f"time={row.get('train_time_min', 0):6.1f}m  "
        f"co2={row.get('train_co2_kg', 0) * 1000:.1f}g  "
        f"co2/correct={co2_per_str}"
    )


def _collect_roots(primary: str, additional: list[str]) -> list[str]:
    """Return a de-duplicated list of existing roots.

    Order: primary → --additional-roots → $TRM_EXPERIMENT_DIR (auto).
    Missing / non-existent roots are skipped silently.
    """
    candidates = [primary, *additional]
    env_root = os.getenv("TRM_EXPERIMENT_DIR", "").strip()
    if env_root:
        candidates.append(env_root)

    seen: set[str] = set()
    out: list[str] = []
    for r in candidates:
        if not r:
            continue
        key = str(Path(r).resolve())
        if key in seen:
            continue
        if not Path(r).is_dir():
            continue
        seen.add(key)
        out.append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--experiments-root",
        default="experiments",
        help="Root directory containing experiment subdirectories. "
             "(default: experiments)",
    )
    parser.add_argument(
        "--additional-roots",
        action="append",
        default=[],
        metavar="DIR",
        help="Extra root(s) to walk (repeat flag for multiple). LLM + "
             "distillation runs are stored under TRM_EXPERIMENT_DIR; pass "
             "that path here or set the env var and it will be picked up "
             "automatically.",
    )
    parser.add_argument(
        "--out-csv",
        default="results/summary.csv",
        help="Where to write the aggregated summary CSV. "
             "(default: results/summary.csv)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable table on stdout.",
    )
    args = parser.parse_args(argv)

    roots = _collect_roots(args.experiments_root, args.additional_roots)
    if not roots:
        print(
            f"No valid roots found. Tried: {args.experiments_root!r} "
            f"plus --additional-roots and $TRM_EXPERIMENT_DIR.",
            file=sys.stderr,
        )
        return 1

    rows = aggregate_all_experiments(roots)

    # Attach CO2/kWh-per-correct-puzzle metrics using TEST_SET_SIZES per family.
    # Leaves the per-correct columns blank for zero-correct rows (cleaner than
    # writing inf to the CSV).
    rows = [attach_efficiency_metrics(r) for r in rows]

    # Per-root "skipped" listing, so typos or crashed dirs surface.
    skipped: list[str] = []
    included = {r["task"] for r in rows}
    for root in roots:
        for s in _list_skipped_dirs(root, included_tasks=included):
            skipped.append(f"{root}/{s}")

    if not rows:
        print(
            f"No experiments found. Walked: {roots}. "
            f"Expected subdirectories containing *_train_log.csv files.",
            file=sys.stderr,
        )
        if skipped:
            print("Skipped:", file=sys.stderr)
            for s in skipped:
                print(f"  {s}", file=sys.stderr)
        return 1

    write_summary_csv(rows, args.out_csv)

    if not args.quiet:
        print(f"Aggregated {len(rows)} experiment(s) from {len(roots)} root(s) -> {args.out_csv}")
        print(f"Roots walked: {roots}")
        print()
        for row in rows:
            print(_format_row(row))
        if skipped:
            print()
            print("Skipped (not in summary):")
            for s in skipped:
                print(f"  {s}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
