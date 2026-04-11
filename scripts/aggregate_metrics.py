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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.aggregate import (  # noqa: E402
    _find_train_log,
    aggregate_experiments,
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
    return (
        f"  {row['task']:<24} "
        f"puzzle={row.get('best_val_puzzle_acc', 0):.4f}  "
        f"cell={row.get('best_val_cell_acc', 0):.4f}  "
        f"epoch={row.get('final_epoch', 0):<5} "
        f"time={row.get('train_time_min', 0):6.1f}m  "
        f"energy={row.get('train_energy_kwh', 0) * 1000:.2f}Wh  "
        f"co2={row.get('train_co2_kg', 0) * 1000:.2f}g"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--experiments-root",
        default="experiments",
        help="Root directory containing experiment subdirectories. "
             "(default: experiments)",
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

    rows = aggregate_experiments(args.experiments_root)
    skipped = _list_skipped_dirs(
        args.experiments_root, included_tasks={r["task"] for r in rows}
    )

    if not rows:
        print(
            f"No experiments found under {args.experiments_root!r}. "
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
        print(f"Aggregated {len(rows)} experiment(s) -> {args.out_csv}")
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
