"""CLI wrapper — render all 5 thesis-report figures from experiments/.

This script is the end-to-end plotting pipeline:

1. Walks ``experiments/<task>/*_train_log.csv`` for per-run curves.
2. Reads ``results/summary.csv`` (or runs the aggregator inline) for bar charts.
3. Looks up parameter counts from a small hardcoded dict (overridable).
4. Writes 5 PNGs into ``results/figures/``.

Usage
-----
    # Default: read experiments/, write results/figures/*.png
    python scripts/plot_results.py

    # Rebuild summary.csv before plotting
    python scripts/plot_results.py --rebuild-summary

    # Override param counts (JSON dict of task_name -> int)
    python scripts/plot_results.py \\
        --param-counts '{"sudoku-official": 7000000}'

The heavy lifting lives in ``src/evaluation/plots.py`` (unit-tested in
``tests/test_plots.py``). This script is intentionally thin: CSV loading,
CLI parsing, and one call per figure.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.aggregate import (  # noqa: E402
    _find_train_log,
    aggregate_experiments,
    write_summary_csv,
)
from src.evaluation.plots import (  # noqa: E402
    plot_accuracy_vs_epoch,
    plot_act_convergence,
    plot_carbon_footprint_bars,
    plot_model_accuracy_bars,
    plot_params_vs_accuracy,
)


# Default parameter counts by task-name substring. The CLI walks summary rows
# and assigns the first matching count — so e.g. "sudoku-official" resolves to
# the TRM 7M entry via the "sudoku-official" key OR the "trm" prefix heuristic.
#
# Override via --param-counts '{...}' on the command line if the auto-match is
# wrong for your experiment naming.
DEFAULT_PARAM_COUNTS: dict[str, int] = {
    # TRM family — all share the same architecture, ~7M params
    "sudoku-official": 7_000_000,
    "sudoku": 7_000_000,
    "sudoku-mlp": 7_000_000,
    "sudoku-att": 7_000_000,
    "maze": 7_000_000,
    # Baseline LLMs (common counts — real config may differ)
    "gpt2": 124_000_000,
    "smollm2": 135_000_000,
    "qwen2.5": 494_000_000,
    "llama-3.2": 1_235_000_000,
    # Distilled students inherit the student architecture
    "distilled": 7_000_000,
}


def _load_train_logs(experiments_root: Path) -> dict[str, list[dict]]:
    """Return ``{task: rows}`` for every ``<task>/*_train_log.csv`` under root.

    Rows are raw ``csv.DictReader`` output — the plot functions parse cells
    themselves (see ``plots._to_float``).
    """
    train_logs: dict[str, list[dict]] = {}
    if not experiments_root.is_dir():
        return train_logs

    for sub in sorted(experiments_root.iterdir()):
        if not sub.is_dir():
            continue
        tl = _find_train_log(sub)
        if tl is None:
            continue
        try:
            with open(tl, newline="") as fh:
                rows = list(csv.DictReader(fh))
        except OSError as exc:
            print(f"  warn: could not read {tl}: {exc}", file=sys.stderr)
            continue
        if not rows:
            continue
        train_logs[sub.name] = rows
    return train_logs


def _render_figure(
    name: str,
    fn,
    out_path: Path,
    *args,
    **kwargs,
) -> bool:
    """Call ``fn(*args, out_path, **kwargs)`` and report on result.

    Returns True on success, False if the function raised ValueError
    (e.g. no data) — we keep going so one missing dataset doesn't kill
    the other figures.
    """
    try:
        fn(*args, str(out_path), **kwargs)
    except ValueError as exc:
        print(f"  skip {name}: {exc}", file=sys.stderr)
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR {name}: {exc}", file=sys.stderr)
        return False
    print(f"  wrote {out_path}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--experiments-root",
        default="experiments",
        help="Root directory containing experiment subdirectories. "
             "(default: experiments)",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/summary.csv",
        help="Path to existing summary CSV. Regenerated in-memory from "
             "experiments/ every run so stale on-disk files don't matter; "
             "this path is used only when --rebuild-summary is set. "
             "(default: results/summary.csv)",
    )
    parser.add_argument(
        "--rebuild-summary",
        action="store_true",
        help="Also write the fresh summary CSV back to --summary-csv.",
    )
    parser.add_argument(
        "--figures-dir",
        default="results/figures",
        help="Output directory for PNG figures. (default: results/figures)",
    )
    parser.add_argument(
        "--param-counts",
        default=None,
        help="JSON dict overriding DEFAULT_PARAM_COUNTS, e.g. "
             '\'{"sudoku-official": 7000000, "gpt2": 124000000}\'',
    )
    args = parser.parse_args(argv)

    experiments_root = Path(args.experiments_root)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Resolve param counts
    param_counts = dict(DEFAULT_PARAM_COUNTS)
    if args.param_counts:
        try:
            override = json.loads(args.param_counts)
        except json.JSONDecodeError as exc:
            print(f"error: --param-counts is not valid JSON: {exc}",
                  file=sys.stderr)
            return 2
        if not isinstance(override, dict):
            print("error: --param-counts must be a JSON object", file=sys.stderr)
            return 2
        param_counts.update(override)

    # Load data
    print(f"Loading train logs from {experiments_root}/ ...")
    train_logs = _load_train_logs(experiments_root)
    print(f"  found {len(train_logs)} task(s) with data: "
          f"{sorted(train_logs.keys())}")

    print(f"Aggregating summary rows from {experiments_root}/ ...")
    summary_rows = aggregate_experiments(str(experiments_root))
    print(f"  found {len(summary_rows)} summary row(s)")

    if args.rebuild_summary and summary_rows:
        write_summary_csv(summary_rows, args.summary_csv)
        print(f"  wrote {args.summary_csv}")

    # Render figures
    print(f"\nRendering figures into {figures_dir}/ ...")
    ok = 0
    total = 0

    total += 1
    ok += _render_figure(
        "accuracy_vs_epoch",
        plot_accuracy_vs_epoch,
        figures_dir / "accuracy_vs_epoch.png",
        train_logs,
    )

    total += 1
    ok += _render_figure(
        "model_accuracy_bars",
        plot_model_accuracy_bars,
        figures_dir / "model_accuracy_bars.png",
        summary_rows,
    )

    total += 1
    ok += _render_figure(
        "carbon_footprint_bars",
        plot_carbon_footprint_bars,
        figures_dir / "carbon_footprint_bars.png",
        summary_rows,
    )

    total += 1
    ok += _render_figure(
        "params_vs_accuracy",
        plot_params_vs_accuracy,
        figures_dir / "params_vs_accuracy.png",
        summary_rows,
        param_counts,
    )

    total += 1
    ok += _render_figure(
        "act_convergence",
        plot_act_convergence,
        figures_dir / "act_convergence.png",
        train_logs,
    )

    print(f"\n{ok}/{total} figures rendered")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
