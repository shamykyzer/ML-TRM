"""One-shot Sunday-morning pipeline: aggregate everything for the report.

Runs the full post-training finalization in order, fail-soft. Each step is
independent — a failure prints a warning and the rest continue, so you can
re-run the script after fixing the broken step instead of re-running the
ones that already succeeded.

Pipeline:
  1. scripts/aggregate_metrics.py            -> results/summary.csv
                                                (local emissions.csv + train_log.csv)
  2. scripts/aggregate_wandb_runs.py --family {trm,llm,distill}
                                              -> results/{family}_runs_overview.csv
                                                (wandb cloud, per-run rows)
  3. backfill_test_accuracy(overview_csv, ...)
                                              -> updates test_accuracy column
                                                (needs best.pt files locally)
  4. scripts/run_novelty_aggregate.py        -> results/novelty/k_vote_results.csv
                                                (K-vote accuracy)
  5. Print a summary of what landed in results/

Usage:
    python scripts/finalize_results.py
    python scripts/finalize_results.py --skip-wandb       # if no internet
    python scripts/finalize_results.py --skip-test-eval   # if no checkpoints synced
    python scripts/finalize_results.py --skip-kvote       # if K-vote not run
    python scripts/finalize_results.py --families trm     # only TRM family

Run this after all training finishes (Sunday) on whichever machine has the
best.pt files synced (or wandb access for steps 1-2 only).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _hr(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _ok(msg: str) -> None:
    print(f"  OK    {msg}")


def _warn(msg: str) -> None:
    print(f"  WARN  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def _safe(label: str, fn: Callable[[], None]) -> bool:
    """Run fn() under try/except. Returns True on success."""
    try:
        fn()
        return True
    except SystemExit as e:
        if e.code in (0, None):
            return True
        _fail(f"{label}: exited with code {e.code}")
        return False
    except Exception as e:
        _fail(f"{label}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Step 1 — local aggregator (emissions.csv + train_log.csv)
# ---------------------------------------------------------------------------

def step_local_aggregate(experiments_root: str | None) -> None:
    _hr("Step 1: aggregate_metrics.py (local files -> results/summary.csv)")
    args = [sys.executable, str(REPO_ROOT / "scripts" / "aggregate_metrics.py")]
    if experiments_root:
        args += ["--experiments-root", experiments_root]
    print(f"  $ {' '.join(args)}")
    result = subprocess.run(args, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"aggregate_metrics.py exited {result.returncode}")
    out = RESULTS / "summary.csv"
    if out.exists():
        _ok(f"wrote {out} ({out.stat().st_size} bytes)")
    else:
        _warn(f"{out} not produced — check stdout above")


# ---------------------------------------------------------------------------
# Step 2 — wandb aggregator per family
# ---------------------------------------------------------------------------

def step_wandb_aggregate(family: str) -> None:
    _hr(f"Step 2.{family}: aggregate_wandb_runs.py --family {family}")
    args = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "aggregate_wandb_runs.py"),
        "--family", family,
    ]
    print(f"  $ {' '.join(args)}")
    result = subprocess.run(args, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"aggregate_wandb_runs.py --family {family} exited {result.returncode}")
    out = RESULTS / f"{family}_runs_overview.csv"
    if out.exists():
        _ok(f"wrote {out} ({out.stat().st_size} bytes)")
    else:
        _warn(f"{out} not produced — check stdout above")


# ---------------------------------------------------------------------------
# Step 3 — backfill test_accuracy (needs local best.pt files)
# ---------------------------------------------------------------------------

def _trm_config_resolver(row) -> str:
    """Pick the right TRM YAML for a wandb row."""
    dataset = str(row.get("dataset") or "").lower()
    mlp_t = bool(row.get("mlp_t"))
    init_weights = str(row.get("init_weights") or "")

    if dataset == "sudoku":
        if mlp_t:
            # HF-init runs use the finetune config; from-scratch uses the standard.
            if "remapped_for_local" in init_weights or "hf_checkpoints" in init_weights:
                return "configs/trm_official_sudoku_mlp_finetune.yaml"
            return "configs/trm_official_sudoku_mlp.yaml"
        return "configs/trm_official_sudoku.yaml"

    if dataset == "maze":
        if "remapped_for_local" in init_weights or "hf_checkpoints" in init_weights:
            return "configs/trm_official_maze_finetune.yaml"
        return "configs/trm_official_maze.yaml"

    raise ValueError(f"unknown dataset for TRM row: {dataset!r}")


def _llm_config_resolver(row) -> str:
    """Best-effort LLM config picker. Falls back to llm_config.yaml."""
    dataset = str(row.get("dataset") or "").lower()
    model_type = str(row.get("model_type") or row.get("model_name") or "").lower()

    # Heuristic: family by model_type substring
    families = [
        ("gpt2", "configs/llm_config.yaml", "configs/llm_gpt2_maze.yaml"),
        ("smollm", "configs/llm_smollm.yaml", "configs/llm_smollm_maze.yaml"),
        ("qwen", "configs/llm_qwen.yaml", "configs/llm_qwen_maze.yaml"),
        ("llama", "configs/llm_llama.yaml", "configs/llm_llama_maze.yaml"),
    ]
    for keyword, sudoku_cfg, maze_cfg in families:
        if keyword in model_type:
            return sudoku_cfg if dataset == "sudoku" else maze_cfg

    # Default — assume sudoku unless dataset says otherwise
    return "configs/llm_config.yaml" if dataset == "sudoku" else "configs/llm_gpt2_maze.yaml"


def step_backfill(family: str) -> None:
    overview = RESULTS / f"{family}_runs_overview.csv"
    if not overview.exists():
        _warn(f"{overview} not present — run step 2 first or the family has no runs")
        return

    _hr(f"Step 3.{family}: backfill_test_accuracy on {overview.name}")

    # Import lazily so the script still parses on machines without torch.
    from src.evaluation.wandb_eval import backfill_test_accuracy

    resolver = _trm_config_resolver if family == "trm" else _llm_config_resolver
    df = backfill_test_accuracy(str(overview), resolver)

    n_filled = int(df["test_accuracy"].notna().sum()) if "test_accuracy" in df.columns else 0
    n_total = len(df)
    _ok(f"{n_filled}/{n_total} rows have test_accuracy after backfill")
    if n_filled < n_total:
        _warn(f"{n_total - n_filled} rows missing test_accuracy — likely missing best.pt locally")


# ---------------------------------------------------------------------------
# Step 4 — K-vote accuracy aggregation
# ---------------------------------------------------------------------------

def step_kvote() -> None:
    _hr("Step 4: run_novelty_aggregate.py (K-vote accuracy CSV)")
    script = REPO_ROOT / "scripts" / "run_novelty_aggregate.py"
    if not script.exists():
        _warn(f"{script} not found — skipping K-vote step")
        return
    args = [sys.executable, str(script)]
    print(f"  $ {' '.join(args)}")
    result = subprocess.run(args, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"run_novelty_aggregate.py exited {result.returncode}")
    out = RESULTS / "novelty" / "k_vote_results.csv"
    if out.exists():
        _ok(f"wrote {out} ({out.stat().st_size} bytes)")
    else:
        _warn(f"{out} not produced — check stdout above")


# ---------------------------------------------------------------------------
# Step 5 — summary print
# ---------------------------------------------------------------------------

def step_summary(families: list[str]) -> None:
    _hr("Step 5: results/ contents after finalize")
    expected = [
        RESULTS / "summary.csv",
        *[RESULTS / f"{f}_runs_overview.csv" for f in families],
        RESULTS / "novelty" / "k_vote_results.csv",
    ]
    for p in expected:
        if p.exists():
            _ok(f"{p.relative_to(REPO_ROOT)} ({p.stat().st_size} bytes)")
        else:
            _warn(f"{p.relative_to(REPO_ROOT)} MISSING")

    print()
    print("Next: open results/*_runs_overview.csv and verify the headline rows")
    print("      for the report's Table 1.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--families", nargs="+", default=["trm", "llm", "distill"],
        help="Which wandb families to aggregate (default: all)",
    )
    parser.add_argument("--experiments-root", default=None,
                        help="Override experiments/ root for step 1 (default: project root)")
    parser.add_argument("--skip-local", action="store_true",
                        help="Skip step 1 (local aggregate_metrics.py)")
    parser.add_argument("--skip-wandb", action="store_true",
                        help="Skip step 2 (wandb fetch)")
    parser.add_argument("--skip-test-eval", action="store_true",
                        help="Skip step 3 (test_accuracy backfill)")
    parser.add_argument("--skip-kvote", action="store_true",
                        help="Skip step 4 (K-vote aggregation)")
    args = parser.parse_args(argv)

    print(f"  REPO_ROOT = {REPO_ROOT}")
    print(f"  RESULTS   = {RESULTS}")
    print(f"  families  = {args.families}")

    if not args.skip_local:
        _safe("step 1 local", lambda: step_local_aggregate(args.experiments_root))

    if not args.skip_wandb:
        for fam in args.families:
            _safe(f"step 2.{fam}", lambda f=fam: step_wandb_aggregate(f))

    if not args.skip_test_eval:
        for fam in args.families:
            _safe(f"step 3.{fam}", lambda f=fam: step_backfill(f))

    if not args.skip_kvote:
        _safe("step 4 kvote", step_kvote)

    step_summary(args.families)
    return 0


if __name__ == "__main__":
    sys.exit(main())
