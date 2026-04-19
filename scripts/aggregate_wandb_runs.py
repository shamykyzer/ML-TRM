"""Aggregate TRM / LLM / distill runs from wandb into a CSV for paper analysis.

Usage
-----
Fetch all TRM runs (default):
    python scripts/aggregate_wandb_runs.py

Fetch LLM baselines later (same script, one flag):
    python scripts/aggregate_wandb_runs.py --family llm

What it does
------------
1. Queries the wandb project ``<entity>/<project>`` for runs whose
   ``config.model.model_type`` is in the requested family's allow-list.
2. Extracts one row per run with: run_id, host, seed, dataset, mlp_t variant,
   best val accuracy, train time, emissions, and a *synthesized* checkpoint path
   (wandb does not log ckpt paths; we derive them from
   ``config.training.rolling_checkpoint_dir`` + ``best.pt`` convention).
3. Saves ``results/<family>_runs_overview.csv``.
4. Fetches per-step history for the best run in each (dataset, mlp_t) group
   -> ``results/history_<label>_best.csv``.
5. Prints mean/std of ``best_val_accuracy`` across seeds.

Test accuracy
-------------
Not logged during training. The ``test_accuracy`` column is left blank; fill it
post-hoc via ``src/evaluation/wandb_eval.py::evaluate_trm_run`` once checkpoints
are present locally. Keeping fetch separate from eval means re-running one does
not re-run the other.

Wandb config quirk
------------------
Runs logged via ``wandb.config.update(...)`` wrap every top-level key under a
``value`` sub-dict. Accessing ``run.config["model"]["value"]["model_type"]``
works for these runs; for cleanly-logged runs ``run.config["model"]["model_type"]``
works. ``_flatten_value_wrappers`` handles both transparently.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import wandb
except ImportError:
    sys.exit("wandb not installed. Run: pip install wandb")


ROOT = Path(__file__).resolve().parent.parent

# Enum values are uppercase in configs written by older wandb SDK versions and
# lowercase in newer ones — include both forms for every family.
FAMILIES: dict[str, list[str]] = {
    "trm": [
        "TRM_OFFICIAL_SUDOKU", "trm_official_sudoku",
        "TRM_OFFICIAL_MAZE", "trm_official_maze",
    ],
    "llm": ["LLM_FINETUNE", "llm_finetune"],
    "distill": ["LLM_DISTILL", "llm_distill"],
}


def _to_dict(v: Any) -> dict | None:
    """Coerce dict-like wandb objects (Config, SummarySubDict, plain dict) to a dict; return None otherwise."""
    if isinstance(v, dict):
        return v
    try:
        return dict(v)
    except (TypeError, ValueError):
        return None


def _unwrap(v: Any) -> Any:
    """Strip wandb's ``{value: ..., desc: ...}`` wrapper, recursively. Idempotent on plain values."""
    d = _to_dict(v)
    if d is not None and "value" in d and len(d) <= 4:
        return _unwrap(d["value"])
    return v


def _nested_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Navigate a wandb Config through arbitrary nesting, unwrapping ``.value`` shells at every level."""
    cur = cfg
    for key in keys:
        cur = _unwrap(cur)
        d = _to_dict(cur)
        if d is None:
            return default
        cur = d.get(key)
        if cur is None:
            return default
    return _unwrap(cur)


def _summary_scalar(summary, key: str, agg: str = "max") -> float | None:
    """Wandb summary metrics may be floats OR aggregation dict-likes (e.g. SummarySubDict{max: 0.74})."""
    v = summary.get(key)
    if v is None:
        return None
    d = _to_dict(v)
    if d is not None:
        for candidate in (agg, "last", "mean"):
            got = d.get(candidate)
            if got is not None:
                try:
                    return float(got)
                except (TypeError, ValueError):
                    return None
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _host_from_name(run_name: str | None) -> str:
    """Fallback: extract hostname from run names like ``trm_official_sudoku_STU-CZC5277FGD_1776064152``."""
    if not run_name:
        return ""
    parts = run_name.rsplit("_", 2)
    if len(parts) >= 2 and parts[-1].isdigit():
        return parts[-2]
    return ""


def _extract_row(run, family: str) -> dict:
    cfg = run.config
    summary = run.summary

    model_type = _nested_get(cfg, "model", "model_type") or ""
    dataset = _nested_get(cfg, "data", "dataset") or ""
    mlp_t = _nested_get(cfg, "model", "mlp_t")
    seed = _nested_get(cfg, "seed")
    host = _nested_get(cfg, "_wandb", "host") or _host_from_name(run.name)
    # Checkpoints: prefer training.rolling_checkpoint_dir (if configured), else top-level
    # checkpoint_dir. Both point to the same directory in practice; best.pt lives there.
    ckpt_dir = (
        _nested_get(cfg, "training", "rolling_checkpoint_dir")
        or _nested_get(cfg, "checkpoint_dir")
        or ""
    )
    ckpt_path = str(Path(ckpt_dir) / "best.pt") if ckpt_dir else ""

    best_val = _summary_scalar(summary, "val/exact_accuracy", "max")
    if best_val is None:
        best_val = _summary_scalar(summary, "val/puzzle_acc", "max")

    runtime = summary.get("_runtime")
    try:
        runtime = float(runtime) if runtime is not None else None
    except (TypeError, ValueError):
        runtime = None

    return {
        "family": family,
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "host": host,
        "model_type": model_type,
        "dataset": dataset,
        "mlp_t": mlp_t,
        "seed": seed,
        "best_val_accuracy": best_val,
        "val_cell_acc_max": _summary_scalar(summary, "val/cell_acc", "max")
            or _summary_scalar(summary, "val/accuracy", "max"),
        "val_avg_act_steps_max": _summary_scalar(summary, "val/avg_act_steps", "max")
            or _summary_scalar(summary, "val/avg_steps", "max"),
        "train_loss_min": _summary_scalar(summary, "train/lm_loss", "min")
            or _summary_scalar(summary, "train/loss", "min"),
        "runtime_s": runtime,
        "emissions_kg": _summary_scalar(summary, "carbon/emissions_kg", "last"),
        "energy_kwh": _summary_scalar(summary, "carbon/energy_kwh", "last"),
        "test_accuracy": None,  # fill post-hoc via src/evaluation/wandb_eval.py
        "checkpoint_dir": ckpt_dir,
        "ckpt_path": ckpt_path,
    }


def _variant_label(dataset: str, mlp_t) -> str:
    """sudoku + mlp_t=True -> 'sudoku-mlp'; sudoku + mlp_t=False -> 'sudoku-att'; maze -> 'maze'."""
    if dataset == "sudoku":
        if mlp_t is True:
            return "sudoku-mlp"
        if mlp_t is False:
            return "sudoku-att"
        return "sudoku-unknown"
    return dataset or "unknown"


def _fetch_history(api, entity: str, project: str, df: pd.DataFrame, out_dir: Path) -> None:
    ranked = df.dropna(subset=["best_val_accuracy"])
    if ranked.empty:
        print("[history] no runs with best_val_accuracy — skipping")
        return
    for (dataset, mlp_t), group in ranked.groupby(["dataset", "mlp_t"], dropna=False):
        best = group.loc[group["best_val_accuracy"].idxmax()]
        try:
            run = api.run(f"{entity}/{project}/{best['run_id']}")
            hist = run.history()  # default columns + sampled rows
        except Exception as exc:
            print(f"[history] failed to fetch run {best['run_id']}: {exc}")
            continue
        label = _variant_label(dataset, mlp_t)
        out_path = out_dir / f"history_{label}_best.csv"
        hist.to_csv(out_path, index=False)
        keys = [c for c in hist.columns if not c.startswith("_")]
        print(f"[history] {out_path.name}: {len(hist)} rows, {len(keys)} metric keys")
        print(f"[history]   keys: {sorted(keys)[:20]}{' ...' if len(keys) > 20 else ''}")


def _print_seed_stats(df: pd.DataFrame) -> None:
    ranked = df.dropna(subset=["best_val_accuracy"]).copy()
    if ranked.empty:
        print("[stats] no runs with best_val_accuracy — skipping")
        return
    for (dataset, mlp_t), group in ranked.groupby(["dataset", "mlp_t"], dropna=False):
        label = _variant_label(dataset, mlp_t)
        seeds = group[["seed", "best_val_accuracy"]].sort_values("seed")
        seed_str = ", ".join(
            f"s{int(s)}={a:.4f}" if pd.notna(s) else f"s?={a:.4f}"
            for s, a in seeds.values
        )
        vals = group["best_val_accuracy"]
        print(
            f"{label}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
            f"n={len(vals)}, seeds=[{seed_str}]"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default=os.getenv("TRM_WANDB_ENTITY", "shamykyzer"))
    ap.add_argument("--project", default=os.getenv("TRM_WANDB_PROJECT", "TRM"))
    ap.add_argument("--family", choices=list(FAMILIES), default="trm")
    ap.add_argument("--output-dir", default=str(ROOT / "results"))
    ap.add_argument("--skip-history", action="store_true", help="Skip per-step history fetch.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    path = f"{args.entity}/{args.project}"
    model_types = FAMILIES[args.family]
    filters = {"config.model.value.model_type": {"$in": model_types}}
    print(f"[fetch] {path} filters={filters}")

    try:
        runs = list(api.runs(path, filters=filters))
    except Exception as exc:
        sys.exit(f"wandb API query failed: {exc}")

    # Fallback: some older runs logged model_type without the .value wrapper.
    if not runs:
        filters = {"config.model.model_type": {"$in": model_types}}
        print(f"[fetch] retry with unwrapped filter: {filters}")
        runs = list(api.runs(path, filters=filters))

    print(f"[fetch] got {len(runs)} runs")
    if not runs:
        sys.exit("No runs matched. Check --entity/--project and model_type values.")

    # api.runs() returns lightweight refs where run.config comes back empty.
    # Re-fetching each by ID hydrates the full config (model, data, training, seed).
    # Summary is populated in the list response, so we only re-fetch for config.
    full_runs = []
    for i, r in enumerate(runs, 1):
        try:
            full_runs.append(api.run(f"{path}/{r.id}"))
        except Exception as exc:
            print(f"[fetch] {i}/{len(runs)} {r.id}: failed to hydrate ({exc}) — using lightweight ref")
            full_runs.append(r)
    print(f"[fetch] hydrated {len(full_runs)} runs")

    rows = [_extract_row(r, args.family) for r in full_runs]
    df = pd.DataFrame(rows).sort_values(
        ["dataset", "mlp_t", "seed", "run_id"], na_position="last"
    )

    csv_path = out_dir / f"{args.family}_runs_overview.csv"
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path} ({len(df)} rows)")

    print("\n=== first 10 rows ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df.head(10))

    print("\n=== mean/std across seeds ===")
    _print_seed_stats(df)

    if not args.skip_history:
        print("\n=== per-step history for best run per variant ===")
        _fetch_history(api, args.entity, args.project, df, out_dir)

    print("\n=== checkpoint path mapping ===")
    cols = ["dataset", "mlp_t", "seed", "run_id", "host", "ckpt_path"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
