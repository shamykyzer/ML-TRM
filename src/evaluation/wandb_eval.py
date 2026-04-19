"""Post-hoc test-set evaluation for runs fetched from wandb.

Pairs with ``scripts/aggregate_wandb_runs.py``. That script produces a CSV with
one row per wandb run but leaves ``test_accuracy`` blank (wandb only logs
train/val). This module fills in those blanks by loading each checkpoint and
running the full test set.

Design
------
Everything already exists. We compose:

- ``src.evaluation.evaluate.load_and_evaluate`` — loads a checkpoint for any
  supported ``ModelType`` and runs the appropriate eval loop (TRMOfficial uses
  ``evaluate_official`` with EMA + ACT halting).
- ``src.data.{sudoku,maze}_dataset.get_*_loaders`` — returns ``(train, test)``.
- ``src.utils.config.load_config`` — parses an ``ExperimentConfig`` from YAML.

Do NOT reimplement checkpoint loading, model construction, or the eval loop
here — ``load_and_evaluate`` is the single source of truth and it handles
subtle details (``strict=False`` state-dict load, forward_dtype, EMA swap).

See ``scripts/eval_hf_checkpoints.py`` for the equivalent pattern used to
evaluate HF-released weights.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.maze_dataset import get_maze_loaders
from src.data.sudoku_dataset import get_sudoku_loaders
from src.evaluation.evaluate import load_and_evaluate
from src.utils.config import ExperimentConfig, load_config


def _test_loader_for(config: ExperimentConfig):
    dataset = config.data.dataset
    if dataset == "sudoku":
        _, test_loader = get_sudoku_loaders(
            data_dir=config.data.data_dir,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
        )
        return test_loader
    if dataset == "maze":
        _, test_loader = get_maze_loaders(
            data_dir=config.data.data_dir,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            mask_non_path=config.data.mask_non_path,
        )
        return test_loader
    raise ValueError(f"Unsupported dataset for TRM eval: {dataset!r}")


def evaluate_trm_run(ckpt_path: str, config_path: str) -> dict:
    """Run test-set evaluation for a single TRM run.

    Parameters
    ----------
    ckpt_path : str
        Absolute path to a checkpoint saved by ``OfficialTRMTrainer`` — typically
        ``{rolling_checkpoint_dir}/best.pt``.
    config_path : str
        Path to the YAML config used for this run (e.g.
        ``configs/trm_official_sudoku.yaml`` or ``configs/trm_official_maze.yaml``).
        Must match the model architecture of the checkpoint.

    Returns
    -------
    dict
        Keys from ``load_and_evaluate``: ``puzzle_accuracy``, ``cell_accuracy``,
        ``avg_act_steps`` (and others depending on model family).
    """
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    config = load_config(config_path)
    test_loader = _test_loader_for(config)
    return load_and_evaluate(str(ckpt), test_loader, config)


def backfill_test_accuracy(
    overview_csv: str,
    config_resolver,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Fill the ``test_accuracy`` column in an overview CSV produced by
    ``scripts/aggregate_wandb_runs.py``.

    Parameters
    ----------
    overview_csv : str
        Path to ``trm_runs_overview.csv`` (or similar).
    config_resolver : callable
        ``config_resolver(row: pd.Series) -> str`` returning the YAML path for
        each row. Row has ``dataset``, ``mlp_t``, ``seed``, etc. Typical impl:
        pick ``configs/trm_official_sudoku.yaml`` vs ``trm_official_maze.yaml``
        based on ``row["dataset"]``.
    output_csv : str or None
        Where to write the updated CSV. Defaults to overwriting ``overview_csv``.

    Notes
    -----
    Rows with a missing checkpoint file are left blank and logged; this is
    expected when runs came from multiple machines and not every ``best.pt`` is
    present locally. Copy the missing files and re-run this function to fill
    the remaining rows.
    """
    df = pd.read_csv(overview_csv)
    if "test_accuracy" not in df.columns:
        df["test_accuracy"] = None

    for idx, row in df.iterrows():
        ckpt = row.get("ckpt_path") or ""
        if not ckpt or not Path(ckpt).exists():
            print(f"[skip] row {idx} ({row.get('run_id')}): missing {ckpt or '<no path>'}")
            continue
        try:
            cfg_path = config_resolver(row)
            result = evaluate_trm_run(ckpt, cfg_path)
        except Exception as exc:
            print(f"[fail] row {idx} ({row.get('run_id')}): {exc}")
            continue
        df.at[idx, "test_accuracy"] = result.get("puzzle_accuracy")
        df.at[idx, "test_cell_accuracy"] = result.get("cell_accuracy")
        print(f"[ok]   row {idx} ({row.get('run_id')}): "
              f"puzzle={result.get('puzzle_accuracy'):.4f} "
              f"cell={result.get('cell_accuracy'):.4f}")

    out = output_csv or overview_csv
    df.to_csv(out, index=False)
    print(f"[save] {out}")
    return df
