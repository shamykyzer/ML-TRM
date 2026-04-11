"""Evaluate all three Sanjin2024 HF reference checkpoints on their own tasks.

This is Regime A from the pre-flight plan: no training, just load the
published remapped checkpoints and measure their puzzle/cell accuracy on our
local test splits. Run time: ~30-60 min on a single RTX 5070 per task (the
Sudoku-Extreme test set is 423K samples; maze is 1K).

Output: three uniquely-named JSON files under results/
  - results/hf_eval_sudoku_att.json
  - results/hf_eval_sudoku_mlp.json
  - results/hf_eval_maze_hard.json

These three files are sufficient to populate the "paper-faithful baseline"
rows in the coursework report's summary table, independently of whether any
from-scratch training run converges.

The three evals are run in sequence rather than parallel — Python + CUDA
across multiple subprocess calls is fine, but loading three large checkpoints
concurrently on one GPU would OOM the 12 GB RTX 5070.

Why not just call main.py three times? Because main.py dispatches to
evaluate.save_results(results, "results", config.model.model_type.value),
and both sudoku configs resolve to the same model_type enum value
(TRM_OFFICIAL_SUDOKU), so the mlp eval would silently overwrite the att eval.
Writing the JSONs directly here sidesteps that filename clash.

Usage:
    python scripts/eval_hf_checkpoints.py              # all three
    python scripts/eval_hf_checkpoints.py sudoku_mlp   # one task
    python scripts/eval_hf_checkpoints.py sudoku_att maze_hard  # two tasks
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# dotenv is optional — if the user already set env vars in the shell, we
# skip .env loading. Required for main.py's wandb/HF env vars to take effect.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class HFEvalTarget:
    """One row in the eval plan: checkpoint path + config path + output name."""
    name: str                # short id, used for the output filename
    config_path: str         # configs/*.yaml
    checkpoint_path: str     # hf_checkpoints/<task>/remapped_for_local.pt
    description: str         # printed before the run starts


TARGETS: list[HFEvalTarget] = [
    HFEvalTarget(
        name="sudoku_att",
        config_path="configs/trm_official_sudoku.yaml",
        checkpoint_path="hf_checkpoints/Sudoku-Extreme-att/remapped_for_local.pt",
        description="Sudoku-Extreme (attention variant, mlp_t=false, paper target 77.70%)",
    ),
    HFEvalTarget(
        name="sudoku_mlp",
        config_path="configs/trm_official_sudoku_mlp.yaml",
        checkpoint_path="hf_checkpoints/Sudoku-Extreme-mlp/remapped_for_local.pt",
        description="Sudoku-Extreme (MLP-t variant, mlp_t=true, paper target 84.80%)",
    ),
    HFEvalTarget(
        name="maze_hard",
        config_path="configs/trm_official_maze.yaml",
        checkpoint_path="hf_checkpoints/Maze-Hard/remapped_for_local.pt",
        description="Maze-Hard 30x30 (attention variant, paper target 85.30%)",
    ),
]


def _run_eval_one(target: HFEvalTarget) -> dict:
    """Load the config + checkpoint and run evaluation. Returns the results dict."""
    # Imports are inside the function so that a missing HF checkpoint file
    # can short-circuit without paying torch import cost.
    from torch.utils.data import DataLoader

    from src.data.collate import official_collate_fn
    from src.evaluation.evaluate import load_and_evaluate
    from src.utils.config import load_config

    config = load_config(target.config_path)
    collate_fn = official_collate_fn(config.training.task_id)

    if config.data.dataset == "maze":
        from src.data.maze_dataset import MazeDataset
        test_ds = MazeDataset(config.data.data_dir, "test")
    else:
        from src.data.sudoku_dataset import SudokuDataset
        test_ds = SudokuDataset(config.data.data_dir, "test")

    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return load_and_evaluate(target.checkpoint_path, test_loader, config)


def _write_result(name: str, result: dict, elapsed_s: float) -> str:
    """Write the per-target JSON. Returns the output path."""
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"hf_eval_{name}.json")

    # Shallow-copy so we don't mutate the caller's dict, then add metadata.
    payload = dict(result)
    payload["_source"] = "scripts/eval_hf_checkpoints.py"
    payload["_elapsed_sec"] = round(elapsed_s, 2)

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out_path


def main() -> int:
    requested = sys.argv[1:]
    if requested:
        targets = [t for t in TARGETS if t.name in requested]
        unknown = set(requested) - {t.name for t in TARGETS}
        if unknown:
            print(f"Unknown target(s): {sorted(unknown)}")
            print(f"Valid: {[t.name for t in TARGETS]}")
            return 2
    else:
        targets = TARGETS

    # Pre-flight: verify every checkpoint exists before doing any work.
    missing = [t for t in targets if not os.path.exists(t.checkpoint_path)]
    if missing:
        print("Missing HF checkpoint file(s):")
        for t in missing:
            print(f"  - {t.name}: {t.checkpoint_path}")
        print(
            "\nRun `python start.py` to trigger the transfer stage, or "
            "verify the remap scripts under scripts/remap_*.py have been "
            "run for each task."
        )
        return 1

    print(f"\n=== Evaluating {len(targets)} HF checkpoint(s) ===")
    for t in targets:
        print(f"  - {t.name}: {t.description}")
    print()

    summary = []
    for target in targets:
        print(f"\n--- {target.name} ---")
        print(f"    config:      {target.config_path}")
        print(f"    checkpoint:  {target.checkpoint_path}")
        t0 = time.time()
        try:
            result = _run_eval_one(target)
        except Exception as exc:  # noqa: BLE001 — we want to log and continue
            elapsed = time.time() - t0
            print(f"    FAILED after {elapsed:.1f}s: {exc}")
            summary.append((target.name, None, elapsed, str(exc)))
            continue
        elapsed = time.time() - t0
        out_path = _write_result(target.name, result, elapsed)
        print(f"    OK in {elapsed:.1f}s  ->  {out_path}")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"      {k}: {v:.4f}")
        summary.append((target.name, result, elapsed, None))

    print("\n=== Summary ===")
    for name, result, elapsed, error in summary:
        if error:
            print(f"  {name:12s} FAILED ({elapsed:.1f}s): {error}")
            continue
        puzzle = result.get("puzzle_accuracy", result.get("puzzle_acc", float("nan")))
        cell = result.get("cell_accuracy", result.get("cell_acc", float("nan")))
        print(f"  {name:12s} puzzle={puzzle:.4f}  cell={cell:.4f}  ({elapsed:.1f}s)")

    # Non-zero exit if any target failed.
    failures = [s for s in summary if s[3] is not None]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
