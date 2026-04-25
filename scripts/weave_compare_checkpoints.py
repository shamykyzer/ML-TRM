"""Run weave.Evaluation across multiple TRM Sudoku checkpoints.

Produces a leaderboard at https://wandb.ai/<entity>/<project>/weave/evaluations
with per-puzzle drill-down, replacing the manual CSV-join in findings.md §5.2.
Each Evaluation call also auto-publishes the underlying weave.Model so the
checkpoints become selectable in the wandb Playground UI for individual
qualitative inspection.

Usage
-----
Default: auto-discover the HF init + every C:/ml-trm-work/sudoku-mlp-*/best.pt:

    python scripts/weave_compare_checkpoints.py

Custom list (name=path syntax — only listed checkpoints are evaluated):

    python scripts/weave_compare_checkpoints.py \
        hf-init=hf_checkpoints/Sudoku-Extreme-mlp/remapped_for_local.pt \
        seed0-best=C:/ml-trm-work/sudoku-mlp-seed0/best.pt \
        seed4-ep480=C:/ml-trm-work/sudoku-mlp-seed4/epoch_480.pt

Sample size is 200 by default (a few minutes per checkpoint on RTX 5070);
override with --sample-size for the full 6.6 k-puzzle eval.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import torch  # noqa: E402

import weave  # noqa: E402

from src.data.sudoku_dataset import SudokuDataset  # noqa: E402
from src.evaluation.weave_models import TRMSudokuModel  # noqa: E402
from src.utils.config import load_config  # noqa: E402


@weave.op()
def cell_accuracy(label: list[int], output: dict) -> float:
    """Per-cell accuracy on the (originally blank) cells of one puzzle.

    label uses stored-token space (0=ignore, 1=blank, 2..10=digits). The 0
    sentinel marks pre-filled cells we don't grade — those are gimme clues.
    """
    pred = output["prediction"]
    correct = 0
    graded = 0
    for p, l in zip(pred, label):
        if l == 0:
            continue
        graded += 1
        if p == l:
            correct += 1
    return correct / max(1, graded)


@weave.op()
def puzzle_correct(label: list[int], output: dict) -> bool:
    """Whether every (graded) cell is exactly right — the puzzle_acc metric."""
    pred = output["prediction"]
    for p, l in zip(pred, label):
        if l == 0:
            continue
        if p != l:
            return False
    return True


@weave.op()
def halt_step(label: list[int], output: dict) -> int:
    """Pass-through scorer so the Evaluation table shows the halt step."""
    del label
    return int(output["halt_step"])


def _build_dataset(config_path: str, sample_size: int) -> weave.Dataset:
    """Load the test split, take the first `sample_size` puzzles, wrap in weave.Dataset.

    "First N" instead of random sampling so the eval set is reproducible
    across script invocations — important for comparing two leaderboard runs.
    """
    cfg = load_config(config_path)
    ds = SudokuDataset(cfg.data.data_dir, "test")
    n = min(sample_size, len(ds))
    rows = []
    for i in range(n):
        inputs, labels = ds[i]
        rows.append({
            "id": i,
            "puzzle": inputs.tolist(),
            "label": labels.tolist(),
        })
    return weave.Dataset(name=f"sudoku_test_first{n}", rows=rows)


def _discover_checkpoints() -> list[tuple[str, str]]:
    """Auto-find the HF init + every C:/ml-trm-work/sudoku-mlp-*/best.pt.

    Returns list of (display_name, absolute_path) tuples in deterministic order.
    Skipped silently when a candidate path doesn't exist on this machine —
    the user can always pass explicit name=path args instead.
    """
    found: list[tuple[str, str]] = []
    hf_init = REPO_ROOT / "hf_checkpoints" / "Sudoku-Extreme-mlp" / "remapped_for_local.pt"
    if hf_init.is_file():
        found.append(("hf-init", str(hf_init)))

    work_root = Path("C:/ml-trm-work")
    if work_root.is_dir():
        for sub in sorted(work_root.iterdir()):
            if not sub.name.startswith("sudoku-mlp"):
                continue
            best = sub / "best.pt"
            if best.is_file():
                found.append((sub.name, str(best)))
    return found


def _parse_named_paths(items: list[str]) -> list[tuple[str, str]]:
    """Parse `name=path` argv tokens; bare paths get the basename as name."""
    out: list[tuple[str, str]] = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name, path = Path(item).stem, item
        if not os.path.isfile(path):
            print(f"[WARN] checkpoint not found, skipping: {path}", file=sys.stderr)
            continue
        out.append((name, path))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "checkpoints", nargs="*",
        help="name=path pairs (or bare paths); empty = auto-discover",
    )
    parser.add_argument(
        "--config", default="configs/trm_official_sudoku_mlp.yaml",
        help="Config file (used to build the model architecture)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=200,
        help="Number of test puzzles to score (200 ~= 5 min/ckpt on RTX 5070)",
    )
    parser.add_argument(
        "--project", default=os.getenv("TRM_WANDB_PROJECT") or "TRM",
        help="wandb/weave project to publish into",
    )
    parser.add_argument(
        "--entity", default=os.getenv("TRM_WANDB_ENTITY") or "",
        help="wandb entity (default = your wandb login)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)

    if args.checkpoints:
        ckpts = _parse_named_paths(args.checkpoints)
    else:
        ckpts = _discover_checkpoints()
    if not ckpts:
        print("ERROR: no checkpoints to evaluate (none provided, none auto-discovered)", file=sys.stderr)
        return 2

    print(f"[weave] {len(ckpts)} checkpoint(s) to evaluate:")
    for name, path in ckpts:
        print(f"  - {name}: {path}")

    target = f"{args.entity}/{args.project}" if args.entity else args.project
    weave.init(target)
    print(f"[weave] init -> {target}")

    dataset = _build_dataset(args.config, args.sample_size)
    print(f"[weave] dataset: {len(dataset.rows)} puzzles")

    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[cell_accuracy, puzzle_correct, halt_step],
        name="sudoku_extreme_compare",
    )

    for name, path in ckpts:
        print(f"\n[weave] evaluating {name} ...")
        model = TRMSudokuModel(
            name=name,
            checkpoint_path=path,
            config_path=args.config,
            device_str=args.device,
        )
        # Evaluation.evaluate is async — run synchronously from this CLI.
        result = asyncio.run(evaluation.evaluate(model))
        cell = result.get("cell_accuracy", {}).get("mean")
        puzz = result.get("puzzle_correct", {}).get("true_fraction")
        steps = result.get("halt_step", {}).get("mean")
        print(f"  cell_acc={cell}  puzzle_acc={puzz}  avg_halt_step={steps}")

    print()
    print("LEADERBOARD URL:")
    print(f"  https://wandb.ai/{target}/weave/evaluations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
