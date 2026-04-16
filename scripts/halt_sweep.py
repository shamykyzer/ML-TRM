"""One-off hypothesis test: does halt_max_steps drive the 79.6% → 100% gap?

Sweeps halt_max_steps ∈ {1, 2, 4, 8, 16} on:
  - HF init (Sanjin2024 remapped checkpoint, pre-training)
  - best.pt from maze-seed0 (post-training)

Prints puzzle_acc / cell_acc / avg_steps per (checkpoint, halt) pair so we
can judge whether the 20-pp jump is about the model learning to halt early
or about actual weight improvements during fine-tuning.

Disposable script — not wired into start.py or anything else. Delete after use.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


HALT_VALUES = [1, 2, 4, 8, 16]
HF_CKPT = "hf_checkpoints/Maze-Hard/remapped_for_local.pt"
TRAINED_CKPT = "C:/ml-trm-work/maze-seed0/best.pt"


def main() -> int:
    from torch.utils.data import DataLoader

    from src.data.collate import official_collate_fn
    from src.data.maze_dataset import MazeDataset
    from src.evaluation.evaluate import load_and_evaluate
    from src.utils.config import load_config

    config = load_config("configs/trm_official_maze.yaml")
    test_ds = MazeDataset(config.data.data_dir, "test")
    collate_fn = official_collate_fn(config.training.task_id)

    targets: list[tuple[str, str]] = [("HF-init", HF_CKPT)]
    if os.path.exists(TRAINED_CKPT):
        targets.append(("trained", TRAINED_CKPT))
    else:
        print(f"[warn] {TRAINED_CKPT} not found — skipping trained checkpoint")

    print(
        f"{'checkpoint':<10} {'halt_max':>8} "
        f"{'puzzle':>8} {'cell':>8} {'avg_steps':>10} {'t(s)':>6}"
    )
    print("-" * 56)

    for name, ckpt_path in targets:
        for halt_max in HALT_VALUES:
            # Override halt_max_steps before eval — the eval loop reads this
            # at iteration time (src/evaluation/evaluate.py:177), so mutating
            # the config mid-run propagates without rebuilding the model.
            config.model.halt_max_steps = halt_max

            # Rebuild the DataLoader each time because collate_fn closures
            # can't be reused across DataLoader instances on Windows. Cheap.
            test_loader = DataLoader(
                test_ds,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=0,  # sidestep Windows spawn pickle issues
                pin_memory=True,
                collate_fn=collate_fn,
            )

            t0 = time.time()
            result = load_and_evaluate(ckpt_path, test_loader, config)
            elapsed = time.time() - t0

            p = result.get("puzzle_accuracy", 0.0)
            c = result.get("cell_accuracy", 0.0)
            a = result.get("avg_act_steps", 0.0)
            print(
                f"{name:<10} {halt_max:>8} "
                f"{p:>8.4f} {c:>8.4f} {a:>10.2f} {elapsed:>6.1f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
