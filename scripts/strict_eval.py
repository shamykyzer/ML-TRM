"""Strict maze eval: grade ALL 900 cells, not just the path cells.

The standard `puzzle_accuracy` metric only checks path cells (where label=5).
A model that outputs `o` at every position trivially scores 100%, which we
just caught happening in maze-seed0/best.pt. This script adds a stricter
metric: the model must output the FULL maze correctly — walls as walls,
open as open, S as S, G as G, and path cells as `o`.

For each (checkpoint, data_dir) pair it prints both the standard and strict
metrics side by side so you can see how much of the standard number is the
reward-hacking shortcut.

Usage:
    python scripts/strict_eval.py

Disposable — delete after use.
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


HF_CKPT = "hf_checkpoints/Maze-Hard/remapped_for_local.pt"
TRAINED_CKPT = "C:/ml-trm-work/maze-seed0/best.pt"
IN_DIST = "data/maze-30x30-hard-1k"
OOD = "data/maze-30x30-hard-1k-ood"
HALT_MAX = 16  # give the model its full "thinking" budget


def eval_both_metrics(checkpoint_path: str, data_dir: str, config) -> dict:
    """Run one eval and compute BOTH the standard path-only metric and the
    strict full-maze metric.

    Returns dict with:
      std_cell, std_puzzle     — the standard metric (path cells only)
      strict_cell, strict_puzzle — all 900 cells must match
    """
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.data.collate import official_collate_fn
    from src.data.maze_dataset import MazeDataset
    from src.models.trm_official import TRMOfficial

    config.data.data_dir = data_dir
    config.model.halt_max_steps = HALT_MAX

    test_ds = MazeDataset(data_dir, "test")
    collate_fn = official_collate_fn(config.training.task_id)
    loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Build model from config
    model_config = {
        "batch_size": config.training.batch_size,
        "seq_len": config.model.seq_len,
        "vocab_size": config.model.vocab_size,
        "num_task_types": config.model.num_task_types,
        "task_emb_ndim": config.model.task_emb_ndim,
        "task_emb_len": config.model.task_emb_len,
        "hidden_size": config.model.d_model,
        "expansion": config.model.ff_hidden / config.model.d_model,
        "num_heads": config.model.n_heads,
        "L_layers": config.model.L_layers,
        "H_cycles": config.model.H_cycles,
        "L_cycles": config.model.L_cycles,
        "halt_max_steps": config.model.halt_max_steps,
        "halt_exploration_prob": config.model.halt_exploration_prob,
        "no_ACT_continue": config.model.no_ACT_continue,
        "forward_dtype": config.model.forward_dtype,
        "mlp_t": config.model.mlp_t,
        "pos_encodings": "rope",
    }
    model = TRMOfficial(model_config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    fwd_dtype = getattr(torch, config.model.forward_dtype)
    model.to(device="cuda", dtype=fwd_dtype).eval()

    # Tallies
    std_cell_correct = 0
    std_cell_total = 0
    std_puzzle_correct = 0
    strict_cell_correct = 0
    strict_cell_total = 0
    strict_puzzle_correct = 0
    total_puzzles = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{os.path.basename(checkpoint_path)} on {os.path.basename(data_dir)}"):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            B = batch["inputs"].shape[0]

            carry = model.initial_carry(batch)
            for _ in range(config.model.halt_max_steps):
                carry, outputs = model(carry=carry, batch=batch)

            preds = outputs["logits"].argmax(-1)
            labels = carry.current_data["labels"]
            inputs = batch["inputs"]

            # STRICT TARGET: at non-path positions (label==-100) the correct
            # answer is the input token itself (wall stays wall, open stays
            # open, S/G unchanged). At path positions the label is `o` (=5).
            strict_target = torch.where(labels == -100, inputs, labels)

            # Standard metric (path cells only, existing definition)
            std_mask = labels != -100
            std_cell_correct += ((preds == labels) & std_mask).sum().item()
            std_cell_total += std_mask.sum().item()
            std_puzzle_correct += (((preds == labels) | ~std_mask).all(dim=-1)).sum().item()

            # Strict metric (every one of the 900 cells)
            strict_cell_correct += (preds == strict_target).sum().item()
            strict_cell_total += strict_target.numel()
            strict_puzzle_correct += ((preds == strict_target).all(dim=-1)).sum().item()

            total_puzzles += B

    return {
        "std_cell": std_cell_correct / max(1, std_cell_total),
        "std_puzzle": std_puzzle_correct / max(1, total_puzzles),
        "strict_cell": strict_cell_correct / max(1, strict_cell_total),
        "strict_puzzle": strict_puzzle_correct / max(1, total_puzzles),
    }


def main() -> int:
    from src.utils.config import load_config

    base_config = load_config("configs/trm_official_maze.yaml")

    runs: list[tuple[str, str, str]] = [
        ("HF-init", HF_CKPT, IN_DIST),
        ("trained", TRAINED_CKPT, IN_DIST),
    ]
    if os.path.exists(OOD):
        runs.append(("HF-init", HF_CKPT, OOD))
        runs.append(("trained", TRAINED_CKPT, OOD))

    # Skip runs whose checkpoint is missing
    runs = [r for r in runs if os.path.exists(r[1])]

    print(
        f"{'model':<10} {'test-set':<28} "
        f"{'std puzzle':>11} {'std cell':>10} "
        f"{'strict puz':>11} {'strict cell':>12}"
    )
    print("-" * 86)

    for name, ckpt, data_dir in runs:
        t0 = time.time()
        r = eval_both_metrics(ckpt, data_dir, base_config)
        t = time.time() - t0
        tag = "in-dist" if data_dir == IN_DIST else "OOD"
        print(
            f"{name:<10} {tag + ' (' + os.path.basename(data_dir) + ')':<28} "
            f"{r['std_puzzle']:>11.4f} {r['std_cell']:>10.4f} "
            f"{r['strict_puzzle']:>11.4f} {r['strict_cell']:>12.4f} "
            f"[{t:.0f}s]"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
