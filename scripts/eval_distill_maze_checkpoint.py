"""One-shot re-evaluation of a distilled-LLM Maze checkpoint.

Mirrors scripts/eval_llm_checkpoint.py but for the distilled student
(`src.models.distilled_llm.DistilledLLM`) instead of `BaselineLLM`. The
student is encoder-only — output[:, i, :] predicts label[:, i] directly,
so we do NOT apply the causal-LM shift that the BaselineLLM eval uses.

Forces `mask_non_path=False` on the maze loader so all 900 cells per
puzzle are graded (Track A fix).

Usage:
    python scripts/eval_distill_maze_checkpoint.py \\
        <distill_config.yaml> <distill_checkpoint.pt> [max_batches]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functools
print = functools.partial(print, flush=True)  # noqa

import torch

from src.data.maze_dataset import get_maze_loaders
from src.models.distilled_llm import DistilledLLM
from src.utils.config import load_config


def main(config_path: str, ckpt_path: str, max_batches: int = 200) -> None:
    cfg = load_config(config_path)
    print(f"[Eval] config     = {config_path}")
    print(f"[Eval] checkpoint = {ckpt_path}")
    print(f"[Eval] dataset    = {cfg.data.dataset}")
    print(f"[Eval] max_batches= {max_batches}")

    if cfg.data.dataset != "maze":
        raise ValueError(f"This script is maze-only; config says dataset={cfg.data.dataset!r}")

    print("[Eval] building distilled student...")
    model = DistilledLLM(
        vocab_size=cfg.model.vocab_size,
        seq_len=cfg.model.seq_len,
        d_model=cfg.model.distill_d_model,
        n_layers=cfg.model.distill_n_layers,
        n_heads=cfg.model.distill_n_heads,
        ff_hidden=cfg.model.distill_ff_hidden,
    )
    print(f"[Eval] params = {model.param_count():,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[Eval] device = {device}")

    print("[Eval] loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print("[Eval] building test loader (mask_non_path=False — grade all 900 cells)...")
    # Force False unconditionally — re-eval scripts should always grade all
    # 900 cells regardless of what the training config defaulted. The
    # Pydantic ExperimentConfig has `mask_non_path: bool = True` as the
    # default, so getattr() returns True for any config that doesn't
    # explicitly override; we don't trust that here.
    mask_non_path = False
    print(f"[Eval] mask_non_path = {mask_non_path} (forced; ignoring config)")
    _, test_loader = get_maze_loaders(
        cfg.data.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        mask_non_path=mask_non_path,
    )
    n_total = len(test_loader)
    n_eval = min(n_total, max_batches)
    print(f"[Eval] test_loader: {n_total} batches of {cfg.training.batch_size}, evaluating {n_eval}")

    total_puzzles_correct = 0
    total_cells_correct = 0
    total_cells_graded = 0
    total_puzzles = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if i >= max_batches:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Distill student is encoder-only: position i predicts label i directly,
            # no causal shift (unlike eval_llm_checkpoint.py which shifts because
            # BaselineLLM is a causal LM).
            preds = model(inputs).argmax(-1)  # [B, L]
            mask = labels != 0

            puzzle_correct = ((preds == labels) | ~mask).all(dim=-1)
            cells_correct = ((preds == labels) & mask).sum().item()

            total_puzzles_correct += puzzle_correct.sum().item()
            total_cells_correct += cells_correct
            total_cells_graded += mask.sum().item()
            total_puzzles += inputs.shape[0]

            if (i + 1) % 10 == 0 or (i + 1) == n_eval:
                running_cell = total_cells_correct / max(1, total_cells_graded)
                running_puzzle = total_puzzles_correct / max(1, total_puzzles)
                print(f"[Eval] batch {i + 1}/{n_eval}: "
                      f"puzzle={running_puzzle:.4f} cell={running_cell:.4f} "
                      f"({total_puzzles_correct}/{total_puzzles} puzzles)")

    puzzle_acc = total_puzzles_correct / max(1, total_puzzles)
    cell_acc = total_cells_correct / max(1, total_cells_graded)

    print()
    print(f"[Results] puzzles: {total_puzzles_correct}/{total_puzzles} correct")
    print(f"[Results] cells:   {total_cells_correct}/{total_cells_graded} correct")
    print(f"[Results] puzzle_acc = {puzzle_acc:.6f}")
    print(f"[Results] cell_acc   = {cell_acc:.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200)
