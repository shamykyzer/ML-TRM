"""One-shot re-evaluation of an existing LLM checkpoint using the fixed eval code.

Usage: python scripts/eval_llm_checkpoint.py <config_path> <checkpoint_path> [max_batches]
  max_batches: optional, default 200. Each batch is `batch_size` puzzles, so
  200 * 16 = 3200 puzzles is enough to tell if cell_acc is 0% vs 20%+.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force unbuffered output so prints land in the log immediately when stdout is redirected.
import functools
print = functools.partial(print, flush=True)  # noqa

import torch

from src.data.sudoku_dataset import get_sudoku_loaders
from src.models.baseline_llm import BaselineLLM
from src.utils.config import load_config


def main(config_path: str, ckpt_path: str, max_batches: int = 200) -> None:
    cfg = load_config(config_path)
    print(f"[Eval] config     = {config_path}")
    print(f"[Eval] checkpoint = {ckpt_path}")
    print(f"[Eval] llm_name   = {cfg.model.llm_name}")
    print(f"[Eval] dataset    = {cfg.data.dataset}")
    print(f"[Eval] max_batches= {max_batches} (subsample for speed)")

    print("[Eval] building model...")
    model = BaselineLLM(
        model_name=cfg.model.llm_name,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        use_qlora=cfg.model.use_qlora,
    )
    print("[Eval] loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"[Eval] load: missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"[Eval] first missing: {missing[:3]}")
    if len(unexpected) > 0:
        print(f"[Eval] first unexpected: {unexpected[:3]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[Eval] device = {device}")

    print("[Eval] building test loader...")
    if cfg.data.dataset == "maze":
        from src.data.maze_dataset import get_maze_loaders
        _, test_loader = get_maze_loaders(
            cfg.data.data_dir,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
        )
    else:
        _, test_loader = get_sudoku_loaders(
            cfg.data.data_dir,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
        )
    n_total = len(test_loader)
    n_eval = min(n_total, max_batches)
    print(f"[Eval] test_loader: {n_total} batches of {cfg.training.batch_size} available, evaluating {n_eval}")

    # Mirrors LLMTrainer.evaluate() after the shift fix.
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

            outputs = model(input_ids=inputs)
            preds = outputs.logits[:, :-1, :].argmax(-1)
            labels_shifted = labels[:, 1:]

            mask = labels_shifted != 0
            puzzle_correct = ((preds == labels_shifted) | ~mask).all(dim=-1)
            cells_correct = ((preds == labels_shifted) & mask).sum().item()

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
    config_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    max_batches = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    main(config_path, ckpt_path, max_batches)
