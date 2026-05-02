"""One-shot re-evaluation of a distilled LLM (student) checkpoint.

Mirrors ``src.evaluation.evaluate.evaluate_standard`` for distilled students:
the student is encoder-only / bidirectional (TransformerEncoder), so
``logits[i]`` predicts ``labels[i]`` directly — no shift, unlike the
causal-LM teacher path used by ``eval_llm_checkpoint.py``.

This script accepts the same maze eval-mask fix and CodeCarbon flags as
``eval_llm_checkpoint.py`` so the two re-eval paths produce comparable
artifacts under ``results/eval_fixed/``.

Usage (legacy positional, still supported):
    python scripts/eval_distill_checkpoint.py <config_path> <checkpoint_path> [max_batches]

Usage (with the maze eval-mask fix and CodeCarbon emissions tracking):
    python scripts/eval_distill_checkpoint.py <config_path> <checkpoint_path> [max_batches] \
        [--mask-non-path BOOL] \
        [--emissions-out PATH] \
        [--results-out PATH]
"""
import functools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print = functools.partial(print, flush=True)  # noqa

import torch

from src.models.distilled_llm import DistilledLLM
from src.utils.config import load_config


def _resolve_mask_non_path(cfg, override) -> bool:
    if override is not None:
        return override
    data_cfg = getattr(cfg, "data", None)
    if data_cfg is not None and hasattr(data_cfg, "mask_non_path"):
        return bool(data_cfg.mask_non_path)
    return False


def main(
    config_path: str,
    ckpt_path: str,
    max_batches: int = 200,
    mask_non_path_override: bool | None = None,
    emissions_out: str | None = None,
    results_out: str | None = None,
    data_dir_override: str | None = None,
) -> None:
    cfg = load_config(config_path)
    if data_dir_override:
        cfg.data.data_dir = data_dir_override
    mask_non_path = _resolve_mask_non_path(cfg, mask_non_path_override)

    print(f"[Eval-Distill] config         = {config_path}")
    print(f"[Eval-Distill] checkpoint     = {ckpt_path}")
    print(f"[Eval-Distill] dataset        = {cfg.data.dataset}")
    print(f"[Eval-Distill] max_batches    = {max_batches} (subsample for speed)")
    print(f"[Eval-Distill] mask_non_path  = {mask_non_path}")
    if emissions_out:
        print(f"[Eval-Distill] emissions_out  = {emissions_out}")
    if results_out:
        print(f"[Eval-Distill] results_out    = {results_out}")

    print("[Eval-Distill] building model...")
    model = DistilledLLM(
        vocab_size=cfg.model.vocab_size,
        seq_len=cfg.model.seq_len,
        d_model=cfg.model.distill_d_model,
        n_layers=cfg.model.distill_n_layers,
        ff_hidden=cfg.model.distill_ff_hidden,
        n_heads=cfg.model.distill_n_heads,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[Eval-Distill] device = {device}")

    print("[Eval-Distill] loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("[Eval-Distill] building test loader...")
    if cfg.data.dataset == "maze":
        from src.data.maze_dataset import get_maze_loaders
        _, test_loader = get_maze_loaders(
            cfg.data.data_dir,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
            mask_non_path=mask_non_path,
        )
    else:
        from src.data.sudoku_dataset import get_sudoku_loaders
        _, test_loader = get_sudoku_loaders(
            cfg.data.data_dir,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
        )
    n_total = len(test_loader)
    n_eval = min(n_total, max_batches)
    print(f"[Eval-Distill] test_loader: {n_total} batches of {cfg.training.batch_size} available, evaluating {n_eval}")

    tracker = None
    if emissions_out:
        from codecarbon import EmissionsTracker
        emissions_dir = os.path.dirname(os.path.abspath(emissions_out)) or "."
        os.makedirs(emissions_dir, exist_ok=True)
        emissions_basename = os.path.basename(emissions_out)
        project_name = os.path.splitext(emissions_basename)[0] or "eval"
        tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=emissions_dir,
            output_file=emissions_basename,
            log_level="error",
            allow_multiple_runs=True,
        )
        tracker.start()

    total_puzzles_correct = 0
    total_cells_correct = 0
    total_cells_graded = 0
    total_puzzles = 0

    try:
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                if i >= max_batches:
                    break
                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = model(inputs)
                if hasattr(logits, "logits"):
                    logits = logits.logits

                # Encoder-only student: no shift. Mirrors evaluate_standard
                # in src/evaluation/evaluate.py.
                preds = logits.argmax(-1)
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
                    print(f"[Eval-Distill] batch {i + 1}/{n_eval}: "
                          f"puzzle={running_puzzle:.4f} cell={running_cell:.4f} "
                          f"({total_puzzles_correct}/{total_puzzles} puzzles)")
    finally:
        if tracker is not None:
            tracker.stop()

    puzzle_acc = total_puzzles_correct / max(1, total_puzzles)
    cell_acc = total_cells_correct / max(1, total_cells_graded)

    print()
    print(f"[Results] puzzles: {total_puzzles_correct}/{total_puzzles} correct")
    print(f"[Results] cells:   {total_cells_correct}/{total_cells_graded} correct")
    print(f"[Results] puzzle_acc = {puzzle_acc:.6f}")
    print(f"[Results] cell_acc   = {cell_acc:.6f}")

    if results_out:
        os.makedirs(os.path.dirname(os.path.abspath(results_out)) or ".", exist_ok=True)
        payload = {
            "config": config_path,
            "checkpoint": ckpt_path,
            "max_batches": max_batches,
            "mask_non_path": mask_non_path,
            "puzzle_acc": puzzle_acc,
            "cell_acc": cell_acc,
            "puzzles_correct": total_puzzles_correct,
            "puzzles_total": total_puzzles,
            "cells_correct": total_cells_correct,
            "cells_graded": total_cells_graded,
            "emissions_csv": emissions_out,
        }
        with open(results_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Results] wrote {results_out}")


def _parse_bool(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_argv(argv: list[str]) -> dict:
    positional: list[str] = []
    mask_override: bool | None = None
    emissions_out: str | None = None
    results_out: str | None = None
    data_dir: str | None = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--mask-non-path":
            mask_override = _parse_bool(argv[i + 1])
            i += 2
        elif tok == "--emissions-out":
            emissions_out = argv[i + 1]
            i += 2
        elif tok == "--results-out":
            results_out = argv[i + 1]
            i += 2
        elif tok == "--data-dir":
            data_dir = argv[i + 1]
            i += 2
        elif tok.startswith("--mask-non-path="):
            mask_override = _parse_bool(tok.split("=", 1)[1])
            i += 1
        elif tok.startswith("--emissions-out="):
            emissions_out = tok.split("=", 1)[1]
            i += 1
        elif tok.startswith("--results-out="):
            results_out = tok.split("=", 1)[1]
            i += 1
        elif tok.startswith("--data-dir="):
            data_dir = tok.split("=", 1)[1]
            i += 1
        else:
            positional.append(tok)
            i += 1
    return {
        "positional": positional,
        "mask_override": mask_override,
        "emissions_out": emissions_out,
        "results_out": results_out,
        "data_dir": data_dir,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    parsed = _parse_argv(sys.argv[1:])
    pos = parsed["positional"]
    if len(pos) < 2:
        print(__doc__)
        sys.exit(1)
    config_path = pos[0]
    ckpt_path = pos[1]
    max_batches = int(pos[2]) if len(pos) > 2 else 200
    main(
        config_path,
        ckpt_path,
        max_batches=max_batches,
        mask_non_path_override=parsed["mask_override"],
        emissions_out=parsed["emissions_out"],
        results_out=parsed["results_out"],
        data_dir_override=parsed["data_dir"],
    )
