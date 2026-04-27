"""K-vote a single LLM (or distilled-LLM) checkpoint.

`run_novelty_k_vote.py` only iterates the 6-cell novelty matrix
(trm-mlp-sudoku, trm-att-maze, qwen-{sudoku,maze}, distill-{sudoku,maze}).
The Apr-26 sprint added gpt2-sudoku + distill-gpt2-sudoku, which aren't
in that matrix, so this thin driver loads ONE checkpoint and K-votes it
without touching the shared RUNS list.

Usage:
    python scripts/run_kvote_llm_single.py \
        --config configs/llm_config.yaml \
        --checkpoint models/llm/gpt2_sudoku_latest.pt \
        --label gpt2-sudoku \
        --k-values 1,2,4 \
        [--temperature 0.7] [--batch-size 16] \
        [--output-csv results/novelty/k_vote_results-gpt2-sudoku.csv]

Writes per-K rows compatible with the schema in run_novelty_k_vote.py
(label, task, model, k, puzzle_acc, cell_acc, mean_latency_ms,
kwh_per_puzzle, n_puzzles, checkpoint_path) and a CodeCarbon emissions
file under results/novelty/k_vote_runs/<label>/.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _parse_csv_ints(s: str) -> list[int]:
    return [int(t.strip()) for t in s.split(",") if t.strip()]


def _build_test_loader(config, batch_size: int):
    """Mirror run_novelty_k_vote.py:_build_test_loader for the LLM path."""
    if config.data.dataset == "maze":
        from src.data.maze_dataset import get_maze_loaders
        _, test_loader = get_maze_loaders(
            config.data.data_dir, batch_size=batch_size,
            num_workers=config.data.num_workers,
        )
    else:
        from src.data.sudoku_dataset import get_sudoku_loaders
        _, test_loader = get_sudoku_loaders(
            config.data.data_dir, batch_size=batch_size,
            num_workers=config.data.num_workers,
        )
    return test_loader


def _build_model(config, ckpt: dict, device: str, kind: str):
    """kind: 'baseline' (BaselineLLM) or 'distill' (DistilledLLM)."""
    import torch  # noqa: F401  (loaded for side effects in submodules)

    if kind == "baseline":
        from src.models.baseline_llm import BaselineLLM
        saved = (ckpt.get("config") or {}).get("model", {}) or {}
        model = BaselineLLM(
            model_name=saved.get("llm_name", config.model.llm_name),
            lora_r=saved.get("lora_r", config.model.lora_r),
            lora_alpha=saved.get("lora_alpha", config.model.lora_alpha),
            use_qlora=saved.get("use_qlora", config.model.use_qlora),
            use_gradient_checkpointing=saved.get(
                "use_gradient_checkpointing", config.model.use_gradient_checkpointing,
            ),
        )
        model.to(device)
        # strict=False per run_novelty_k_vote.py rationale (QLoRA absmax keys
        # appear in unexpected_keys but the load is correct).
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return model

    if kind == "distill":
        from src.models.distilled_llm import DistilledLLM
        model = DistilledLLM(
            vocab_size=config.model.vocab_size,
            seq_len=config.model.seq_len,
            d_model=config.model.distill_d_model,
            n_layers=config.model.distill_n_layers,
            ff_hidden=config.model.distill_ff_hidden,
            n_heads=config.model.distill_n_heads,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model.to(device)

    raise ValueError(f"--kind must be 'baseline' or 'distill', got {kind!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True,
                   help="path to YAML config (e.g. configs/llm_config.yaml)")
    p.add_argument("--checkpoint", required=True,
                   help="path to *_latest.pt for the LLM/distill model")
    p.add_argument("--label", required=True,
                   help="run label written into the CSV (e.g. gpt2-sudoku)")
    p.add_argument("--task", default="",
                   help="task name for CSV (defaults to config.data.dataset)")
    p.add_argument("--model-tag", default="",
                   help="model name for CSV (defaults to llm_name or 'distill')")
    p.add_argument("--kind", choices=("baseline", "distill"), default="",
                   help="auto-detected from checkpoint filename if omitted")
    p.add_argument("--k-values", default="1,2,4",
                   help="comma-separated K values, e.g. 1,2,4")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output-csv", default="",
                   help="output CSV path (defaults to results/novelty/"
                        "k_vote_results-<label>.csv)")
    args = p.parse_args()

    import torch
    from src.utils.config import load_config
    from src.evaluation.k_vote import run_k_vote_llm

    config = load_config(args.config)

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    # Auto-detect kind from filename when caller omitted --kind. Distillation
    # checkpoints are written by trainer_distill with a 'distill_' prefix.
    kind = args.kind or ("distill" if "distill" in ckpt_path.name else "baseline")

    device = config.device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = _build_model(config, ckpt, device, kind)
    test_loader = _build_test_loader(config, args.batch_size)
    k_values = _parse_csv_ints(args.k_values)

    out_root = REPO_ROOT / "results" / "novelty"
    kvote_dir = out_root / "k_vote_runs" / args.label
    kvote_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.output_csv).resolve() if args.output_csv
        else out_root / f"k_vote_results-{args.label}.csv"
    )

    print(f"[k-vote] label    : {args.label}")
    print(f"[k-vote] kind     : {kind}")
    print(f"[k-vote] ckpt     : {ckpt_path}")
    print(f"[k-vote] device   : {device}")
    print(f"[k-vote] k_values : {k_values}")
    print(f"[k-vote] csv      : {csv_path}")

    results = run_k_vote_llm(
        model, test_loader, k_values, device,
        temperature=args.temperature, output_dir=str(kvote_dir),
    )

    # Schema kept identical to run_novelty_k_vote.py CSV_COLUMNS.
    columns = [
        "label", "task", "model", "k",
        "puzzle_acc", "cell_acc", "mean_latency_ms",
        "kwh_per_puzzle", "n_puzzles", "checkpoint_path",
    ]
    task = args.task or config.data.dataset
    saved_cfg = (ckpt.get("config") or {}).get("model", {}) or {}
    model_tag = (
        args.model_tag
        or saved_cfg.get("llm_name", config.model.llm_name)
        or ("distill" if kind == "distill" else "baseline")
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in results:
            w.writerow([
                args.label, task, model_tag, r["k"],
                f"{r['puzzle_acc']:.4f}", f"{r['cell_acc']:.4f}",
                f"{r['mean_latency_ms']:.3f}", f"{r['kwh_per_puzzle']:.8f}",
                r["n_puzzles"], str(ckpt_path),
            ])
    print(f"\nCSV -> {csv_path}")
    for r in results:
        print(
            f"  K={r['k']:>2} puzzle={r['puzzle_acc']:.4f} cell={r['cell_acc']:.4f}"
            f" lat={r['mean_latency_ms']:.1f}ms kwh/puz={r['kwh_per_puzzle']:.2e}"
        )


if __name__ == "__main__":
    main()
