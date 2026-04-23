"""K-vote inference sweep over the 6 iso-time novelty checkpoints.

Rebuilds model + test loader from each RunSpec's YAML, loads latest.pt,
calls run_k_vote_{trm,llm} for K in {1,2,4,8,16}, aggregates per-(run, K)
metrics to results/novelty/k_vote_results.csv, and emits accuracy-vs-K +
kWh/accuracy Pareto plots.

In-process (no subprocesses): k_vote owns its own CarbonTracker /
torch.no_grad scope, and one process keeps the GPU warm across K values
within a run so latency isn't polluted by cold-start kernel compile time.

Usage:
    python scripts/run_novelty_k_vote.py [--seed 0] [--work-dir C:/ml-trm-work] \
        [--k-values 1,2,4,8,16] [--skip-labels qwen-maze,distill-maze] \
        [--temperature 0.7] [--batch-size 16]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Same import bootstrap as run_novelty_iso_time.py + `scripts/` so we can
# reuse its RUNS matrix without duplicating the spec list.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_novelty_iso_time import (  # noqa: E402
    NOVELTY_RIG_PLAN, RUNS, RunSpec, _artifact_dir,
)

from src.cli.workdir import _resolve_work_dir  # noqa: E402

# src.utils.config imports yaml + pydantic; not available under a bare
# system Python. `--help` should work without the project venv, so we
# wrap the heavy import in a try and fall through with None sentinels.
# The module-level names below are only dereferenced inside function
# bodies, so functions that need them fail at call-time (when the venv
# is guaranteed) rather than at import-time.
try:
    from src.utils.config import ExperimentConfig, ModelType, load_config  # noqa: E402
except ImportError:
    ExperimentConfig = None  # type: ignore[assignment]
    ModelType = None  # type: ignore[assignment]
    load_config = None  # type: ignore[assignment]


@dataclass
class KVoteRow:
    spec: RunSpec
    k: int
    puzzle_acc: float
    cell_acc: float
    mean_latency_ms: float
    kwh_per_puzzle: float
    n_puzzles: int
    checkpoint_path: str


def _find_checkpoint(artifact_dir: Path) -> Optional[Path]:
    """Prefer canonical `latest.pt`; fall back to any `*_latest.pt`.

    Covers official (latest.pt), llm (<tag>_latest.pt) and distill
    (distill_<dataset>_latest.pt) without re-deriving the tag here.
    """
    canonical = artifact_dir / "latest.pt"
    if canonical.is_file():
        return canonical
    hits = sorted(artifact_dir.glob("*latest.pt"))
    return hits[0] if hits else None


# --- Model + loader builders (mirror main.py:_run_eval per ModelType) ------

def _build_test_loader(config: ExperimentConfig, batch_size_override: int):
    """batch_size_override threads CLI batch so K=16 × train batch can't OOM."""
    from torch.utils.data import DataLoader
    mt = config.model.model_type
    bs = batch_size_override or config.training.batch_size

    if mt in (ModelType.TRM_OFFICIAL_SUDOKU, ModelType.TRM_OFFICIAL_MAZE):
        from src.data.collate import official_collate_fn
        collate_fn = official_collate_fn(config.training.task_id)
        if config.data.dataset == "maze":
            from src.data.maze_dataset import MazeDataset
            test_ds = MazeDataset(config.data.data_dir, "test")
        else:
            from src.data.sudoku_dataset import SudokuDataset
            test_ds = SudokuDataset(config.data.data_dir, "test")
        return DataLoader(
            test_ds, batch_size=bs, shuffle=False,
            num_workers=config.data.num_workers, pin_memory=True,
            collate_fn=collate_fn,
        )

    # LLM_FINETUNE + distill use the tuple-style loaders.
    if config.data.dataset == "maze":
        from src.data.maze_dataset import get_maze_loaders
        _, test_loader = get_maze_loaders(
            config.data.data_dir, batch_size=bs, num_workers=config.data.num_workers,
        )
    else:
        from src.data.sudoku_dataset import get_sudoku_loaders
        _, test_loader = get_sudoku_loaders(
            config.data.data_dir, batch_size=bs, num_workers=config.data.num_workers,
        )
    return test_loader


def _build_trm(config: ExperimentConfig, ckpt: dict, device: str):
    """Mirrors main.py's TRM_OFFICIAL branch + trainer_official._load_checkpoint."""
    import torch
    from src.models.losses_official import ACTLossHead
    from src.models.trm_official import TRMOfficial

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
    }
    model = TRMOfficial(model_config)
    loss_head = ACTLossHead(model, q_loss_weight=config.training.q_loss_weight)
    # Match training cast — ACTLossHead holds the model so .to() propagates.
    forward_dtype = getattr(torch, config.model.forward_dtype, torch.bfloat16)
    model.to(device=device, dtype=forward_dtype)
    loss_head.to(device=device, dtype=forward_dtype)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, loss_head


def _build_baseline_llm(config: ExperimentConfig, ckpt: dict, device: str):
    # Prefer saved llm_name — protects against YAML drift between train
    # and this sweep loading a mismatched backbone.
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
    # strict=False: bnb Linear4bit consumes weight.absmax / weight.quant_map /
    # weight.quant_state.bitsandbytes__nf4 to rebuild QuantState but leaves them
    # in the unexpected-keys set, so strict=True false-positives on every QLoRA
    # checkpoint. Load is correct either way.
    model.to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model


def _build_distilled(config: ExperimentConfig, ckpt: dict, device: str):
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


# --- Per-run driver --------------------------------------------------------

def _run_one(
    spec: RunSpec, work_dir: Path, seed: int,
    k_values: list[int], temperature: float, batch_size: int,
    output_dir: Path,
) -> list[KVoteRow]:
    import torch

    artifact_dir = _artifact_dir(work_dir, spec.label, seed)
    ckpt_path = _find_checkpoint(artifact_dir)
    if ckpt_path is None:
        # Iso-time slot may have skipped/crashed — keep going for the others.
        print(f"  [{spec.label}] SKIP — no checkpoint under {artifact_dir}")
        return []

    print(f"\n{'=' * 72}\n[{spec.label}] ckpt={ckpt_path}\n{'=' * 72}")
    # Scope env so load_config resolves checkpoint/experiment_dir per-run.
    os.environ["TRM_CHECKPOINT_DIR"] = str(artifact_dir)
    os.environ["TRM_EXPERIMENT_DIR"] = str(artifact_dir)
    config = load_config(str(REPO_ROOT / spec.config))
    device = config.device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    # Per-label subdir so concurrent runs don't collide on k_vote_*_kN stems.
    kvote_out = output_dir / "k_vote_runs" / spec.label
    kvote_out.mkdir(parents=True, exist_ok=True)
    from src.evaluation.k_vote import run_k_vote_llm, run_k_vote_trm
    test_loader = _build_test_loader(config, batch_size)
    mt = config.model.model_type

    if mt in (ModelType.TRM_OFFICIAL_SUDOKU, ModelType.TRM_OFFICIAL_MAZE):
        model, loss_head = _build_trm(config, ckpt, device)
        results = run_k_vote_trm(
            model, loss_head, test_loader, k_values, device,
            task_id=config.training.task_id, output_dir=str(kvote_out),
        )
    elif mt == ModelType.LLM_FINETUNE:
        # Distill runs reuse the Qwen YAML (mode=distill at train time) so
        # model_type is still LLM_FINETUNE here. Disambiguate by checkpoint
        # filename (trainer_distill prefixes with distill_).
        if "distill_" in ckpt_path.name:
            model = _build_distilled(config, ckpt, device)
        else:
            model = _build_baseline_llm(config, ckpt, device)
        results = run_k_vote_llm(
            model, test_loader, k_values, device,
            temperature=temperature, output_dir=str(kvote_out),
        )
    else:
        print(f"  [{spec.label}] SKIP — unsupported model_type {mt}")
        return []

    rows = [
        KVoteRow(
            spec=spec, k=r["k"], puzzle_acc=r["puzzle_acc"], cell_acc=r["cell_acc"],
            mean_latency_ms=r["mean_latency_ms"], kwh_per_puzzle=r["kwh_per_puzzle"],
            n_puzzles=r["n_puzzles"], checkpoint_path=str(ckpt_path),
        ) for r in results
    ]

    # Free between runs so peak GPU mem = max(runs), not sum(runs).
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


# --- CSV + plots -----------------------------------------------------------

CSV_COLUMNS = [
    "label", "task", "model", "k",
    "puzzle_acc", "cell_acc", "mean_latency_ms",
    "kwh_per_puzzle", "n_puzzles", "checkpoint_path",
]


def _write_csv(rows: list[KVoteRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in rows:
            w.writerow([
                r.spec.label, r.spec.task, r.spec.model, r.k,
                f"{r.puzzle_acc:.4f}", f"{r.cell_acc:.4f}",
                f"{r.mean_latency_ms:.3f}", f"{r.kwh_per_puzzle:.8f}",
                r.n_puzzles, r.checkpoint_path,
            ])
    print(f"\nCSV -> {out_path}")


def _emit_plots(rows: list[KVoteRow], out_dir: Path) -> None:
    if not rows:
        print("  [plots] no rows to plot")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by (label, task, model); sort by k so line plots don't zigzag.
    groups: dict[tuple[str, str, str], list[KVoteRow]] = {}
    for r in rows:
        groups.setdefault((r.spec.label, r.spec.task, r.spec.model), []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: r.k)

    models = sorted({r.spec.model for r in rows})
    tasks = sorted({r.spec.task for r in rows})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_for = {m: colors[i % len(colors)] for i, m in enumerate(models)}
    # Marker per task so Pareto encodes model (color), task (shape), and
    # same-run trajectory (connecting line) at once.
    marker_for = {t: m for t, m in zip(tasks, ("o", "s", "^", "D", "v"))}

    # Plot 1 — accuracy vs K (log x).
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for (label, task, model), grp in groups.items():
        ax.plot([r.k for r in grp], [r.puzzle_acc for r in grp],
                marker=marker_for.get(task, "o"), color=color_for[model],
                linewidth=1.6, markersize=7, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K (samples per puzzle)")
    ax.set_ylabel("puzzle accuracy")
    ax.set_title("K-vote: puzzle accuracy vs K")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    acc_path = out_dir / "k_vote_accuracy_curve.png"
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    print(f"  plot -> {acc_path}")

    # Plot 2 — Pareto: kwh/puzzle (log) vs puzzle_acc.
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for (label, task, model), grp in groups.items():
        xs = [r.kwh_per_puzzle for r in grp]
        ys = [r.puzzle_acc for r in grp]
        ax.plot(xs, ys, linestyle="--", color=color_for[model], linewidth=1.0, alpha=0.5)
        ax.scatter(xs, ys, c=color_for[model], marker=marker_for.get(task, "o"),
                   s=80, edgecolors="black", linewidths=0.5, label=label)
        for r in grp:
            ax.annotate(f"K={r.k}", (r.kwh_per_puzzle, r.puzzle_acc),
                        textcoords="offset points", xytext=(4, 4), fontsize=6, alpha=0.7)
    # CodeCarbon returns 0 on unsupported HW — log can't handle 0.
    if min(r.kwh_per_puzzle for r in rows) > 0:
        ax.set_xscale("log")
    ax.set_xlabel("kWh per puzzle")
    ax.set_ylabel("puzzle accuracy")
    ax.set_title("K-vote Pareto: accuracy vs energy cost")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    par_path = out_dir / "k_vote_pareto.png"
    fig.savefig(par_path, dpi=150)
    plt.close(fig)
    print(f"  plot -> {par_path}")


# --- CLI -------------------------------------------------------------------

def _parse_csv_strs(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()] if raw else []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work-dir", type=str, default="")
    p.add_argument("--k-values", type=str, default="1,2,4,8,16")
    p.add_argument("--skip-labels", type=str, default="",
                   help="comma-separated RUN labels to skip (e.g. qwen-maze); "
                        "mutually exclusive with --rig")
    p.add_argument("--rig", type=int, default=0, choices=(0, 1, 2, 3),
                   help="K-vote only this rig's slice. Mirrors NOVELTY_RIG_PLAN "
                        "from run_novelty_iso_time.py so each rig K-votes the "
                        "same checkpoints it trained.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="LLM sampling temperature for diversity")
    p.add_argument("--batch-size", type=int, default=16,
                   help="test-loader batch size override (0 = use YAML)")
    args = p.parse_args()

    if args.rig and args.skip_labels:
        p.error("--rig and --skip-labels are mutually exclusive")

    if args.work_dir:
        os.environ["TRM_WORK_DIR"] = args.work_dir
    work_dir = Path(_resolve_work_dir()).resolve()

    k_values = [int(t) for t in _parse_csv_strs(args.k_values)]
    if args.rig:
        rig_indices = set(NOVELTY_RIG_PLAN[args.rig])
        skip = {s.label for s in RUNS if s.idx not in rig_indices}
    else:
        skip = set(_parse_csv_strs(args.skip_labels))

    print(f"[k-vote] work_dir : {work_dir}")
    print(f"[k-vote] seed     : {args.seed}")
    print(f"[k-vote] k_values : {k_values}")
    if args.rig:
        kept = sorted({s.label for s in RUNS if s.idx in NOVELTY_RIG_PLAN[args.rig]})
        print(f"[k-vote] rig      : {args.rig}  (labels {kept})")
    if skip:
        print(f"[k-vote] skipping : {sorted(skip)}")

    out_dir = REPO_ROOT / "results" / "novelty"
    all_rows: list[KVoteRow] = []
    for spec in RUNS:
        if spec.label in skip:
            print(f"\n[{spec.label}] SKIP via --skip-labels")
            continue
        try:
            all_rows.extend(_run_one(
                spec, work_dir, args.seed, k_values,
                args.temperature, args.batch_size, out_dir,
            ))
        except Exception as e:
            # One bad run shouldn't nuke the sweep — note and continue.
            print(f"  [{spec.label}] ERROR: {type(e).__name__}: {e}")

    # Per-rig filename so concurrent rigs don't clobber each other's CSV on
    # the shared OneDrive folder; aggregator merges them back together.
    csv_name = f"k_vote_results-rig{args.rig}.csv" if args.rig else "k_vote_results.csv"
    _write_csv(all_rows, out_dir / csv_name)
    _emit_plots(all_rows, out_dir)


if __name__ == "__main__":
    main()
