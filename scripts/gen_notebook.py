"""Generate ML_TRM_Walkthrough.ipynb for the supplementary submission."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "ML_TRM_Walkthrough.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
        "outputs": [],
        "execution_count": None,
    }


cells = [
    # ── Title ──
    md("""\
# Tiny Recursive Models vs Fine-Tuned LLMs for Structured Reasoning
## Supplementary Code Walkthrough

**Authors:** Ahmed AlShamy, Armin Ghaseminejad, Nickolas Greiner

This notebook demonstrates the full ML pipeline used in our project:
1. **Data** -- Building and loading Sudoku-Extreme and Maze-Hard datasets
2. **Models** -- TRM-MLP, TRM-Att, LLM baselines (LoRA), Knowledge Distillation
3. **Training** -- Launching training runs with YAML configs
4. **Evaluation** -- Computing cell/puzzle accuracy, carbon emissions
5. **Figures** -- Generating the paper's training curve and energy scatter plots

All source code lives in `src/`. This notebook imports from it and shows how
each pipeline stage works. The actual training was run across 6 machines over
several weeks; this notebook demonstrates the code paths, not a full re-run."""),

    # ── Setup ──
    md("""\
## 1. Environment Setup

Install dependencies (run once). Requires Python 3.10+ and a CUDA GPU for training."""),

    code("""\
# Install dependencies (uncomment to run)
# !pip install -r requirements.txt

import os, sys
sys.path.insert(0, os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == 'scripts' else os.getcwd())

import torch
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")"""),

    # ── Data ──
    md("""\
## 2. Data Preparation

### 2.1 Building the Datasets

Datasets are downloaded from HuggingFace and preprocessed into `.npy` arrays.
- **Sudoku-Extreme**: 423,000 test puzzles, 1,000 train (423:1 ratio)
- **Maze-Hard**: 1,000 30x30 mazes with Hard path filter

The build scripts handle downloading, tokenisation, and train/test splitting."""),

    code("""\
# Build Sudoku-Extreme dataset (downloads from HuggingFace)
# Run from repo root:
#   python data/build_sudoku_dataset.py

# Build Maze-Hard dataset:
#   python data/build_maze_dataset.py

# After building, the data directory contains:
#   data/sudoku-extreme-full/{train,test}/all__inputs.npy, all__labels.npy, dataset.json
#   data/maze-30x30-hard-1k/{train,test}/all__inputs.npy, all__labels.npy, dataset.json
print("Dataset builders: data/build_sudoku_dataset.py, data/build_maze_dataset.py")"""),

    md("""\
### 2.2 Loading Datasets

The `SudokuDataset` and `MazeDataset` classes load preprocessed `.npy` files.

**Token schemas:**
- Sudoku: 0=pad, 1=blank, 2-10=digits 1-9 (vocab_size=11)
- Maze: 0=pad, 1=wall, 2=open, 3=start, 4=goal, 5=path (vocab_size=6)

Labels are masked so the loss only applies to cells the model must predict."""),

    code("""\
from src.data.sudoku_dataset import SudokuDataset, get_sudoku_loaders
from src.data.maze_dataset import MazeDataset, get_maze_loaders

# Example: load Sudoku dataset
DATA_DIR_SUDOKU = "data/sudoku-extreme-full"
if os.path.exists(os.path.join(DATA_DIR_SUDOKU, "train", "all__inputs.npy")):
    train_loader, test_loader = get_sudoku_loaders(DATA_DIR_SUDOKU, batch_size=32)
    inputs, labels = next(iter(train_loader))
    print(f"Sudoku batch: inputs {inputs.shape}, labels {labels.shape}")
    print(f"  Vocab range: {inputs.min().item()}-{inputs.max().item()}")
    print(f"  Train size: {len(train_loader.dataset)}, Test size: {len(test_loader.dataset)}")
else:
    print("Dataset not built yet. Run: python data/build_sudoku_dataset.py")

# Example: load Maze dataset
DATA_DIR_MAZE = "data/maze-30x30-hard-1k"
if os.path.exists(os.path.join(DATA_DIR_MAZE, "train", "all__inputs.npy")):
    train_loader_m, test_loader_m = get_maze_loaders(DATA_DIR_MAZE, batch_size=16, mask_non_path=False)
    inp_m, lab_m = next(iter(train_loader_m))
    print(f"Maze batch: inputs {inp_m.shape}, labels {lab_m.shape}")
    print(f"  Sequence length: {inp_m.shape[1]} (30x30 = 900 cells)")
else:
    print("Dataset not built yet. Run: python data/build_maze_dataset.py")"""),

    # ── Models ──
    md("""\
## 3. Model Architectures

### 3.1 TRM-MLP (Tiny Recursive Model with MLP-Mixer)

The TRM uses Adaptive Computation Time (ACT) to dynamically decide how many
reasoning iterations each puzzle needs. The MLP-Mixer variant uses fixed
token-mixing (81 tokens for Sudoku) instead of self-attention.

Key hyperparameters from `configs/trm_official_sudoku_mlp.yaml`:
- `d_model=512`, `ff_hidden=1536`, `L_cycles=6`, `H_cycles=3`
- `halt_max_steps=16`, `halt_exploration_prob=0.1`
- `mlp_t=True` (MLP token-mixer, not self-attention)
- Optimizer: AdamATan2 (paper-faithful)"""),

    code("""\
from src.utils.config import load_config

# Load a TRM-MLP config
config = load_config("configs/trm_official_sudoku_mlp.yaml")
print(f"Model type: {config.model.model_type}")
print(f"Architecture: d_model={config.model.d_model}, ff_hidden={config.model.ff_hidden}")
print(f"ACT: H_cycles={config.model.H_cycles}, L_cycles={config.model.L_cycles}")
print(f"MLP token-mixer: {config.model.mlp_t}")
print(f"Training: lr={config.training.lr}, epochs={config.training.epochs}, batch={config.training.batch_size}")"""),

    code("""\
from src.models.trm_official import TRMOfficial

# Instantiate TRM-MLP model
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
print(f"TRM-MLP parameters: {model.param_count():,}")"""),

    md("""\
### 3.2 TRM-Att (Self-Attention Variant)

The attention variant replaces the MLP token-mixer with multi-head self-attention.
Used for Maze-Hard where variable-length solution paths (up to 900 tokens)
benefit from learned pairwise attention, unlike Sudoku's fixed 81-token grid
where global MLP mixing suffices.

Config: `configs/trm_official_maze.yaml` (key differences: `mlp_t=False`,
`seq_len=900`, `vocab_size=6`, `L_cycles=4`)."""),

    code("""\
# Load TRM-Att (Maze) config
maze_config = load_config("configs/trm_official_maze.yaml")
print(f"TRM-Att: seq_len={maze_config.model.seq_len}, vocab={maze_config.model.vocab_size}")
print(f"MLP token-mixer: {maze_config.model.mlp_t}  (False = self-attention)")
print(f"L_cycles={maze_config.model.L_cycles} (vs 6 for MLP variant)")"""),

    md("""\
### 3.3 LLM Baselines (LoRA Fine-Tuning)

We fine-tune four HuggingFace LLMs with LoRA adapters on the same datasets:
- **GPT-2 Small** (124M params, 0.8M trainable)
- **SmolLM2-360M** (360M params, 1.2M trainable)
- **Llama-3.2-1B** (1.24B params, 3.4M trainable)
- **Qwen2.5-0.5B** (494M params, 1.6M trainable)

LoRA was chosen over full fine-tuning due to the 12 GB VRAM constraint on our
consumer GPUs. The `BaselineLLM` class wraps any HuggingFace causal LM with
PEFT LoRA and remaps the 11-token puzzle vocabulary onto real digit tokens."""),

    code("""\
from src.models.baseline_llm import BaselineLLM

# Instantiate a LoRA-wrapped LLM (example: Qwen2.5-0.5B)
# NOTE: This downloads the model from HuggingFace (~1 GB)
# Uncomment to run:
#
# llm = BaselineLLM(
#     model_name="Qwen/Qwen2.5-0.5B",
#     lora_r=8,
#     lora_alpha=16,
#     use_qlora=False,
# )
# print(f"Trainable: {llm.trainable_param_count():,} / {llm.total_param_count():,}")

# Config files for all 4 LLMs:
llm_configs = [
    "configs/llm_qwen.yaml",       # Qwen2.5-0.5B (Sudoku)
    "configs/llm_smollm.yaml",     # SmolLM2-360M (Sudoku)
    "configs/llm_llama.yaml",      # Llama-3.2-1B (Sudoku)
]
for cfg_path in llm_configs:
    if os.path.exists(cfg_path):
        c = load_config(cfg_path)
        print(f"{c.model.llm_name}: lr={c.training.lr}, epochs={c.training.epochs}, "
              f"LoRA r={c.model.lora_r}, alpha={c.model.lora_alpha}")"""),

    md("""\
### 3.4 Knowledge Distillation

We distil fine-tuned LLMs into compact student models (~2.4M params) that
match the TRM's parameter scale, enabling fair efficiency comparisons.

The student is a lightweight encoder-only transformer trained with a
weighted combination of hard labels (CE loss) and soft labels (KL divergence
from the teacher's output distribution).

`distill_alpha=0.7` weights the KL loss; `distill_temperature=4.0` softens
the teacher's logits."""),

    code("""\
from src.models.distilled_llm import DistilledLLM

# Instantiate a distilled student model
student = DistilledLLM(
    vocab_size=11,
    seq_len=81,
    d_model=256,
    n_layers=3,
    n_heads=4,
    ff_hidden=1024,
)
print(f"Distilled student parameters: {student.param_count():,}")

# Distillation config: configs/distill_qwen_sudoku.yaml
# Launch: python main.py --mode distill --config configs/distill_qwen_sudoku.yaml \\
#         --checkpoint <path-to-qwen-teacher-best.pt>"""),

    # ── Training ──
    md("""\
## 4. Training

### 4.1 Entry Point

All training is launched through `main.py` with a YAML config:

```bash
# TRM-MLP on Sudoku-Extreme (3 seeds)
python main.py --config configs/trm_official_sudoku_mlp.yaml --mode train --seed 0
python main.py --config configs/trm_official_sudoku_mlp.yaml --mode train --seed 1
python main.py --config configs/trm_official_sudoku_mlp.yaml --mode train --seed 2

# LLM baseline (Qwen2.5-0.5B on Sudoku)
python main.py --config configs/llm_qwen.yaml --mode train --seed 42

# Knowledge distillation
python main.py --config configs/distill_qwen_sudoku.yaml --mode distill \\
    --checkpoint models/llm/qwen_best.pt --seed 0
```

### 4.2 Carbon Tracking

Every training run uses CodeCarbon to track energy consumption and CO2 emissions.
The `CarbonTracker` wrapper logs cumulative emissions at each `log_interval`."""),

    code("""\
from src.training.carbon_tracker import CarbonTracker

# Carbon tracking is integrated into every trainer
# Example usage:
# carbon = CarbonTracker("trm_mlp_sudoku", output_dir="experiments/sudoku-mlp")
# carbon.start()
# ... training loop ...
# stats = carbon.flush()  # mid-training checkpoint
# print(f"Energy so far: {stats['energy_kwh']:.4f} kWh, CO2: {stats['emissions_kg']:.4f} kg")
# final = carbon.stop()   # end of training
# print(f"Total: {final['energy_kwh']:.4f} kWh, {final['emissions_kg']:.4f} kg CO2")

print("Carbon tracking: CodeCarbon EmissionsTracker")
print("Emissions CSVs saved to each experiment's output directory")"""),

    md("""\
### 4.3 Trainer Architecture

Each model type has a dedicated trainer:

| Trainer | File | Model Type | Key Features |
|---------|------|------------|--------------|
| `OfficialTRMTrainer` | `src/training/trainer_official.py` | TRM-MLP, TRM-Att | ACT loss, AdamATan2, EMA, deep recursion |
| `LLMTrainer` | `src/training/trainer_llm.py` | GPT-2, SmolLM, Llama, Qwen | LoRA, HF tokenizer remap, gradient accumulation |
| `DistillationTrainer` | `src/training/trainer_distill.py` | Distilled student | KL divergence + CE, temperature scaling |
| `TRMTrainer` | `src/training/trainer_trm.py` | Legacy TRM | Simple CE training (early prototype) |"""),

    # ── Evaluation ──
    md("""\
## 5. Evaluation

### 5.1 Metrics

Two accuracy metrics are used:
- **Cell accuracy**: fraction of non-masked cells predicted correctly
- **Puzzle accuracy**: fraction of puzzles where ALL cells are correct

For Maze-Hard, we additionally check path validity (4-connected chain from S to G)."""),

    code("""\
from src.evaluation.metrics import cell_accuracy, puzzle_accuracy

# Demonstration with random tensors
logits = torch.randn(4, 81, 11)  # batch=4, seq=81, vocab=11
labels = torch.randint(0, 11, (4, 81))
labels[:, :20] = 0  # mask first 20 positions (pre-filled)

cell_acc = cell_accuracy(logits, labels, ignore_index=0)
puzzle_acc = puzzle_accuracy(logits, labels, ignore_index=0)
print(f"Cell accuracy (random baseline): {cell_acc:.4f}")
print(f"Puzzle accuracy (random baseline): {puzzle_acc:.4f}")"""),

    md("""\
### 5.2 Running Evaluation

```bash
# Evaluate a trained checkpoint
python main.py --config configs/trm_official_sudoku_mlp.yaml --mode eval \\
    --checkpoint models/trm_mlp/best.pt

# Evaluate with the standalone script
python scripts/eval_llm_checkpoint.py --checkpoint models/llm/qwen_best.pt \\
    --config configs/llm_qwen.yaml
```"""),

    # ── Results ──
    md("""\
## 6. Results Summary

The consolidated results are in `results/summary_fixed.csv`. Key findings:

| Model | Task | Cell Acc | Puzzle Acc | Energy (kWh) | CO2 (kg) |
|-------|------|----------|------------|--------------|----------|
| TRM-MLP (HF eval) | Sudoku | 91.55% | 84.74% | 0.48 | -- |
| TRM-MLP (fine-tuned, mean) | Sudoku | 86.01% | 74.62% | 5.45 | 1.30 |
| Qwen2.5-0.5B + LoRA | Sudoku | 19.07% | 0% | 0.90 | 0.21 |
| GPT-2 + LoRA | Sudoku | 13.18% | 0% | 0.26 | 0.06 |
| Distill-Qwen | Sudoku | 35.87% | 0% | 0.009 | 0.002 |
| TRM-Att (HF eval) | Maze | 99.30% | 79.60% | 0.002 | -- |"""),

    code("""\
import pandas as pd

results_path = "results/summary_fixed.csv"
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    cols = ["task", "best_val_puzzle_acc", "best_val_cell_acc", "train_energy_kwh", "train_co2_kg"]
    display_cols = [c for c in cols if c in df.columns]
    print(df[display_cols].to_string(index=False))
else:
    print("Results file not found. Run experiments first.")"""),

    # ── Figures ──
    md("""\
## 7. Figure Generation

The paper's figures are generated by scripts in `scripts/`:

```bash
# Training curve (TRM-MLP 3-seed fine-tuning trajectory)
python scripts/gen_training_curve.py

# Energy scatter plot (all models)
python scripts/gen_energy_scatter.py
```

These produce `paper/fig_training_curve.pdf` and `paper/fig_energy_scatter.pdf`."""),

    code("""\
# Figure generation requires training log CSVs in /tmp/trm_logs/
# (extracted from machine-4 training archives)
#
# The scripts are standalone and can be run directly:
#   python scripts/gen_training_curve.py
#   python scripts/gen_energy_scatter.py
#
# Output: paper/fig_training_curve.pdf, paper/fig_energy_scatter.pdf

print("Figure scripts:")
print("  scripts/gen_training_curve.py  -- 3-seed training trajectory")
print("  scripts/gen_energy_scatter.py  -- energy vs accuracy scatter")"""),

    # ── Configs ──
    md("""\
## 8. Configuration Reference

All experiment configurations are YAML files in `configs/`:

| Config | Model | Task | Key Settings |
|--------|-------|------|-------------|
| `trm_official_sudoku_mlp.yaml` | TRM-MLP | Sudoku | mlp_t=True, L_cycles=6, epochs=500 |
| `trm_official_maze.yaml` | TRM-Att | Maze | mlp_t=False, L_cycles=4, seq_len=900 |
| `llm_qwen.yaml` | Qwen2.5-0.5B | Sudoku | LoRA r=8, epochs=30 |
| `llm_smollm.yaml` | SmolLM2-360M | Sudoku | LoRA r=8, epochs=30 |
| `llm_llama.yaml` | Llama-3.2-1B | Sudoku | LoRA r=8, epochs=30 |
| `llm_gpt2_maze.yaml` | GPT-2 | Maze | LoRA r=8, mask_non_path |
| `distill_qwen_sudoku.yaml` | Distill-Qwen | Sudoku | alpha=0.7, T=4.0, 2.4M params |
| `distill_gpt2_maze.yaml` | Distill-GPT-2 | Maze | alpha=0.7, T=4.0 |

See each YAML file for the full set of hyperparameters."""),

    code("""\
import glob

configs = sorted(glob.glob("configs/*.yaml"))
print(f"Total configs: {len(configs)}")
for c in configs:
    print(f"  {c}")"""),

    # ── Project Structure ──
    md("""\
## 9. Project Structure

```
ML-TRM-Code/
├── main.py                 # Entry point: train / eval / distill
├── start.py                # CLI bootstrap (alternative entry)
├── requirements.txt        # Python dependencies
├── run.sh                  # Task runner shell script
│
├── src/
│   ├── models/             # Model definitions
│   │   ├── trm_official.py      # TRM architecture (ACT + MLP/Attention)
│   │   ├── layers_official.py   # MLP-Mixer, SwiGLU, Rotary embeddings
│   │   ├── losses_official.py   # ACT loss computation
│   │   ├── baseline_llm.py      # LoRA-wrapped HuggingFace LLMs
│   │   ├── distilled_llm.py     # KD student model
│   │   ├── recursion.py         # Deep recursion logic
│   │   └── trm_block.py         # TRM block definition
│   │
│   ├── training/           # Training loops
│   │   ├── trainer_official.py  # TRM trainer (AdamATan2, EMA, ACT)
│   │   ├── trainer_llm.py      # LLM LoRA trainer
│   │   ├── trainer_distill.py  # Knowledge distillation trainer
│   │   ├── carbon_tracker.py   # CodeCarbon wrapper
│   │   └── wandb_utils.py      # W&B logging utilities
│   │
│   ├── data/               # Dataset classes
│   │   ├── sudoku_dataset.py    # Sudoku-Extreme loader + augmentation
│   │   ├── maze_dataset.py      # Maze-Hard loader
│   │   ├── encoding.py          # Token encoding/decoding
│   │   └── collate.py           # Collation for official TRM format
│   │
│   ├── evaluation/         # Evaluation and analysis
│   │   ├── evaluate.py          # Load-and-evaluate pipeline
│   │   ├── metrics.py           # Cell/puzzle accuracy
│   │   ├── k_vote.py            # K-vote ensemble evaluation
│   │   └── aggregate.py         # Cross-run aggregation
│   │
│   └── utils/              # Shared utilities
│       ├── config.py            # Pydantic config schema
│       ├── gpu_config.py        # GPU memory overrides
│       └── seed.py              # Reproducibility seeding
│
├── configs/                # YAML experiment configurations (24 files)
├── data/                   # Dataset builders + metadata
├── scripts/                # Reproduction and figure generation
├── results/                # Consolidated results (summary_fixed.csv)
└── tests/                  # Unit tests
```"""),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}


def main() -> None:
    with open(OUT, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
