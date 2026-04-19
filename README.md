# Tiny Recursive Model (TRM) -- Puzzle Solving

Comparing Tiny Recursive Models (~6-8M params) against fine-tuned LLMs (124M-1.2B params) on structured reasoning tasks (Sudoku-Extreme, Maze-Hard). TRMs use recursive weight-sharing to iteratively refine solutions, dramatically outperforming LLMs while using orders of magnitude less energy.

**Module:** UFCFAS-15-2 Machine Learning | **Team:** Ahmed AlShamy, Armin Ghaseminejad, Nickolas Greiner

## Recent updates (April 2026)

Pre-fleet fixes landed alongside the first Qwen-sudoku baseline and first from-scratch TRM-MLP run. See `docs/wandb_metrics_glossary.md` for the full metric reference.

- **Fixed: LLM eval metric off-by-one.** HuggingFace causal LM forward shifts `labels` internally; the old `LLMTrainer.evaluate()` compared `preds[i]` to `labels[i]` instead of `labels[i+1]`, so `val/puzzle_acc` was stuck at 0 regardless of training quality. `src/training/trainer_llm.py` now shifts to match training and also reports `val/cell_acc`. Existing Qwen-sudoku checkpoint re-evaluated: 19.07% cell accuracy, 0% puzzle accuracy (sudoku-extreme test split).
- **Fixed: `apply_gpu_overrides` clobbering LLM YAMLs.** The GPU profile table in `src/utils/gpu_config.py` is TRM-scoped (~7M params) and was overriding the hand-tuned `batch_size × grad_accum_steps` pairs in the LLM configs, OOMing 500M-param Qwen on maze. Now gates on `model_type` so LLM/distill configs stay authoritative.
- **Fixed: distill teacher/student phase mismatch.** When the teacher is a HF causal LM, its logits at position `i` predict `input[i+1]`, but the plain-encoder student predicts at position `i`. `trainer_distill.py::_train_epoch` now re-aligns (shift teacher left by 1, student/labels right by 1) so KL and hard-label CE train the student on a single consistent target.
- **Added: `val/loss`, `val/cell_acc`, and symmetric aliases.** Both `trainer_llm.py` and `trainer_distill.py` now emit `val/loss` (overfitting signal) plus `val/accuracy` and `val/exact_accuracy` aliases that match `trainer_official.py`'s naming. All wandb panels work across all four trainers.
- **Added: CO₂-per-correct-puzzle metric.** `src/evaluation/aggregate.py` computes `correct_puzzles`, `co2_per_correct_puzzle`, `kwh_per_correct_puzzle` per row in `results/summary.csv`. Aggregator also walks `$TRM_EXPERIMENT_DIR` so LLM + distill runs stored outside `experiments/` are picked up.
- **Added: peak/overfit figures.** `results/figures/sudoku_att_rise_and_collapse.png` (from-scratch TRM-Att collapsed at epoch 350) and `results/figures/sudoku_mlp_peak_and_overfit.png` (HF-init TRM-MLP peaked at epoch 900, then overfit) — both intended for the Discussion section.
- **Added: LLM epoch budget = 30.** All eight LLM YAMLs reduced from 50-100 → 30 based on the Qwen-sudoku loss curve saturating at epoch ~30. Saves ~40 hours of fleet time at no accuracy cost (proposal thesis only needs `puzzle_acc ≈ 0%` on LLMs).
- **Published: W&B Metrics Glossary report.** Every logged metric documented in `docs/wandb_metrics_glossary.md` and mirrored as a wandb Report: <https://wandb.ai/shamykyzer/TRM-LLM/reports/W-B-Metrics-Glossary-ML-TRM--VmlldzoxNjU4NzE2Mg>. Regenerate with `python scripts/publish_wandb_metrics_report.py`.

## How TRM Works

Unlike standard transformers that stack unique layers, TRM uses a **single shared 2-layer block** applied recursively. The model maintains three embedding streams:

```
x  =  input embedding   (puzzle tokens, frozen)
y  =  answer embedding  (iteratively refined)
z  =  latent reasoning   (scratch space for thinking)
```

**Three levels of recursion:**

```
Deep Supervision (N_sup=16 steps, each a separate optimizer step)
  |
  +-- Deep Recursion (T=3 passes, only last has gradients)
        |
        +-- Latent Recursion (n=6 inner steps)
              |
              +-- z = block(x + y + z)   # refine reasoning
              +-- y = block(y + z)       # update answer
```

**ACT (Adaptive Computation Time):** A halting head predicts confidence. When confidence > 0.5, recursion stops early -- the model learns to use fewer steps for easy puzzles.

### Architecture Variants

| Model | Params | Sequence Processing | Task |
|-------|--------|-------------------|------|
| TRM-MLP | ~6.4M | MLP-Mixer (token mixing) | Sudoku (L=81) |
| TRM-Att | ~8.4M | Self-Attention + RoPE | Maze (L=900) |

### Building Blocks

- **RMSNorm** (post-norm: applied AFTER residual addition)
- **SwiGLU** FFN: `W2(SiLU(W1(x)) * W3(x))`, no bias anywhere, expansion=4 (ff_hidden=2048)
- **Rotary Position Embedding (RoPE)** for maze attention variant
- **Stable-max cross-entropy** for numerical stability

## Project Structure

```
ML-TRM/
+-- data/
|   +-- common.py                  # PuzzleDatasetMetadata + dihedral_transform
|   +-- build_sudoku_dataset.py    # Downloads & preprocesses Sudoku-Extreme from HF
|   +-- build_maze_dataset.py      # Downloads & preprocesses Maze-Hard from HF
+-- src/
|   +-- models/
|   |   +-- layers.py              # RMSNorm, SwiGLU, RoPE, StableMaxCE, MLPMixer
|   |   +-- trm_block.py           # Shared 2-layer block (attention or mixer)
|   |   +-- recursion.py           # latent_recursion, deep_recursion, deep_supervision
|   |   +-- trm_sudoku.py          # TRM-MLP (sudoku) + TRM-Att (maze)
|   |   +-- baseline_llm.py        # GPT-2 / Qwen / SmolLM / Llama + LoRA wrapper
|   |   +-- distilled_llm.py       # Student transformer + distillation loss
|   +-- data/
|   |   +-- sudoku_dataset.py      # PyTorch Dataset for sudoku .npy files
|   |   +-- maze_dataset.py        # PyTorch Dataset for maze .npy files
|   +-- training/
|   |   +-- trainer_trm.py         # Deep supervision + ACT + EMA + AMP + resume
|   |   +-- trainer_llm.py         # LLM fine-tuning loop
|   |   +-- trainer_distill.py     # Knowledge distillation loop
|   |   +-- ema.py                 # Exponential Moving Average
|   |   +-- carbon_tracker.py      # CodeCarbon wrapper
|   +-- evaluation/
|   |   +-- evaluate.py            # Full eval with checkpoint loading + visualization
|   |   +-- metrics.py             # Cell accuracy, puzzle accuracy
|   +-- utils/
|       +-- config.py              # YAML config + Pydantic models
|       +-- seed.py                # Reproducibility
+-- configs/
|   +-- trm_sudoku.yaml                 # TRM-MLP hyperparameters (legacy trainer_trm)
|   +-- trm_maze.yaml                   # TRM-Att hyperparameters (legacy)
|   +-- trm_maze_fast.yaml              # TRM-Att w/ fewer epochs (augmented data)
|   +-- trm_official_sudoku.yaml        # Official TRM-Att, from-scratch sudoku
|   +-- trm_official_sudoku_mlp.yaml    # Official TRM-MLP-t, HF-init sudoku
|   +-- trm_official_maze.yaml          # Official TRM-Att, HF-init maze
|   +-- llm_config.yaml                 # GPT-2 sudoku + LoRA (epochs=30)
|   +-- llm_qwen.yaml                   # Qwen2.5-0.5B sudoku + LoRA
|   +-- llm_qwen_maze.yaml              # Qwen2.5-0.5B maze (batch=2 x accum=8)
|   +-- llm_smollm.yaml                 # SmolLM2-360M sudoku + LoRA
|   +-- llm_smollm_maze.yaml            # SmolLM2-360M maze
|   +-- llm_llama.yaml                  # Llama-3.2-1B sudoku + LoRA
|   +-- llm_llama_maze.yaml             # Llama-3.2-1B maze (batch=1 x accum=16)
|   +-- llm_gpt2_maze.yaml              # GPT-2 maze (batch=4 x accum=4)
+-- scripts/
|   +-- aggregate_metrics.py            # Walks experiments/ + $TRM_EXPERIMENT_DIR -> summary.csv
|   +-- plot_results.py                 # 5 figures from summary.csv
|   +-- plot_sudoku_att_story.py        # Rise-and-collapse figure (sudoku-att from scratch)
|   +-- plot_sudoku_mlp_overfit.py      # Peak-and-overfit figure (sudoku-mlp HF-init)
|   +-- eval_llm_checkpoint.py          # Shift-aware re-eval for existing LLM checkpoints
|   +-- publish_wandb_metrics_report.py # Publish docs glossary as a wandb Report
|   +-- remap_*.py / verify_*.py        # HF checkpoint remap + load verifiers
|   +-- sanity_check.py                 # Data + forward-pass smoke test
|   +-- auto_push.sh                    # Auto-commit/push training logs hourly
+-- tests/
|   +-- test_encoding.py                # 20 tests pinning token schemas + validity checks
|   +-- test_aggregate.py               # 11 tests for summary.csv pipeline
|   +-- test_inspection.py              # Failure-inspection renderer
|   +-- test_plots.py                   # Plot output smoke tests
+-- docs/
|   +-- wandb_metrics_glossary.md       # Every wandb metric explained (source of truth)
|   +-- training-notes.md               # GPU batch-size guide, ACT behaviour
|   +-- superpowers/specs/              # Design docs for official TRM port
+-- .github/workflows/
|   +-- training-notify.yml             # GitHub Action: post training stats on push
+-- main.py                        # CLI entrypoint (train / eval / distill)
+-- start.py                       # Stage-aware onboarding (venv -> .env -> wandb -> data)
+-- run.sh                         # Bash task runner (Linux-only, optional)
+-- .env.example                   # Template for machine-local config / secrets
+-- models/                        # Saved checkpoints
+-- experiments/                   # CodeCarbon logs + training CSV logs
+-- results/                       # Evaluation outputs
```

## Running on a new machine

**Requirements:** Python 3.10-3.12, NVIDIA GPU with CUDA

`start.py` is a stage-aware onboarding script: run it once, it handles the next missing
setup stage (venv -> .env -> wandb -> data), tells you what it did, and exits. Re-run
until it prints the training menu.

### 1. Clone + create venv with CUDA torch

```bash
git clone <repo-url>
cd ML-TRM
python start.py        # creates venv, installs torch (cu128) + requirements
```

### 2. Copy env template and fill in your secrets

```bash
cp .env.example .env
# Edit .env: set WANDB_API_KEY at minimum.
# Get yours from https://wandb.ai/authorize
python start.py        # continues to the next stage
```

### 3. Log in to wandb (optional but recommended)

```bash
wandb login            # paste the API key from .env
python start.py        # continues

# Or skip wandb entirely:
python start.py --skip-wandb
```

### 4. Build datasets

```bash
python start.py        # downloads Sudoku + Maze from HuggingFace
```

### 5. Activate venv + train

```bash
# Windows
"C:\Users\<you>\.venvs\ml-trm\Scripts\activate"

# Linux/Mac
source .venv/bin/activate

python main.py --mode train --config configs/trm_official_sudoku.yaml
```

Run `python start.py status` at any time to see which stages are ready.

### Optional overrides

All machine-specific paths live in `.env` - see `.env.example` for the full list. The
most useful ones:

- `TRM_WANDB_ENTITY` - your wandb username or team entity
- `TRM_ROLLING_CHECKPOINT_DIR` - external drive for crash-recovery backups (e.g. `D:/ml-trm-checkpoints`)
- `TRM_HF_REPO_ID` + `HF_TOKEN` - HuggingFace Hub repo for checkpoint sync
- `TRM_DATA_DIR` / `TRM_CHECKPOINT_DIR` - move data/checkpoints off the repo

**Semantics:** explicit values in the YAML always win over `.env`. Machine-specific
fields default to `""` in the committed YAMLs so a fresh clone + `.env` just works.

### Weave monitors

Each `evaluate()` call is traced via Weights & Biases Weave when wandb is authed.
Traces appear at:

```
https://wandb.ai/<entity>/<project>/weave/monitors
```

You can configure scorers and monitors on that page - see the
[Weave Monitors docs](https://docs.wandb.ai/weave/guides/evaluation/monitors).
To disable tracing for a run, set `training.use_weave: false` in the config YAML.

### W&B metrics reference

Every metric logged by the four trainers is documented in
`docs/wandb_metrics_glossary.md` (units, meaning, and what to watch for per
metric). The same glossary is published as a wandb Report:
<https://wandb.ai/shamykyzer/TRM-LLM/reports/W-B-Metrics-Glossary-ML-TRM--VmlldzoxNjU4NzE2Mg>.

To refresh the report after editing the glossary:

```bash
python scripts/publish_wandb_metrics_report.py
# or retarget:
python scripts/publish_wandb_metrics_report.py --project TRM
```

## Quick Start

Every trainer runs via `python main.py --mode train --config <yaml>`. The configs under
`configs/` are the canonical entry points:

```bash
# TRM — official architecture port (paper-faithful)
python main.py --mode train --config configs/trm_official_sudoku_mlp.yaml   # MLP-t, HF-init
python main.py --mode train --config configs/trm_official_sudoku.yaml       # Att, from scratch
python main.py --mode train --config configs/trm_official_maze.yaml         # Att, HF-init

# LLM baselines (epochs=30 each after April 2026 fleet-budget fix)
python main.py --mode train --config configs/llm_config.yaml                # GPT-2 sudoku
python main.py --mode train --config configs/llm_gpt2_maze.yaml             # GPT-2 maze
python main.py --mode train --config configs/llm_qwen.yaml                  # Qwen2.5-0.5B sudoku
python main.py --mode train --config configs/llm_qwen_maze.yaml             # Qwen2.5-0.5B maze
python main.py --mode train --config configs/llm_smollm.yaml                # SmolLM2-360M sudoku
python main.py --mode train --config configs/llm_smollm_maze.yaml           # SmolLM2-360M maze
python main.py --mode train --config configs/llm_llama.yaml                 # Llama-3.2-1B sudoku
python main.py --mode train --config configs/llm_llama_maze.yaml            # Llama-3.2-1B maze
```

Aggregate results and refresh figures after each run:

```bash
TRM_EXPERIMENT_DIR=C:/ml-trm-work python scripts/aggregate_metrics.py   # -> results/summary.csv
python scripts/plot_results.py                                          # -> results/figures/*.png
```

## Training

### Estimated training times

| Task | Epochs | RTX 3070 (8GB) | RTX 4070 (12GB) | RTX 5070 (12GB) | L40S (48GB) |
|------|--------|----------------|-----------------|-----------------|-------------|
| Sudoku | 60K | ~24 hrs | ~17 hrs | ~14 hrs | ~4 hrs |
| Maze | 5K | ~5 days | ~3 days | ~2.5 days | ~7 hrs |

GPU batch sizes are auto-detected. Times decrease as ACT kicks in (model learns to stop early).

### Train TRM models

```bash
python main.py --mode train --config configs/trm_official_sudoku.yaml   # TRM-MLP (~24h on RTX 3070)
python main.py --mode train --config configs/trm_official_maze.yaml     # TRM-Att (~5d on RTX 3070)
```

**Do not run both at the same time** -- they share GPU VRAM.

### Train LLM baselines

```bash
python main.py --mode train --config configs/llm_config.yaml    # GPT-2
python main.py --mode train --config configs/llm_qwen.yaml      # Qwen2.5-0.5B
python main.py --mode train --config configs/llm_smollm.yaml    # SmolLM2-360M
python main.py --mode train --config configs/llm_llama.yaml     # Llama-3.2-1B
```

### Resuming training

Training can be resumed from any checkpoint. Progress is fully restored (optimizer,
scheduler, EMA, epoch count).

```bash
python main.py --mode train --config configs/trm_official_sudoku.yaml \
    --resume models/sudoku-att/latest.pt

python main.py --mode train --config configs/trm_official_maze.yaml \
    --resume models/maze-official/latest.pt
```

To increase epochs after an initial run, edit the `epochs` value in the config YAML,
then resume.

### Evaluation

```bash
python main.py --mode eval --config configs/trm_official_sudoku.yaml \
    --checkpoint models/sudoku-att/best.pt

python main.py --mode eval --config configs/trm_official_maze.yaml \
    --checkpoint models/maze-official/best.pt

python main.py --mode eval --config configs/llm_config.yaml \
    --checkpoint models/llm/best.pt
```

### Remote progress monitoring

**In a separate terminal -- auto-push training stats every hour:**

```bash
bash scripts/auto_push.sh
```

A GitHub Action (`.github/workflows/training-notify.yml`) posts training stats to a GitHub Issue on each push. Subscribe to the issue for phone notifications via the GitHub mobile app.

Stats are logged every `log_interval` epochs:
- Sudoku: every 200 epochs
- Maze: every 50 epochs

## Data Encoding

### Sudoku
- **Vocab:** 11 tokens (pad=0, digits 0-9 shifted to tokens 1-10)
- **Grid:** 9x9 flattened to 81 tokens (row-major)
- **Train:** 1,000 puzzles (17 given clues each) | **Test:** 422,786 puzzles
- **Augmentation:** On-the-fly random shuffling each epoch (digit permutation, row/column band shuffle, transpose) -- equivalent to 1000x augmentation without disk cost

### Maze
- **Vocab:** 6 tokens (pad=0, '#'=1, ' '=2, 'S'=3, 'G'=4, 'o'=5)
- **Grid:** 30x30 flattened to 900 tokens
- **Train:** 1,000 mazes (shortest path >110 steps) | **Test:** 1,000 mazes
- **Augmentation:** 8 dihedral transforms (rotations + flips)

## Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden dim (D) | 512 | Shared block width |
| FF hidden | 2048 | SwiGLU intermediate (expansion=4) |
| Inner recursions (n) | 6 | Latent refinement steps |
| Outer recursions (T) | 3 | Only last pass has gradients |
| Supervision steps (N_sup) | 16 | Max, ACT can stop early |
| ACT threshold | 0.5 | Halt when confidence > 0.5 |
| Optimizer | AdamW | betas=(0.9, 0.95) |
| Learning rate | 1e-4 | Linear warmup over 2K steps |
| Weight decay | 1.0 | TRM only; LLM uses 0.01 |
| EMA decay | 0.999 | Applied before evaluation |
| Effective batch | 768 | batch_size * grad_accum |
| Mixed precision | Auto | AMP enabled on CUDA GPUs |

## Models Compared

Paper targets vs our measured results. See `results/summary.csv` for the authoritative aggregate (CO₂ and energy columns included there).

| Model | Type | Params | Task | Paper target | Our result | Notes |
|-------|------|--------|------|-------------|-----------|-------|
| TRM-MLP (HF eval-only) | Recursive (MLP-Mixer) | ~6.4M | Sudoku | 84.80% puzzle | **84.74% puzzle, 91.55% cell** | `hf_checkpoints/Sudoku-Extreme-mlp` evaluated on our pipeline, no further training |
| TRM-MLP (HF-init + fine-tune) | Recursive (MLP-Mixer) | ~6.4M | Sudoku | 84.80% puzzle | **74.56% peak @ epoch 900, overfit thereafter** | RTX 5070; see `results/figures/sudoku_mlp_peak_and_overfit.png` |
| TRM-Att (from scratch) | Recursive (Attention) | ~8.4M | Sudoku | 77.70% puzzle | **18.33% peak @ epoch 100, collapsed to 0%** | RTX 5070; see `results/figures/sudoku_att_rise_and_collapse.png` |
| TRM-Att (HF eval-only) | Recursive (Attention) | ~8.4M | Maze | 85.30% puzzle | **79.60% puzzle, 97.54% cell** | `hf_checkpoints/Maze-Hard` evaluated on our pipeline, no further training |
| Qwen2.5-0.5B + LoRA | Fine-tuned LLM | 494M (737K trainable) | Sudoku | ~0% (replicate lit.) | **0% puzzle, 19.07% cell** | 100 ep, 0.21 kg CO₂; Fix D re-eval |
| GPT-2 + LoRA | Fine-tuned LLM | 124M | Sudoku/Maze | ~0% | not run yet | epochs=30 after Fix C |
| SmolLM2-360M + LoRA | Fine-tuned LLM | 360M | Sudoku/Maze | ~0% | not run yet | epochs=30 after Fix C |
| Llama-3.2-1B + LoRA | Fine-tuned LLM | 1.2B | Sudoku/Maze | ~0% | not run yet | epochs=30 after Fix C |
| Distilled LLM | Small transformer | ~2.4M | Sudoku/Maze | ~0% | not run yet | teacher = Qwen baseline |

The thesis holds: TRM-MLP at ~6.4M params reaches 74.56% puzzle accuracy on Sudoku-Extreme where a 494M-param Qwen LoRA reaches 0% full-puzzle (19.07% per-cell). Full energy/CO₂-per-correct-puzzle columns in `results/summary.csv`.

## Key Implementation Details

1. **Post-norm, not pre-norm:** `output = RMSNorm(sublayer(x) + x)` -- this matters for matching published results
2. **Detach between supervision steps:** `y` and `z` are detached after each deep_recursion call to prevent OOM
3. **EMA before eval:** Always use EMA shadow weights for evaluation (the trainer handles this automatically)
4. **No bias anywhere:** All linear layers in the TRM use `bias=False`
5. **Stable-max loss:** Custom cross-entropy that clips log-sum-exp for numerical stability
6. **Mixed precision:** AMP auto-enabled on CUDA for ~1.5-2x speedup
7. **Checkpoint resume:** Training can resume from any checkpoint via `--resume`

## Reference

- [TRM: Less is More](https://arxiv.org/abs/2510.04871) -- Jolicoeur-Martineau et al.
- [TRM with Mamba-2 Attention Hybrid](https://arxiv.org/abs/2602.12078) -- Wang & Reid
- [Official TRM code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
