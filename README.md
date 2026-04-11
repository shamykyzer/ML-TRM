# Tiny Recursive Model (TRM) -- Puzzle Solving

Comparing Tiny Recursive Models (~6-8M params) against fine-tuned LLMs (124M-1.2B params) on structured reasoning tasks (Sudoku-Extreme, Maze-Hard). TRMs use recursive weight-sharing to iteratively refine solutions, dramatically outperforming LLMs while using orders of magnitude less energy.

**Module:** UFCFAS-15-2 Machine Learning | **Team:** Ahmed AlShamy, Armin Ghaseminejad, Nickolas Greiner

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
|   +-- trm_sudoku.yaml            # TRM-MLP hyperparameters
|   +-- trm_maze.yaml              # TRM-Att hyperparameters
|   +-- trm_maze_fast.yaml         # TRM-Att with fewer epochs (for augmented data)
|   +-- llm_config.yaml            # GPT-2 + LoRA config
|   +-- llm_qwen.yaml              # Qwen2.5-0.5B + LoRA config
|   +-- llm_smollm.yaml            # SmolLM2-360M + LoRA config
|   +-- llm_llama.yaml             # Llama-3.2-1B + LoRA config
+-- scripts/
|   +-- auto_push.sh               # Auto-commit/push training logs hourly
+-- .github/workflows/
|   +-- training-notify.yml        # GitHub Action: post training stats on push
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

## Quick Start

Every trainer runs via `python main.py --mode train --config <yaml>`. The configs under
`configs/` are the canonical entry points:

```bash
python main.py --mode train --config configs/trm_official_sudoku.yaml   # TRM-MLP sudoku
python main.py --mode train --config configs/trm_official_maze.yaml     # TRM-Att maze
python main.py --mode train --config configs/llm_config.yaml            # GPT-2 + LoRA
python main.py --mode train --config configs/llm_qwen.yaml              # Qwen2.5-0.5B
python main.py --mode train --config configs/llm_smollm.yaml            # SmolLM2-360M
python main.py --mode train --config configs/llm_llama.yaml             # Llama-3.2-1B
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

| Model | Type | Params | Expected Sudoku | Expected Maze |
|-------|------|--------|----------------|---------------|
| TRM-MLP | Recursive (MLP-Mixer) | ~6.4M | ~87% | -- |
| TRM-Att | Recursive (Attention) | ~8.4M | -- | ~85% |
| GPT-2 + LoRA | Fine-tuned LLM | 124M (0.8M trainable) | ~0% | -- |
| Qwen2.5-0.5B + LoRA | Fine-tuned LLM | 494M | ~0% | -- |
| SmolLM2-360M + LoRA | Fine-tuned LLM | 360M | ~0% | -- |
| Llama-3.2-1B + LoRA | Fine-tuned LLM | 1.2B | ~0% | -- |
| Distilled LLM | Small transformer | ~2.4M | ~0% | -- |

The thesis: TRM with 6-8M params dramatically outperforms LLMs with 20-200x more parameters on structured reasoning, at a fraction of the energy cost.

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
