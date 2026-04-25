# Training Configuration Notes

Lessons learned from configuring TRM training on consumer GPUs.

## Problem: Slow Training with Zero Accuracy (800 epochs, 25 hours, 0% accuracy)

### Root Cause

With the original config (batch_size=64, 100 sudoku samples, 5K epochs):

- **Only 1 batch per epoch** — `floor(100 / 64) = 1` with `drop_last=True`
- **36 samples wasted every epoch** — never seen by the model
- **GPU auto-tune was never called** — `main.py` used the YAML batch_size directly, ignoring `gpu_config.py`
- **Total optimizer updates after 800 epochs:** `800 x 1 batch x 16 supervision steps = 12,800` — far too few for convergence

The model was barely above random chance (CE loss 2.20 vs random 2.30 for 10 classes).

### Fix

1. **Use 1K samples** (full paper dataset) instead of 100 — gives 20 batches/epoch instead of 1
2. **Reduce epochs to 500** — with 20 batches/epoch, 500 epochs gives `500 x 20 x ~8 avg steps = 80K` optimizer updates (vs 12.8K before)
3. **Integrate GPU auto-tune** into `main.py` — batch_size is now set per GPU at runtime
4. **On-the-fly augmentation** — random digit/row/col/transpose shuffle every load, so the model sees different versions each epoch

### Key Insight: Optimizer Updates Matter, Not Epochs

The total number of **optimizer updates** determines learning, not epochs:

```
optimizer_updates = epochs x batches_per_epoch x avg_supervision_steps
```

With 100 samples: `5000 x 1 x 16 = 80K updates` (but takes ~6.5 days)
With 1K samples:  `500 x 20 x ~8 = 80K updates` (same updates, more data diversity)

More samples per epoch = more diversity = faster convergence = ACT kicks in sooner.

## How Deep Supervision Works

Each batch goes through up to N_sup=16 "supervision steps":

```
for step in range(N_sup):         # up to 16 steps
    optimizer.zero_grad()
    x = model.embedding(inputs)
    (y, z), logits, q = deep_recursion(block, x, y, z, n=6, T=3)
    loss = cross_entropy(logits, labels) + bce(q, correctness)
    loss.backward()
    optimizer.step()              # weight update EVERY step
    if q > threshold: break       # ACT early stopping
```

Each supervision step is a **separate optimizer update** — not accumulated gradients. The model refines its answer (y) and reasoning (z) over multiple steps, with weights changing between steps. States y and z are detached between steps to prevent memory explosion.

This means `grad_accum_steps` from the config is **irrelevant** for TRM training — the deep supervision loop handles its own optimization.

## How ACT (Adaptive Computational Time) Works

ACT is a learned halting mechanism:

1. The model outputs a confidence score `q` (sigmoid of a learned head) after each supervision step
2. `q` is trained with BCE loss: target is 1 if the prediction is fully correct, 0 otherwise
3. If `q > 0.5`, training stops early for that batch (model is confident it solved it)

### ACT Behaviour During Training

- **Early training:** Model gets nothing right, q stays near 0, all 16 steps run (slow)
- **Mid training:** Model starts solving some puzzles, q rises for easy ones, averages ~4-8 steps
- **Late training:** Model solves most puzzles quickly, q rises fast, averages ~2 steps (fast)

The paper reports average steps dropping from 16 to **less than 2** on Sudoku-Extreme.

### ACT at Test Time

ACT is **only used during training** to save time. At test time, all 16 supervision steps are run to maximize accuracy — the model gets every chance to refine its answer.

## Batch Size Guidelines Per GPU

| GPU | VRAM | Sudoku (L=81) | Maze (L=900) |
|-----|------|---------------|--------------|
| RTX 3070 | 8 GB | 48 | 8 |
| RTX 4060 | 8 GB | 48 | 8 |
| RTX 4070 | 12 GB | 128 | 16 |
| RTX 5070 | 12 GB | 128 | 16 |
| RTX 5090 | 32 GB | 256 | 64 |
| L40S | 48 GB | 768 | 128 |

Batch sizes are auto-detected at runtime via `gpu_config.py`. Maze uses self-attention (O(L^2) memory) so it needs much smaller batches than sudoku (MLP-Mixer).

## How to Avoid Slow Training

1. **Check batches/epoch** — if it shows "1/1" in the progress bar, your batch_size is too large for your dataset
2. **Watch the loss** — CE loss should decrease steadily. If it plateaus near ln(num_classes), the model isn't learning
3. **Watch steps_taken** — if stuck at 16.0 after hundreds of epochs, ACT isn't kicking in yet
4. **Watch Q** — q=0.000 means zero confidence. Should start rising once cell accuracy improves
5. **Use on-the-fly augmentation** — pre-generating millions of samples wastes disk and slows data loading
6. **Match total optimizer updates, not epochs** — fewer epochs with more batches is equivalent and provides more data diversity

## Paper Reference Numbers

The TRM paper ("Less is More", Jolicoeur-Martineau 2025) trains:
- **60K epochs** on both Sudoku-Extreme and Maze-Hard
- **Batch size 768** (effective, fits on L40S with 48GB VRAM)
- **1K base samples** with augmentation (1000x shuffle for sudoku, 8x dihedral for maze)
- **Sudoku:** <36 hours on 1x L40S → 87.4% test accuracy
- **Maze:** <24 hours on 4x L40S → 85.3% test accuracy

## Fine-tuning a converged TRM checkpoint vs from-scratch training

**The two regimes need different hyperparameters.** From-scratch values
applied to a converged init (e.g. `--init-weights …Sudoku-Extreme-mlp/…pt`)
will overshoot the optimum during warmup and may *regress* the init. This
was burnt into the repo by run `dz3tkge9` (seed-4 sudoku-mlp,
2026-04-22 → 04-23): paper-faithful config + HF init lost 12 pp of val
puzzle accuracy and never recovered. Full diagnosis at
[../analysis_run_dz3tkge9.md](../analysis_run_dz3tkge9.md).

| Field | from-scratch | fine-tune | why it differs |
|---|---|---|---|
| `lr` | 1e-4 | **1e-5** | full pretrain LR knocks the converged weights off-optimum at peak |
| `warmup_steps` | 2000 | **200** | shorter ramp = smaller cumulative off-optimum displacement |
| `weight_decay` | 1.0 | **0.1** | aggressive WD pulls weights toward zero faster than the (small) loss gradient can pin them |
| `task_emb_lr` / `task_emb_weight_decay` | 1e-4 / 1.0 | **1e-5 / 0.1** | match the global lr / WD per paper convention |
| `q_loss_weight` | 0.5 | **0.0** | `configs/trm_official_sudoku_mlp_finetune.yaml` freezes Q-head gradient flow entirely (more conservative than `config.py:131-133`'s 0.01 recommendation) |
| `halt_exploration_prob` | 0.1 | **0.0** | random halt-decision noise destabilises a halter that's already calibrated |
| `epochs` | 500–5000 | **150–200** | fine-tunes plateau early; spending more is overfitting |
| `eval_interval` | 50 | **10** | early-stop needs frequent enough eval to catch the peak |

Active config: [`configs/trm_official_sudoku_mlp_finetune.yaml`](../configs/trm_official_sudoku_mlp_finetune.yaml). Use it whenever
launching with `--init-weights`. Use `configs/trm_official_sudoku_mlp.yaml`
only when the goal is from-scratch reproduction.
