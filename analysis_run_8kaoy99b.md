# Run analysis — `run-20260422_164116-8kaoy99b`

**TL;DR.** A 23.4 h, ~490-epoch fine-tune of the published Sudoku-Extreme-mlp checkpoint (seed 5) ended interrupted mid-evaluation at epoch 500. The run never improved on its starting point: the loaded HF checkpoint already scored val puzzle_acc = 0.8485 at epoch 0, the optimizer then immediately drove it down to 0.6245 by epoch 50, and 440 epochs of further training only clawed it back to 0.7268 — still 12.2 pp below where we started. About 87 % of the wall-clock was spent in `eval_interval=50` evaluations, not training. Recommendation: kill this trajectory, fix the warm-start regression before another seed is launched, and subsample evaluation during training.

---

## 1. Run identity

| Field | Value |
|---|---|
| wandb run id | `8kaoy99b` |
| Display name | `trm_official_sudoku_seed5_STU-CZC5277FDY_1776872475` |
| Started | 2026-04-22 16:41:16 +0100 |
| Last activity | 2026-04-23 16:06:32 +0100 (output.log mtime) |
| Wall-clock when interrupted | ~23.4 h |
| Last completed epoch in CSV | 490 / 1000 |
| Last log line | mid-eval at epoch 500, eval-step 3250/6607 (~49 % through) |
| Git commit | `4a51d87e443debedd2e76d74134850c775326f5b` (`main`) |
| Host | `STU-CZC5277FDY` (RTX 5070, 12 GB, CUDA 13.0, Python 3.13) |
| Command | `main.py --mode train --config configs/trm_official_sudoku_mlp.yaml --seed 5 --init-weights hf_checkpoints/Sudoku-Extreme-mlp/remapped_for_local.pt --epochs 1000` |

Config (`configs/trm_official_sudoku_mlp.yaml`): TRM-official, `mlp_t=True`, `H_cycles=3`, `L_cycles=6`, `L_layers=2`, `halt_max_steps=16`, `forward_dtype=bfloat16`, `no_ACT_continue=true`. Optimizer `adam_atan2`, `lr=1e-4`, `betas=(0.9, 0.95)`, `weight_decay=1.0`, `warmup_steps=2000`, `batch_size=64`, `grad_accum_steps=24` (effective batch = 1536), `ema_decay=0.999`, `max_grad_norm=1.0`, `task_emb_lr=1e-4`, `task_emb_weight_decay=1.0`. `eval_interval=50`, `log_interval=10`, `save_interval=20`.

---

## 2. What happened (chronological)

### Phase 0 — initialization (epoch 0)
The loaded HF checkpoint was evaluated on the full 6607-batch test split before any training step. Result: **val cell_acc 0.916, val puzzle_acc 0.8485**, `best_puzzle_acc` set to 0.8485. This matches the published Sudoku-Extreme-mlp number (~84.80 %). From this moment, the run cannot achieve a "new best" without surpassing the public checkpoint.

### Phase 1 — warmup-driven blow-up (epochs 1–50)
Train `lm_loss` *climbs* from 1.24 → 3.40 over the first 10 epochs while `cell_acc` (train) stays around 0.96 — the loss spike reflects the auxiliary `q_halt_loss` and label-smoothing regimes re-equilibrating, not a genuine accuracy collapse on training data. But on the held-out split:

- epoch 50 eval (first eval after init): **val puzzle_acc 0.6245 (–22.4 pp), val cell_acc 0.800 (–11.6 pp)**

The test-set drop is the diagnostic event. A few mechanisms compound:

1. **Cold optimizer state.** `--init-weights` only loads model weights, not Adam moment estimates. The first ~2000 warmup steps run a freshly-zeroed optimizer with `lr` ramping 0 → 1e-4. Even small gradients in this regime push weights along directions that are not the ones the original training had stabilized.
2. **Heavy decoupled weight decay.** `weight_decay=1.0` with adam_atan2 applies a per-step pull toward zero proportional to `lr·wd` (≈ 1e-4 per step). This is the paper-faithful Sanjin2024 setting, but it assumes a *running* training history, not a sudden warm-start. On a converged checkpoint, that decay is destabilizing because the optimum already encodes regularization that the public training found.
3. **Seed change.** Seed 5 reshuffles the dataloader, EMA, and dropout RNGs. The public checkpoint corresponds to a specific gradient trajectory; warm-starting with a different stream is effectively a different training run from epoch 0.
4. **ACT halting drift.** Average steps-per-puzzle on training is 4.0 at epoch 10, 3.2 by epoch 50, and 2.2 by epoch 490. Training accuracy improves while average steps shrink — the model becomes faster *and* overconfident. On easy puzzles this is fine; on hard ones it halts before reasoning is complete. q_halt_accuracy stays ≥ 0.97 throughout, so the halt signal is well-fit *to the training distribution* — but generalization to the test split is the loser.

### Phase 2 — slow recovery (epochs 50–490)

Training metrics improve steadily and substantially:

| epoch | train lm_loss | train cell_acc | train exact_acc | avg_steps |
|---|---|---|---|---|
| 50  | 2.314 | 0.970 | 0.922 | 3.2 |
| 100 | 1.833 | 0.976 | 0.940 | 2.9 |
| 200 | 1.565 | 0.980 | 0.948 | 2.6 |
| 300 | 1.338 | 0.984 | 0.956 | 2.4 |
| 400 | 1.229 | 0.986 | 0.962 | 2.3 |
| 490 | 1.125 | 0.987 | 0.965 | 2.2 |

But validation lifts only marginally:

| epoch | val cell_acc | val puzzle_acc | Δ from epoch 50 |
|---|---|---|---|
| 50  | 0.8003 | 0.6245 | — |
| 100 | 0.8170 | 0.6599 | +3.5 pp |
| 150 | 0.8232 | 0.6691 | +4.5 pp |
| 200 | 0.8302 | 0.6839 | +5.9 pp |
| 250 | 0.8432 | 0.7109 | +8.6 pp |
| 300 | 0.8448 | 0.7144 | +9.0 pp |
| 350 | 0.8459 | 0.7147 | +9.0 pp |
| 400 | 0.8497 | 0.7239 | +9.9 pp |
| 450 | 0.8513 | 0.7268 | +10.2 pp |
| 490 | 0.8513 | 0.7268 | +10.2 pp |

Linear-fit on val_puzzle_acc against epoch from 50 → 490 gives ~ +0.024 pp / epoch. To re-reach 0.8485 from 0.7268 at that rate would need ~ 500 more epochs — i.e., the run would need to roughly double in length to get back to its starting point, with no guarantee of crossing it.

The widening **train-vs-val gap** is the other diagnostic: train exact_accuracy 0.965 vs val puzzle_acc 0.727 at epoch 490 → 24 pp generalization gap, and growing. Classic overfitting on top of a worse minimum.

### Phase 3 — interrupted mid-eval (epoch 500)
At epoch 500 the run entered its 11th full evaluation (every 50 epochs). At eval-step 3250/6607 the process stopped writing to `output.log` and `run-8kaoy99b.wandb`. The last line shows live test puzzle_acc ≈ 0.729 — consistent with the trajectory; no evidence of a metric explosion. wandb `debug-internal.log` continued sending heartbeat filestream requests for ~2 minutes after stdout went silent, then also stopped. **No exception, no NaN, no OOM, no CUDA error in stdout** — the process was killed (manual cancel, system reboot, or OneDrive sync conflict; the run lives under a OneDrive path). `best.pt` was last touched at 18:35 on day 1, i.e. during the early epochs when `val_puzzle=0.8485` was still the running maximum from the init eval.

---

## 3. Compute breakdown

Total wall-clock to last logged epoch: **1358.5 minutes** for 49 logged epochs and 9 full evaluations.

- Each full eval over 6607 batches × 1.03 s/it ≈ **113 min/eval**.
- 9 evals (at epochs 50, 100, …, 450) × 113 = **1017 min on eval** (~75 % of wall-clock).
- Add the start-of-run eval (~113 min) and the partial epoch-500 eval (~55 min) and **eval cost ≈ 1185 min ≈ 87 %** of total wall-clock.
- Pure training: ~173 min for 490 epochs ≈ **21 s/epoch**, GPU-saturating at batch 1536 effective.

This is the single most fixable cost in the pipeline. With `eval_interval=50`, every saved hour of eval = an hour reclaimed for training.

---

## 4. Comparison to historical runs (`results/trm_runs_overview.csv`)

| run_id | seed | state | best val puzzle_acc | val cell_acc max | runtime (s) | notes |
|---|---|---|---|---|---|---|
| `94idw79x` | 0 | finished | 0.7340 | 0.8550 | 82,469 | full-length completion |
| `ihj6hpsn` | 0 | crashed | 0.7456 | 0.8584 | 288,394 | longer, then crash |
| `c5kt8l2i` | 1 | crashed | 0.7420 | 0.8585 | 290,867 | similar |
| `8hncpi2x` | 2 | killed   | 0.7486 | 0.8613 | 293,732 | similar |
| **`8kaoy99b`** | **5** | **killed** | **0.7268** | **0.8513** | **84,510** | this run |

Every prior local Sudoku-mlp run plateaued in 0.73–0.75 puzzle-accuracy, well short of the 0.8485 published checkpoint. The behaviour is reproducible across seeds and crash/kill states. **The pattern is the warm-start trajectory itself, not a bad seed or a single hardware fault.**

---

## 5. Root cause hypothesis

The configuration is paper-faithful to Sanjin2024's `all_config.yaml`, but it is intended for *training from scratch*. Three settings are pathological when the model is warm-started from a converged checkpoint:

1. **`warmup_steps=2000` over a converged init.** During warmup, the optimizer is feeling out a minimum it does not yet know, with rising lr. Starting *at* a minimum, this becomes a perturbation engine.
2. **`weight_decay=1.0`.** Aggressive decoupled decay against weights that already encode the published model's regularization; the published training history balanced decay against active gradients, but here the cold optimizer cannot replay that balance.
3. **EMA reset.** `ema_decay=0.999` is reinitialized to the loaded weights, but the EMA-vs-raw trajectory the public checkpoint depended on is gone.

The model rapidly leaves the basin and only re-finds a similar (but distinct, and worse) minimum by epoch ~250–300. After that, the curve flattens at ≈ 0.72 and the train/val gap continues to widen.

---

## 6. Termination diagnosis

- No Python exception in `output.log` (grepped: error/Error/NaN/Exception/CUDA/OOM all clean).
- `run-8kaoy99b.wandb` last updated 2026-04-23 16:03:57; `output.log` last updated 16:06:32; `debug-internal.log` last filestream POST at 16:06:22 returning 200 OK.
- The process simply stopped emitting stdout in the middle of a long evaluation pass.
- Most likely causes, in order: (a) manual Ctrl+C / window close, (b) host sleep/reboot, (c) OneDrive sync lock on `output.log` (the working directory is under `OneDrive - UWE Bristol`, which is known to interfere with long-running file handles).
- The wandb run is presumably still in `running` state on the server until it auto-times out. The local cache has everything needed to `wandb sync` it for the record.

---

## 7. Recommendations

### Immediate
- **Do not resume.** The trajectory is below init and improving at ~0.024 pp / epoch — not worth the GPU-hours.
- `wandb sync wandb/run-20260422_164116-8kaoy99b` to flush the final state to the dashboard, then mark the run "killed — analysed".
- Move the working directory off OneDrive (or exclude it from sync) before launching the next long run.

### Before the next warm-started run
Pick one of the two fixes and stick to it:

1. **Fine-tune mode** (preferred if the goal is to improve on 0.8485):
   - `warmup_steps: 0` (the optimizer should not ramp into a converged basin)
   - `lr: 1e-5` (10× lower)
   - `weight_decay: 0.1` (10× lower; matches the historical attention-variant config that was overridden in this file)
   - `ema_decay: 0.9999` (slower EMA = less "forgetting" of the loaded weights)
   - Also load Adam moments if you have a checkpoint that contains them; if not, accept that the first epoch will be noisy and freeze it (`lr=0` for epoch 0, then ramp).

2. **From-scratch mode** (if the goal is to reproduce 0.8485 ourselves):
   - Drop `--init-weights` entirely. Keep the current paper-faithful config. Budget for the full Sanjin2024 epoch count (much more than 1000).

### Across all runs
- **Subsample eval during training.** Add `--eval-subsample 1000` (or a `training.eval_subsample_batches` config key) so the every-50-epoch eval uses ~1000 of 6607 batches. Run the full split only on a new best, on the final epoch, and at the milestone fractions. Expected payback: drops eval cost from ~87 % of wall-clock to ~13 %, i.e. ~6.7× more training per wall-clock hour.
- **Track val-vs-init delta as a first-class metric.** An obvious red flag is "first eval beats every later eval"; the dashboard should surface this within the first hour rather than at hour 23.
- **Save `best.pt` alongside the eval that produced it.** The current `best.pt` was written at 18:35 on day 1 (epoch ~10) and corresponds to the loaded HF weights, not to anything training produced. If we are going to keep `best_puzzle_acc=0.8485` semantically, the file on disk should match it; right now it does (it's effectively the init), but only by coincidence.

---

## 8. Artifacts

- Training metrics: `wandb/run-20260422_164116-8kaoy99b/files/trm_official_sudoku_train_log.csv`
- Stdout: `wandb/run-20260422_164116-8kaoy99b/files/output.log` (777 KB, 2953 lines)
- Wandb metadata: `wandb/run-20260422_164116-8kaoy99b/files/wandb-metadata.json`
- Checkpoint: `wandb/run-20260422_164116-8kaoy99b/files/best.pt` (29 MB; equals init)
- Config: `configs/trm_official_sudoku_mlp.yaml`
- Init weights: `hf_checkpoints/Sudoku-Extreme-mlp/remapped_for_local.pt`
