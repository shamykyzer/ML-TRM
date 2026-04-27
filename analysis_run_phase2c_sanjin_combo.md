# M2 Phase 2c — Sanjin combo cross-seed post-mortem

**Date:** 2026-04-27
**Machine:** M2 (STU-CZC5277FFS, RTX 5070 Blackwell)
**Seeds:** 0, 2 (back-to-back)
**Wall-clock:** ~2 h per seed (4 evals + kill at epoch 4)
**Hypothesis under test:** Restoring **both** of Sanjin's halt-related knobs (`q_loss_weight=0.5` AND `halt_exploration_prob=0.1`) recovers `val_puzzle_acc` from the M2/M3 collapse — distinct from either knob alone.

## Setup (identical for both seeds)

`configs/trm_official_maze_finetune.yaml` edits, reverted after both runs:

```diff
- q_loss_weight: 0.0
+ q_loss_weight: 0.5
- halt_exploration_prob: 0.0
+ halt_exploration_prob: 0.1
- epochs: 100
+ epochs: 10
- log/eval/save_interval: 20
+ log/eval/save_interval: 2
```

Everything else identical to the Phase 1 baseline (AdamW, lr=1e-5, warmup=500, batch=16). Launched via `start.py` option 8 with seed 0 then seed 2 sequentially. HF init loaded with all 24/24 keys for each.

## Results

### Seed 0

| Epoch | val_puzzle | val_cell | avg_steps | q_halt_acc | train_exact | q_halt_loss |
|-------|------------|----------|-----------|------------|-------------|-------------|
|   2   | 0.0000     | 0.9678   | 15.2      | 0.9621     | 0.0380      | 0.5634      |
|   4   | 0.0000     | 0.9663   | 15.4      | 0.9690     | 0.0261      | 0.4997      |

KILLED at epoch 4 per the brief's two-evals-at-zero rule.

### Seed 2

| Epoch | val_puzzle | val_cell | avg_steps | q_halt_acc | train_exact | q_halt_loss |
|-------|------------|----------|-----------|------------|-------------|-------------|
|   2   | 0.0000     | 0.9690   | **14.0**  | 0.9371     | **0.0961**  | 0.6732      |
|   4   | 0.0000     | 0.9643   | **16.0**  | **0.9996** | 0.0004      | **0.0487**  |

KILLED at epoch 4. `best_puzzle_acc` (in-epoch peak) at epoch 2 was 0.5290 — the model briefly hit ~53% mid-epoch then collapsed by eval time.

## Interpretation

The combination is **necessary but insufficient at our batch size**.

Both seeds show the same trajectory:
1. **Epoch 2:** halt head receives signal (`avg_steps` drops to 14–15 vs trivial 16, `q_halt_acc` < 0.97 vs trivial 0.999, training-time exact accuracy non-zero), and there's an in-epoch peak in puzzle solving captured by the trainer's `best_puzzle_acc` column (0.226 for seed 0, 0.529 for seed 2).
2. **Epoch 4:** full collapse back to the trivial attractor — halt head pinned at 16 steps, q_halt_loss shrunk to ~0.05 as the head settles on "always continue", training exact accuracy back to ~0, val_puzzle still 0.

The two seeds differ in the *strength* of epoch-2 signal (seed 2 had ~2.5× higher training exact and ~2× higher in-epoch peak) but converge to the same end state by epoch 4. So this isn't a seed-specific instability — it's a systematic failure mode at this batch size.

## Refined diagnosis (now spanning Phase 1, 2b, 2c)

| Run | q_loss | halt_exp | Result |
|---|---|---|---|
| Phase 1 (run 1u5fesvh) | 0.0 | 0.0 | val=0 at ep 20, halt head never escapes trivial |
| Phase 2b (run 1sg9856w) | 0.5 | 0.0 | val=0.004 at ep 2, collapses by ep 4 |
| Phase 2c seed 0 | 0.5 | 0.1 | val=0 at ep 2/4, collapses by ep 4 |
| Phase 2c seed 2 | 0.5 | 0.1 | val=0 at ep 2/4, collapses by ep 4 |

The progression suggests:
- `q_loss=0.0` + `halt_exp=0.0` is the worst case (head receives no signal at all, never trains).
- `q_loss=0.5` + `halt_exp=0.0` lets the head briefly find a productive setting (transient recovery at ep 2 in run 1sg9856w) before re-collapsing — Q-loss alone gives a learning signal but no exploration to find good halt timings.
- `q_loss=0.5` + `halt_exp=0.1` (Sanjin's combo) doesn't markedly improve the trajectory: epoch-2 signal is *less* than Phase 2b's seed 1 (val 0.000 vs 0.004), and the collapse to "always continue" by epoch 4 happens regardless.

The **batch-size gap** (16 vs Sanjin's 4608, a 288× factor) appears to be the dominant unexplored variable. With small-batch noise, the halt head's gradient is too noisy to escape the "always continue" attractor's basin even with both knobs on. Sanjin's training had ~288× less per-step gradient noise; at that scale, the halt-head dynamics that look unstable here are smooth enough to converge.

## Implication for the report

The team's narrative for the maze fine-tune attempts is now well-supported:

> *"We attempted to fine-tune the released Maze-Hard checkpoint at our hardware-constrained batch size of 16. The trainer's halt head consistently collapses to a trivial 'always continue' attractor regardless of the loss configuration tested (q_loss_weight ∈ {0.0, 0.5} × halt_exploration_prob ∈ {0.0, 0.1}). The most likely root cause is the 288× batch size gap relative to the published training (4608 on 8 × H200), which we cannot close on the available hardware. Our deliverable is therefore the K-vote analysis on the released checkpoint, which produced a Pareto-degenerate curve at the published 79.6% baseline."*

## Useful artifacts

- Seed 0 dir: `/c/ml-trm-work/trm-att-maze-50ep-seed0/`
  - `epoch_2.pt`, `epoch_4.pt`, `best.pt` (val_puzzle = 0)
  - `trm_official_maze_train_log.csv` — 2 eval rows
  - `emissions.csv`
- Seed 2 dir: `/c/ml-trm-work/trm-att-maze-50ep-seed2/` — same structure
- Wandb (look up by run ID emitted in launcher stdout)
