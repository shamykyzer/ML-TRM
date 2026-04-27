# M2 Phase 1 — TRM-Att Maze fine-tune post-mortem (run 1u5fesvh)

**Date:** 2026-04-26
**Machine:** M2 (STU-CZC5277FFS, RTX 5070 Blackwell, sm_120)
**Seed:** 1
**Wandb:** https://wandb.ai/shamykyzer/TRM/runs/1u5fesvh
**Wall-clock:** 9 h 05 min (one eval at epoch 20, then killed)
**Energy:** 2.06 kWh, 0.49 kg CO₂eq

## Hypothesis under test

The M3 seed-2 post-mortem (`analysis_run_8kaoy99b.md`) attributed that run's `val_puzzle_acc = 0.7960 → 0.000` collapse inside one epoch (~100 batches) to `adam_atan2`'s direction-only update — `atan2(m, sqrt(v))` is bounded in magnitude by `π/2·lr` regardless of gradient norm, so at a converged optimum it random-walks at that fixed step size and escapes the HF basin in ~100 steps.

The proposed fix was to swap the optimizer to AdamW. This run tested whether AdamW alone would let seed 1 hold its position in the HF basin.

## Setup

`configs/trm_official_maze_finetune.yaml` post-fix, no edits beyond the M3 post-mortem changes already on `main`:
- `optimizer: adamw` (was `adam_atan2`)
- `warmup_steps: 500` (was 200)
- `log/eval/save_interval: 20` (was 5)
- `q_loss_weight: 0.0` (unchanged from dz3tkge9 post-mortem)
- `halt_exploration_prob: 0.0` (unchanged from dz3tkge9 post-mortem)
- 100 epochs

Launched via `start.py` option 8 (after the `menus.py:626` TypeError fix in `dc09f0c`), seed 1, AdamW. HF init loaded with all 24/24 keys, EMA reseeded.

## Result

Single eval row at epoch 20 (CSV — definitive):

| Metric | Value | Interpretation |
|---|---|---|
| `val_puzzle_acc` | **0.0000** | KILL RULE triggered (< 0.5) |
| `val_cell_acc` | 0.9622 | HF init present and largely intact (baseline 0.9754, 1.3 pp drift) |
| `best_puzzle_acc` | 0.0000 | nothing solved end-to-end |
| `train exact_accuracy` | 0.0005 | almost no full-puzzle correctness during training |
| `q_halt_loss` | 0.4020 | non-zero in metric but **doesn't reach the gradient** because `q_loss_weight=0` |
| `q_halt_accuracy` | 0.999 | trivial — head learned to predict "continue" everywhere |
| `avg_steps` | 16.0 | pinned at max ACT budget every batch |

Killed per the brief's KILL RULE for first eval (`val_puzzle_acc < 0.5`).

## Interpretation — AdamW is *not* the load-bearing fix

The M3 hypothesis fails to predict this outcome. AdamW has no random-walk pathology and the model still collapses to the same `val_puzzle_acc=0`, `avg_steps=16` end state.

The smoking gun is in `hf_checkpoints/Maze-Hard/all_config.yaml` and `losses.py` — Sanjin's actual training artifacts:

```python
# Sanjin's losses.py, last line of ACTLossHead.forward:
return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), ...
```

`q_loss_weight = 0.5` is **hardcoded** in the published training. Our `src/models/losses_official.py:144` exposes it as a config parameter, and our fine-tune config sets it to `0.0`:

```python
total_loss = lm_loss + self.q_loss_weight * (q_halt_loss + q_continue_loss)
# with q_loss_weight=0.0 → total_loss = lm_loss
# the Q-halt head receives ZERO gradient signal
```

Combined with `halt_exploration_prob=0.0`, the halt head has neither a learning signal nor an exploration mechanism. It trivially settles on "predict continue everywhere", `avg_steps` pins at 16, no puzzle ever halts, no puzzle is ever counted as solved on eval. `val_puzzle_acc = 0` regardless of cell-prediction quality.

## Cross-config table for the team's report

| Run | Optimizer | First eval | val_cell | val_puzzle | avg_steps |
|---|---|---|---|---|---|
| HF init baseline (Sanjin) | AdamW (β=0.9/0.95) | – | 0.9754 | 0.7960 | – |
| M3 seed 2 | adam_atan2 | inside ep 1 | drifted | 0.0000 | – |
| **M2 seed 1, this run** | **AdamW** | **epoch 20** | **0.9622** | **0.0000** | **16.0** |

→ Optimizer choice is **not** the load-bearing variable. The collapse mechanism is upstream of the optimizer, and `q_loss_weight=0.0` is the prime suspect.

## Confounds that don't explain the collapse

- **Batch size mismatch** (Sanjin 4608 vs ours 16) explains why we'd never reach 0.7960 even with a perfect setup — doesn't explain landing at 0.0000.
- **Epoch count** (Sanjin 50 000 vs ours 20) — same comment.
- **LR** (1e-4 vs our 1e-5) — at 100× lower LR, a converged head still shouldn't collapse to the trivial halt policy unless the training signal that anchors it is removed.

The *only* thing that explains the trivial-halt collapse is removing the Q-loss gradient. Verified empirically by Phase 2b (`analysis_run_1sg9856w.md`).

## Useful artifacts

- `/c/ml-trm-work/trm-att-maze-50ep-seed1.q_loss_0_run-2026-04-26/epoch_20.pt` — checkpoint preserved as evidence
- `/c/ml-trm-work/trm-att-maze-50ep-seed1.q_loss_0_run-2026-04-26/trm_official_maze_train_log.csv` — single eval row
- `/c/ml-trm-work/trm-att-maze-50ep-seed1.q_loss_0_run-2026-04-26/emissions.csv` — full run energy log
- Wandb: https://wandb.ai/shamykyzer/TRM/runs/1u5fesvh

## Follow-ups produced

- Phase 2b on this same machine (`analysis_run_1sg9856w.md`): tested `q_loss_weight=0.5` alone — recovered briefly at epoch 2, collapsed by epoch 4.
- Phase 2c on this machine (seeds 0 and 2): testing Sanjin's full combination (`q_loss=0.5` + `halt_exp=0.1`).
- M1/M3 brief at `docs/sprint_brief_m1_m3_recovery_2026-04-26.md` for the other two boxes.
