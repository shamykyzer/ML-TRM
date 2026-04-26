# M2 Phase 2b — q_loss_weight=0.5 validation post-mortem (run 1sg9856w)

**Date:** 2026-04-26
**Machine:** M2 (STU-CZC5277FFS, RTX 5070)
**Seed:** 1
**Wandb:** https://wandb.ai/shamykyzer/TRM/runs/1sg9856w
**Wall-clock:** 2 h 49 min (5 evals at epochs 2, 4, 6 — killed before 8 and 10)
**Energy:** 0.640 kWh, 0.152 kg CO₂eq

## Hypothesis under test

The M2 Phase 1 post-mortem (run `1u5fesvh`) identified `q_loss_weight: 0.0` as the root cause of the seed-1 collapse: with the Q-halt gradient zeroed out and `halt_exploration_prob=0.0`, the halt head trivially settles on "always continue", `avg_steps` pins at 16, no puzzle ever halts → `val_puzzle_acc = 0`.

Sanjin's `losses.py` hardcoded `q_loss_weight = 0.5`. This run tested whether restoring that single value recovers `val_puzzle_acc` from 0 toward the HF baseline of 0.7960.

## Setup

Single-line config edits to `configs/trm_official_maze_finetune.yaml` (everything else identical to the failed Phase 1 run on the same machine):

```diff
- q_loss_weight: 0.0
+ q_loss_weight: 0.5
- epochs: 100
+ epochs: 10
- log_interval: 20
+ log_interval: 2
- eval_interval: 20
+ eval_interval: 2
- save_interval: 20
+ save_interval: 2
```

Launched via `start.py` option 8 (post `dc09f0c` fix), seed 1, AdamW optimizer, HF init loaded with all 24/24 keys.

## Result

| Epoch | `val_puzzle_acc` | `val_cell_acc` | `avg_steps` | `q_halt_accuracy` | `train_exact_acc` | `q_halt_loss` |
|-------|------------------|-----------------|-------------|-------------------|--------------------|----------------|
|   2   | **0.0040**       | 0.9788          | **9.7**     | 0.8165            | **0.2871**         | 1.352          |
|   4   | 0.0000           | 0.9625          | 16.0        | 0.9990            | 0.0010             | 0.071          |
|   6   | 0.0000           | 0.9625          | 16.0        | 0.9998            | 0.0003             | 0.031          |

**Killed at epoch 6** per the brief's "< 0.1 → fix didn't take" rule.

## Interpretation

The `q_loss_weight=0.5` fix worked **transiently at epoch 2** then **collapsed by epoch 4**. Every signal that recovered at epoch 2 reverted to the failed-run pattern.

The transient recovery is real and interpretable:
- `avg_steps` 16 → 9.7: the halt head was actually halting at intermediate ACT steps
- `q_halt_accuracy` 0.999 → 0.8165: the head was learning real halt decisions, not the trivial "always predict continue" attractor (which gives 0.999 because most ACT steps mid-recurrence are correctly continued)
- `train_exact_acc` 0.0010 → 0.2871: the model was solving 28.7% of training puzzles end-to-end
- `val_puzzle_acc` 0.000 → 0.004: 4 out of 1000 test puzzles solved correctly

The collapse to "always continue" by epoch 4 happens because:
- `q_halt_loss` is BCE between halt logit and `seq_is_correct`
- Without `halt_exploration_prob > 0`, the halt head never explores halting at non-default times, so it can't observe the consequences of halting decisions in regimes it doesn't yet visit
- The local minimum of "predict negative logit everywhere → never halt → use full ACT budget → BCE target depends on step-16 correctness" is a strong attractor when the model is occasionally wrong (i.e., during fine-tuning when the LM head is shifting)

`q_halt_loss` shrinks from 1.35 → 0.03 as the head finds tighter "always continue" predictions. The Q-loss gradient is flowing — it just isn't pointing toward a useful halt policy without exploration to ground it.

## Refined diagnosis

The `q_loss_weight=0` decision (dz3tkge9 post-mortem) and the `halt_exploration_prob=0` decision (also dz3tkge9) **interact**. Removing both is too aggressive:
- Removing `q_loss_weight` alone: halt head receives no signal, drifts to trivial "always continue" (the failed Phase 1 pattern).
- Removing `halt_exploration_prob` alone (this run): halt head receives signal, recovers briefly, then the lack of exploration lets it find a different trivial attractor.

Sanjin's combination is `q_loss_weight=0.5` AND `halt_exploration_prob=0.1`. At least one of the two needs to be present for the halt head to stay calibrated through fine-tuning.

## Recommendation for the next experiment

Restore Sanjin's combination (both `q_loss_weight=0.5` and `halt_exploration_prob=0.1`) and re-run for 10 epochs at seed 1. If `val_puzzle_acc` at epoch 2 is again ~0.004 but **stays non-zero or climbs at epoch 4/6/8/10**, the diagnosis is fully confirmed: the dz3tkge9 post-mortem was right that exploration noise can be problematic, but the fix (zeroing both) creates a worse failure mode.

Cross-seed validation (M1 seed 0 / M3 seed 2) on the q_loss=0.5-only condition will tell us whether the transient recovery + collapse pattern is seed-independent (likely) or seed-specific (would change the story).

## Useful artifacts

- `/c/ml-trm-work/trm-att-maze-50ep-seed1/best.pt` — checkpoint from epoch 2 (val_puzzle_acc=0.004; the only non-collapsed checkpoint produced by this experiment)
- `/c/ml-trm-work/trm-att-maze-50ep-seed1/epoch_{2,4,6}.pt`
- `/c/ml-trm-work/trm-att-maze-50ep-seed1/trm_official_maze_train_log.csv`
- `/c/ml-trm-work/trm-att-maze-50ep-seed1/emissions.csv`
- Wandb: https://wandb.ai/shamykyzer/TRM/runs/1sg9856w

## Next: handoff to M1/M3

The brief at `docs/sprint_brief_m1_m3_recovery_2026-04-26.md` tells M1/M3 to run the same `q_loss_weight=0.5` validation on their seeds (0 and 2). Their results are still useful:
- If they also see transient-then-collapse: cross-seed reproduction of this finding.
- If they see continued recovery: seed-1 specific weakness, paper finding.

Either way, after their runs we should pivot to testing Sanjin's full combo (`q_loss=0.5` + `halt_exp=0.1`) on at least one seed before the deadline.
