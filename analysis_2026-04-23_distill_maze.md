# Latest training run — distill-maze (2026-04-23)

## Summary

A 1-epoch, 9.6-minute, 43 Wh distillation reproduces the Qwen-maze
teacher's perfect maze accuracy at ~37x less energy and ~250x lower
inference latency. The numbers are real, but Maze-Hard-1k-aug looks
saturated for this teacher (1.0 puzzle-acc with K-vote flat across
K=1..16), so the comparison cannot support the proposal's
TRM-vs-LLM-efficiency or K-vote-helps claims as written. Treat the
distill result as honest mechanically, but the benchmark choice as
the bottleneck.

## What ran

- **wandb run**: `09lcqw5y`, name `llm_qwen2_5_0_5b_maze_seed0_STU-CZC5277FFN_1776948145`
- **Started**: 2026-04-23 13:42:26 UTC, ran 578 s
- **Commit**: `2f9993f` (fix: move .to(device) before load_state_dict for QLoRA)
- **CLI**: `python main.py --mode distill --config configs/llm_qwen_maze.yaml --seed 0 --checkpoint C:/ml-trm-work/novelty-qwen-maze-seed0/qwen2.5_0.5b_maze_latest.pt`
- **Hardware**: RTX 5070 (12 GB), i7-14700, Windows 11, CUDA 13.0
- **Rig**: `TRM_RIG=3` (the rig-3 slice = qwen-maze + distill-maze)

### Models

| | Teacher (qwen-maze) | Student (distill-maze) |
|---|---|---|
| Architecture | Qwen2.5-0.5B + LoRA r=8, QLoRA nf4 | 3-layer transformer, d_model=256, n_heads=4, ff=1024 |
| Parameters | ~500 M | ~2.4 M |
| Precision | bf16 forward, nf4 weights | bf16 |

### Training config (distill)

- batch_size 2, grad_accum 8 (effective 16), lr 5e-5, AdamW, weight_decay 0.01
- distill_alpha 0.7, distill_temperature 4
- epochs cap 30, log_interval 10, save_interval 20
- wall-clock cap from `TRM_MAX_TRAIN_SECONDS` (env var) — fired before epoch 2
- dataset `data/maze-30x30-hard-1k-aug`, mask_non_path: true, num_workers 0

## Outcome

The trainer halted cleanly with
`[wall-clock] budget exhausted before epoch 2/30 — halting.` Exactly
**1 epoch of distillation** completed before `TRM_MAX_TRAIN_SECONDS`
tripped. The post-loop save still wrote
`C:/ml-trm-work/novelty-distill-maze-seed0/distill_maze_latest.pt`
(10.4 MB) and the K-vote sweep at 14:56–14:59 evaluated that
checkpoint.

### Numbers

| | Teacher | Student | Ratio |
|---|---:|---:|---:|
| Train wall-clock | 22 537 s (~6.3 hr) | **576 s (~9.6 min)** | ~39x faster |
| Train energy | 1.564 kWh | **0.0428 kWh** | ~37x less |
| Train CO2 | 0.372 kg | 0.010 kg | ~37x less |
| Best puzzle acc | 1.0000 (epoch 10/30) | **1.0000 (epoch 1/30)** | tied |
| Inference K=1 | 3.36 s/puzzle, 75 Wh / 1000 puz | **14 ms/puzzle, 0.21 mWh / 1000 puz** | ~240x faster, ~360x less |

### K-vote sweep (distilled student, 1000 maze test puzzles)

| K | puzzle_acc | cell_acc | mean_latency_ms | kWh/puzzle |
|---:|---:|---:|---:|---:|
| 1  | 1.0000 | 1.0000 | 14.358 | 2.2e-7 |
| 2  | 1.0000 | 1.0000 | 16.870 | 2.3e-7 |
| 4  | 1.0000 | 1.0000 | 14.148 | 2.1e-7 |
| 8  | 1.0000 | 1.0000 | 14.639 | 2.1e-7 |
| 16 | 1.0000 | 1.0000 | 13.472 | 2.0e-7 |

K-vote curve is flat — saturated at K=1 — so the Pareto front for the
student is degenerate (K=1 dominates on accuracy, latency, energy).

## Why this result is suspicious

**The numbers themselves are real** (GPU pulled ~220 W at 95 % util
for 9.6 min; CodeCarbon recorded the energy; the K-vote script
re-evaluated 1000 mazes and the latencies are consistent). What's
suspect is the *interpretation*, not the measurement.

1. **Maze-Hard at 100 % is much higher than the literature.** The TRM
   paper reports ~85 % on Maze-Hard with a purpose-built recursive
   architecture. A LoRA-finetuned 500 M LLM hitting 100 %, and a
   2.4 M student matching it in one epoch, is well above published
   numbers.
2. **`mask_non_path: true` is the most likely explanation.** If
   accuracy is computed only on path cells (everything else
   ignored), and most of a 30x30 maze is walls, the metric is much
   easier than "did you solve the maze". This wants a quick read of
   the eval code to confirm what `puzzle_acc` actually counts.
3. **Augmentation leakage is the second hypothesis.** The dataset is
   `maze-30x30-hard-1k-aug`. If the augmented copies (rotations /
   reflections) were generated *before* the train/test split,
   "test" mazes are just rotated versions of training mazes and the
   model is recognising rather than solving. Both `train/` and
   `test/` show 1000 puzzles each — they're not literally identical,
   but they may share underlying layouts.
4. **K-vote being completely flat at 1.0** is a symptom of either of
   the above — a non-trivial benchmark would leave at least a few
   errors at K=1.

## Issues with the experiment matrix

1. **iso-time matrix is incomplete.** Only rig-3's qwen-maze
   succeeded in `iso_time_results-rig3.csv`; distill-maze still
   shows `error / exit code 1` from the older QLoRA-strict-load
   attempt at 13:32 (commit `2f9993f`). The fix `acb0cd4`
   (`strict=False`) landed *after* the wandb run, but the wandb run
   itself proceeded — the CSV was never re-written after the
   successful re-run, so the canonical record under-reports what
   actually happened. Rig-1 (TRM-MLP-sudoku, TRM-Att-maze) and
   rig-2 (qwen-sudoku → distill-sudoku) are entirely absent.
   `iso_time_acc_vs_kwh.png` is therefore a single-point plot.
2. **qwen-maze K-vote is incomplete.**
   `k_vote_runs/qwen-maze/emissions.csv` only contains a K=1 row
   (3 359 s, 75 Wh). K∈{2,4,8,16} were not run for the LLM, so
   `k_vote_results.csv` has no qwen rows and the report cannot yet
   compare student-with-K-vote vs teacher-with-K-vote.
3. **Empty per-epoch CSV.** `distill_maze_train_log.csv` is just a
   header. The trainer only writes when `(epoch+1) % log_interval
   == 0` (=10), and the wall-clock cap fired after epoch 1. For
   short-budget runs the log_interval should be lowered or epoch 1
   should always log.
4. **Stale stdout.log.** `C:/ml-trm-work/novelty-distill-maze-seed0/stdout.log`
   (13:32) is the *pre-fix* error trace and doesn't correspond to
   the wandb run. Easy to confuse with current state.

## What to verify before quoting these numbers

1. Confirm what `puzzle_acc` counts (mask_non_path effect on the
   metric, not just on the loss).
2. Diff a handful of `test/all__inputs.npy` rows against
   `train/all__inputs.npy` to rule out augmentation leakage. If any
   test maze is a rotation of a train maze, the result is
   memorisation, not solving.
3. If both come back clean, the conclusion is "Maze-Hard 1k is too
   easy for this teacher" and the next-step is to switch to
   Sudoku-Extreme — where the proposal expects the gap to show and
   K-vote is more likely to be informative.

## Suggested next steps (priority order)

1. Run the qwen-maze K-vote sweep at K=2,4,8,16 on rig-3 (~7 hr,
   mostly K=16). Without the LLM half, the K-vote Pareto plot is
   half-empty.
2. Switch attention to rig-2 (qwen-sudoku → distill-sudoku) and
   rig-1 (TRM-MLP-sudoku, TRM-Att-maze). Sudoku is where the
   benchmark should not saturate.
3. After (1) or (2), re-run `scripts/run_novelty_aggregate.py` so
   the canonical `iso_time_results.csv` and `k_vote_results.csv`
   reflect the actual successful runs (not the rig-3 stale-error
   row for distill-maze).
4. Either lower distill `log_interval` to 1 for short-budget runs
   or always log epoch 1, so a wall-clock-halted run still leaves
   a non-empty train log.

## File pointers

- Training run dir: `C:/ml-trm-work/novelty-distill-maze-seed0/`
- Teacher checkpoint: `C:/ml-trm-work/novelty-qwen-maze-seed0/qwen2.5_0.5b_maze_latest.pt`
- wandb run: `wandb/run-20260423_134226-09lcqw5y/`
- K-vote results: `results/novelty/k_vote_results.csv`,
  `results/novelty/k_vote_runs/`
- Iso-time results (rig-3): `results/novelty/iso_time_results-rig3.csv`
- Plots: `results/novelty/iso_time_*.png`, `results/novelty/k_vote_*.png`
