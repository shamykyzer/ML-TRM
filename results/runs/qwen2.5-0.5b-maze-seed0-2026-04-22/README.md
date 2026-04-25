# qwen2.5-0.5b-maze-seed0 (started 2026-04-22)

Run #4 of the 6-run novelty matrix: Qwen2.5-0.5B with LoRA r=8 +
QLoRA nf4, fine-tuned on Maze-Hard 1k. Teacher for the
[`distill-maze-seed0-2026-04-23`](../distill-maze-seed0-2026-04-23/)
student.

## Identity

- **Model**: Qwen/Qwen2.5-0.5B + LoRA r=8 / alpha=16, QLoRA nf4 weights
- **Task**: maze-30x30-hard-1k-aug (`mask_non_path: true`)
- **Config**: `configs/llm_qwen_maze.yaml`
- **Seed**: 0
- **Hardware**: RTX 5070, i7-14700, Windows 11
- **Started**: 2026-04-22 (~21:36)
- **Finished**: 2026-04-22 23:40

## Headline numbers

From `qwen2.5_0.5b_maze_training_results.json` and `emissions.csv`:

| Metric | Value |
|---|---:|
| Wall-clock | 22 537 s (~6.26 hr) |
| Energy | **1.564 kWh** |
| CO2 | 0.372 kg |
| Best `val_puzzle_acc` | 1.0000 (epoch 10) |
| Best `val_cell_acc` | 1.0000 |

Per-epoch log only retains epoch 0 (val baseline) and epoch 10
(first `log_interval=10` write); the model converged inside that
window.

## Caveats

The teacher hits 100 % on what is meant to be a "hard" benchmark.
Two probable causes — `mask_non_path` metric shape and possible
augmentation leakage in the train/test split — are documented in
[`/analysis_2026-04-23_distill_maze.md`](../../../analysis_2026-04-23_distill_maze.md).
Treat the puzzle-accuracy number as honest mechanically but the
benchmark's headroom as the open question.

## Weights on the release

Attached to GitHub Release `runs-snapshot-2026-04-25`:

- `qwen2.5-0.5b-maze-seed0-latest.pt` (749 MB)

Download: https://github.com/shamykyzer/ML-TRM/releases/tag/runs-snapshot-2026-04-25

## How to use as a distillation teacher

```bash
python main.py \
  --mode distill \
  --config configs/llm_qwen_maze.yaml \
  --seed 0 \
  --checkpoint <local-path>/qwen2.5-0.5b-maze-seed0-latest.pt
```
