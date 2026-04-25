# trm-att-maze-seed2 (started 2026-04-19)

TRM-Att baseline run on Maze-Hard 1k augmented dataset. Long
training, well below the TRM paper's reported ceiling at the
checkpoint counts captured here.

## Identity

- **Model**: TRM-Att, ~8.4 M params (`configs/trm_official_maze.yaml`)
- **Task**: maze-30x30-hard-1k-aug
- **Seed**: 2
- **Started**: 2026-04-19 17:19
- **Last checkpoint**: 2026-04-21 13:56 (epoch 100)

## Headline numbers

From `trm_official_maze_train_log.csv`:

| Epoch | val_cell_acc | val_puzzle_acc | best_puzzle_acc | elapsed_min |
|---:|---:|---:|---:|---:|
| 50 | 0.9497 | 0.1020 | 0.1130 | 1351.2 |
| 100 | 0.9595 | 0.1810 | 0.1810 | 2702.3 |

So 100 epochs in ~45 hours, hitting only **18.1 % puzzle accuracy**.
Cell accuracy is high (95.95 %) because most cells are walls; the
gap between cell and puzzle accuracy is exactly the "easy-metric"
dynamic flagged in [`/analysis_2026-04-23_distill_maze.md`](../../../analysis_2026-04-23_distill_maze.md).

The TRM paper reports ~85 % on Maze-Hard at much higher epoch
counts; this checkpoint is a *partial* training of the same
architecture, not a converged result.

## Weights on the release

Attached to GitHub Release `runs-snapshot-2026-04-25`:

- `trm-att-maze-seed2-best.pt` (41 MB) — best by `val_puzzle_acc`
- `trm-att-maze-seed2-epoch_50.pt` (68 MB)
- `trm-att-maze-seed2-epoch_100.pt` (68 MB)
- `trm-att-maze-seed2-milestone-10pct-epoch100.pt` (41 MB) — thesis snapshot

Download: https://github.com/shamykyzer/ML-TRM/releases/tag/runs-snapshot-2026-04-25

## How to continue training

```bash
python main.py \
  --mode train \
  --config configs/trm_official_maze.yaml \
  --seed 2 \
  --checkpoint <local-path>/trm-att-maze-seed2-epoch_100.pt
```
