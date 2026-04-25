# distill-maze-seed0 (2026-04-23)

Run #6 of the 6-run novelty matrix: distillation from the
[`qwen2.5-0.5b-maze-seed0-2026-04-22`](../qwen2.5-0.5b-maze-seed0-2026-04-22/)
teacher into a 2.4 M-param student. **The latest training run as of
this snapshot.** Full analysis (numbers, suspicions, what-to-verify)
in [`/analysis_2026-04-23_distill_maze.md`](../../../analysis_2026-04-23_distill_maze.md).

## Identity

- **Model**: 3-layer transformer, d_model=256, n_heads=4, ff_hidden=1024 (~2.4 M params)
- **Task**: maze-30x30-hard-1k-aug
- **Config**: `configs/llm_qwen_maze.yaml` (with distill flags)
- **Seed**: 0
- **Hardware**: RTX 5070, i7-14700, Windows 11
- **wandb run id**: `09lcqw5y`
- **Commit**: `2f9993f`
- **Started**: 2026-04-23 13:42:26 UTC
- **Halted**: 2026-04-23 13:52:04 (578 s, before epoch 2/30, by `TRM_MAX_TRAIN_SECONDS`)

## Headline numbers

| Metric | Value |
|---|---:|
| Wall-clock | 576 s (~9.6 min) |
| Energy | **0.0428 kWh** |
| CO2 | 0.0102 kg |
| Best `val_puzzle_acc` (post-K-vote) | 1.0000 |
| K-vote sweep K∈{1,2,4,8,16} puzzle_acc | 1.0 / 1.0 / 1.0 / 1.0 / 1.0 |
| Inference K=1 latency | 14 ms / puzzle |
| Inference K=1 energy | 2.2e-7 kWh / puzzle |

Compared to the teacher: ~37x less training energy, ~250x lower
inference latency, identical post-training accuracy. K-vote curve is
flat at 1.0 — the maze benchmark is saturated, see analysis.

## Why the per-epoch CSV is empty

`distill_maze_train_log.csv` is just a header row. The trainer only
writes when `(epoch+1) % log_interval == 0` (`log_interval=10`), and
the wall-clock cap fired after epoch 1. The post-loop save still
wrote the `.pt`; that's the file the K-vote sweep evaluated.

## Forensic files

- `stdout-failed-load-2026-04-23T13-32.log` — stdout from the
  earlier (13:32) launch attempt that errored with
  `Unexpected key(s) in state_dict ... bitsandbytes__nf4`. Fix
  landed as commit `acb0cd4` (`strict=False` on QLoRA load).
- `smoke_test.log` — partial state-dict key dump preserved from the
  same failed-then-fixed load sequence. Useful only as evidence of
  the QLoRA serialisation layout.

## Weights on the release

Attached to GitHub Release `runs-snapshot-2026-04-25`:

- `distill-maze-seed0-latest.pt` (10 MB)

Download: https://github.com/shamykyzer/ML-TRM/releases/tag/runs-snapshot-2026-04-25

## How to evaluate K-vote on it

```bash
python scripts/run_novelty_k_vote.py \
  --rig 3 --seed 0 \
  --checkpoint <local-path>/distill-maze-seed0-latest.pt
```

Or via the `start.py` menu — option 5 ("K-vote existing
checkpoints"), added in commit `527297e`.
