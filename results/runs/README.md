# Training-run snapshots

Light, version-controlled artifacts from completed training runs. The
matching weight files (`*.pt`) live on the GitHub Release tagged
`runs-snapshot-2026-04-25` rather than in git history, because they
range from 10 MB to 749 MB and would bloat the repo.

## Layout

One directory per run, named `<model>-<task>-seed<N>-<YYYY-MM-DD>`,
where the date is the training-start date. Each directory holds:

- `emissions.csv` — CodeCarbon per-process energy / CO2 trace
- `<tag>_train_log.csv` — per-`log_interval` epoch snapshots
- `<tag>_*results*.json` — final summary JSON
- `README.md` — one-page run summary, headline numbers, link to the
  `.pt` asset on the release, and pointers back to any deeper
  analysis document in the repo

Forensic logs (`stdout.log`, `smoke_test.log`) are included only when
they document a non-trivial event (e.g., a failed-then-fixed
checkpoint load).

## Index

| Run | Model | Task | Seed | Start | Best puzzle acc | Train kWh | Notes |
|---|---|---|---:|---|---:|---:|---|
| `trm-att-maze-seed2-2026-04-19/` | TRM-Att 8.4 M | Maze-Hard 1k | 2 | 2026-04-19 | 0.181 (epoch 100) | (see emissions.csv) | Long baseline — far below TRM paper ceiling at 100 epochs. |
| `qwen2.5-0.5b-maze-seed0-2026-04-22/` | Qwen2.5-0.5B + LoRA r=8 | Maze-Hard 1k | 0 | 2026-04-22 | 1.000 (epoch 10) | 1.564 | Teacher for the distill run below. Hits 1.0 — see analysis caveats. |
| `distill-maze-seed0-2026-04-23/` | Distilled student 2.4 M | Maze-Hard 1k | 0 | 2026-04-23 | 1.000 (1 epoch) | 0.043 | Halted by `TRM_MAX_TRAIN_SECONDS` before epoch 2; full analysis in [`/analysis_2026-04-23_distill_maze.md`](../../analysis_2026-04-23_distill_maze.md). |

## Where the weights are

The corresponding checkpoint files are attached to the GitHub
Release `runs-snapshot-2026-04-25`. Direct download via the
release page:

- https://github.com/shamykyzer/ML-TRM/releases/tag/runs-snapshot-2026-04-25

Each per-run README lists the exact filename(s) it expects.

## How to reproduce a run

The configs that drove these runs are unchanged from the repo. To
reproduce one, point `main.py --config` at the same YAML referenced
in the per-run README, set the same seed, and let it train. To
*continue* from a snapshot, download the matching `.pt` from the
release and pass it via `--checkpoint`.
