# ML-TRM Scripts Index

Every script under `scripts/` grouped by purpose. Most are called from
`start.py` / `main.py`, but they all work standalone too — the invocation
path is `python scripts/<name>.py` from the repo root (with the venv
activated, or via `<venv>/Scripts/python.exe`).

## Setup / transfer-learning (HF checkpoint remap)

The paper's published HuggingFace checkpoints don't load directly into
TRMOfficial — they need a key-level remap into our local tensor names.
Generic driver + per-task variants so fine-tuning can start from the
paper accuracy instead of random init.

- `remap_hf_checkpoint.py` — Generic driver for the ARC-AGI-2 checkpoint.
- `remap_sudoku_mlp.py` — Per-task remap for Sudoku-Extreme MLP-t.
- `remap_sudoku_att.py` — Per-task remap for Sudoku-Extreme attention.
- `remap_maze.py` — Per-task remap for Maze-Hard 30x30.
- `verify_remap_loads.py` — Loads the generic remap into a fresh
  TRMOfficial and checks every expected key is present with no
  unexpected extras.
- `verify_sudoku_mlp_loads.py` / `verify_sudoku_att_loads.py` /
  `verify_maze_loads.py` — Per-task variants of the verifier.
- `inspect_hf_checkpoint.py` — Dump the HF checkpoint's state-dict keys
  (for figuring out how to extend the remap table).

## Training / fleet wrappers

- `run_seed.sh` / `run_seed.ps1` — Bash / PowerShell wrapper for one
  `(task, seed)` pair. Preferred over direct `python main.py ...`
  because it auto-sets `TRM_CHECKPOINT_DIR` and guards against OneDrive
  paths.
- `auto_push.sh` / `auto_push.ps1` — Hourly git push of checkpoints and
  logs (used by the 6-machine fleet for centralised monitoring).
- `hourly_checkpoint_push.sh` — Same as above, cron-friendly one-shot.

## Evaluation

- `eval_hf_checkpoints.py` — Regime A. Evaluates all three published
  Sanjin2024 checkpoints; writes `results/hf_eval_*.json`.
- `eval_llm_checkpoint.py` — Evaluate a single LLM checkpoint with the
  shift-corrected eval (Fix B, matches local test_accuracy).
- `strict_eval.py` — Extra eval that double-checks the puzzle_acc number
  using the `is_valid_sudoku_solution` / `is_valid_maze_path` helpers
  from `src.data.encoding`.

## Aggregation / reporting

- `aggregate_metrics.py` — Walks `experiments/<task>/*_train_log.csv`
  and `emissions.csv` to produce `results/summary.csv` (one row per
  task with best_val_puzzle_acc, CO2, wall time, etc.).
- `aggregate_wandb_runs.py` — Cross-machine fleet summary: fetches
  every wandb run in the project, writes `results/trm_runs_overview.csv`
  and `results/history_<task>_best.csv`.
- `publish_wandb_metrics_report.py` — Pushes `wandb_metrics_glossary.md`
  as a formatted wandb Report so collaborators can read it in-browser.

## Plots / figures

- `plot_results.py` — Renders the 5 thesis-report figures from
  `results/summary.csv` + per-experiment train logs.
- `plot_sudoku_mlp_overfit.py` — Overfit narrative: peak + decline
  annotations.
- `plot_sudoku_att_story.py` — Collapse narrative for sudoku-att's
  from-scratch run.

## Diagnostics / forensics

- `sanity_check.py` — Fast "does the repo run?" probe (loads one
  batch, runs one step, exits).
- `diagnose_missing_keys.py` — Lists state-dict keys missing or
  unexpected when loading a checkpoint.
- `diagnose_real_weights.py` — Checks loaded weights aren't all-zero
  or near-random (catches silent remap breakage).
- `forensic_maze_corruption.py` — Post-mortem for the corrupted
  maze `best.pt` incident; regenerates the probe tensors used for the
  root-cause investigation.
- `halt_sweep.py` — Sweep the TRM halt parameters and log to wandb.

## Dataset generation

- `generate_ood_mazes.py` — Generate 1,000 held-out 30x30 mazes via
  recursive backtracker for OOD generalization tests.
