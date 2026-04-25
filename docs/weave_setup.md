# Weave + W&B Reporting Setup

How the repo uses Weave (per-puzzle traces, evaluations, models) on top of
the existing wandb integration. This is the operational doc — for what each
*metric* means, see [`wandb_metrics_glossary.md`](wandb_metrics_glossary.md).

## What's wired up

| Feature | Where | Trigger | Default |
|---|---|---|---|
| **Outer eval trace** (one per `evaluate()` call) | `trainer_official.py:1036` (`@weave_op` on `evaluate`) | every `eval_interval` epochs | always on when `use_weave: true` |
| **Per-puzzle eval traces** (sampled) | `trainer_official.py` `_trace_eval_puzzle` | sampled stride during eval loop | `eval_trace_sample_size: 100` puzzles per eval |
| **Mid-run regression alert** | `trainer_official.py` `_maybe_alert_on_regression` | first eval where val drops > threshold below running max | `regression_alert_threshold: 0.03` (3 pp) |
| **`weave.Model` wrappers** | `src/evaluation/weave_models.py` | manual — used by Evaluation script + Playground | n/a |
| **Cross-checkpoint Evaluation** | `scripts/weave_compare_checkpoints.py` | manual CLI | n/a |
| **Auto-rebuilding runs Report** | `scripts/publish_runs_overview_report.py` | manual CLI | n/a |

## 1. Per-puzzle eval traces

Every `eval_interval` epochs, the trainer samples N puzzles uniformly across
the test split and emits one `@weave.op` trace per sampled puzzle, capturing:

- input puzzle (81 stored tokens)
- ground-truth label
- model prediction
- ACT halt step (1..`halt_max_steps`, or `halt_max_steps` if it never halted)
- per-cell correctness count
- whether the whole puzzle is correct

In the Weave UI (`wandb.ai/<entity>/<project>/weave/traces`) each trace
shows up as a child of the parent `evaluate()` call. You can filter by
`puzzle_correct == false` to see exactly which puzzles the model got wrong
in any epoch, or by `halt_step > 10` to see hard cases.

**Tuning the sample size** (per-config knob):
```yaml
training:
  eval_trace_sample_size: 100   # 0 disables; 100 ≈ 1 % of the 6.6 k Sudoku test
```
Storage cost: 100 traces × ~2 KB each × ~10 evals/run ≈ 2 MB / run on the
wandb side. Negligible relative to checkpoint upload bandwidth.

## 2. Mid-run regression alert

After every eval, the trainer checks if `val_puzzle_acc` dropped more than
`regression_alert_threshold` below `self.best_acc`. First time the drop is
seen it calls `wandb.alert()` with `WARN` level — fires once per run so
you don't get spammed if the run keeps drifting.

**Where the alert shows up:**
- wandb run page, "Alerts" tab
- Slack DM if you've set up the wandb-Slack integration (Settings →
  Personal → Slack)
- Email if email alerts are on (Settings → Personal → Email)
- Always: a `[ALERT]` line in stdout via tqdm.write

**Tuning** (per-config knob):
```yaml
training:
  regression_alert_threshold: 0.03   # 0 disables; 0.03 = 3 pp
```

This alone would have caught `dz3tkge9` at epoch 50 instead of 23 hours later.

### Optional: matching UI-side Monitor

You can complement the in-trainer alert with a wandb Monitor (no code). In
the wandb UI:

1. Open the project → **Monitors** tab → **New Monitor**.
2. Filter: `state = running`, `tags contains trm_official_sudoku`.
3. Metric: `val/puzzle_acc`.
4. Trigger: `current_value < max_value - 0.03` (i.e. > 3 pp below running max).
5. Notify: Slack channel or your email.

The Monitor and the in-trainer alert are redundant on purpose — the trainer
alert is mostly stdout/tqdm visibility (catches the case where wandb sync
is delayed); the Monitor is the durable cross-run guard that survives even
if a run crashes before its alert can fire.

## 3. `weave.Model` wrappers

`src/evaluation/weave_models.py` provides `TRMSudokuModel(weave.Model)`. Each
instance binds to one checkpoint file and exposes `.predict(puzzle)` as a
`@weave.op`. Two consumers:

### a) `weave.Evaluation` for cross-checkpoint comparison

```bash
# Auto-discover hf_checkpoints/Sudoku-Extreme-mlp + every C:/ml-trm-work/sudoku-mlp-*/best.pt
python scripts/weave_compare_checkpoints.py

# Or pass explicit name=path pairs
python scripts/weave_compare_checkpoints.py \
    hf-init=hf_checkpoints/Sudoku-Extreme-mlp/remapped_for_local.pt \
    seed4-ep480=C:/ml-trm-work/sudoku-mlp-seed4/epoch_480.pt
```

Builds a `weave.Dataset` from the first 200 test puzzles (override with
`--sample-size`), runs `weave.Evaluation` against each model with three
scorers (`cell_accuracy`, `puzzle_correct`, `halt_step`), and prints the
leaderboard URL: `wandb.ai/<entity>/<project>/weave/evaluations`.

This is the right replacement for the manual CSV-join in `findings.md` §5.2
once the fine-tune rerun finishes.

### b) Playground

Each `TRMSudokuModel` instance auto-publishes on first `predict()` call. After
that it's selectable in the Playground UI
(`wandb.ai/<entity>/<project>/weave/playground`) — paste a puzzle as JSON,
pick the model, see the prediction. Useful for the report's qualitative
"how did the distilled student fail differently" exhibit.

## 4. Auto-rebuilding "TRM runs overview" Report

```bash
# Requires: pip install wandb_workspaces (not in requirements.txt by default —
# only needed for Report publishing)
python scripts/publish_runs_overview_report.py
python scripts/publish_runs_overview_report.py --project TRM --top-n 30
```

Pulls every run in the configured project via `wandb.Api()`, formats a
markdown table + per-run sections, saves as a wandb Report. Re-run any time
to refresh — the latest Report version is always at the top of
`wandb.ai/<entity>/<project>/reports/`.

Pairs with `findings.md` (narrative + decisions, hand-edited) and
`results/trm_runs_overview.csv` (canonical CSV, generated by
`aggregate_wandb_runs.py`).

## Disabling everything

Each feature is independently togglable:

```yaml
training:
  use_wandb: false                  # turns off wandb entirely (Weave too)
  use_weave: false                  # wandb on, weave off
  eval_trace_sample_size: 0         # outer eval trace stays; per-puzzle off
  regression_alert_threshold: 0     # alert disabled
```

## Files added by this setup

- `src/evaluation/weave_models.py` — `TRMSudokuModel` wrapper
- `scripts/weave_compare_checkpoints.py` — `weave.Evaluation` runner
- `scripts/publish_runs_overview_report.py` — auto-rebuilding Report
- `docs/weave_setup.md` — this file

## Files modified

- `src/utils/config.py` — added `regression_alert_threshold`,
  `eval_trace_sample_size`
- `src/training/trainer_official.py` — alert + per-puzzle traces wiring
- `docs/INDEX.md` — pointer to this doc
