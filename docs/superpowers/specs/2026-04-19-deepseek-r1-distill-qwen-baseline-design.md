# Design: DeepSeek-R1-Distill-Qwen-1.5B LLM Baseline

**Date:** 2026-04-19
**Status:** Approved (brainstorming complete, plan pending)
**Related:** `configs/llm_qwen.yaml`, `src/training/trainer_llm.py`, `src/evaluation/aggregate.py`

## Goal

Add DeepSeek-R1-Distill-Qwen-1.5B to the existing LLM baseline fleet, on both
Sudoku-Extreme and Maze-Hard at 30 epochs each. The aim is not to beat TRM —
every other LLM baseline in the fleet already fails at puzzle-level accuracy.
The aim is to produce *measurable, cite-able evidence that the model is not
learning the task*, so the dissertation's "structured reasoning is out of
reach for causal-LM fine-tuning at this scale" claim rests on more than a
visually flat plot.

Three shared trainer additions make that evidence concrete: a pre-training
evaluation at epoch 0 (so plateau plots are anchored, not "starts at
epoch 10"), an `lm/loss` alias logged alongside `train/loss` (so the metric
name matches the claim), and an optional per-step loss logging flag (so the
DeepSeek plateau figure shows within-epoch noise floor). A single
`loss_delta_pct` column in `results/summary.csv` quantifies "how much did
val_loss drop" into one number per run, cite-able in the Discussion section.

## Scope

In scope:
- Two new YAML configs (`llm_deepseek.yaml`, `llm_deepseek_maze.yaml`)
- Three additive changes to `src/training/trainer_llm.py` (shared with all LLM baselines)
- One field added to `TrainingConfig` in `src/utils/config.py`
- One column added to `results/summary.csv` via `src/evaluation/aggregate.py`
- Three new test cases in `tests/test_aggregate.py`
- Post-run updates to README, wandb metrics glossary, wandb Report

Explicitly out of scope:
- Adding Llama-3.2-3B (considered in brainstorming, dropped by user)
- DeepSeek-R1-Distill-Qwen-7B / -Llama-8B (infeasible on the 12 GB RTX 5070 fleet)
- Refactoring `baseline_llm.py` (the `deepseek` branch on lines 53-54 already works)
- Renaming any existing wandb keys (aliases only, no renames)

## Success criteria

1. Both runs complete all 30 epochs on a 5070 without OOM.
2. Wandb dashboards for both runs show non-empty `val/loss` at `step=0` and at `step=30`.
3. `results/summary.csv` contains two new rows, `deepseek_r1_distill_qwen_1_5b_sudoku` and
   `deepseek_r1_distill_qwen_1_5b_maze`, each with a numeric `loss_delta_pct` cell.
4. The `loss_delta_pct` column is present and numeric (including possibly
   negative) for both DeepSeek rows in `results/summary.csv`. The *value* is
   an experimental finding, not a completion gate — even a surprising
   `loss_delta_pct > 20%` (DeepSeek actually learning) would be a valid,
   dissertation-worthy result, not a spec failure.
5. No regression in existing LLM baseline logging — all prior wandb keys
   (`train/loss`, `val/loss`, `val/puzzle_acc`, `val/cell_acc`,
   `val/accuracy`, `val/exact_accuracy`, `train/elapsed_min`) continue to
   appear at the same cadence and with the same values.
6. `tests/test_aggregate.py` passes with three new test cases covering the
   new column.

## Design

### Architecture overview

No architectural change. The existing `BaselineLLM` wrapper in
`src/models/baseline_llm.py` already dispatches on model-name substring,
including a `"deepseek"` branch at lines 53-54 that selects
`["q_proj", "k_proj", "v_proj"]` as PEFT target modules — correct for the
Qwen-based distill variant. The integration is purely additive: two new
YAMLs, three small diffs to shared code.

### Files added

**`configs/llm_deepseek.yaml`** (sudoku run):

```yaml
model:
  model_type: llm_finetune
  vocab_size: 11
  seq_len: 81
  num_classes: 11
  llm_name: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  lora_r: 8
  lora_alpha: 16
  use_qlora: false

training:
  lr: 0.00005
  betas: [0.9, 0.999]
  weight_decay: 0.01
  warmup_steps: 500
  batch_size: 8
  grad_accum_steps: 2
  epochs: 30
  early_stop_patience: 0
  log_per_step: true
  use_wandb: true
  wandb_project: TRM-LLM
  wandb_entity: ""
  log_interval: 10
  save_interval: 50

data:
  dataset: sudoku
  data_dir: data/sudoku-extreme-full
  num_workers: 4

seed: 42
device: cuda
checkpoint_dir: models/llm
experiment_dir: experiments/llm
```

**`configs/llm_deepseek_maze.yaml`** (maze run):

```yaml
model:
  model_type: llm_finetune
  vocab_size: 6
  seq_len: 900
  num_classes: 6
  llm_name: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  lora_r: 8
  lora_alpha: 16
  use_qlora: true
  use_gradient_checkpointing: true

training:
  lr: 0.00005
  betas: [0.9, 0.999]
  weight_decay: 0.01
  warmup_steps: 500
  batch_size: 1
  grad_accum_steps: 16
  epochs: 30
  early_stop_patience: 0
  log_per_step: true
  use_wandb: true
  wandb_project: TRM-LLM
  wandb_entity: ""
  log_interval: 10
  save_interval: 50

data:
  dataset: maze
  data_dir: data/maze-30x30-hard-1k-aug
  num_workers: 4

seed: 42
device: cuda
checkpoint_dir: models/llm
experiment_dir: experiments/llm
```

Key choices with their rationale:

- **`lr: 5e-5`** — matches every other YAML in the fleet. Defensibility: a
  reviewer can't claim "DeepSeek didn't learn because you used different
  settings" when the LR is identical to Qwen-0.5B (which learned to 19%
  cell accuracy).
- **`early_stop_patience: 0`** — early stopping is deliberately *disabled*.
  The plateau *is* the result; stopping at epoch 5 because the model isn't
  improving would hide the full 30-epoch curve we want to plot.
- **`log_per_step: true`** — emits `lm/step_loss` every micro-batch. Produces
  a dense plot showing within-epoch noise floor, making the "nothing changed"
  claim visually unmissable rather than inferred from 3 data points at
  epochs 10/20/30.
- **Sudoku `batch=8, accum=2`** — effective batch 16 preserves fleet parity
  for optimizer dynamics comparability. 1.5B fp16 at seq=81 fits the 5070
  comfortably; could push `batch=16, accum=1` for ~2× speed after a smoke
  test, but default errs toward VRAM headroom.
- **Maze `batch=1, accum=16` + QLoRA + gradient checkpointing** — mirrors
  `llm_llama_maze.yaml` (1.2B). DeepSeek-1.5B at seq=900 is larger still, so
  these constraints are the baseline, not a conservative choice.

### Files modified

#### `src/utils/config.py`

Add one field to the `TrainingConfig` Logging block (insert after
`eval_interval: int = 0` at line 100):

```python
# Per-step LM loss logging. When True, every gradient-accumulation
# micro-batch emits `lm/step_loss` to wandb. Off by default — enabling
# produces 1000-4000 points per run which clutters the default dashboard.
# Turn on for runs where the plateau claim needs within-epoch visibility
# (e.g. DeepSeek-R1-Distill-Qwen-1.5B baseline runs).
log_per_step: bool = False
```

Backwards-compatible — existing YAMLs inherit the `False` default.

#### `src/training/trainer_llm.py`

Three additions, all additive-only (no renames, no semantic changes to
existing code).

**Change A — Pre-training evaluation at step 0.** Insert in `train(self)`
immediately after `self.carbon.start()` and before the epoch loop
(currently line 99 area):

```python
# Baseline eval at step 0, before any gradient update. Anchors the plateau
# plot: if val metrics at step 0 ≈ step N, the model never moved. Without
# this anchor, the first data point in prior LLM runs is at step=log_interval
# (e.g. 10), and reviewers can fairly ask "how do we know it didn't briefly
# learn and then stall?" — this baseline answers that definitively.
val_metrics_initial = self.evaluate()
tqdm.write(
    f"[baseline] Epoch 0/{self.tc.epochs} | "
    f"ValLoss: {val_metrics_initial['loss']:.4f} | "
    f"Puzzle: {val_metrics_initial['puzzle_acc']:.4f} | "
    f"Cell: {val_metrics_initial['cell_acc']:.4f}"
)
self._append_log([
    0, "",  # no train_loss at epoch 0 — no gradient steps taken yet
    f"{val_metrics_initial['loss']:.4f}",
    f"{val_metrics_initial['puzzle_acc']:.4f}",
    f"{val_metrics_initial['cell_acc']:.4f}",
    "0.0",
])
if self.use_wandb:
    wandb.log(
        {
            "val/loss": val_metrics_initial["loss"],
            "val/lm_loss": val_metrics_initial["loss"],
            "val/puzzle_acc": val_metrics_initial["puzzle_acc"],
            "val/cell_acc": val_metrics_initial["cell_acc"],
            "val/accuracy": val_metrics_initial["cell_acc"],
            "val/exact_accuracy": val_metrics_initial["puzzle_acc"],
        },
        step=0,
    )
```

The CSV header in `_init_log` (line 87-90) already includes a "loss" column
in position 2 that the epoch-0 row fills with `""` — blank-cell convention
matches how `attach_efficiency_metrics` handles zero-correct runs.

**Change B — `lm/loss` and `val/lm_loss` aliases.** In the periodic
`wandb.log` call currently at lines 138-149, add two new keys (same scalar,
aliased names):

```python
wandb.log(
    {
        "train/loss": metrics["loss"],
        "lm/loss": metrics["loss"],          # NEW: explicit semantic alias
        "val/loss": val_metrics["loss"],
        "val/lm_loss": val_metrics["loss"],  # NEW: explicit semantic alias
        "val/puzzle_acc": val_metrics["puzzle_acc"],
        "val/cell_acc": val_metrics["cell_acc"],
        "val/accuracy": val_metrics["cell_acc"],
        "val/exact_accuracy": val_metrics["puzzle_acc"],
        "train/elapsed_min": elapsed,
    },
    step=epoch + 1,
)
```

No rename — every existing wandb panel continues to function on
`train/loss` / `val/loss`. New panels can use `lm/loss` for
self-documenting nomenclature.

**Change C — Optional per-step loss logging.** In `_train_epoch`, inside
the micro-batch for-loop right after `outputs.loss.item()` is computed
(around line 214):

```python
if self.use_wandb and self.tc.log_per_step:
    wandb.log({"lm/step_loss": outputs.loss.item()})
```

Gated on the new `TrainingConfig.log_per_step` field — defaults off, only
opted-in by the two DeepSeek YAMLs.

#### `src/evaluation/aggregate.py`

Add `loss_delta_pct` (and its two inputs) to the `parse_train_log` return
dict. In the function body, add two trackers alongside the existing ones
(around line 135):

```python
initial_val_loss: float | None = None
final_val_loss: float | None = None
```

Inside the row-iteration loop, capture val_loss at epoch 0 and at max epoch:

```python
vl = _to_float(row.get("val_loss"))
if epoch == 0 and vl is not None:
    initial_val_loss = vl
if epoch == max_epoch and vl is not None:
    final_val_loss = vl
```

At the bottom of the function, compute the delta and add to the returned
dict (three new keys):

```python
if (
    initial_val_loss is not None
    and final_val_loss is not None
    and initial_val_loss > 0
):
    loss_delta_pct: float | str = (
        (initial_val_loss - final_val_loss) / initial_val_loss * 100.0
    )
else:
    loss_delta_pct = ""  # blank cell for runs pre-dating the baseline hook

return {
    # ... existing fields ...
    "initial_val_loss": initial_val_loss if initial_val_loss is not None else "",
    "final_val_loss": final_val_loss if final_val_loss is not None else "",
    "loss_delta_pct": loss_delta_pct,
}
```

Formula rationale: `(initial − final) / initial × 100` gives a single
percentage per run. Using val_loss (not train_loss) is deliberate —
train_loss can drop from overfitting even when the model isn't learning
the task, whereas val_loss tracks generalization. Models that learn return
delta > ~20%; plateau runs return < 5%. This cleanly separates the thesis
into two populations.

Backwards compatibility:

| Run type | Has epoch-0 row | `loss_delta_pct` value |
|---|---|---|
| New LLM runs (post-merge) | Yes | Populated float |
| Old LLM runs (pre-merge) | No | Blank string |
| Official TRM runs | No (different CSV schema) | Blank string |
| Legacy TRM runs | No | Blank string |

Blank string (`""`) matches the existing convention in
`attach_efficiency_metrics` for zero-correct runs — imports to pandas as
NaN which survives `.mean()` and `.plot()` correctly, whereas `0.0` would
poison aggregate statistics.

#### `tests/test_aggregate.py`

Three new test cases exercising the new column:

1. `test_loss_delta_pct_computed_when_epoch_zero_row_present` — synthetic
   CSV with `epoch=0, val_loss=2.5` and `epoch=30, val_loss=2.45` asserts
   `loss_delta_pct ≈ 2.0`.
2. `test_loss_delta_pct_blank_when_no_epoch_zero_row` — synthetic CSV with
   only `epoch=10, 20, 30` asserts `loss_delta_pct == ""`.
3. `test_loss_delta_pct_negative_when_val_loss_increased` — synthetic CSV
   with `epoch=0, val_loss=2.0` and `epoch=30, val_loss=2.2` asserts
   `loss_delta_pct ≈ -10.0` (negative is valid semantics for "model got
   worse", not a bug).

## Implementation order

The dependency chain:

| # | Change | Depends on |
|---|---|---|
| 1 | `TrainingConfig.log_per_step` field in `config.py` | — |
| 2 | Three diagnostics in `trainer_llm.py` | #1 |
| 3 | Two new YAMLs in `configs/` | #1, #2 |
| 4 | `loss_delta_pct` in `aggregate.py` | — (parallel with 2-3) |
| 5 | Three new test cases in `test_aggregate.py` | #4 |
| 6 | Smoke test: 3 epochs, 5 min on 5070 | #1-#5 |
| 7 | Full sudoku run: 30 epochs, ~4 hr on 5070 | #6 passes |
| 8 | Full maze run: 30 epochs, ~12 hr on 5070 | #7 passes |
| 9 | Regenerate `summary.csv`, update `README.md` + wandb glossary + wandb Report | #7, #8 |

Running sudoku before maze enforces fail-fast at the cheapest possible
point — sudoku and maze share all pipeline code, so if sudoku's dashboard
looks right, maze is very likely fine.

## Smoke test specification

```bash
python main.py --mode train --config configs/llm_deepseek.yaml \
  training.epochs=3 \
  training.log_interval=1 \
  training.save_interval=100
```

Pass criteria (all four must hold):

1. Stdout shows `[baseline] Epoch 0/3 | ValLoss: ...` before the first training epoch.
2. Wandb dashboard shows `val/loss` data points at `step=0, 1, 2, 3`.
3. Wandb dashboard shows `lm/loss`, `val/lm_loss`, `lm/step_loss` keys
   appearing with non-empty values.
4. `python scripts/aggregate_metrics.py` produces a `loss_delta_pct` cell
   (numeric or blank — not a crash / missing column) for the smoke run.

If any fail: fix before launching the full runs. Cost of this 5-minute
check insures against a wasted 12-hour maze run.

## Regression check for shared changes

`trainer_llm.py` is shared with GPT-2 / SmolLM / Qwen / Llama baselines.
Expected effects on their next run after this change:

| Effect | Regression? | Explanation |
|---|---|---|
| Adds ~30s–2min pre-training eval before epoch 0 | No | Wall-clock addition, no metric change. Produces an epoch-0 row for them too — small net positive. |
| Adds `lm/loss`, `val/lm_loss` keys to wandb dashboard | No | Duplicate keys with identical values. Existing panels keyed on `train/loss` / `val/loss` continue to render. |
| `log_per_step: false` default | No | Existing behavior preserved — no per-step logging unless YAML opts in. |
| `loss_delta_pct` in `summary.csv` | No | Additive CSV column. Existing CSV readers ignore unknown columns. |

## Post-run housekeeping

After both full runs complete:

1. `python scripts/aggregate_metrics.py` — regenerates `results/summary.csv` with the two new rows + new column.
2. Update `docs/wandb_metrics_glossary.md` — document `lm/loss`, `val/lm_loss`, `lm/step_loss`, the epoch-0 baseline semantics, and `loss_delta_pct`.
3. `python scripts/publish_wandb_metrics_report.py` — refresh the published wandb Report from the glossary.
4. `README.md` lines 367-378 — add two rows:
   - `DeepSeek-R1-Distill-Qwen-1.5B + LoRA | Fine-tuned LLM | 1.5B | Sudoku | ~0% | <measured>`
   - `DeepSeek-R1-Distill-Qwen-1.5B + LoRA | Fine-tuned LLM | 1.5B | Maze | ~0% | <measured>`
5. `README.md` Quick Start block (line 242 area) — add the two new `python main.py --mode train --config configs/llm_deepseek*.yaml` commands.
6. `README.md` Recent updates section — note addition of DeepSeek baseline plus the `loss_delta_pct` evidence metric.

## Open questions

- Teammate coordination: other team members (Armin, Nickolas) also run on
  RTX 5070 machines per `project_trm_maze_trained_3_seeds.md`. When
  `trainer_llm.py` changes land on main, any in-flight LLM run on their
  machines will pick up the epoch-0 eval and `lm/loss` alias on their next
  trainer instantiation. This is additive (no regression) but worth a
  heads-up message before merging.

## References

- `configs/llm_qwen.yaml` — architectural analog (Qwen2 family)
- `configs/llm_llama_maze.yaml` — VRAM analog (1.2B on 5070 maze)
- `src/models/baseline_llm.py:49-64` — existing DeepSeek dispatch
- `src/training/trainer_llm.py:138-149` — shared wandb.log block
- `src/evaluation/aggregate.py:104-193` — `parse_train_log` function
- `docs/wandb_metrics_glossary.md` — metric reference to update post-run
- DeepSeek model card: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
