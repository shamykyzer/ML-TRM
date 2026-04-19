# W&B Metrics Glossary

One-line explanations of every metric logged by the four trainers
(`trainer_trm`, `trainer_official`, `trainer_llm`, `trainer_distill`).
Use this as a reference when reading wandb dashboards or writing the
Experiments section of the report. Copy-paste into a wandb Report for
annotated live dashboards.

The four sections (`train/`, `val/`, `carbon/`, `system/`) correspond to
the panel groups W&B auto-creates from the metric prefixes.

---

## `train/` — Training-time signal

Measured on the training batches as the model updates.

| Metric | Unit | Meaning | What to watch for |
|---|---|---|---|
| `train/loss` | nats/token | Cross-entropy loss averaged over the epoch. LLM: HF causal-LM loss. Distill: α·KL + (1-α)·CE. TRM: StableMax CE + halting BCE. | Should decrease monotonically. Plateau near `ln(vocab_size)` means nothing learned. Sudden spike = LR too high. |
| `train/accuracy` | fraction (0-1) | Per-cell accuracy on the training set (TRM only — LLM trainers don't emit this). | Climbs above `1/vocab_size` once learning starts. 0.99+ combined with flat val = memorisation. |
| `train/exact_accuracy` | fraction (0-1) | Full-puzzle accuracy on training set (TRM only). | Lags `train/accuracy` — an 81-cell puzzle with 99% per-cell still has only 44% chance of being fully correct by random chance. |
| `train/lr` | scalar | Current learning rate (after warmup + schedule). | Should match config. If 0 forever, scheduler bug. |
| `train/elapsed_min` | minutes | Wall-clock minutes since training started. | Diagonal line if training is progressing at constant rate. Slope change = hardware issue. |

## `val/` — Held-out evaluation signal

Measured on the test split every `log_interval` epochs.

| Metric | Unit | Meaning | What to watch for |
|---|---|---|---|
| `val/loss` | nats/token | Cross-entropy on the validation split. **Added 2026-04-19.** LLM: HF causal-LM loss with the same shift+mask as training. Distill: hard-label CE (no teacher call — fast). | If `val/loss` starts rising while `train/loss` keeps falling → overfitting. This is the single most important signal to monitor. |
| `val/puzzle_acc` | fraction (0-1) | Full-puzzle accuracy — a puzzle counts as correct only if every non-clue cell is correct. The proposal's headline evaluation metric. | Zero until the model can chain dozens of correct predictions. LLM baselines expected to sit near 0 per thesis (§1 of proposal). |
| `val/cell_acc` | fraction (0-1) | Per-cell accuracy — fraction of non-clue cells the model predicts correctly. Complementary to puzzle_acc; survives when puzzle_acc = 0. | Use this to judge whether a 0% puzzle_acc model is "completely random" or "learned statistics but can't compose". |
| `val/accuracy` | fraction (0-1) | **Alias of `val/cell_acc`.** Added for dashboard symmetry with `train/accuracy` (same cell-level concept). | Identical curve to `val/cell_acc`. Filter by this when building cross-trainer panels. |
| `val/exact_accuracy` | fraction (0-1) | **Alias of `val/puzzle_acc`.** Added for dashboard symmetry with `train/exact_accuracy`. | Identical curve to `val/puzzle_acc`. |
| `val/avg_act_steps` | float | TRM only — average number of ACT halting steps used per puzzle at eval. | Drops from 16 (no halting) toward 1-2 once the model solves puzzles confidently. Flat at 16 → halting head never triggered. |

**Overfitting checklist** (read these four panels together):
1. `train/loss` falling — model fitting train set
2. `val/loss` rising — no longer generalizing
3. `train/accuracy` at ceiling (~1.0)
4. `val/cell_acc` / `val/puzzle_acc` falling

If all four, overfit confirmed. Reduce epochs, add regularization, or use
HF init + shorter fine-tune.

## `carbon/` — CodeCarbon energy and emissions

Flushed every `log_interval` from `src.training.carbon_tracker.CarbonTracker`.

| Metric | Unit | Meaning | What to watch for |
|---|---|---|---|
| `carbon/emissions_kg` | kg CO₂-equivalent | Cumulative grid-adjusted emissions for this run. Regional intensity baked in by CodeCarbon (UK grid ≈ 0.2 kg/kWh as of 2026). | Straight-line growth proportional to GPU utilization. Used in summary.csv → `train_co2_kg`. |
| `carbon/energy_kwh` | kWh | Cumulative electrical energy (CPU + GPU + RAM). | Denominator for the `co2_per_correct_puzzle` metric in `results/summary.csv`. |

## `system/` — Wandb-auto-collected hardware telemetry

Wandb's agent logs these every ~30s regardless of your trainer code. Useful
for sanity-checking that the GPU is actually being used and you're not CPU-bound.

| Metric | Unit | What it tells you |
|---|---|---|
| `system/gpu.0.gpu` | percent | GPU core utilization. Should be 90%+ during training for efficient work. <50% = CPU-bound data loading. |
| `system/gpu.0.memoryAllocatedBytes` | bytes | GPU VRAM in use. Plateau near VRAM cap means batch size is right-sized. Sudden drop = OOM crash. |
| `system/gpu.0.temp` | °C | GPU die temperature. Throttling starts around 83°C on consumer cards. |
| `system/cpu` | percent | CPU aggregate utilization. |

---

## Where these metrics come from (code map)

| Metric | Emitted by | Line |
|---|---|---|
| `train/loss` | `trainer_llm.py`, `trainer_distill.py`, `trainer_trm.py`, `trainer_official.py` | varies |
| `val/loss`, `val/puzzle_acc`, `val/cell_acc` + aliases | `trainer_llm.py:131-147`, `trainer_distill.py:172-188` | — |
| `train/accuracy`, `train/exact_accuracy` | `trainer_official.py` | `wandb.log` in its training loop |
| `carbon/*` | `trainer_*.py` | via `self.carbon.flush()` at log intervals |
| `system/*` | wandb agent | automatic, no code |

## Summary aggregations (runs table view)

Defined once in `src/training/wandb_utils.py::define_common_metrics`.
These determine which single value shows up in the wandb "runs table"
column for each metric when you sort across runs.

| Metric pattern | Aggregation | Rationale |
|---|---|---|
| `val/*_acc` | max | Peak validation accuracy is the "best result" — overfitting shouldn't hide it |
| `*/loss` | min | Lowest loss reached during training |
| `carbon/*` | last | Cumulative counters — last value is the total |
| `system/*` | max | Peak GPU utilization / temperature over the run |
| `train/lr` | last | Final learning rate after schedule |
| `*/_sec` | mean | Per-step timings — mean is more informative than last |
