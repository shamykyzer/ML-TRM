# Novelty experiment — iso-wall-clock comparison + K-vote inference

This directory holds the artifacts and description of the coursework's novelty
contribution, layered on top of the baseline TRM-vs-LLM comparison documented
in the repo root `README.md`. It is the source material for the Methods and
Experiments sections of the 6-page group report; it is not the report itself.

## 1. Experiment overview

The proposal claims TRM is more parameter- and energy-efficient than
fine-tuned LLMs on structured puzzles. Rather than restate that claim, we
**measure** it under a controlled training budget and then ask a second
question: does repeated sampling at inference time close the gap? The novelty
has two parts. First, an **iso-wall-clock** training comparison: every model
gets exactly 2.5 hours of training wall time, and we record
`(epochs_completed, wall_clock_sec, kwh, accuracy)` as four coupled outcomes.
Second, a **K-vote inference sweep** (K in {1, 2, 4, 8, 16}) that extends the
training-time WTA of Dillon (2026) to inference time for TRM via random
latent-init perturbation, and adds the standard temperature-sampled
self-consistency baseline (Wang et al., 2022) for LLMs.

## 2. Design decisions

### Iso-wall-clock vs iso-epoch vs iso-energy

| Regime | What it fixes | Why we rejected / chose it |
|--------|---------------|----------------------------|
| iso-epoch | Same number of epochs per model | Rejected. 30 LLM epochs is roughly 600 TRM epochs in FLOPs; forcing epoch equality biases whichever architecture gets fewer epochs than its design calls for. Epochs are not commensurable across TRM and causal-LM objectives. |
| iso-energy | Same kWh budget | Rejected. Harder to schedule (you only know the budget is spent after the fact) and harder to explain in a 6-page report. CodeCarbon readings also drift with background GPU load. |
| **iso-wall-clock** | Same 2.5 hr/run | **Chosen.** Standard in the empirical-ML literature, trivial to schedule, lets per-model epoch counts follow each model's own tuned recipe (TRM 2000, LLM 30, distilled 100). Preserves the proposal's efficiency claim as a measured outcome, not an assumption. |

Per-config epoch caps are set high enough that they never bind before the
2.5 hr wall clock does; the trainer exits on whichever comes first.

Wall time is measured from the first forward pass, not from process start —
dataset loading, HF download, and CUDA initialisation are excluded. This
matches how `docs/training-notes.md` reports the baseline times and makes the
9000-second budget comparable across boxes with different disk speeds.

### K-vote diversity source (TRM vs LLM)

| Candidate | How it makes K predictions differ | Why we rejected / chose it |
|-----------|-----------------------------------|----------------------------|
| MC-dropout | Re-sample dropout masks per forward pass | Rejected for TRM. Its recursion blocks use minimal dropout by design (weight-tying plus post-norm already regularise), so mask resampling barely moves the output distribution — the K runs would be near-identical. |
| Input augmentation | Apply dihedral transforms (maze) or digit permutation (sudoku), predict, un-transform, vote | Rejected. The "reasoning" is the same on every pass; only the input view changes. Weakest link to the Dillon (2026) WTA-style framing we want to cite. |
| **Latent-init perturbation (TRM)** | Draw `z_init`, `y_init` fresh per pass from a small-variance Gaussian around zero | **Chosen for TRM.** Extends Dillon (2026)'s training-time WTA to inference time on the same architecture. One-sentence contribution: "we re-use their K=4 training trick at K=1..16 inference time." |
| **Temperature sampling, T=0.7 (LLM)** | Standard self-consistency: sample K times, majority-vote | **Chosen for LLM.** Matches Wang et al. (2022) self-consistency, so the LLM side is a faithful baseline rather than a strawman. |

### Compute budget

- Iso-time training: 6 runs x 2.5 hr = 15 hr.
- K-vote sweep: ~2 hr total across both models and both tasks (inference only; cached checkpoints from part 1). Dominant cost is Qwen at K=16 on the 1000-example maze test split.
- Total: ~17 hr, split across two evenings on one machine. Fits inside the student's ~14-hr-per-machine window with headroom for re-runs; the split is deliberate so a crash in the training phase does not invalidate the inference phase.
- Runs 1-4 can be executed on any of the six fleet boxes; runs 5 and 6 must share a machine with runs 3 and 4 so the teacher checkpoint is local.

## 3. The 6-run matrix

| # | Label | Model | Task | Config | Notes |
|---|-------|-------|------|--------|-------|
| 1 | trm-mlp-sudoku | TRM-MLP (~6.4M) | Sudoku-Extreme | `configs/trm_official_sudoku_mlp.yaml` | HF-init from `hf_checkpoints/Sudoku-Extreme-mlp`; epoch cap 2000. |
| 2 | trm-att-maze | TRM-Att (~8.4M) | Maze-Hard | `configs/trm_official_maze.yaml` | HF-init from `hf_checkpoints/Maze-Hard`; epoch cap 2000. |
| 3 | qwen-sudoku | Qwen2.5-0.5B + LoRA | Sudoku-Extreme | `configs/llm_qwen.yaml` | Epoch cap 30; teacher for run #5. |
| 4 | qwen-maze | Qwen2.5-0.5B + LoRA | Maze-Hard | `configs/llm_qwen_maze.yaml` | Epoch cap 30; teacher for run #6. |
| 5 | distill-sudoku | DistilledLLM (~2.4M) | Sudoku-Extreme | derived from run #3 | Depends on run #3 finishing (teacher checkpoint); epoch cap 100. |
| 6 | distill-maze | DistilledLLM (~2.4M) | Maze-Hard | derived from run #4 | Depends on run #4 finishing (teacher checkpoint); epoch cap 100. |

Runs 5 and 6 are scheduled after 3 and 4 on the same machine; the runner
writes the teacher path into the distill config at launch time.

## 4. How to re-run

Two modes: single-rig (one machine does all 6 runs at the 2.5 hr/run cap) or
3-rig split (three machines in parallel; each run gets 6-7 hr). The 3-rig
mode is the one to use when a fleet is available — 2.4-2.8x more training
time per model inside roughly the same wall-clock window.

### 4a. Single-rig mode (~17 hr on one box)

The menu-driven path:

```bash
python start.py  # option 7 -> Novelty experiments -> 3 (both)
```

The direct path, useful for a headless box or a re-run of a single stage:

```bash
python scripts/run_novelty_iso_time.py --seed 0 --max-train-seconds 9000
python scripts/run_novelty_k_vote.py --seed 0
```

`--max-train-seconds 9000` is the 2.5 hr cap. The K-vote script reads the
checkpoints written by the iso-time script from `$TRM_WORK_DIR/novelty-*`.

### 4b. Three-rig split (each rig runs ~12-14 hr)

Runs divide cleanly across 3 machines so each run can train 2.4-2.8x longer
than the single-rig cap. Each (teacher, student) pair stays on the same rig
so the distill teacher checkpoint never leaves local disk.

| Rig | TRM_RIG | Runs | Per-run cap | Rig total |
|-----|---------|------|-------------|-----------|
| 1 | `TRM_RIG=1` | trm-mlp-sudoku + trm-att-maze | 7.0 hr | 14.0 hr |
| 2 | `TRM_RIG=2` | qwen-sudoku -> distill-sudoku | 6.0 hr | 12.0 hr |
| 3 | `TRM_RIG=3` | qwen-maze -> distill-maze | 6.0 hr | 12.0 hr |

Menu path on each rig (prompts for `TRM_RIG` and persists to `.env`):

```bash
python start.py  # option 7 -> Novelty experiments -> 4 (this rig's slice)
```

Direct path (headless):

```bash
# on rig 1:
python scripts/run_novelty_iso_time.py --rig 1 --seed 0
python scripts/run_novelty_k_vote.py --rig 1 --seed 0
# same pattern on rigs 2 and 3 with --rig 2 / --rig 3.
```

Each rig writes partial CSVs `iso_time_results-rig{N}.csv` and
`k_vote_results-rig{N}.csv` to `results/novelty/` (OneDrive-synced, so all
three machines see the same folder once sync settles).

### 4c. Aggregate (after all 3 rigs finish)

On any machine that has `results/novelty/` synced:

```bash
python scripts/run_novelty_aggregate.py --seed 0
```

Merges the 3 partial CSVs into the canonical `iso_time_results.csv` and
`k_vote_results.csv`, and re-emits all 5 plots against the full dataset.
`--ignore-missing` lets the aggregator run while one rig is still active.

## 5. Output artifacts

Heavy artifacts (checkpoints, `emissions.csv`, `train_log.csv`) are written
to `$TRM_WORK_DIR/novelty-*` — outside the repo, on the local SSD, so the
OneDrive-synced repo stays small.

Light artifacts (version-controlled) land in this directory:

- `iso_time_results.csv` — one row per run. Schema: `run_id, label, task, model, epochs_completed, wall_clock_sec, kwh, emissions_kg, final_puzzle_acc, final_cell_acc, best_puzzle_acc, checkpoint_path`.
- `k_vote_results.csv` — one row per (label, K). Schema: `label, task, model, k, puzzle_acc, cell_acc, mean_latency_ms, kwh_per_puzzle`.
- `iso_time_accuracy_by_model.png` — grouped bars, one bar per run, y = `best_puzzle_acc`.
- `iso_time_energy_by_model.png` — grouped bars, y = `kwh`.
- `iso_time_acc_vs_kwh.png` — scatter of `best_puzzle_acc` vs `kwh` with model family as colour; supports the Pareto discussion.
- `k_vote_accuracy_curve.png` — `puzzle_acc` vs K for the four (model, task) pairs.
- `k_vote_pareto.png` — `puzzle_acc` vs `kwh_per_puzzle` with K as the annotated point labels.

## 6. Report writing hints

Rubric cells this unlocks (from the Group Project Specification, 70-100% band):

- **Methods 30%** — "alternative methods considered and fully justified methodology". Section 2 above enumerates alternatives (iso-epoch, iso-energy; MC-dropout, input augmentation) and justifies the rejection of each against a concrete criterion (bias, schedulability, diversity strength, citation clarity).
- **Experiments 30%** — "experimental design matches the research question, controls are appropriate". The iso-wall-clock constraint is the control; the four outcome columns are the dependent variables; the K sweep is the planned sensitivity analysis.

Proposal claims this evidence addresses:

- "TRM is more efficient than fine-tuned LLMs" — tested directly by the `accuracy / kwh` ratio in `iso_time_acc_vs_kwh.png`, no longer assumed from parameter counts alone.
- "Distilled students inherit teacher weakness on structured reasoning" — runs 5 and 6 land in the near-zero-accuracy, low-energy corner of the scatter if the claim holds.
- "Test-time compute can trade energy for accuracy" — the K-vote Pareto front answers yes/no and quantifies the slope.

Citations to add to the report:

- Jolicoeur-Martineau et al. (2025), *TRM: Less is More* — the base architecture; stored at `papers/01_TRM_Jolicoeur-Martineau_2025.pdf`.
- Dillon (2026) — training-time K-sample WTA on TRM; our inference-time extension. This is the single-sentence novelty claim.
- Wang, Wei, Schuurmans et al. (2022), *Self-consistency improves chain of thought reasoning in language models* — the LLM K-vote baseline.

A useful framing sentence for the Methods section: "We adopt an iso-wall-clock
protocol (Section X) so that accuracy and kWh are coupled outcomes of the same
training budget, not parameters that must be controlled separately." The
corresponding Experiments-section hook is the `iso_time_acc_vs_kwh.png`
figure, which makes the trade-off visible at a glance.

## 7. Known limitations

- **Single seed per run.** The time budget (17 hr) does not fit the 3 seeds x 6 runs = 51 hr needed to report mean +/- std for the novelty rows. The baseline rows in `results/summary.csv` do have seed-variance (fleet plan); novelty rows do not. The report should state this explicitly and avoid significance claims on the novelty table.
- **2.5 hr may be short for TRM-Att on maze.** The TRM paper trains for ~2000 epochs; 2.5 hr on an RTX 5070 reaches roughly 400-600. HF-init mitigates this but does not eliminate it. The iso-time result is a lower bound on what TRM can achieve, not its ceiling.
- **One LLM family.** Qwen2.5-0.5B is the only LLM in the novelty matrix, so findings do not automatically generalise to GPT-2, SmolLM2, or Llama-3.2 (all of which the baseline comparison in `results/summary.csv` already covers at the epoch-30 budget).
- **K-vote diversity source differs across families.** TRM uses latent-init perturbation; LLMs use temperature sampling. They are not the same knob, and a reader could argue the LLM side has "more" diversity. The report should either defend the asymmetry (each family gets the diversity source that matches its literature) or note it as a threat to validity.
- **Majority vote is per-cell, not per-puzzle.** Cell-wise voting can produce a voted solution that no individual sample actually produced; this is the standard self-consistency convention (Wang et al., 2022) but is worth flagging because it can inflate cell accuracy relative to puzzle accuracy. The `k_vote_results.csv` schema reports both so the report can show the gap.
- **Energy figures are process-level, not wall-power.** CodeCarbon estimates kWh from NVML GPU-power samples plus a country-specific carbon intensity factor; it does not measure the PSU. Readings are directionally correct but absolute values should be cited as "CodeCarbon estimate" rather than ground truth.
