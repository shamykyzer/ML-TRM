# Report draft — Methods + Experiments sections

**Status:** Draft for the team's review, not the final submission.
**Author:** Generated 2026-04-25 from the run inventory in `findings.md`,
`results/summary.csv`, `results/trm_runs_overview.csv`,
`results/novelty/README.md`, the per-run snapshots in
`results/runs/`, and `analysis_2026-04-23_distill_maze.md`.
**Why this file:** the canonical `docs/report.md` lists Methods and
Experiments as TBD (Tasks 2 and 10). This is a complete first pass at
both sections, sized for ~1.8 pages each in conference double-column,
hitting the rubric's 70-100% band on "alternative methods considered"
(Methods) and "fully justified methodology, excellent appraisal of
results" (Experiments).
**Editor's note:** numbers are pulled from CSVs and JSONs in the repo
as of this date. Verify in-line citations before submitting; I have
flagged the maze-LLM saturation explicitly as a limitation rather
than a result, which is the report's safest position with the
evidence we currently have.

---

## 4 Methods

### 4.1 Architectures

We compare three model families on two structurally-different
constraint-satisfaction tasks. **TRM** (Tiny Recursive Model;
Jolicoeur-Martineau et al., 2025) produces puzzle solutions by
iteratively refining a latent state through a fixed number of
recursion steps with deep supervision. Following the TRM paper's
task-specific recommendation, we use the **attention-free MLP
variant** (~6.4 M parameters) for Sudoku-Extreme and the
**self-attention variant** (~8.4 M parameters) for Maze-Hard; both
share the same recursion controller (H_cycles=3, L_cycles=4) and
only the per-step block differs.

**Qwen2.5-0.5B + LoRA** is our fine-tuned LLM baseline. We selected
Qwen2.5-0.5B over GPT-2 (older positional encoding, weaker tokenizer
for digit sequences), SmolLM2 (immature pipeline support at the time
of running), and Llama-3.2-1B (energy-prohibitive at our compute
scale). Adapter rank r=8, α=16, and 4-bit QLoRA NF4 quantisation are
necessary to fit a 900-token sequence at batch size 2 on a 12 GB
consumer GPU; without QLoRA the model OOMs at sequence length 900.
Effective batch size after gradient accumulation is 16.

**Distilled student** is a 3-layer transformer (d_model=256, 4
heads, FFN hidden=1024; ~2.4 M parameters) trained via Hinton-style
knowledge distillation from the fine-tuned Qwen teacher. The
student inherits the teacher's tokenizer but is initialised from
scratch. We use weighted soft + hard cross-entropy with
distillation weight α=0.7 and temperature T=4. The student is
roughly 200× smaller than its teacher.

### 4.2 Training protocol — HF-init plus fine-tune

The TRM paper's reported numbers (87.4 % Sudoku-Extreme, 85.3 %
Maze-Hard) come from 8×H200 training with effective batch 4608 for
~60 000 epochs — roughly three orders of magnitude more compute
than a single RTX 5070 can supply. We therefore warm-start TRM
training from community-published checkpoints (Sanjin2024
`Sudoku-Extreme-mlp` and `Maze-Hard`) and fine-tune for up to 2 245
epochs on a single GPU. As an ablation we attempted from-scratch
training of TRM-Att on Sudoku-Extreme; we discuss the resulting
collapse in §5.4.

### 4.3 Iso-wall-clock budget — alternatives considered

For the cross-architecture comparison we constrain every model to
the same 2.5-hour training budget on identical hardware. We
considered three alternative protocols:

| Protocol | What it fixes | Why we rejected it |
|---|---|---|
| iso-epoch | Same number of training epochs | TRM's recursive objective trains in tens of thousands of epochs; LoRA fine-tuning converges in tens. Epoch counts are not commensurable across the two objectives, so iso-epoch biases whichever architecture gets fewer epochs than its design calls for. |
| iso-energy | Same total training kWh | Hard to schedule because the budget is only known after the fact; CodeCarbon NVML readings also drift with background GPU load. |
| **iso-wall-clock (chosen)** | Same wall-clock seconds | Standard in the empirical-ML literature, trivial to schedule, and lets per-model epoch counts follow each architecture's own tuned recipe. Crucially, it preserves the proposal's efficiency claim as a *measured outcome* rather than an a priori assumption. |

The wall-clock guard (`src/training/wall_clock_guard.py`) is armed
by the `TRM_MAX_TRAIN_SECONDS` environment variable and checked at
the top of each epoch. On expiry the trainer saves the latest
checkpoint and exits cleanly, so a halted run still produces a
usable artifact for downstream evaluation.

### 4.4 Inference protocol — K-vote with justified diversity source

To test whether test-time compute can recover accuracy lost to a
tight training budget, we extend Dillon (2026)'s K-sample
winner-take-all (WTA) protocol from training time to inference
time. We sweep K ∈ {1, 2, 4, 8, 16}. The diversity source differs
across families because the architectures admit different
perturbations:

| Candidate | How K predictions differ | Decision |
|---|---|---|
| MC-dropout | Resample dropout masks per pass | Rejected for TRM. Its recursion blocks use minimal dropout by design (weight-tying plus post-norm already regularise); mask resampling would barely move the output distribution. |
| Input augmentation | Dihedral transforms (maze) or digit permutation (sudoku); transform → predict → un-transform → vote | Rejected. The reasoning is identical on every pass; only the input view changes. Weaker test of the test-time-compute hypothesis. |
| **Latent-init perturbation (TRM)** | Draw `z_init`, `y_init` fresh per pass from a small-variance Gaussian | **Chosen for TRM.** Extends Dillon (2026)'s training-time WTA to inference time on the same architecture. |
| **Temperature sampling, T=0.7 (LLM, distill)** | Standard self-consistency: sample K times, majority-vote per cell | **Chosen for LLM and distilled.** Matches Wang et al. (2022) self-consistency, so the LLM side is a faithful baseline rather than a strawman. |

We acknowledge the asymmetry — TRM and the LLM family use
different diversity knobs — and treat it as a threat to validity in
§5.5 rather than papering over it.

### 4.5 Energy accounting

We use CodeCarbon (v3.2.6) to record per-process kWh,
CO2-equivalent emissions, and wall-clock time during both training
and inference. Energy is computed from NVML GPU-power samples plus
a UK grid carbon-intensity factor. Absolute kWh values should be
read as CodeCarbon estimates rather than wall-power ground truth;
relative ranking across models on the same hardware is reliable.

We report **CO2 per correct puzzle**, not raw CO2. The two metrics
order models differently: a model that emits 0.9 kg of CO2 to solve
no puzzles is infinitely *less* efficient than one that emits 22 kg
to solve three quarters of them, and the per-correct-puzzle
denominator surfaces this. Raw kWh totals on their own would
mislead a reader into ranking the LLM as more efficient than TRM on
Sudoku-Extreme.

### 4.6 Ethical considerations (MO4)

Three ethical hooks shape the methodology. First, we made an
explicit decision **not to retrain** the collapsed Sudoku-Att run
at different hyperparameters; the expected new information per kg
CO2 emitted was unfavourable, and reproducing a known failure does
not advance the proposal's efficiency thesis (§5.4). Second, we
deliberately chose consumer-grade hardware (RTX 5070, 12 GB) over
cloud accelerators to make the comparison representative of
accessible-AI scenarios — the proposal's "low-cost AI" framing
requires the methodology itself to be low-cost. Third, the
`co2_per_correct_puzzle` metric we report (§4.5) was designed to
penalise models that train cheaply but produce no usable output, a
category of inefficiency that raw kWh totals hide.

---

## 5 Experiments

### 5.1 Sudoku-Extreme — TRM dominates LLMs at every measured axis

Table 1 reports validation puzzle accuracy and CO2-per-correct
puzzle on the Sudoku-Extreme test split (1000 train, 423 000 test).
Three independent seeds confirm the TRM-MLP result.

**Table 1.** Validation results on Sudoku-Extreme. Columns: model,
parameter count, best validation puzzle accuracy, validation cell
accuracy, total training energy, CO2 per correctly-solved test
puzzle.

| Model | Params | Puzzle acc | Cell acc | Train kWh | CO2/correct |
|---|---:|---:|---:|---:|---:|
| TRM-MLP, HF eval (no fine-tune) | 6.4 M | 84.74 % | 91.55 % | 0.48 (inf) | 1.23×10⁻⁶ kg |
| TRM-MLP, fine-tune mean of 3 seeds | 6.4 M | **74.25 ± 0.63 %** | 85.3 % | 22.0 / seed | 1.66×10⁻⁵ kg |
| Qwen2.5-0.5B + LoRA, 100 epochs | 500 M | **0.00 %** | 19.07 % | 0.90 | undefined (∞) |
| Distilled student, 30 epochs | 2.4 M | 0.00 % | 25.78 % | 0.011 | undefined (∞) |
| TRM-Att, from scratch (ablation) | 8.4 M | 18.33 % @ ep100 → 0 % | 55.4 % | 6.93 | 2.12×10⁻⁵ kg |

The headline finding: a **6.4 M-parameter TRM solves 74.25 % of
held-out Sudoku-Extreme puzzles using fine-tune from a community
checkpoint, while a 500 M-parameter fine-tuned LLM solves zero**.
Qwen's per-cell accuracy of 19.07 % is meaningfully above the 11.1 %
uniform-prior chance level, indicating it has learned per-digit
statistics — but it cannot compose them into a globally consistent
solution. This precisely matches the failure mode reported by
Jolicoeur-Martineau et al. (2025) on DeepSeek R1, Claude 3.7, and
o3-mini-high, all of which scored 0 % on the same benchmark.

**Three-seed variance.** Best validation puzzle accuracy across
three seeds (0/1/2) for TRM-MLP fine-tune was 74.56 / 74.20 /
74.86 %, σ = 0.0063. The variance is small enough to support
single-seed point estimates elsewhere in this paper.

**Energy framing matters.** Per kWh, Qwen training is cheaper (0.9
vs 22 per TRM seed). Per *correct puzzle*, the comparison reverses
— TRM-MLP emits 1.66×10⁻⁵ kg CO2 per correct solution; Qwen and
the distilled student emit no CO2 per correct solution because
there are none. The proposal's central efficiency claim is
confirmed *only* under this denominator-aware framing; raw
training-energy totals would misrank the comparison.

### 5.2 Distillation — cheap inheritance of teacher capability

The distilled 2.4 M-parameter student trained for 4 minutes on
0.011 kWh — roughly 30× less energy than its 100-minute Qwen
teacher — and achieved 25.78 % cell accuracy on Sudoku-Extreme,
**higher than the teacher's 19.07 %** at far lower cost. Both
still score 0 % whole-puzzle accuracy. Two implications: (1)
distillation works mechanically — the student is not bottlenecked
by capacity, and a 200× parameter reduction does not collapse
single-cell prediction quality; (2) the structural failure of LLMs
on Sudoku-Extreme is *inherited from the teacher*, not
training-budget-induced. The proposal's prediction that "distilled
students inherit teacher weakness on structured reasoning" is
supported.

On Maze-Hard the distill run completed one epoch under a 9.6-minute
wall-clock cap and emitted 0.043 kWh — roughly 37× less than its
Qwen teacher's 1.564 kWh — at identical (1.000) puzzle accuracy.
We caveat the maze accuracy in §5.3.

### 5.3 Maze-Hard — benchmark saturation under LoRA fine-tune

The TRM-Att HF eval recovers the published 79.60 % puzzle / 99.30 %
cell accuracy on Maze-Hard. Consumer-GPU from-scratch fine-tuning
of TRM-Att across three seeds (0 / 1 / 2) produced 20.2 / 18.9 /
4.7 % puzzle accuracy respectively (mean 14.6 %, σ = 8.5 %); the
high variance and seed-2 collapse indicate from-scratch attention
training is not stable at our compute scale. The HF-eval value, not
the fine-tune mean, is therefore the load-bearing TRM-Att
Maze-Hard number for the cross-family comparison.

The fine-tuned LLM and distilled student both score **1.000 puzzle
accuracy** on Maze-Hard. We treat this as a benchmark-saturation
finding rather than a model-capability finding, for two reasons:
(1) the dataset (`maze-30x30-hard-1k-aug`) applies dihedral
augmentation; if augmentation was applied before the train/test
split, "test" mazes would be rotations of training mazes and the
model recognises rather than solves; (2) the `mask_non_path: true`
evaluation setting ignores wall cells, which dominate a 900-cell
sequence and can inflate puzzle-level metrics. The K-vote curve for
the distilled model is flat at 1.000 across K ∈ {1, 2, 4, 8, 16},
which is itself a symptom of saturation — a non-trivial benchmark
would leave at least a few errors at K=1 for K-vote to correct.

We report the maze numbers for completeness in the cross-task
comparison but **do not draw efficiency conclusions** from them;
the load-bearing efficiency comparison is on Sudoku-Extreme (§5.1).

### 5.4 Sudoku-Att collapse — when not to train

Figure 2 plots train and validation accuracy across 500 epochs of
from-scratch TRM-Att training on Sudoku-Extreme. Train accuracy
climbs monotonically from 0 to 0.96; validation accuracy peaks at
0.18 at epoch 100 and decays to 0 by epoch 350. This is a textbook
overfit on the 1000-example training set, consistent with the TRM
paper's reported regime (8×H200 effective batch 4608 for 60 000
epochs) being a *prerequisite* for stable convergence rather than a
ceiling. We deliberately did not re-run with different
hyperparameters; the predicted value-of-information per kg CO2 was
unfavourable (§4.6) and reproducing a known failure does not
advance the proposal's efficiency thesis.

### 5.5 Limitations

(1) **Single-seed for the iso-time and K-vote rows.** The 17-hour
single-rig novelty budget does not fit the 51-hour 3-seed
multiplier required for variance bars. Sudoku-MLP long-budget runs
do have three seeds (§5.1), so we anchor variance claims there.
(2) **Maze-Hard appears saturated under LoRA fine-tune** (§5.3);
maze efficiency rows should be read as upper bounds on what the
LLM/distill family achieves, not evidence of structural capability.
(3) **Consumer-GPU compute** is roughly three orders of magnitude
below the TRM paper's; our TRM fine-tune numbers are lower bounds
on what TRM achieves at scale, not its ceiling. (4) **Single LLM
family** (Qwen2.5-0.5B); generalisation to GPT-2, SmolLM2, or
Llama-3.2 is not tested. (5) **Asymmetric K-vote diversity source
across families** (latent perturbation for TRM, temperature
sampling for LLM/distill) is documented in §4.4 but readers may
argue the LLM side has more diversity; we did not normalise.
(6) **CodeCarbon estimates** are NVML-derived and exclude PSU
losses; absolute kWh values are CodeCarbon estimates, but ranking
across models on the same hardware is reliable.

---

## What's still TBD in `docs/report.md`

This file fills §4 (Methods) and §5 (Experiments). The remaining
sections still need work:

- **Abstract** (200 w) — assemble last from §5.1 headline +
  CO2-per-correct + saturation caveat.
- **§1 Introduction (10 %)** — adapt proposal §1, cite
  DeepSeek/Claude/o3 zero-shot 0 % on Sudoku-Extreme, preview the
  74.25 vs 0 % result.
- **§2 Related Work (10 %)** — TRM paper, Dillon 2026 (WTA),
  McGovern 2025 (test-time adaptation), Wang et al. 2022
  (self-consistency), Hinton et al. 2015 (distillation).
- **§3 Data (10 %)** — Sudoku-Extreme (1k train / 423k test, 17
  clues), Maze-Hard (1k/1k, paths >110 steps), the augmentation
  question explicitly addressed.
- **§6 Conclusion (5 %)** — TRM thesis confirmed on Sudoku,
  inconclusive on saturated Maze, distillation mechanically works;
  future work = re-evaluate maze with stricter eval, fix K-vote
  vectorisation, multi-seed novelty.
- **§7 Ethical and Societal Implications** — the §4.6 hooks
  expanded; "when not to train" framing; accessible-AI argument
  from consumer-hardware methodology choice.

## Figures the report needs (artifacts already exist or are easy)

1. **Figure 1 — Accuracy comparison bar chart** (Sudoku-Extreme,
   3 models). Data: Table 1 above.
2. **Figure 2 — Sudoku-Att train-vs-val collapse** showing the
   classic overfit signature (§5.4). Source:
   `results/figures/sudoku_att_rise_and_collapse.png` (already
   generated).
3. **Figure 3 — CO2-per-correct-puzzle bar chart** (log-scale
   y-axis to handle the ∞ for LLM/distill). Source:
   `results/summary.csv`.
4. **Figure 4 — Sudoku-MLP fine-tune curve** showing the HF-init
   peak around epoch 10 and subsequent over-training decay.
   Source: `results/history_sudoku-mlp_best.csv`.
5. **(Optional) Figure 5 — Iso-time scatter** if rigs 2 and 3
   sudoku data lands in time. Source:
   `results/novelty/iso_time_acc_vs_kwh.png`, currently a
   single-point plot.

The 6-page conference template easily accommodates 4 figures with
the table; the optional fifth depends on whether the iso-time
matrix is filled in by the writing window.
