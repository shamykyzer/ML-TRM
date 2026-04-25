# Report content plan — 2026-04-25 (T-6 days)

**Author:** Ahmed
**Purpose:** Map the actual run results in `results/` against the UWE
specification rubric (`docs/Group Project Specification 2025-26-v4.pdf`)
and the project proposal (`docs/Project Proposal.pdf`), and turn that
into a section-by-section drafting plan for `docs/report.md`.
**Deadline:** 17:00 on 1 May 2026 (+48 h grace). 6 days from this note.
**Inputs read for this plan:**
- `docs/Project Proposal.pdf`
- `docs/Group Project Specification 2025-26-v4.pdf`
- `results/summary.csv`
- `results/novelty/iso_time_results-rig1.csv`
- `results/novelty/k_vote_runs/trm-mlp-sudoku/emissions.csv`
- `results/novelty/analysis_2026-04-25.md`
- `findings.md`

---

## 1. Reality check: proposal commitments vs what we have

The proposal promised **3 models × 2 tasks = 6 cells**, plus CodeCarbon
energy/CO2 for each. Current state:

| Model | Sudoku-Extreme | Maze-Hard |
|---|---|---|
| TRM | OK — 84.8% (HF-init→fine-tune, epoch 10) and 74.6% ± 0.6% (3-seed from-scratch) | OK — 79.6% (HF-eval); from-scratch unusable, see `findings.md` §5.7 |
| Fine-tuned LLM (Qwen2.5-0.5B + LoRA) | OK — 0% puzzle / 19% cell, 0.9 kWh | MISSING — never run |
| Distilled LLM | MISSING — teacher checkpoint exists but distill not launched | MISSING — teacher missing too |

**3 of 6 cells are filled.** With 6 days left, we cannot honestly fill
the other 3 *and* write a good 6-page report. Recommendation: narrow
the thesis from "TRM vs LLMs in general" to "TRM vs Qwen2.5-0.5B
baseline; distillation arm scoped out as future work" and lean into the
methodology contribution + the iso-time/K-vote novelty.

The rubric's 70-100% band rewards "critical analysis", "fully justified
methodology" and "discussion of limitations" — a clean 3-cell story
honestly framed scores higher than a rushed 6-cell story with broken
numbers.

## 2. Rubric quick-reference (from spec pp. 5-10)

| Section | Weight | MOs | 70-100% band requires |
|---|---|---|---|
| Introduction | 10% | MO2, MO4 | Clear problem definition; excellent description of approach + aims + results |
| Related Work | 10% | MO2 | Critical appraisal; critical analysis closely related to the project |
| Data | 10% | MO2, MO3 | Clear justification of dataset choice; excellent treatment of data |
| **Methods** | **30%** | MO3, MO4 | **Understanding of alternative methods**; high technical depth; excellent discussion of ethical issues |
| **Experiments** | **30%** | MO3 | **Fully justified methodology**; excellent appraisal and evaluation of results |
| Conclusion | 5% | MO2, MO4 | Demonstration of clear understanding and implications |
| Writing & Formatting | 5% | all | Well structured; high clarity |

Notes on MO weighting:
- **MO4 (ethics)** is assessed in Introduction, Methods, and Conclusion. Energy/CO2 is our natural ethics hook — exploit it in all three places, not only one.
- **Methods + Experiments = 60% of the report mark.** Allocate page-budget accordingly: ~1.8 pages each.

## 3. Section-by-section content plan

### 3.1 Abstract (≤200 words, ungraded but anchors the marker)

One claim, three numbers, one limitation. Skeleton:

> "We compare a 5-7 M-parameter Tiny Recursive Model (TRM) against a
> fine-tuned 0.5 B Qwen2.5 LLM on Sudoku-Extreme and Maze-Hard under a
> matched wall-clock training budget. TRM reaches 84.8% / 79.6% puzzle
> accuracy at ~1.9 / ~1.7 kWh; Qwen reaches 0% puzzle accuracy on
> Sudoku at 0.9 kWh. We additionally introduce inference-time K-vote
> sampling on TRM as an extension of Dillon (2026)'s training-time
> winner-take-all and report its energy-vs-accuracy trade-off. The
> distillation arm of the proposed matrix is scoped out as future work."

### 3.2 Introduction (10%, MO2 + MO4) — ~0.6 page

Beats, in order:
1. Constraint-reasoning failure of frontier LLMs (DeepSeek R1, Claude
   3.7, o3-mini all 0% on Sudoku-Extreme — straight from proposal §1).
2. **MO4 ethics hook**: energy footprint of LLM training/inference
   contrasted with the size of the actual problem (1k training examples).
3. What we did: 3 cells of the proposed matrix at iso-wall-clock, plus
   K-vote inference sweep as novelty.
4. Headline numbers (one sentence).

### 3.3 Related Work (10%, MO2) — ~0.6 page

Anchor papers from proposal:
- **Jolicoeur-Martineau (2025)** — TRM architecture; we reproduce its results within ~3 percentage points
- **Dillon (2026)** — training-time WTA at K=4; **our novelty extends this to inference time** (single-sentence differentiation)
- **McGovern (2025)** — TRM fine-tuning under tight compute; we corroborate (HF-init reaches ceiling in ~30 min)

Add for the LLM K-vote baseline:
- **Wang et al. (2022) self-consistency** — already in `results/novelty/README.md` §2

Critical-analysis angles:
- Reproducibility gap: we get 84.8% on Sudoku-Extreme vs the paper's
  87.4%. Discuss (single seed, shorter training, GPU difference).
- The proposal claims "TRM outperforms LLMs"; we test that under matched
  compute rather than assuming it from parameter count.

### 3.4 Data (10%, MO2 + MO3) — ~0.6 page

Cover:
- **Sudoku-Extreme**: 1k train / 423k test, 9×9 grids with exactly 17
  given clues. Source: Jolicoeur-Martineau (2025) HF release.
- **Maze-Hard**: 1k train / 1k test, 30×30 mazes with shortest-path
  length ≥110. Same source.
- Preprocessing: tokenisation for LLM (digit-stream encoding),
  one-hot grid for TRM, train/val split.
- Justify choice: matches the TRM benchmark setting → results
  directly comparable to published numbers.

### 3.5 Methods (30%, MO3 + MO4) — ~1.8 pages, distinction hook

This is the highest-leverage section. Lift heavily from
`results/novelty/README.md` §2 (the "alternative methods considered"
tables already do most of the rubric work).

**Subsections to write:**

1. **Model architectures (~0.4 page)**
   - TRM-MLP (sudoku, ~6.4 M params): 2-layer recursive net, deep supervision, attention-free per Jolicoeur-Martineau
   - TRM-Att (maze, ~8.4 M params): same backbone, self-attention block
   - Qwen2.5-0.5B + LoRA: low-rank fine-tune of a small pre-trained transformer

2. **Training protocol — alternatives table (~0.4 page) — DISTINCTION HOOK**
   Reproduce the iso-wall-clock vs iso-epoch vs iso-energy table from
   the novelty README §2, with rejection rationale per row. This
   single table satisfies "understanding of alternative methods".

3. **K-vote diversity sources — alternatives table (~0.4 page) — NOVELTY**
   Reproduce the MC-dropout / input-aug / latent-init / temperature
   table from the novelty README §2. State the one-sentence novelty:
   "we re-use Dillon (2026)'s training-time K=4 trick at K=1..16
   inference time."

4. **Energy measurement (~0.2 page)**
   CodeCarbon, NVML GPU power × country grid intensity (UK).
   Limitation flag: process-level, not wall-power.

5. **Ethical issues paragraph (~0.4 page) — MO4, mandatory for 70-100%**
   Three threads:
   a) **Energy/CO2**: our headline finding (TRM at ~2 kWh vs LLM
      training norm of 100s of kWh) is itself an ethics result.
   b) **Dual-use**: structured-reasoning systems used in safety-critical
      decisions (medical triage, legal scheduling). 84.8% accuracy is
      not deployment-grade — flag this.
   c) **Dataset provenance**: publicly released, no PII; reproducible.

### 3.6 Experiments (30%, MO3) — ~1.8 pages

**5.1 Iso-wall-clock comparison** (headline table + Pareto plot)

| Model | Task | Best puzzle acc | kWh | Wall (h) | Source |
|---|---|---|---|---|---|
| TRM-MLP (HF-init) | Sudoku | 84.8% | 1.93 | 8.7 | iso-time run |
| TRM-MLP 3-seed (from-scratch) | Sudoku | 74.6% ± 0.6% | 22.0 | 80.1 | `summary.csv` |
| TRM-Att (HF-eval) | Maze | 79.6% | — | 0 | `findings.md` §5.6 |
| Qwen2.5-0.5B + LoRA | Sudoku | 0% / 19% cell | 0.90 | 6.8 | `summary.csv` |

Plot: `iso_time_acc_vs_kwh.png` (already on disk).

**Critical analysis (the "appraisal" the rubric wants):**
- TRM hits its ceiling at epoch 10 of the iso-time run; the remaining
  140 epochs cause val accuracy to drop from 0.85 to 0.67. Report this
  as an empirical finding, not a failure to hide. Practical implication:
  HF-init reaches the model ceiling in ~30 minutes of fine-tune.
- Maze from-scratch fine-tunes corrupted HF weights (val_exact 0.789 →
  0.11 in one epoch — `findings.md` §5.7). Report and explain.
- Iso-time TRM-Att maze ran only 16 of 5000 scheduled epochs; we use
  the HF-eval number (0.796) as the headline.

**5.2 K-vote sensitivity (the novelty)**

What we have *right now*: cost-side only.

| K | kWh | Per-sample kWh |
|---|---|---|
| 1 | 0.43 | 0.43 |
| 2 | 1.42 | 0.71 |
| 4 | 2.20 | 0.55 |

Honest framing if accuracy data is missing at write time:
"K-vote inference cost grows super-linearly under our current
implementation; accuracy aggregation across the K samples remains in
progress."

**Decision point:** if `scripts/run_novelty_aggregate.py` produces
accuracy numbers before drafting starts, this section upgrades to a
full Pareto plot and the novelty section becomes a real result. That
single chart would lift this section from the 50-69% band to the
70-100% band.

**5.3 Limitations (this paragraph raises grades, doesn't lower them)**
- Single seed on novelty rows; no significance claims
- 3 of 6 proposal cells unfilled (LLM-maze, distill-sudoku, distill-maze)
- Iso-time budget too short for TRM-Att maze
- CodeCarbon process-level, not wall-power
- K-vote diversity source differs across families (TRM latent-init
  perturbation vs LLM temperature sampling)

### 3.7 Conclusion (5%, MO2 + MO4) — ~0.3 page

Three sentences, one each:
1. **What we measured**: TRM matches the published 87.4% within ~3
   percentage points at ~1.9 kWh; Qwen2.5-0.5B does not solve
   Sudoku-Extreme at any compute budget we tested.
2. **What we learned**: HF-init reaches the model ceiling in ~30
   minutes of fine-tune — practical implication for compute budgets.
3. **What's next**: the missing distillation arm, K-vote accuracy
   aggregation, multi-seed for the novelty rows.

### 3.8 Writing & Formatting (5%)

Use the provided Word template (Formatting Guidelines folder on
Blackboard). Conference-style two-column. Figures with captions and
in-text references. References in consistent style (the spec defers to
the formatting guidelines doc — check it).

### 3.9 Supplementary ZIP (allowed by spec p. 3)

Include:
- Jupyter notebook with the runnable pipeline (TRM training, LLM
  fine-tune, K-vote inference, plotting)
- `results/summary.csv`, `results/novelty/iso_time_results-rig1.csv`,
  K-vote `emissions.csv`
- `results/novelty/analysis_2026-04-25.md` (this run analysis)
- README pointing at the GitHub release for `C:/ml-trm-work` checkpoints
  if we publish one (keeps ZIP small, gives marker the option)

## 4. Six-day execution plan

| Day | Date | Action |
|---|---|---|
| 0 | Apr 25 (today) | Decide scope (narrow to 3 cells). Run `scripts/run_novelty_aggregate.py --ignore-missing` and confirm whether K-vote accuracy exists. Sign off on this plan. |
| 1 | Apr 26 | Optional: launch distill-sudoku (~6 h, teacher exists). If K-vote accuracy missing, debug `scripts/run_novelty_k_vote.py`. |
| 2 | Apr 27 | Draft Methods section (highest weight, strongest material). Generate plots. |
| 3 | Apr 28 | Draft Experiments section using whatever K-vote state exists. |
| 4 | Apr 29 | Draft Intro + Related Work + Data + Conclusion + Ethics. |
| 5 | Apr 30 | Format check using Word template. Cross-read by all three authors. Polish. |
| 6 | May 1 | Final read 09:00. Submit by 17:00. ZIP supplementary. |

## 5. Open decisions (need group sign-off)

1. **Narrow to 3 cells, or attempt distill-sudoku?**
   - Narrow: cleaner story, more time to write. Recommended.
   - Attempt distill-sudoku: 1 extra cell, ~6 h compute, must run by Apr 27 to leave time for K-vote re-eval.
2. **Headline TRM-sudoku number: 84.8% (HF-init) or 74.6% ± 0.6% (3-seed from-scratch)?**
   - 84.8% is higher and more comparable to the paper, but single seed.
   - 74.6% has error bars (rubric loves error bars) but is lower.
   - Recommendation: report **both** in the table, with 74.6% ± 0.6% as
     the "with-uncertainty" number and 84.8% as the "best achievable".
3. **K-vote section framing** (depends on aggregate run output):
   - Full Pareto if accuracy emerges → real novelty result.
   - Cost-only if not → "design + measured cost; accuracy aggregation
     pending" — still defensible.
4. **GitHub release for `C:/ml-trm-work`?**
   - Decision pending from prior conversation about release scope.
   - Either way, supplementary ZIP can link to it as optional fetch.
