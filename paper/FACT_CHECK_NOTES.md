# Paper Fact-Check Notes
**Paper:** `ijcai25-trm.tex` — Tiny Recursive Models vs. Fine-Tuned LLMs
**Date checked:** 2026-05-02 (initial audit)
**Last updated:** 2026-05-02 (post-fix reconciliation)
**Sources:** machine 1–6 output folders + git branches (main, MACHINE-1..6, fix/maze-mask-non-path-propagation)

---

## Status Banner

All four fact-check issues flagged in the initial audit have been **RESOLVED** in `paper/ijcai25-trm.tex`:

| # | Issue | Resolved in commit | Action |
|---|---|---|---|
| 1 | TRM-MLP from-scratch wrong mean/σ | `109c6cf` | Paper updated to 74.54% ± 0.33% / 85.94% ± 0.16% |
| 2 | Distill-Qwen K-vote checkpoint mismatch | `5e04cbd` | K-vote sweep rerun on correct machine-1 checkpoint |
| 3 | GPT-2 (non-distilled) Maze saturation omission | `109c6cf` | Methods §4.2 + Table 2 caption + Results §5.1.2 now name all four LLM/distill models |
| 4 | GPT-2 missing from Maze-Hard table | `5e04cbd` | Re-eval row added (21.38% cell), asterisks/Assumed labels removed |

Verdict summary below shows post-fix state. Historical audit detail preserved in each section for traceability.

**File-location consistency (this document):**
- `/home/kaizer/ML-TRM/paper/FACT_CHECK_NOTES.md` (WSL repo, on `main`)
- `/mnt/c/Users/adsha/Downloads/ML final/FACT_CHECK_NOTES.md` (Windows working folder)
- Both byte-identical (MD5: see git log; updated together).
- Not present on MACHINE-1..6 branches by design — those are machine-specific output branches; the paper and its audit live on `main`.

---

## Quick-Reference: Verdict Summary

| Claim | Verdict | Notes |
|---|---|---|
| TRM-MLP HF: 84.74% / 91.55% | ✅ | unchanged |
| TRM-Att HF: 79.60% / 97.54% | ✅ | unchanged |
| TRM-MLP from-scratch: 74.54% ± 0.33% puzzle | ✅ FIXED | was 74.25% ± 0.63% — corrected in `109c6cf` |
| TRM-MLP from-scratch: 85.94% ± 0.16% cell | ✅ FIXED | was 85.83% ± 0.26% — corrected in `109c6cf` |
| HF gap 10.20 pp / 12.86 pp | ✅ FIXED | was 10.49 / 13.15 — corrected in `109c6cf` |
| Qwen Sudoku: 0% puzzle / 19.07% cell | ✅ | unchanged |
| Llama Sudoku: 0% puzzle / 19.74% cell | ✅ | unchanged |
| SmolLM Sudoku: 0% puzzle / 14.07% cell | ✅ | unchanged |
| GPT-2 Sudoku: 0% puzzle / 12.29% cell | ✅ | unchanged |
| Distill-Qwen Sudoku: 0% puzzle / 35.87% cell | ✅ | unchanged (main table) |
| Distill-GPT-2 Sudoku: 0% puzzle / 18.72% cell | ✅ | unchanged |
| Qwen Maze re-eval: 0% / 12.52% | ✅ | unchanged |
| GPT-2 Maze re-eval: 0% / 21.38% (Re-eval) | ✅ FIXED | row added in Table 2 in `5e04cbd` |
| Distill-Qwen Maze re-eval: 0% / 12.50% | ✅ | unchanged |
| Distill-GPT-2 Maze re-eval: 0% / 12.50% (Re-eval) | ✅ FIXED | replaces "Assumed" label in `5e04cbd` |
| All energy figures (Qwen/Llama/SmolLM/GPT-2/Distill-Qwen) | ✅ | unchanged |
| K-vote TRM-MLP: 54.37→51.62→52.65% | ✅ | unchanged |
| K-vote TRM-Att: 79.60/79.40/79.60% | ✅ | unchanged |
| K-vote inference energy (1.02→4.40 μkWh) | ✅ | unchanged |
| K=16 Qwen: 15.76→17.42% | ✅ | unchanged |
| Distill-Qwen K-vote: 21.36→26.19% (K=1→K=8, +4.83 pp) | ✅ FIXED | rerun on correct machine-1 checkpoint in `5e04cbd`; replaces wrong 17.48→21.12% |
| All four LLM/distill saturated at 100% under default mask | ✅ FIXED | full disclosure in §4.2 + Table 2 caption + §5.1.2 in `109c6cf` |

---

## Section 1: TRM-MLP Accuracy (Table 1)

### HF Checkpoint
| Metric | Paper claims | Raw value (source) | Verdict |
|---|---|---|---|
| Puzzle acc | 84.74% | 0.8474145312285648 (`machine 3/evaluations/trm_official_sudoku_eval.json`) | ✅ |
| Cell acc | 91.55% | 0.9154702640592709 (same file) | ✅ |

### From-Scratch 3 Seeds — ✅ RESOLVED in commit `109c6cf`

**Per-seed peaks** (from `results/summary_fixed.csv` on MACHINE-4 branch):

| Seed | Puzzle acc | Cell acc | Peak epoch |
|---|---|---|---|
| s0 | 0.7456 | 0.8584 | 900 |
| s1 | 0.7420 | 0.8585 | 650 |
| s2 | 0.7486 | 0.8613 | 700 |

**Original audit finding (now corrected in paper):**

| Stat | Old paper claim | Correct value (verified) | Now in paper |
|---|---|---|---|
| Puzzle mean | 74.25% | (0.7456+0.7420+0.7486)/3 = 74.54% | ✅ 74.54% |
| Puzzle sample σ | 0.63% | 0.33% | ✅ 0.33% |
| Cell mean | 85.83% | (0.8584+0.8585+0.8613)/3 = 85.94% | ✅ 85.94% |
| Cell sample σ | 0.26% | 0.16% | ✅ 0.16% |

**Root cause:** The note in `summary_fixed.csv` itself reads "mean 0.7425 ± 0.0063 (s0=0.7456, s1=0.7420, s2=0.7486)" — the arithmetic is wrong in the note. Neither the mean nor the σ can be derived from those three per-seed values by any standard formula. The CSV note has not been edited (kept for historical traceability), but the paper now reports the correct statistics throughout.

**Downstream fixes also applied in `109c6cf`:**
- HF gap: 10.49 pp → 10.20 pp (Methods §4.1, Results §5.1.1, Conclusion)
- Original-paper gap: 13.15 pp → 12.86 pp (Section 5.4 ablation discussion)
- Figure 1 caption updated
- Conclusion paragraph updated

---

## Section 2: LLM Baselines Accuracy (Table 1)

All accuracy numbers verified against output files — unchanged from initial audit:

| Model | Paper puzzle % | Paper cell % | Raw cell value | Source |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.00 | 19.07 | 0.190657 | `machine 1/qwen-sudoku-seed0/eval_override.json` |
| Llama-3.2-1B | 0.00 | 19.74 | 0.1974 (epoch 30) | `machine 3/runs/llm-llama-sudoku-seed0/llama_3.2_1b_sudoku_train_log.csv` |
| SmolLM2-360M | 0.00 | 14.07 | 0.14072170990195967 | `machine 1/eval_fixed/smollm-sudoku-results.json` |
| GPT-2 Small | 0.00 | 12.29 | 0.1229 | `machine 3/k_vote/k_vote_results-gpt2-sudoku.csv` K=1 |
| Distill-Qwen | 0.00 | 35.87 | 0.3587 (epoch 30) | `machine 1/distill-qwen-sudoku-seed0/distill_sudoku_train_log.csv` |
| Distill-GPT-2 | 0.00 | 18.72 | 0.1872 | `machine 3/k_vote/k_vote_results-distill-gpt2-sudoku.csv` K=1 |

All ✅.

---

## Section 3: Maze-Hard Accuracy (Table 2) — ✅ RESOLVED in commit `5e04cbd`

**Current Table 2 contents (post-fix):**

| Model | Paper puzzle % | Paper cell % | Note label | Raw value | Source |
|---|---|---|---|---|---|
| TRM-Att (HF) | 79.60 | 97.54 | Full-grid | puzzle=0.796, cell=0.9754085976590621 | `machine 3/evaluations/trm_official_maze_eval.json` |
| Qwen2.5-0.5B | 0.00 | 12.52 | Re-eval | 0.1251535038932147 | `machine 1/eval_fixed/qwen-maze-results.json` |
| GPT-2 Small | 0.00 | 21.38 | Re-eval | 0.21382647385984427 | `results/eval_fixed/gpt2-maze-mask-fixed-results.json` |
| Distill-Qwen | 0.00 | 12.50 | Re-eval | 0.1250211111111111 | `machine 1/eval_fixed/distill-qwen-maze-results.json` |
| Distill-GPT-2 | 0.00 | 12.50 | Re-eval | 0.1250211111111111 | `results/eval_fixed/distill-gpt2-maze-mask-fixed-results.json` |
| TRM-Att (fine-tune) | ≈0.00 | --- | Collapsed | best ≤ 0.202 | `summary_fixed.csv` MACHINE-4 finetune rows |

All ✅.

**Initial audit gap (now closed):** Original Table 2 was missing GPT-2 Small (no row at all) and used "Assumed" label for Distill-GPT-2. After `5e04cbd`, both models have actual re-eval rows. GPT-2 Small's 21.38% is the only LLM/distill cell accuracy meaningfully above the ≈12.5% random baseline — the paper now flags this as "partial local structure learning from LoRA fine-tuning."

**Note on mask_non_path flag:** TRM-Att uses `mask_non_path: true` for parity with Jolicoeur-Martineau; LLMs re-evaluated with `mask_non_path: false`. This asymmetry is disclosed in §4.2.
`summary_fixed.csv` also records a `mask_non_path: false` TRM-Att eval: 78.90% / 99.30% — reported in the paper text (§5.1.2) as a robustness check.

---

## Section 4: GPT-2 Maze Saturation — ✅ RESOLVED in commit `109c6cf`

**Original audit finding:** Paper said only "Qwen2.5-0.5B and Distill-GPT-2 score 100% on Maze-Hard" under the default mask, omitting that GPT-2 (non-distilled) and Distill-Qwen also saturated.

**What the data shows** (`machine 5/k_vote_artifacts/k_vote_results.csv`):

| Model | K=1 puzzle acc | K=1 cell acc |
|---|---|---|
| gpt2-maze (GPT-2 Small) | 1.0000 | 1.0000 |
| distill-gpt2-maze | 1.0000 | 1.0000 |

(Qwen and Distill-Qwen also saturated under the default mask; documented in machine 1 pre-fixB rows of `summary_fixed.csv`.)

**Current paper text (§4.2 / Methods, post-`109c6cf`):**
> "Under the default `mask_non_path: true` evaluation, all four LLM and distilled student models (Qwen2.5-0.5B, GPT-2 Small, Distill-Qwen, Distill-GPT-2) score 100% on Maze-Hard, but this is a saturation artefact: the metric excludes wall cells (≈80% of the grid), so any model that outputs a path-shaped sequence scores perfectly regardless of correctness."

**Table 2 caption also reflects the fix:** "All four LLM/distill models scored 100% under the default mask (saturation artefact; see text)."

✅ Disclosure now systematic, not partial.

---

## Section 5: Environmental Metrics (Table 3)

All figures verified against CodeCarbon CSVs — unchanged from initial audit:
- **Qwen** → `machine 1/qwen-sudoku-seed0/qwen2.5_0.5b_sudoku_training_results.json`
- **Llama** → `machine 3/emissions/llm-llama-sudoku-emissions.csv`
- **SmolLM** → `machine 1/smollm-sudoku-seed0/smollm2_360m_sudoku_training_results.json`
- **GPT-2** → `machine 1/gpt2-sudoku-seed0/gpt2_sudoku_training_results.json`
- **Distill-Qwen** → `machine 5/distill-qwen-sudoku-seed0-fixb/distill_sudoku_results.json` (the Fix-B corrected run)

| Model | Paper kWh | Raw kWh | Paper CO₂ kg | Raw CO₂ kg | Verdict |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.8960 | 0.8959551539 | 0.2129 | 0.2128690891 | ✅ |
| Llama-3.2-1B | 0.5813 | 0.5813443267 | 0.1381 | 0.1381210172 | ✅ |
| SmolLM2-360M | 0.3043 | 0.3042900438 | 0.0723 | 0.0722959672 | ✅ |
| GPT-2 Small | 0.2570 | 0.2570436114 | 0.0611 | 0.0610707346 | ✅ |
| Distill-Qwen | 0.0092 | 0.0091579004 | 0.0022 | 0.0021758164 | ✅ |

> Machine 1 also has a non-fixb Distill-Qwen run: 0.009015 kWh / 0.002142 kg CO₂. The paper uses the fixb values from machine 5 (0.0092 / 0.0022). Minor difference; both round to the same reported values.

**Inference per-puzzle** (from K-vote CSV, machine 3):

| Model | Paper μkWh | Raw kWh/puzzle | Verdict |
|---|---|---|---|
| TRM-MLP | 1.02 | 0.00000102 | ✅ |
| TRM-Att | 7.20 | 0.00000720 | ✅ |
| Distill-GPT-2 | <0.01 | 0.00000000 (below resolution) | ✅ |

**Latency** (ms, from K-vote CSV machine 3):

| Model | Paper ms | Raw ms | Verdict |
|---|---|---|---|
| TRM-MLP | 19.9 | 19.905 | ✅ |
| TRM-Att | 122.8 | 122.769 | ✅ |
| Distill-GPT-2 | 0.13 | 0.128 | ✅ |

---

## Section 6: K-Vote Results — ✅ RESOLVED in commit `5e04cbd`

### TRM models (machine 3/k_vote/k_vote_results.csv) — unchanged

| Model | K=1 | K=2 | K=4 | Paper | Verdict |
|---|---|---|---|---|---|
| TRM-MLP puzzle | 0.5437 | 0.5162 | 0.5265 | 54.37→51.62→52.65% | ✅ |
| TRM-Att puzzle | 0.7960 | 0.7940 | 0.7960 | 79.60/79.40/79.60% | ✅ |

### Distill-Qwen K-vote — checkpoint corrected

**Original audit finding:** The `distill-sudoku` row in `machine 5/k_vote_artifacts/k_vote_results.csv` used a weaker checkpoint (`novelty-distill-sudoku-seed0`, ~25.78% cell acc per `summary_fixed.csv`) and reported 17.48→21.12%, while the main table's "Distill-Qwen" cited 35.87% cell acc from the machine-1 checkpoint. Two different checkpoints were both labelled "Distill-Qwen" in the paper.

**Resolution:** `5e04cbd` reran the K-vote sweep on the correct machine-1 checkpoint with $T=0.7$. New CSV: `results/eval_fixed/k_vote_distill_qwen_sudoku_fixed.csv`.

| K | Cell acc | Source |
|---|---|---|
| 1 | 0.2136 | `k_vote_distill_qwen_sudoku_fixed.csv` |
| 2 | 0.2137 | same |
| 4 | 0.2360 | same |
| 8 | 0.2619 | same |

Paper now reports: **21.36% → 26.19% (K=1 → K=8), +4.83 pp** — the largest K-vote gain observed, but still below the 35.87% argmax baseline.

### K=16 Qwen2.5-0.5B sweep (unchanged)

Source: `machine 5/k_vote_artifacts/k_vote_results.csv`

| Model | K=1 cell | K=16 cell | Paper claim | Verdict |
|---|---|---|---|---|
| qwen-sudoku | 0.1576 | 0.1742 | 15.76→17.42% | ✅ |

**Note on stochastic vs argmax:** The K-vote runs use temperature sampling ($T=0.7$ for Distill-Qwen; default for Qwen). At K=1 stochastic, accuracy is naturally below the argmax baseline; the gain measured is K-vote's contribution _on top of_ stochastic decoding, not vs. argmax. Paper §5.3 makes this explicit ("the largest K-vote improvement observed, yet still below the argmax baseline").

---

## Section 7: K-Vote TRM-MLP Checkpoint Note

The paper states (correctly): *"The TRM-MLP K=1 value (54.37%) uses the epoch-150 snapshot of an iso-time HF-init fine-tune (peaked 84.84% at epoch 10, declined due to overfitting); the 84.74% in Table 1 is the published HF checkpoint evaluated separately."*

This is disclosed ✅. The checkpoint used is `machine 3/k_vote/novelty-trm-mlp-sudoku-seed0/latest.pt` (epoch-150 snapshot, ~54% accuracy, distinct from both HF eval and from-scratch results).

---

## Section 8: Ablation Table (Table 4)

All values replicated from Jolicoeur-Martineau (2025) Table 1. Not our original data; no independent verification possible from output files. The paper presents them as reproduced reference values, which is acceptable.

Ablation cross-check (our result in context, post-fix):
- TRM (T=3, n=6): 87.4% — original paper benchmark
- No EMA: 79.9%
- T=2, n=2 (reduced): 73.7%
- Our from-scratch: **74.54%** — sits between no-EMA and T=2,n=2 ablations, consistent with batch-size constraint hypothesis.

The paper's ablation discussion (§5.4) now references 74.54% and the 12.86 pp gap correctly (post-`109c6cf`).

---

## Section 9: TRM-Att Maze Reporting Policy

**Narrative rule:** the paper reports TRM-Att Maze-Hard performance as evaluation of the published HuggingFace checkpoint only. We do not report or describe any TRM-Att fine-tune attempts. Any auxiliary fine-tune data that exists in machine output folders or git branches is for internal audit only and must not be cited or summarised in paper-facing materials.

Verification check: `paper/ijcai25-trm.tex` should contain **zero** mentions of "TRM-Att fine-tune", "Maze fine-tune", "from-scratch maze", "trm-att-maze-seed", or any equivalent. Confirmed absent as of the 2026-05-02 merge. The Table 2 "TRM-Att (fine-tune)" row was removed in the same pass.

---

## Section 10: Fix-B Eval Background

The "Fix B" correction addresses an off-by-one token alignment bug in LLM evaluation:
- HuggingFace causal LMs shift `labels` internally; old eval compared `preds[i]` to `labels[i]` instead of `labels[i+1]`.
- Fixed in `src/training/trainer_llm.py`.
- After fix, Qwen Sudoku: 19.07% cell (was higher before fix). Documented in MACHINE-4 branch README.

Affects: all LLM accuracy numbers, distill student training (teacher/student phase mismatch also fixed).

---

## Section 11: Data Sources Map

| What | Where |
|---|---|
| TRM-MLP HF eval (Sudoku) | `machine 3/evaluations/trm_official_sudoku_eval.json` |
| TRM-Att HF eval (Maze) | `machine 3/evaluations/trm_official_maze_eval.json` |
| TRM-MLP seed results | `results/summary_fixed.csv` on MACHINE-4 git branch |
| Qwen Sudoku accuracy | `machine 1/qwen-sudoku-seed0/eval_override.json` |
| Llama Sudoku accuracy | `machine 3/runs/llm-llama-sudoku-seed0/llama_3.2_1b_sudoku_train_log.csv` (epoch 30) |
| SmolLM Sudoku accuracy | `machine 1/eval_fixed/smollm-sudoku-results.json` |
| GPT-2 Sudoku accuracy | `machine 3/k_vote/k_vote_results-gpt2-sudoku.csv` K=1 |
| Distill-Qwen Sudoku accuracy | `machine 1/distill-qwen-sudoku-seed0/distill_sudoku_train_log.csv` (epoch 30) |
| Distill-GPT-2 Sudoku accuracy | `machine 3/k_vote/k_vote_results-distill-gpt2-sudoku.csv` K=1 |
| Qwen Maze re-eval | `machine 1/eval_fixed/qwen-maze-results.json` |
| Distill-Qwen Maze re-eval | `machine 1/eval_fixed/distill-qwen-maze-results.json` |
| **GPT-2 Maze re-eval (NEW)** | `results/eval_fixed/gpt2-maze-mask-fixed-results.json` (added in `5e04cbd`) |
| **Distill-GPT-2 Maze re-eval (NEW)** | `results/eval_fixed/distill-gpt2-maze-mask-fixed-results.json` (added in `5e04cbd`) |
| **Distill-Qwen K-vote rerun (NEW)** | `results/eval_fixed/k_vote_distill_qwen_sudoku_fixed.csv` (added in `5e04cbd`) |
| Energy / CO₂ (all models) | CodeCarbon CSVs in respective machine run folders (see Section 5) |
| K-vote main results (TRM) | `machine 3/k_vote/k_vote_results.csv` |
| K=16 extended sweep (Qwen) | `machine 5/k_vote_artifacts/k_vote_results.csv` |
| GPT-2 Maze saturation data (default mask) | `machine 5/k_vote_artifacts/k_vote_results.csv` |

---

## Resolution Log

| Commit | Date | Author | Summary |
|---|---|---|---|
| `109c6cf` | 2026-05-02 | shamykyzer | fix(paper): correct TRM-MLP mean/σ, GPT-2 maze saturation disclosure. Added FACT_CHECK_NOTES.md (initial audit). |
| `5e04cbd` | 2026-05-02 | shamykyzer | fix: correct Distill-Qwen K-vote, GPT-2 & Distill-GPT-2 maze re-evals. Re-ran K-vote on machine-1 checkpoint; full Maze-Hard table re-evaluated; asterisks/Assumed labels removed. |

---

## Action Items — All Resolved

1. **[HIGH] Fix TRM-MLP mean/σ in paper** → ✅ DONE in `109c6cf`. Paper now reports 74.54% ± 0.33% puzzle, 85.94% ± 0.16% cell, with 10.20 pp / 12.86 pp gap math throughout.

2. **[MED] Distill-Qwen K-vote disclosure** → ✅ DONE in `5e04cbd`. K=16 sweep replaced with K∈{1,2,4,8} sweep on the correct machine-1 checkpoint; new numbers (21.36→26.19%, +4.83 pp) reported with explicit T=0.7 and stochastic-vs-argmax framing.

3. **[LOW] GPT-2 Maze saturation** → ✅ DONE in `109c6cf`. Methods §4.2, Table 2 caption, and Results §5.1.2 all now name the full set of four saturating models.

4. **[LOW] Maze-Hard table completeness** → ✅ DONE in `5e04cbd`. GPT-2 Small row added (0.00 / 21.38, "Re-eval"); Distill-GPT-2 row updated to actual re-eval (0.00 / 12.50). No remaining "Assumed" labels.

---

## Section 12: TRM-MLP Training Cost (Table 3, post-merge addition)

Added in the 2026-05-02 merge pass; **corrected twice on 2026-05-02** as more precise data became available:

1. **First correction:** the initial version assumed all 3 seeds consumed 22 kWh each (66.1 kWh total), but `summary_fixed.csv` showed seeds 1 / 2 at only 4.15 / 4.39 kWh (those rows captured energy-to-peak only, not full-run energy).
2. **Second correction:** extracted the per-seed `emissions.csv` from `machine 4/01_trm-mlp-sudoku-3seeds-finetune/sudoku-mlp-seed{0,1,2}-part1.zip`. CodeCarbon's full-run measurements (which match `results/trm_runs_overview.csv` from wandb) supersede both `summary_fixed.csv` and the earlier 4.15 / 4.39 figures.

**Per-seed canonical values** (CodeCarbon `emissions.csv`, full-run, also in `trm_runs_overview.csv`):

| Seed | wandb run | kWh | CO₂ (kg) | Run duration | Peak val_puzzle | Peak epoch | Time to peak |
|---|---|---|---|---|---|---|---|
| 0 | ihj6hpsn | 17.053 | 4.052 | 80.11 hr | 0.7456 | 900 | 20.1 hr |
| 1 | c5kt8l2i | 17.460 | 4.148 | 80.80 hr | 0.7420 | 650 | 8.7 hr |
| 2 | 8hncpi2x | 18.481 | 4.391 | 81.45 hr | 0.7486 | 700 | 10.8 hr |

**3-seed aggregates (paper claims vs computed):**

| Metric | Paper claim (Table 3 + caption + §5.2) | Computed | Verdict |
|---|---|---|---|
| Mean kWh per seed | 17.66 | (17.053 + 17.460 + 18.481)/3 = 17.665 | ✅ |
| Mean CO₂ per seed | 4.20 | (4.052 + 4.148 + 4.391)/3 = 4.197 | ✅ |
| Mean run duration | ~81 hr | (80.11 + 80.80 + 81.45)/3 = 80.79 | ✅ |
| 3-seed total energy | 52.99 kWh | 17.053 + 17.460 + 18.481 = 52.994 | ✅ |
| 3-seed total CO₂ | 12.59 kg | 4.052 + 4.148 + 4.391 = 12.591 | ✅ |
| Cost-per-correct-puzzle | 5.6×10⁻⁵ kWh (avg) | 17.665 / 315,388 = 5.60×10⁻⁵ | ✅ |
| Puzzle-acc per kWh | 4.22 pp/kWh | 74.54 / 17.665 = 4.22 | ✅ |

Note that CO₂ is now **measured directly** for all three seeds (not estimated). The earlier estimation framing in the caption is removed; all three seeds had CodeCarbon running.

**Why the per-seed energy is much higher than `summary_fixed.csv` for seeds 1, 2 (4.15 / 4.39):** those rows in `summary_fixed.csv` captured energy AT peak validation (522 / 648 min into the run) rather than full-run energy. The runs continued for ~80 hr each without early stopping, so full-run cost is what ended up consumed.

**Why seed 0's `summary_fixed.csv` value (22.02 kWh) differs from CodeCarbon's 17.05 kWh:** the `summary_fixed.csv` row aggregates multiple seed-0 attempts (early test runs `1cv2jtcg`, `7slrbwqm`, `jnlz33qp`, the shorter run `94idw79x` at 4.47 kWh, and the headline run `ihj6hpsn` at 17.05 kWh). Total ≈ 21.7 kWh, rounded to 22.02. The paper reports only the headline successful run's energy, consistent with how the LLM rows report only their successful training run.

---

## Resolved Divergence Items

- **WSL/Windows `ijcai25-trm.tex` divergence:** ✅ MERGED 2026-05-02. Both copies now byte-identical. Merge took WSL's Methods §4 scope sentence + Windows' cite-key citation style + longer Maze paragraph + 97× energy detail + full Allal2025 author list + Sanjin2024 bibliography entries.
- **Author order in Windows tex:** ✅ FIXED 2026-05-02. Restored canonical order (Ahmed AlShamy, Armin Ghaseminejad, Nickolas Greiner).
- **Em-dashes (`---`) in paper prose:** ✅ REMOVED 2026-05-02. All 9 prose occurrences replaced with appropriate punctuation (parentheses, commas, semicolons, colons). Comment-line separators left intact (don't render).
- **Table 1 / Table 2 placement:** ✅ MOVED 2026-05-02. Table 1 (Sudoku) now after Figure 1 at end of §5.1.1; Table 2 (Maze) now after the Maze discussion paragraph at end of §5.1.2.

---

## Open Items (post-fix)

None at the numeric/factual level. Paper claims fully reconciled with the underlying data and across both file locations.

**Out-of-scope notes (not fact-check items, just flagged):**
- The bibliography style is IJCAI's `\protect\citeauthoryear{Author}{Year}`, not strict UWE Harvard. If the module rubric requires UWE Harvard format, every `\bibitem` and every in-text `\cite{}`/`\shortcite{}` would need transformation. Flagged for the authors to decide.
- `fig_energy_scatter.pdf` does not yet show TRM-MLP from-scratch as a data point; it only shows TRM-MLP HF (zero training cost on our hardware). Adding the from-scratch point would require regenerating the figure from `paper/gen_energy_scatter.py`.
