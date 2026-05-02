# Paper Fact-Check Notes
**Paper:** `ijcai25-trm.tex` — Tiny Recursive Models vs. Fine-Tuned LLMs  
**Date checked:** 2026-05-02  
**Sources:** machine 1–6 output folders + git branches (main, MACHINE-1..6, fix/maze-mask-non-path-propagation)

---

## Quick-Reference: Verdict Summary

| Claim | Verdict |
|---|---|
| TRM-MLP HF: 84.74% / 91.55% | ✅ |
| TRM-Att HF: 79.60% / 97.54% | ✅ |
| TRM-MLP from-scratch: **74.25% ± 0.63%** puzzle | ⚠️ **WRONG MEAN** — actual 74.54% ± 0.33% |
| TRM-MLP from-scratch: **85.83% ± 0.26%** cell | ⚠️ **WRONG MEAN** — actual 85.94% ± 0.16% |
| Qwen Sudoku: 0% puzzle / 19.07% cell | ✅ |
| Llama Sudoku: 0% puzzle / 19.74% cell | ✅ |
| SmolLM Sudoku: 0% puzzle / 14.07% cell | ✅ |
| GPT-2 Sudoku: 0% puzzle / 12.29% cell | ✅ |
| Distill-Qwen Sudoku: 0% puzzle / 35.87% cell | ✅ (main table only) |
| Distill-GPT-2 Sudoku: 0% puzzle / 18.72% cell | ✅ |
| Qwen Maze re-eval: 0% / 12.52% | ✅ |
| Distill-Qwen Maze re-eval: 0% / 12.50% | ✅ |
| All energy figures (Qwen/Llama/SmolLM/GPT-2/Distill-Qwen) | ✅ |
| K-vote TRM-MLP: 54.37→51.62→52.65% | ✅ |
| K-vote TRM-Att: 79.60/79.40/79.60% | ✅ |
| K-vote inference energy (1.02→4.40 μkWh) | ✅ |
| K=16 Qwen: 15.76→17.42%, Distill: 17.48→21.12% | ✅ numbers, ⚠️ **checkpoint mismatch** |
| GPT-2 Maze saturation disclosure | ⚠️ **PARTIAL** — paper omits GPT-2 (non-distilled) also hit 100% |

---

## Section 1: TRM-MLP Accuracy (Table 1)

### HF Checkpoint
| Metric | Paper claims | Raw value (source) | Verdict |
|---|---|---|---|
| Puzzle acc | 84.74% | 0.8474145312285648 (`machine 3/evaluations/trm_official_sudoku_eval.json`) | ✅ |
| Cell acc | 91.55% | 0.9154702640592709 (same file) | ✅ |

### From-Scratch 3 Seeds — ⚠️ ARITHMETIC ERROR IN PAPER

**Per-seed peaks** (from `results/summary_fixed.csv` on MACHINE-4 branch):

| Seed | Puzzle acc | Cell acc | Peak epoch |
|---|---|---|---|
| s0 | 0.7456 | 0.8584 | 900 |
| s1 | 0.7420 | 0.8585 | 650 |
| s2 | 0.7486 | 0.8613 | 700 |

**Calculated vs. claimed:**

| Stat | Paper claims | Correct value | Error |
|---|---|---|---|
| Puzzle mean | **74.25%** | **(0.7456+0.7420+0.7486)/3 = 74.54%** | 0.29 pp too low |
| Puzzle sample σ | **0.63%** | **0.33%** | 0.30 pp inflated |
| Cell mean | **85.83%** | **(0.8584+0.8585+0.8613)/3 = 85.94%** | 0.11 pp too low |
| Cell sample σ | **0.26%** | **0.16%** | 0.10 pp inflated |

**Root cause:** The note in `summary_fixed.csv` itself reads "mean 0.7425 ± 0.0063 (s0=0.7456, s1=0.7420, s2=0.7486)" — the arithmetic is wrong in the note. Neither the mean nor the σ can be derived from those three per-seed values by any standard formula.

**Likely fix:** Change paper to **74.54% ± 0.33%** puzzle, **85.94% ± 0.16%** cell.

> The gap to the HF checkpoint changes: 84.74% − 74.54% = **10.20 pp** (paper says 10.49 pp).  
> The gap to the original published result (87.4%) changes: 87.4% − 74.54% = **12.86 pp** (paper says 13.15 pp).

---

## Section 2: LLM Baselines Accuracy (Table 1)

All accuracy numbers verified against output files:

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

## Section 3: Maze-Hard Accuracy (Table 2)

| Model | Paper puzzle % | Paper cell % | Raw value | Source |
|---|---|---|---|---|
| TRM-Att HF | 79.60 | 97.54 | puzzle=0.796, cell=0.9754085976590621 | `machine 3/evaluations/trm_official_maze_eval.json` |
| Qwen2.5-0.5B re-eval | 0.00 | 12.52 | 0.1251535038932147 | `machine 1/eval_fixed/qwen-maze-results.json` |
| Distill-Qwen re-eval | 0.00 | 12.50 | 0.1250211111111111 | `machine 1/eval_fixed/distill-qwen-maze-results.json` |

All ✅.

**Note on mask_non_path flag:** TRM-Att uses `mask_non_path: true` (default); LLMs re-evaluated with `mask_non_path: false`. This asymmetry is disclosed in Section 4.2.  
`summary_fixed.csv` also records a `mask_non_path: false` TRM-Att eval: 78.90% / 99.30% — not reported in the paper but consistent with the approach.

---

## Section 4: GPT-2 Maze Saturation — ⚠️ OMISSION

The paper (Section 4.2) says: *"Qwen2.5-0.5B and Distill-GPT-2 score 100% on Maze-Hard"* (mask_non_path: true bug).

**What the data actually shows** (`machine 5/k_vote_artifacts/k_vote_results.csv`):

| Model | K=1 puzzle acc | K=1 cell acc |
|---|---|---|
| qwen-sudoku (Qwen2.5-0.5B) | — | — |
| gpt2-maze | **1.0000** | **1.0000** |
| distill-gpt2-maze | **1.0000** | **1.0000** |

**Non-distilled GPT-2 on Maze also hit 100%** under the default mask. The paper does not mention this. The Maze-Hard table also omits GPT-2 (non-distilled) entirely — its row is missing.

**Recommendation:** Add a footnote or sentence acknowledging GPT-2 (non-distilled) also saturated at 100% on Maze under the default metric.

---

## Section 5: Environmental Metrics (Table 3)

All figures verified against CodeCarbon CSVs. Paper uses:
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

## Section 6: K-Vote Results

### TRM models (machine 3/k_vote/k_vote_results.csv)

| Model | K=1 | K=2 | K=4 | Paper | Verdict |
|---|---|---|---|---|---|
| TRM-MLP puzzle | 0.5437 | 0.5162 | 0.5265 | 54.37→51.62→52.65% | ✅ |
| TRM-Att puzzle | 0.7960 | 0.7940 | 0.7960 | 79.60/79.40/79.60% | ✅ |

### Extended K=16 sweep — ⚠️ CHECKPOINT MISMATCH

Source: `machine 5/k_vote_artifacts/k_vote_results.csv`

| Model | K=1 cell | K=16 cell | Paper claim | Verdict |
|---|---|---|---|---|
| qwen-sudoku | 0.1576 | 0.1742 | 15.76→17.42% | ✅ numbers |
| distill-sudoku | 0.1748 | 0.2112 | 17.48→21.12% ("Distill-Qwen") | ✅ numbers, **⚠️ labeling** |

**Problem with "Distill-Qwen" label in K-vote:**

The K-vote `distill-sudoku` checkpoint is `C:\ml-trm-work\novelty-distill-sudoku-seed0\distill_sudoku_latest.pt` — this is machine 5's `distill-sudoku-seed0` (train log final val_cell_acc ≈ 25.80%).

The main table's "Distill-Qwen" uses `machine 1/distill-qwen-sudoku-seed0` (train log final val_cell_acc = **35.87%**).

These are **different model checkpoints**. At K=1, a model should score its regular eval accuracy. But:
- Main table Distill-Qwen cell acc = **35.87%**
- K-vote Distill-Qwen K=1 cell acc = **17.48%**

A gap of 18+ pp between K=1 and the stated cell accuracy is inconsistent. The paper labels both as "Distill-Qwen" without disclosing they come from different checkpoints.

**Recommendation:** Either re-run the K=16 sweep on the machine 1 checkpoint, or explicitly note in the K-vote section that a different (earlier/weaker) Distill-Qwen checkpoint was used for the extended sweep.

---

## Section 7: K-Vote TRM-MLP Checkpoint Note

The paper states (correctly): *"The TRM-MLP K=1 value (54.37%) uses the epoch-150 snapshot of an iso-time HF-init fine-tune (peaked 84.84% at epoch 10, declined due to overfitting); the 84.74% in Table 1 is the published HF checkpoint evaluated separately."*

This is disclosed ✅. The checkpoint used is `machine 3/k_vote/novelty-trm-mlp-sudoku-seed0/latest.pt` (epoch-150 snapshot, ~54% accuracy, distinct from both HF eval and from-scratch results).

---

## Section 8: Ablation Table (Table 4)

All values replicated from Jolicoeur-Martineau (2025) Table 1. Not our original data; no independent verification possible from output files. The paper presents them as reproduced reference values, which is acceptable.

Ablation cross-check (our result in context):
- TRM (T=3, n=6): 87.4% — original paper benchmark
- No EMA: 79.9%
- T=2, n=2 (reduced): 73.7%
- Our from-scratch: 74.25% (claimed) / **74.54% (actual)** — sits between no-EMA and T=2,n=2 ablations

---

## Section 9: Maze Fine-Tune Collapse

Paper claims: "Three from-scratch seeds all collapsed; the batch-size gap is the dominant cause."

Confirmed from `results/summary_fixed.csv` (MACHINE-4 branch):
- `trm-att-maze-seed0-finetune`: best puzzle = 0.202 (flagged NOT FOR HEADLINE)
- `trm-att-maze-seed1-finetune`: best puzzle = 0.189 (flagged NOT FOR HEADLINE)
- `trm-att-maze-seed2-finetune`: best puzzle = 0.047 (bad seed, collapsed)

All three well below 79.60% HF baseline. ✅

Additional K-vote+q_loss collapse explored in `machine 2/analysis_run_phase2c_sanjin_combo.md` and `machine 3/runs/trm-att-maze-seed2/` — consistent with paper's diagnosis.

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
| Energy / CO₂ (all models) | CodeCarbon CSVs in respective machine run folders (see Section 5) |
| K-vote main results | `machine 3/k_vote/k_vote_results.csv` |
| K=16 extended sweep | `machine 5/k_vote_artifacts/k_vote_results.csv` |
| GPT-2 Maze saturation data | `machine 5/k_vote_artifacts/k_vote_results.csv` |

---

## Action Items (Priority Order)

1. **[HIGH] Fix TRM-MLP mean/σ in paper** — change to 74.54% ± 0.33% puzzle, 85.94% ± 0.16% cell. Update all downstream references (10.49 pp → 10.20 pp gap; 13.15 pp → 12.86 pp gap; ablation discussion in Section 5.4).

2. **[MED] Distill-Qwen K-vote disclosure** — either re-run K=16 sweep on the machine 1 checkpoint (which gives 35.87%), or add a sentence in Section 5.3 noting the K-vote used a weaker (`machine 5/novelty-distill-sudoku-seed0`) checkpoint, not the one reported in Table 1.

3. **[LOW] GPT-2 Maze saturation** — add a brief note that GPT-2 (non-distilled) also scored 100% under `mask_non_path: true`. Currently paper only names Qwen and Distill-GPT-2.

4. **[LOW] Maze-Hard table completeness** — GPT-2 on Maze has no row in Table 2. Worth adding a row with "Re-eval: 0% / ~12.5%" to match the Distill-GPT-2 "Assumed" row treatment.
