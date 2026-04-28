# Tiny Recursive Models vs Fine-Tuned LLMs for Structured-Reasoning Puzzles

**Authors:** Ahmed AlShamy, Armin Ghaseminejad, Nickolas Greiner
**Module:** UFCFAS-15-2 Machine Learning — UWE Bristol, 2025-26
**Date:** 2026-05-01

<!-- Target: 6 pages conference-style. Rubric weights in comments below. -->

## Abstract
<!-- 200 words. Problem, approach (TRM vs 4 LLMs + distill), key result, CO₂ savings.
     Canonical numbers in this paragraph come from
     `results/summary_fixed.csv` and `findings.md` §9 (2026-04-28). -->

Solving constraint-satisfaction puzzles such as Sudoku-Extreme and
Maze-Hard requires multi-step logical reasoning that current
large language models fail at — DeepSeek R1, Claude 3.7, and
o3-mini-high all score 0 % on Sudoku-Extreme. We compare a
6.4 M-parameter Tiny Recursive Model (TRM-MLP) against four LoRA
fine-tuned LLMs (GPT-2 124 M, SmolLM2 360 M, Qwen2.5 500 M, plus a
2.4 M-parameter distilled student of Qwen) under matched
consumer-GPU compute (RTX 5070, 12 GB) on Sudoku-Extreme and
Maze-Hard from Jolicoeur-Martineau et al. (2025). TRM-MLP
recovers the published 84.74 % puzzle accuracy on Sudoku-Extreme
under HF-eval and 74.25 ± 0.63 % under 3-seed from-scratch
fine-tune; TRM-Att HF-eval recovers 79.60 % puzzle on Maze-Hard.
Every fine-tuned LLM and the distilled student score **0.00 %
puzzle accuracy on both tasks** — they learn per-cell statistics
(13–36 % cell accuracy) but cannot compose them into a globally
consistent solution. Per-correct-puzzle CO₂ is 1.23×10⁻⁶ kg for
TRM-MLP HF eval; undefined (∞) for every LLM. We additionally
diagnose and correct a maze-evaluator artefact
(`mask_non_path: true`) that previously inflated all LLM Maze
puzzle accuracies to 1.000.

## 1 Introduction
<!-- Rubric weight: 10%. 70-100% band: "Clear definition of the problem, excellent
     description of the proposed approach, excellent description of all the aims,
     objectives and results". Aim: ~0.6 page. -->
TBD (Task 12).

## 2 Related Work
<!-- Rubric weight: 10%. 70-100% band: "Critical appraisal of related work,
     excellent discussion of similarities and differences with critical analysis
     related closely and clearly to the project". Aim: ~0.6 page. -->
TBD (Task 12).

## 3 Data
<!-- Rubric weight: 10%. 70-100% band: "Clear justification of appropriate
     techniques to collect datasets or select existing datasets; Excellent
     treatment of data". Aim: ~0.6 page. -->
TBD (Task 11).

## 4 Methods
<!-- Rubric weight: 30%. 70-100% band: "Excellent selection and use of methods
     demonstrating an understanding of alternative methods, high level of
     technical breadth and depth, excellent discussion of relevant issues
     including ethical issues". Aim: ~1.8 pages.
     CONTENT: lift §4 from `docs/report_methods_experiments_draft.md`
     (architectures, training protocol, alternatives table, energy
     accounting, ethics paragraph). Source of truth for numbers:
     `results/summary_fixed.csv` and `findings.md` §9. -->
See `docs/report_methods_experiments_draft.md` §4 for the working
draft (architectures, alternatives considered, K-vote diversity-source
table, ethical considerations). Figures referenced in the draft live
at `results/figures/`. The draft is up to date as of 2026-04-28.

## 5 Experiments
<!-- Rubric weight: 30%. 70-100% band: "Excellent generation and clear analysis
     of experimental results using fully justified methodology, excellent appraisal
     and evaluation of results". Aim: ~1.8 pages. Includes the three figures.
     Source of truth: `results/summary_fixed.csv` and `findings.md` §9 (2026-04-28). -->
See `docs/report_methods_experiments_draft.md` §5 for the working
draft. Headline numbers (canonical, 2026-04-28):

- **Sudoku-Extreme**: TRM-MLP HF-eval **84.74 %** / 3-seed
  from-scratch **74.25 ± 0.63 %**. GPT-2 / SmolLM2 / Qwen2.5
  fine-tunes all score **0.00 %** puzzle (cell 13.18 / 14.11 /
  19.07 %). Distilled 2.4 M student scores 0.00 % puzzle but
  **35.87 % cell — 1.88× the Qwen teacher's 19.07 % at 200×
  fewer parameters and 100× less training energy**.
- **Maze-Hard**: TRM-Att HF-eval **79.60 %** puzzle / 99.30 %
  cell. Post-fix LLM and distill re-evals (after the
  `mask_non_path: true` correction) score **0.00 %** puzzle /
  12.50–12.52 % cell. The 0.02-pp distill-teacher cell match
  is the strongest single methodology signal in the section.

## 6 Conclusion
<!-- Rubric weight: 5%. 70-100% band: "Demonstration of clear understanding and
     implications of results". Aim: ~0.3 page. -->
TBD (Task 13).

## 7 Ethical and Societal Implications
<!-- MO4 assessed. Aim: ~0.3 page. -->
TBD (Task 14).

## References
TBD — populate as sections cite.

## Appendix (supplementary — not counted toward 6-page limit)
See `supplementary.ipynb` for the runnable pipeline, full hyperparameter tables,
per-seed training curves, and CO₂ attribution per training run.
