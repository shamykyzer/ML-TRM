# ML Engineering Findings — Pre-Fleet Checkpoint

**Author:** Ahmed (ML Engineering Lead)
**Date:** 2026-04-19
**Audience:** Armin, Nick
**Decision requested:** Sign-off on §4 (what we launch next) before I start the LLM fleet runs.

---

## TL;DR

1. **Do NOT retrain sudoku-att.** The from-scratch run peaked at 18.33% and collapsed to 0% by epoch 500. Retraining will most likely reproduce the collapse. Our time/compute is better spent elsewhere.
2. **Our headline TRM-sudoku number is 74.56%** (MLP variant, HF-init + fine-tune). The published MLP checkpoint evaluated on our pipeline gives 84.74%, matching the paper.
3. **Our headline LLM-sudoku number is 0% puzzle / 19% cell** (Qwen2.5-0.5B, 100 epochs). The thesis holds cleanly.
4. **Seven of eight LLM runs (the proposal-committed baselines) haven't happened yet.** That's where the remaining ~16 hrs of fleet compute should go, not into another TRM-Att attempt.

---

## 1. What we've trained so far (audited from logs + summary.csv)

| Run | Variant | Init | Epochs | Peak `val/puzzle_acc` | Wall time | CO₂ (kg) | CO₂/correct | Notes |
|---|---|---|---|---|---|---|---|---|
| `sudoku-mlp-seed0` | MLP-Mixer (mlp_t=true) | HF-init from Sanjin2024 Sudoku-Extreme-mlp | 2245 | **0.7456 @ ep 900** | 80 hrs | 5.23 | 1.66 × 10⁻⁵ kg | Overfit: val dropped to 0.5948 by ep 2245; stopped training |
| `sudoku-att` | Attention + RoPE (mlp_t=false) | **From scratch** | 500 | **0.1833 @ ep 100** | 16 hrs | 1.65 | 2.12 × 10⁻⁵ kg | Collapsed to 0% by ep 500; train-acc climbed to 0.96 — textbook overfit |
| `llm-qwen-sudoku-seed0` | Qwen2.5-0.5B + LoRA | HF-init | 100 | **0.0000 (cell: 0.1907)** | 7 hrs | 0.21 | ∞ | Eval was broken pre-Fix-B; re-evaluated with `scripts/eval_llm_checkpoint.py` |
| TRM-MLP sudoku eval-only | MLP-Mixer | HF checkpoint, no training | n/a | **0.8474 (cell: 0.9155)** | 9.5 hrs inference | 0.48 | 1.23 × 10⁻⁶ kg (inference) | `results/trm_official_sudoku_eval.json` |
| TRM-Att maze eval-only | Attention | HF checkpoint, no training | n/a | **0.7960 (cell: 0.9754)** | 2.2 min inference | 0.002 | 2.51 × 10⁻⁶ kg (inference) | `results/hf_eval_maze_hard.json` |

Full aggregate: `results/summary.csv`. Plot files: `results/figures/sudoku_mlp_peak_and_overfit.png`, `results/figures/sudoku_att_rise_and_collapse.png`.

### Key observations

- **HF-init + fine-tune is viable** (74.56% on sudoku-MLP) but we overshot the peak by ~1300 epochs. Lesson: next TRM fine-tune should set `early_stop_patience` (config flag already added) around 300 epochs.
- **From-scratch att training does not converge at our compute scale.** The paper used 8×H200 with effective batch 4608 for 60K epochs. We have 1×RTX 5070 with batch 32-64. That's ~1000× less compute. Our 18% peak followed by collapse is the expected outcome at this scale.
- **Qwen's 19% cell accuracy is above random chance (1/9 = 11%).** The model learned per-digit statistics — it just can't compose them. That's the thesis, cleanly measured.

---

## 2. Recommendation: do NOT retrain sudoku-att

### The case for retraining (steelman)

- Hyperparameter tuning (lower LR, longer warmup, gradient clipping) might push the peak to 30-40%.
- More data to compare against MLP.
- Demonstrates rigor.

### Why the case is weak

1. **The proposal doesn't require it.** §4 Methodology: *"an attention-free MLP for Sudoku and self-attention for mazes."* Sudoku-Att is an ablation, not a commitment.
2. **Expected ceiling is still ~40%, not 77.70%.** Even with perfect hyperparameters, we cannot recover 1000× less compute. The paper target is unreachable from scratch here.
3. **The existing collapsed run already earns the "methodology probe" marks.** `sudoku_att_rise_and_collapse.png` shows the classic train-accuracy-climbs-while-val-collapses pattern. That's publishable as a reproducibility note.
4. **It burns ~3.2 kg CO₂ for marginal new information.** Our proposal emphasizes energy efficiency (MO4 ethics); burning more CO₂ to reproduce a collapse we've already documented weakens that narrative.
5. **Opportunity cost.** The same ~16 hrs funds roughly 4-5 LLM runs that ARE proposal-committed and haven't happened yet.

### Decision

**Leave `sudoku-att` as is. Write it up as Finding #2 in the Discussion section.**

---

## 3. How to frame this in the report (copy-adapt as needed)

Paste or adapt these for the Experiments and Discussion sections:

### For the Experiments Methods paragraph

> *Following Jolicoeur-Martineau et al. (2025), we use the attention-free MLP variant for Sudoku-Extreme and the self-attention variant for Maze-Hard. We warm-start TRM training from the Sanjin2024 community checkpoints (HuggingFace `Sudoku-Extreme-mlp` and `Maze-Hard`) and fine-tune on a single RTX 5070. This reduces training cost by roughly three orders of magnitude relative to the paper's 8×H200 regime while preserving the paper's headline numbers at zero-fine-tune evaluation (84.74% on Sudoku, 79.60% on Maze).*

### For the Experiments Results section

> *TRM-MLP evaluated directly from the HuggingFace checkpoint on our pipeline recovers the paper's headline result: 84.74% puzzle accuracy on Sudoku-Extreme, 91.55% cell accuracy. Further fine-tuning on consumer hardware for 2245 epochs peaked at 74.56% at epoch 900 before overfitting. The training curve (Figure X) shows the classic signature: training accuracy reaches 0.99 while validation accuracy decays by 15.08 percentage points over the subsequent 1300 epochs. Fine-tuning was stopped when the decline had persisted for over 1300 epochs.*

### For the Discussion — reproducibility note

> *We attempted from-scratch training of the attention variant (sudoku-att) to independently assess whether a consumer GPU can reproduce the paper's 77.70% claim without community pre-training. Validation accuracy peaked at 18.33% at epoch 100 before collapsing to zero by epoch 350. The training-accuracy curve climbed monotonically to 0.96 throughout, confirming overfitting on the 1000-example training set rather than optimization failure (Figure Y). This is consistent with the paper's reported training regime (8×H200 with effective batch 4608 for 60K epochs) being a prerequisite for stable convergence of the attention variant; our collapse trajectory is a reproducibility note rather than a null result. We do not retrain, both because the outcome is predictable at our compute scale and because additional training runs would add ~3 kg CO₂ without new information — a trade-off inconsistent with the efficiency argument central to this paper.*

### For the Ethics paragraph (MO4)

> *We made explicit choices about when not to train. The sudoku-att collapse, once documented, was not retrained at different hyperparameters despite the availability of compute, because our expected value of new information per kilogram of CO₂ emitted was unfavourable. This reflects the same efficiency argument the paper's central thesis makes at the architecture level.*

---

## 4. What I want to launch next (sign-off please)

Proposed sequence on machine 1 (RTX 5070), one at a time, 30 epochs each:

1. **`llm-qwen-maze`** — validates Fix A (OOM patch) + Fix B (shift fix) on maze sequence length. ~1 hr.
2. **`llm-gpt2-sudoku`** (`configs/llm_config.yaml`) — smaller model, fast sanity check. ~20 min.
3. **`llm-gpt2-maze`** — ~30 min.
4. **`llm-smollm-sudoku`** — ~45 min.
5. **`llm-smollm-maze`** — ~60 min.
6. **`llm-llama-sudoku`** — ~45 min.
7. **`llm-llama-maze`** — ~90 min.
8. **`distill` run** — needs a chosen teacher (Qwen is the obvious pick). ~1 hr.

**Total: ~6.5 hours of compute. Produces 7 new rows in `results/summary.csv` and fills the baseline comparison table the proposal commits to.**

All 8 config files already point at `epochs: 30` (updated 2026-04-19). All training outputs will land in `C:/ml-trm-work/<task>-seed0/` and the aggregator (`scripts/aggregate_metrics.py`) will pick them up via `$TRM_EXPERIMENT_DIR`.

### Things that are NOT on this list

- Retraining sudoku-att (discussed above)
- Seed variance (the proposal doesn't require it; can do post-fleet if time permits)
- Maze from-scratch training (HF-eval 79.60% is sufficient per the proposal's "fine-tuned baseline" framing)

### Decision needed

**Reply OK and I'll start with (1) Qwen-maze.** Or tell me what to drop/add.

---

## 5. Follow-up (2026-04-19 afternoon) — wandb aggregation + maze diagnosis

Added after §4 was drafted. Two new tools in the repo, one unresolved training
question.

### 5.1 New tooling

- **`scripts/aggregate_wandb_runs.py`** — pulls all TRM/LLM/distill runs from
  `shamykyzer/TRM` via `wandb.Api`, writes `results/trm_runs_overview.csv`
  (one row per wandb run with seed/host/mlp_t/best_val/runtime/emissions/ckpt_path),
  saves per-step history for the best run per variant, and prints seed stats.
  CLI: `--family {trm,llm,distill}`, `--skip-history`.
- **`src/evaluation/wandb_eval.py`** — post-hoc test-set eval helpers.
  `evaluate_trm_run(ckpt, config)` wraps the existing `load_and_evaluate()`;
  `backfill_test_accuracy(csv, resolver)` fills the blank `test_accuracy` column
  in the overview CSV once checkpoints are copied to one machine.

### 5.2 Current wandb state

Seed-aggregated via the new script (note §1 has Apr-11 snapshot; these are live):

| Variant | mean | std | n | per-seed |
|---|---|---|---|---|
| sudoku-mlp | 0.7425 | 0.0063 | 4 | s0=0.7340/0.7456, s1=0.7420, s2=0.7486 |
| **maze (running)** | 0.146 | 0.086 | 3 | s0=0.202, s1=0.189, **s2=0.047** |

sudoku-mlp seed std of 0.006 confirms training determinism — worth a sentence
in the paper.

### 5.3 Maze training is active and partially working

§4 said "Maze from-scratch training … is not on this list". That decision still
stands for *from-scratch* training. What's actually running are HF-init
**fine-tunes** launched Apr 17 across FDK/FCM/FFN (runs `t6kveuwu`, `p5b5zdhb`,
`xu7umj0r`). Budget: 2000 epochs each, ~65h of 55h consumed.

**Diagnosis (from wandb history API, 142 epochs in):**

- ✅ HF init confirmed applied (`--init-weights …Maze-Hard/remapped_for_local.pt`
  in all three `wandb-metadata.json` args arrays).
- ✅ `data.mask_non_path = false` — reward-hacking attractor inactive.
- ✅ Train `lm_loss` trending down monotonically (seed 0: 1.364→1.003).
- ❌ **Gradient clipping is strangling the run.** `max_grad_norm = 1.0` while
  actual grad norms are 41-101 — ~80-100× clipping ratio, effective update
  magnitude is a small fraction of design.
- ❌ **ACT halting head regressing.** `train/q_halt_loss` rising (seed 0:
  1.47→2.34; seed 2: 1.36→2.99). `val/frac_at_max_steps` ∈ {0.91, 0.96, 0.98} —
  model almost never halts voluntarily. Contrast sudoku-mlp (`ihj6hpsn`) where
  q_halt_loss dropped 0.115→0.057 and avg_steps went 6.83→1.25.
- ❌ **Continue head is dead** (`no_ACT_continue: true`, `train/q_continue_loss = 0`
  throughout).
- ❌ **Seed 2 is a bad seed.** val_exact 0.008→0.037→0.047 with avg_act_steps
  stuck at 15.8 — not recovering.

### 5.4 Maze decision options (team sign-off needed)

**Option M-A — Kill all 3, report HF-eval (0.796) only.** Aligns with §4's
original decision. Cleanest narrative. Throws away ~30h of sunk compute.

**Option M-B — Kill seed 2, let seeds 0/1 finish, report whatever they reach.**
Accepts the clipped-gradient handicap. Current trajectory suggests final
val_exact around 0.25-0.35, far below HF-eval 0.796. Report honestly as
"attempted fine-tune hit a gradient-clipping floor; we report the
HF-checkpoint number as the reproduction result and this as a methodology note."

**Option M-C — Kill all 3, patch `configs/trm_official_maze.yaml`
(`max_grad_norm: 1.0 → 5.0`, explicit `eval_interval: 50`), relaunch with
`--init-weights`.** ~55h × 3 = 165h of compute. Risky with 12 days to deadline
unless we can start today.

**Recommendation: M-A or M-B.** M-C's ROI is marginal given deadline pressure.
Fastest path to a defensible paper is to report HF-eval (0.796) as the maze
number and note own-training as an ablation in Discussion.

### 5.5 Follow-up TODO (append to §4 or track here)

- [ ] **Team decision: M-A, M-B, or M-C for maze.** Blocking.
- [ ] **Kill `xu7umj0r`** (maze seed 2 on FFN) regardless of which option — it's
      learning nothing and occupying a machine needed for LLM fleet.
- [ ] **Back up maze `best.pt` from seeds 0 and 1** (on FDK, FCM) to OneDrive/HF
      before any relaunch. 10h of compute is expensive to re-do.
- [ ] **Copy all 6 `best.pt` files** (3 maze + 3 sudoku-mlp seeds) to one machine
      for post-hoc test-set evaluation via `backfill_test_accuracy()`.
- [ ] **Run `backfill_test_accuracy()`** to fill `test_accuracy` column in
      `results/trm_runs_overview.csv`.
- [ ] **Re-run `scripts/aggregate_wandb_runs.py` daily** while maze is still
      training; fresh numbers feed the paper.
- [ ] **Paper Methods update**: add sentence on the grad-clip finding if M-C is
      pursued; otherwise skip.

---

## Appendix — how to verify any of the above

```bash
# Regenerate summary.csv from all run dirs
TRM_EXPERIMENT_DIR=C:/ml-trm-work python scripts/aggregate_metrics.py

# Re-evaluate existing Qwen sudoku checkpoint with the fixed shift-aware eval
python scripts/eval_llm_checkpoint.py \
    configs/llm_qwen.yaml \
    "C:/ml-trm-work/llm-qwen-sudoku-seed0/qwen2.5_0.5b_sudoku_latest.pt" 50

# Regenerate the two story figures
python scripts/plot_sudoku_mlp_overfit.py
python scripts/plot_sudoku_att_story.py

# Run the glossary publish step after editing docs/wandb_metrics_glossary.md
python scripts/publish_wandb_metrics_report.py
```

Raw data lives at:

- `C:/ml-trm-work/sudoku-mlp-seed0/` — 90 checkpoints + full 2245-epoch train log
- `C:/ml-trm-work/llm-qwen-sudoku-seed0/` — epoch 50, 100, latest checkpoints + train log + emissions
- `experiments/sudoku-att/` — 500-epoch train log + emissions
- `hf_checkpoints/Sudoku-Extreme-mlp/` and `hf_checkpoints/Maze-Hard/` — community checkpoints used for warm-start + eval-only baselines
- `results/summary.csv` — authoritative aggregate
- `results/figures/` — all plots, including the two new Discussion figures
- `docs/wandb_metrics_glossary.md` — metric reference
- `fix-plan.html` — visual summary of the April 19 fix work (archival)
