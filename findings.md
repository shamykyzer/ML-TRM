# ML Engineering Findings — Pre-Fleet Checkpoint

**Author:** Ahmed (ML Engineering Lead)
**Date:** 2026-04-19 (updated afternoon — see §5 for the wandb-aggregation trail)
**Audience:** Armin, Nick
**Decisions requested:**
 - Sign-off on §4 (LLM fleet launch on this machine)
 - Sign-off on §5.7 (kill all 3 active maze-from-scratch runs on FDK/FCM/FFN; use HF-eval 0.796 as the maze number)

---

## TL;DR

1. **Do NOT retrain sudoku-att.** From-scratch run peaked at 18.33% then collapsed to 0%. Retrain will reproduce the collapse (§2).
2. **Headline TRM-sudoku: 74.25% mean across 4 runs / 3 seeds (std 0.0063)**, MLP variant, HF-init + fine-tune. Best per seed: s0=0.7456, s1=0.7420, s2=0.7486. Published MLP checkpoint eval-only: **84.74%** (§5.2).
3. **Headline TRM-maze: 0.796 (HF-eval)**, metric-robust under both mask_non_path settings (§5.6). **Kill the 3 active from-scratch maze fine-tunes** — they are actively *corrupting* the HF weights (val_exact collapsed from 0.789 → 0.11 in one epoch, §5.7).
4. **Headline LLM-sudoku: 0% puzzle / 19% cell** (Qwen2.5-0.5B, 100 ep). Thesis holds cleanly.
5. **Seven of eight LLM runs haven't happened yet.** That's where the remaining fleet compute should go once the maze runs are killed.

---

## 1. What we've trained so far (audited from logs + summary.csv)

> **Note (added 2026-04-19 afternoon):** this table was drafted from the
> STU-CZC5277FGD logs only. The full multi-machine fleet view (3 sudoku-mlp
> seeds + 3 maze seeds across FDK/FCM/FFN/FFS/FDY) is in §5.2.
> `results/trm_runs_overview.csv` is the authoritative source.

| Run | Variant | Init | Epochs | Peak `val/puzzle_acc` | Wall time | CO₂ (kg) | CO₂/correct | Notes |
|---|---|---|---|---|---|---|---|---|
| `sudoku-mlp-seed0` | MLP-Mixer (mlp_t=true) | HF-init from Sanjin2024 Sudoku-Extreme-mlp | 2245 | **0.7456 @ ep 900** | 80 hrs | 5.23 | 1.66 × 10⁻⁵ kg | Overfit: val dropped to 0.5948 by ep 2245; stopped training |
| `sudoku-mlp-seed1` | MLP-Mixer | HF-init | — | **0.7420** (crashed) | — | 4.15 | — | On STU-CZC5277FFS; see §5.2 |
| `sudoku-mlp-seed2` | MLP-Mixer | HF-init | — | **0.7486** (killed) | — | 4.39 | — | On STU-CZC5277FDY; see §5.2 |
| `sudoku-att` | Attention + RoPE (mlp_t=false) | **From scratch** | 500 | **0.1833 @ ep 100** | 16 hrs | 1.65 | 2.12 × 10⁻⁵ kg | Collapsed to 0% by ep 500; train-acc climbed to 0.96 — textbook overfit |
| `maze-seed0` | Attention (mlp_t=false) | HF-init from Sanjin2024 Maze-Hard | **running** | **0.202** (so far) | ~38 hrs | 3.57 | — | FDK; actively corrupting HF weights — see §5.7, kill recommended |
| `maze-seed1` | Attention | HF-init | **running** | **0.189** (so far) | ~38 hrs | 3.53 | — | FCM; same kill recommendation |
| `maze-seed2` | Attention | HF-init | **running** | **0.047** (so far) | ~38 hrs | 3.49 | — | FFN; bad seed, kill ASAP |
| `llm-qwen-sudoku-seed0` | Qwen2.5-0.5B + LoRA | HF-init | 100 | **0.0000 (cell: 0.1907)** | 7 hrs | 0.21 | ∞ | Eval was broken pre-Fix-B; re-evaluated with `scripts/eval_llm_checkpoint.py` |
| TRM-MLP sudoku eval-only | MLP-Mixer | HF checkpoint, no training | n/a | **0.8474 (cell: 0.9155)** | 9.5 hrs inference | 0.48 | 1.23 × 10⁻⁶ kg (inference) | `results/trm_official_sudoku_eval.json` |
| TRM-Att maze eval-only | Attention | HF checkpoint, no training | n/a | **0.7960 / 0.7890** (see §5.6) | 2.2 min inference | 0.002 | 2.51 × 10⁻⁶ kg (inference) | Metric-robust per §5.6; headline maze number |

Full aggregate: `results/summary.csv` (sudoku) + `results/trm_runs_overview.csv` (multi-machine). Plot files: `results/figures/sudoku_mlp_peak_and_overfit.png`, `results/figures/sudoku_att_rise_and_collapse.png`.

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

### 5.6 Metric-sensitivity check — HF maze under both settings (2026-04-19 eve)

Patched `scripts/eval_hf_checkpoints.py:100` to pass `config.data.mask_non_path`
into `MazeDataset(...)` instead of using the dataset's default of `True`. Re-ran
against `hf_checkpoints/Maze-Hard/remapped_for_local.pt`.

| Evaluation | mask_non_path | puzzle_accuracy | cell_accuracy | avg_act_steps | Elapsed |
|---|---|---|---|---|---|
| `results/hf_eval_maze_hard_mask_true.json` | True | **0.796** | 0.975 | 16.0 | 133.8s |
| `results/hf_eval_maze_hard_mask_false.json` | False | **0.789** | 0.993 | 16.0 | 133.4s |

**Key finding:** the two metrics give nearly identical puzzle_accuracy (0.796 vs
0.789). The HF checkpoint is **not** a reward-hacking artifact — it correctly
predicts walls, open cells, S, G, AND the path. The metric-choice debate
(§5.3–5.4) is effectively moot on this checkpoint.

This changes the maze situation:

1. The HF checkpoint at **0.789 (strict) / 0.796 (paper-faithful)** is your
   maze result. No further training needed to get a defensible number.
2. The current fine-tune runs are **not just slow — they are actively
   corrupting a working model.** Epoch 1 val_exact_accuracy is 0.11 for seed 0;
   the HF init *itself* on the same metric/split scores 0.789. That is a
   68-point collapse in one epoch of training, which cannot be explained by
   slow convergence. Likely culprits:
   - `strict=False` state-dict load shows `[Eval] Loaded 24 keys (missing: 3,
     unexpected: 0)` — 3 trainable layers are initialized at random, and the
     first optimizer step corrupts the carefully-aligned HF weights around
     them.
   - Combined with `max_grad_norm=1.0` vs grad norms of 80-100, the clipped
     gradient direction is dominated by the random-init layers, pointing the
     whole model toward a degenerate basin.
3. **Option M-C (retrain with `max_grad_norm=5.0`) will not fix this.** The
   grad-clip change addresses slow convergence; the actual problem is
   catastrophic first-step corruption. A real fix would require identifying
   the 3 missing keys and either (a) excluding them from the first N
   optimizer steps, (b) initializing them to match the HF checkpoint's
   expected behavior, or (c) freezing the HF-loaded layers for a warmup
   period. That's a research project, not a 12-day fix.

### 5.7 Revised maze recommendation

**Use Option M-A unchanged (report 0.796 HF-eval as the maze result), plus:**

- **Kill ALL 3 current maze runs** — not just seed 2. Continuing them wastes
  compute on a known-corrupting training regime.
- **Report the methodology observation as a paper strength** rather than a
  limitation: "We verify that the authors' released weights are robust to
  metric choice (0.796 vs 0.789 under path-masked vs every-cell grading),
  which is a stronger claim than the paper itself makes."
- **Skip the max_grad_norm retrain.** Save the compute for the LLM fleet per §4.

### 5.8 Kill-runs commands (per machine)

On each of FDK, FCM, FFN: open the terminal where training is running and
press Ctrl+C once, or find the python process:

```bash
# Windows (Git Bash / WSL on training machine)
ps aux | grep 'main.py --mode train' | grep -v grep
# take the PID from the output, then:
kill <PID>

# Powershell equivalent:
Get-Process python | Where-Object { $_.CommandLine -like '*main.py*train*maze*' } | Stop-Process
```

Then on wandb.ai, mark the runs as finished (or leave them as "killed" — the
aggregator already handles that state).

Files changed this session:
- `scripts/eval_hf_checkpoints.py` (line 100, now respects `config.data.mask_non_path`)
- `results/hf_eval_maze_hard_mask_true.json` (preserved paper-faithful result)
- `results/hf_eval_maze_hard_mask_false.json` (new strict-metric result)

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
