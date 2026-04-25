# Latest Training Run — Plain-English Analysis

**Date written:** 2026-04-25
**Latest run:** `wandb/run-20260422_190526-mk9hqzz7` (2026-04-22, 19:05)
**This rig:** `STU-CZC5277FCM` = rig 2 of the 3-rig novelty split

---

## 1. What was the latest run?

It was the **distilled-student-on-Sudoku** training. In plain words:

- We took the small Qwen2.5-0.5B language model that we fine-tuned earlier the same day (the "teacher").
- We used its predictions to teach a much smaller model (~2.4M parameters) — the "student".
- The student ran for 30 epochs on the Sudoku-Extreme dataset.
- Total training time: **4 minutes** on the RTX 5070.
- Total energy: **0.011 kWh** (about the same as boiling a kettle for 30 seconds).

This is run #5 in the 6-run novelty matrix described in `results/novelty/README.md`.

## 2. How did the student do?

| Epoch | Training loss | Validation loss | Cell accuracy | Whole-puzzle accuracy |
|------:|--------------:|----------------:|--------------:|----------------------:|
| 10 | 1.03 | 2.17 | 14.4 % | 0 % |
| 20 | 0.72 | 1.93 | 19.0 % | 0 % |
| 30 | **0.66** | **1.84** | **25.8 %** | **0 %** |

- "Cell accuracy" = fraction of individual sudoku cells filled in correctly.
- "Puzzle accuracy" = fraction of *whole* puzzles solved end-to-end.

**Translation:** the student is getting better at guessing single digits, but it never solves a complete puzzle. Loss is still falling at epoch 30, so it hasn't fully converged.

## 3. Student vs its own teacher (the Qwen LoRA model)

Both ran on the same machine, same day, same dataset:

|                          | Qwen teacher (run #3) | Distilled student (run #5) | Winner |
|--------------------------|----------------------:|---------------------------:|:------:|
| Wall-clock time          | 100 min               | **4 min**                  | student (25× faster) |
| Energy used              | 0.32 kWh              | **0.011 kWh**              | student (30× less) |
| Cell accuracy            | 18.6 %                | **25.8 %**                 | student (+39 %) |
| Puzzle accuracy          | 0 %                   | 0 %                        | tie (both fail) |

**The headline:** the small student beats its own teacher on every measurable axis. Distillation worked here, but neither model can actually solve a sudoku.

## 4. Where TRM fits in (the coursework comparison)

The coursework wants to compare three families on two tasks:

| Family | Sudoku model | Maze model |
|--------|--------------|------------|
| TRM (the paper's tiny recursive model) | TRM-MLP (~6.4M params) | TRM-Att (~8.4M params) |
| Fine-tuned LLM | Qwen2.5-0.5B + LoRA | Qwen2.5-0.5B + LoRA |
| Distilled LLM | ~2.4M-param student | ~2.4M-param student |

Each model gets the **same 2.5 hours** of training (the "iso-wall-clock" rule), and we record accuracy + energy.

### What we have on *this* machine right now

Only the LLM half:

- ✅ qwen-sudoku (run #3) — done
- ✅ distill-sudoku (run #5, the latest run) — done
- ❌ TRM-MLP-sudoku (run #1) — lives on rig 1
- ❌ TRM-Att-maze (run #2) — lives on rig 1
- ❌ qwen-maze (run #4) — lives on rig 3
- ❌ distill-maze (run #6) — lives on rig 3

The TRM rows in `results/novelty/iso_time_results-rig2.csv` are marked `skipped` because they're another rig's job. Once rigs 1 and 3 finish and OneDrive syncs their CSVs, `scripts/run_novelty_aggregate.py` will combine all six.

### What we already know about TRM (from older, longer runs)

Even though the iso-time TRM rows aren't here yet, two earlier studies already tell us roughly where TRM lands:

**A. Long-budget TRM training (`results/summary.csv`):**

| Model | Puzzle accuracy | Cell accuracy | Training time | Energy |
|-------|----------------:|--------------:|--------------:|-------:|
| TRM-MLP on sudoku | **74.6 %** | 85.8 % | 103 hours | 22.0 kWh |
| TRM-Att on sudoku | 18.3 % | 55.4 % | 32 hours | 6.9 kWh |
| Qwen LoRA on sudoku | 0 % | 19.1 % | 6.8 hours | 0.9 kWh |

**B. TRM-Att on maze, 3 seeds (`results/trm_runs_overview.csv`):**

| Seed | Puzzle acc | Cell acc | Time | Energy |
|------|-----------:|---------:|-----:|-------:|
| 0 | 20.2 % | 96.6 % | 65 h | 15.0 kWh |
| 1 | 18.9 % | 97.3 % | 65 h | 14.9 kWh |
| 2 | 4.7 %  | 95.5 % | 65 h | 14.7 kWh |

Note: seed 2 collapsed; mean ≈ 14.6 %, sd ≈ 8.5 % — high variance, worth flagging in the report.

**C. HuggingFace pre-trained checkpoints (paper-quality, our warm start):**

| Checkpoint | Puzzle acc | Cell acc |
|------------|-----------:|---------:|
| Sudoku-Extreme-mlp | **84.7 %** | 91.5 % |
| Maze-Hard | **78.9 %** | 99.3 % |

These are the "ceiling" — what TRM achieves when trained for days on 8 GPUs.

## 5. Does the proposal's main claim hold up?

The proposal says: **"TRM is more parameter- and energy-efficient than fine-tuned LLMs on structured puzzles."** Based on what we have:

1. **On whole-puzzle accuracy: TRM wins clearly.** TRM-MLP solves 74.6 % of sudokus; Qwen solves 0 %. The distilled student also gets 0 %. This is the cleanest result for the report.
2. **On energy: it's nuanced.** TRM-MLP burned 22 kWh to reach 74.6 %; Qwen burned 0.9 kWh to reach 0 %. Per-correct-puzzle, TRM is essentially infinitely more efficient (because Qwen never solves any). Per-kWh-of-training, Qwen is cheaper but produces nothing useful.
3. **The 2.5-hour cap probably hurts TRM.** The TRM paper trains for ~60,000 epochs; 2.5 hours on an RTX 5070 reaches maybe a few hundred epochs. So the iso-time TRM numbers will be a *lower bound* on TRM's real ceiling. The novelty README already calls this out as a known limitation.
4. **Distillation is a side-story.** The student matches/beats its weak teacher in 4 minutes for ~30× less energy — a clean demonstration of distillation efficiency, *but only against a weak teacher*. Neither solves real puzzles.

## 6. What to do next

Before the report can be written:

1. **Get rigs 1 and 3 done** (or fetch their `iso_time_results-rig{1,3}.csv` files via OneDrive sync).
2. **Run the aggregator:** `python scripts/run_novelty_aggregate.py --seed 0 --ignore-missing` will combine all available CSVs and re-emit the 5 plots.
3. **Re-run K-vote against the new distilled checkpoint.** The current K-vote results (`results/novelty/k_vote_results.csv`) were generated before tonight's distill run — at K=1 the CSV says 17.5 % cell-acc but the latest run measures 25.8 %. The Pareto plot is stale.
4. **Decide which TRM number to put in the report's headline table.** Options:
   - The 2.5 h iso-time run (fair comparison, but underestimates TRM)
   - The HF-init pre-trained eval (84.7 % sudoku, 78.9 % maze — paper quality)
   - Both, with the gap framed as evidence that the iso-time protocol is biased against architectures that need long training
5. **State seed variance honestly.** Maze TRM has σ ≈ 8.5 % across 3 seeds — bigger than some of the gaps the report will discuss. Novelty rows are single-seed; this should be acknowledged.

## 7. Quick file map

- **Latest run logs:** `wandb/run-20260422_190526-mk9hqzz7/`
- **Iso-time CSV (this rig only):** `results/novelty/iso_time_results-rig2.csv`
- **K-vote CSV (LLM-side only):** `results/novelty/k_vote_results.csv`
- **Long-budget baseline summary:** `results/summary.csv`
- **TRM run overview (3 seeds maze):** `results/trm_runs_overview.csv`
- **TRM-MLP best history (sudoku):** `results/history_sudoku-mlp_best.csv`
- **TRM-Att best history (maze):** `results/history_maze_best.csv`
- **HF checkpoint evals:** `results/trm_official_sudoku_eval.json`, `results/hf_eval_maze_hard_mask_*.json`
- **Plots:** `results/figures/` (older baseline plots) and `results/novelty/*.png` (iso-time and K-vote)
- **Novelty experiment design doc:** `results/novelty/README.md`

---

## 8. Other runs in `C:\ml-trm-work` (not the latest)

The local `C:\ml-trm-work` working folder holds five other run directories besides the latest distill. Total ~3.1 GB on disk. Inventory and what each is worth:

### 8.1 `maze-seed1` — TRM-Att on Maze-Hard, seed 1 (the only other real run)

| Epoch | Train loss | Avg ACT steps | Val cell acc | Val puzzle acc | Best so far |
|------:|-----------:|--------------:|-------------:|---------------:|------------:|
| 50    | 0.361      | 5.4           | 97.0 %       | **27.5 %**     | 27.5 %      |
| 100   | 0.215      | 3.5           | 96.1 %       | 17.2 %         | 27.5 %      |

- **Total duration:** 65 hours (233 828 s) over 3 calendar days (2026-04-19 → 2026-04-22)
- **Total energy:** 14.92 kWh, **3.54 kg CO₂** (CodeCarbon, UK grid)
- **Rise-and-collapse pattern:** peaked at epoch 50 then regressed at epoch 100 — train loss kept dropping (0.36 → 0.22) while val puzzle-acc fell ~10 percentage points. Same pattern as `results/figures/sudoku_att_rise_and_collapse.png` on the maze side.
- **ACT halting kicked in:** average steps fell 5.4 → 3.5 — the model became confidently wrong, halting earlier on examples it now gets wrong.
- `best.pt` (41 MB) almost certainly holds epoch-50 weights; `epoch_100.pt` is the worse end-state. Pick deliberately.

### 8.2 The three GPT-2 smoke folders — zero training, no rubric value

| Folder | Duration | Epochs | Energy | Notes |
|---|---|---|---|---|
| `llm-fmt-test` | 25 s | **0** | 0.00076 kWh | Header-only log; 501 MB .pt is GPT-2 base weights serialised before any training step |
| `llm-smoke` | 33 s | **0** | 0.00091 kWh | Same as above, byte-identical 501 MB checkpoint, run 90 min earlier |
| `llm-maze-smoke` | — | **0** | — | Header-only log, no checkpoint, no emissions — never started |

These were format-check setups before switching from GPT-2 to Qwen2.5-0.5B. **Don't include in the report's tables.** At most, one sentence in Methods justifying the GPT-2 → Qwen switch.

### 8.3 `novelty-qwen-sudoku-seed0` and `novelty-distill-sudoku-seed0`

The Qwen teacher (run #3) and distilled student (run #5, the latest run) — already analysed in §2–§3 above.

---

## 9. Mapping each run onto the spec rubric and proposal claims

The spec rubric (page 9 of `docs/Group Project Specification 2025-26-v4.pdf`) puts **60% of marks on Methods + Experiments**. The proposal commits to a 3×2 matrix: **{TRM, fine-tuned LLM, distilled LLM} × {Sudoku-Extreme, Maze-Hard}**, evaluated on **puzzle accuracy + kWh + CO₂-eq + wall-clock**.

### 9.1 Coverage of the proposal's 6-cell matrix

| Cell | Model | Task | Run on this machine | Status |
|---|---|---|---|---|
| (1,1) | TRM-MLP | Sudoku-Extreme | — | ❌ on rig 1 (HF ckpt = 84.7 % available as fallback) |
| (1,2) | TRM-Att | Maze-Hard | `maze-seed1` | ⚠️ partial: 1 of 3 seeds, 100 epochs |
| (2,1) | Fine-tuned LLM (Qwen) | Sudoku-Extreme | `novelty-qwen-sudoku-seed0` | ✅ done |
| (2,2) | Fine-tuned LLM (Qwen) | Maze-Hard | — | ❌ on rig 3 |
| (3,1) | Distilled LLM | Sudoku-Extreme | `novelty-distill-sudoku-seed0` | ✅ done |
| (3,2) | Distilled LLM | Maze-Hard | — | ❌ on rig 3 |

**3 of 6 cells filled locally.** Smoke runs are not part of the matrix.

### 9.2 What each run earns in the rubric

| Run | Methods (30%) | Experiments (30%) | MO4 ethics |
|---|---|---|---|
| `maze-seed1` | Demonstrates ACT halting (avg_steps 5.4 → 3.5) — real evidence of "alternative methods understanding" | Rise-and-collapse weakness measured — rubric rewards discussing limitations | 14.9 kWh / 3.54 kg CO₂ — concrete number for ethical-issues paragraph |
| `novelty-qwen-sudoku-seed0` | Justifies LLM choice; reproduces proposal's premise (LLMs at 0 % puzzle) | Required LLM baseline for cell (2,1) | 0.32 kWh / 0.077 kg CO₂ — middle of the cost spectrum |
| `novelty-distill-sudoku-seed0` | Demonstrates knowledge distillation (α=0.7, T=4) — proposal's third method | Required distilled baseline for cell (3,1); shows student > teacher on every axis | 0.011 kWh / 0.0026 kg CO₂ — cheapest cell in matrix |
| Smoke runs ×3 | One sentence on GPT-2→Qwen switch (if interesting) | None | None |

### 9.3 Headline numbers the report can quote directly

- **CO₂ range across the 3 completed cells:** 0.0026 kg (distill-sudoku) → 3.54 kg (TRM-att-maze) = **~1360× ratio**. Strong sentence for the ethical-issues paragraph (rubric explicitly rewards this in the 70-100 % Methods band).
- **Both LLM cells reproduce 0 % puzzle accuracy** — confirms proposal's quoted claim about DeepSeek R1 / Claude / o3-mini-high failing on Sudoku-Extreme.
- **TRM-Att maze peaked at 27.5 % puzzle-acc at epoch 50, fell to 17.2 % at epoch 100** — measured rise-and-collapse, not assumed. Rubric rewards "discussing limitations".
- **ACT halting confirmed working** in `maze-seed1` (5.4 → 3.5 steps) — direct evidence the recursion + adaptive-compute mechanism behaves as Jolicoeur-Martineau (2025) describes.

### 9.4 What's missing for full proposal coverage

The proposal promised three figure types. Status:

- ✅ **Carbon footprint bar chart** — `results/novelty/iso_time_energy_by_model.png` exists (3 of 6 bars filled).
- ⚠️ **Accuracy comparison table** — 3 of 6 cells. Either wait for rigs 1 + 3, or fall back to HF-ckpt eval for missing TRM cells and disclose the fallback in the table caption.
- ❌ **Performance-by-difficulty curves** — no run stratifies by difficulty. Sudoku-Extreme is uniformly 17-clue, so this may need to be reframed as "performance vs training-budget curves" using the existing iso-time data.

### 9.5 Note on `C:\ml-trm-work` and the supplementary ZIP

Spec page 3: supplementary material is "a separate ZIP file" of code, and "any text … is for coding and documentation purposes only and does not directly contribute the marking of the project report". **Binary weight files earn no rubric marks.** What earns marks is the report's tables and figures, which are already fed by the lightweight metadata in `results/novelty/`. A GitHub release of `.pt` files would not move any rubric cell.
