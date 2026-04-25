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
