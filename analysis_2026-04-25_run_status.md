# Run-status snapshot — 2026-04-25

> Coursework deadline: **2026-05-01 17:00 BST** — 6 days remaining.
> Module: UFCFAS-15-2 Machine Learning. Team: AlShamy / Ghaseminejad / Greiner.

This file is a frozen-in-time analysis of the project's training state on
2026-04-25, written after auditing the latest local wandb runs and
`results/summary.csv`. It exists so the next Claude Code session (or a
teammate) can pick up the planning thread without re-deriving everything.

**Scope of this doc.** Local wandb run dirs on rig `STU-CZC5277FGD` plus
`results/summary.csv` plus `configs/`. It does NOT re-cover the rig-1
iso-time / K-vote work or the rig-2 distill story — those have their
own analyses listed below.

## Related analyses (read these too)

The project has three other recent analyses that cover ground this file
deliberately does not:

- **`results/novelty/analysis_2026-04-25.md`** — rig 1, iso-time +
  K-vote experiments, K-vote per-sample cost growth, iso-time matrix
  (2 of 6 cells filled).
- **`training_run_analysis.md`** (project root) — rig 2 (`STU-CZC5277FCM`),
  the distill-sudoku run (run #5 in the novelty matrix) and the Qwen
  teacher comparison; also has the canonical "what TRM looks like with
  long-budget training" table for the report headline.
- **`analysis_2026-04-23_distill_maze.md`** — distill-maze seed 0
  (halted by `TRM_MAX_TRAIN_SECONDS` before epoch 2; full forensic).
- **`results/runs/README.md`** — index of light, version-controlled
  artifacts for each completed run; weights live on GitHub Release
  `runs-snapshot-2026-04-25`.

If any conflict arises between this doc and the three above, the three
above are likely fresher on the iso-time / novelty / distill axes.
This doc is the source of truth for the **local sudoku-mlp wandb seed
audit** and the **`configs/` LLM-fleet completeness check**.

---

## TL;DR

- The "latest run" (`dhz4ksox`, sudoku-mlp seed 3, HF-init) is **stalled
  mid-eval at epoch 500**; its `best.pt` is the HF init weights at epoch 0
  with `best_puzzle_acc = 0.8474145312285648`. 490 epochs of fine-tuning
  produced *no* improvement and demonstrably overfit (val puzzle acc fell
  from 0.847 → 0.726).
- The seed-3 number contributes a third data point to the **HF-init
  reproducibility row**: 0.8483 (s=1) and 0.8474 (s=3); seed-0 evaluated
  but val column was empty.
- For the **from-scratch** sudoku-mlp row we have 2 clean seeds: 0.7456
  (s=0) and 0.7486 (s=2). Seed-1 (`22aagzdz`) is mid-training at epoch 545.
- **LLM-fleet coverage is wider than `summary.csv` suggests** — once
  you also count `results/runs/`: Qwen-sudoku ✅, Qwen-maze ✅ (in
  `results/runs/qwen2.5-0.5b-maze-seed0-2026-04-22/`), distill-sudoku ✅
  (per `training_run_analysis.md`), distill-maze ✅
  (`analysis_2026-04-23_distill_maze.md`). What's NOT run anywhere local
  is **Llama / SmolLM / DeepSeek / GPT-2** — 7 of 10 configs.
- **Headline recommendation:** stop `dhz4ksox`, harvest seed 1 + seed 3 HF
  numbers as-is, let the in-flight from-scratch and HF-init seed-1 runs
  finish if they are still alive, and spend the remaining GPU on the
  cheapest second LLM cell (suggest `llm_smollm.yaml`). Days 3–6 should
  be report writing, not training.

---

## 1. Latest run — `dhz4ksox`

**Identity.** `wandb/run-20260422_164235-dhz4ksox/`,
wandb name `trm_official_sudoku_seed3_STU-CZC5277FGD_1776872554`,
git commit `4a51d87`, started 2026-04-22 16:42 UTC,
host `STU-CZC5277FGD`, GPU `RTX 5070` (Blackwell, 12.8 GB, CUDA 13.0).

**Command.**

```bash
python main.py --mode train \
  --config configs/trm_official_sudoku_mlp.yaml \
  --seed 3 \
  --init-weights hf_checkpoints/Sudoku-Extreme-mlp/remapped_for_local.pt \
  --epochs 1000
```

**Config.** `configs/trm_official_sudoku_mlp.yaml` — paper-faithful
Sanjin2024 sudoku-mlp settings: `mlp_t=true`, `H_cycles=3`, `L_cycles=6`,
`no_ACT_continue=true`, `weight_decay=1.0`, `lr=1e-4`, `batch_size=64`,
`eval_interval=50`. `--epochs 1000` overrode the YAML's 500.

**Status as of 2026-04-25 morning.** CSV last appended 2026-04-23 15:37
at epoch 490; `output.log` shows full eval (50 → 1450 / 6607 batches) was
running for the post-epoch-500 evaluation, then stopped. The `wandb` run
file's mtime keeps moving (e.g. 2026-04-25 08:38) which can mean the
process is alive but stuck, or wandb sync is housekeeping. Three Python
processes are alive on the host; one is at ~1.6 GB RSS.

**`best.pt` contents (verified by `torch.load`):**

| Field | Value |
|---|---|
| `epoch` | `0` |
| `global_step` | `240` |
| `seed` | `3` |
| `best_puzzle_acc` | `0.8474145312285648` |
| keys | `model_state_dict`, `ema_state_dict`, `config`, … |

So the saved "best" is the **HF init weights re-evaluated under the
local code with halt-RNG seed 3**, captured before any gradient step.
Use this number as the seed-3 datapoint in the table; do not write
"trained for 490 epochs" against it.

**Training-curve summary** (49 logged epochs, every 10):

| Metric | Epoch 10 | Epoch 490 |
|---|---|---|
| `lm_loss` | 3.52 | 1.09 |
| train `accuracy` | 0.955 | 0.988 |
| train `exact_accuracy` | 0.891 | 0.965 |
| `val_cell_acc` | 0.916 | 0.851 |
| `val_puzzle_acc` | **0.8474** | **0.7255** |
| `best_puzzle_acc` (running max) | 0.8474 | 0.8474 |
| `avg_steps` (ACT) | 4.1 | 2.2 |

Train-vs-val divergence is unambiguous: classic overfit. `q_continue_loss`
sits at 0.0000 throughout because the config sets `no_ACT_continue: true`
(paper-faithful) — it is *not* a bug. `val_*` columns repeat between
logged rows because `eval_interval=50` while `log_interval=10`; the
trainer carries the last eval forward.

---

## 2. Seed table — what we have locally

11 sudoku-mlp wandb run dirs scanned. They split cleanly into two columns
of the results table.

### 2.1 TRM-MLP sudoku-extreme — **HF-init** (start = published 84.80 %)

| Run | Seed | Epochs | `best_puzzle_acc` | Notes |
|---|---|---|---|---|
| `jnlz33qp` (Apr 11) | 0 | 45 | — | val column empty; no eval recorded |
| `sa6b3d2y` (Apr 22) | 1 | 45+ (still going) | **0.8483** | only the pre-training eval has fired |
| `dhz4ksox` (Apr 22) | 3 | 490 | **0.8474** | also captures overfit collapse to 0.7255 |

**Two clean datapoints: 0.8483, 0.8474. Mean ≈ 0.8479 ± 0.0006 (range).**
Halt-RNG variance only — model weights are fixed (HF). For seed-0 you
can either re-run the eval (~2 h) or cite the published 0.8480 as the
third anchor.

### 2.2 TRM-MLP sudoku-extreme — **from scratch**

| Run(s) | Seed | Epochs | `best_puzzle_acc` | Peak epoch |
|---|---|---|---|---|
| `94idw79x` / `7slrbwqm` / `1cv2jtcg` / `ihj6hpsn` (same training, multiple uploads) | 0 | 2245 | **0.7456** | 900 |
| `22aagzdz` (Apr 13) | 1 | 545 (still going) | **0.7445** so far | not yet at peak |
| `hdky5fnw` (Apr 13) | 2 | 2230 | **0.7486** | — |

**Two complete datapoints: 0.7456, 0.7486. Mean ≈ 0.7471, range 0.0030.**
Seed-1 needs ~1500 more epochs (~3 days at observed throughput) to be
comparable to the others. Either let it finish or ship a 2-seed table.

> ⚠️ Do not double-count seed-0: the four runs `94idw79x`, `7slrbwqm`,
> `1cv2jtcg`, `ihj6hpsn` share identical final stats — they are one
> training resumed/re-uploaded several times.

### 2.3 What `results/summary.csv` already has vs. what's missing

`summary.csv` (5 rows): `sudoku` (broken, peak=0.0), `sudoku-att`
(0.183), `llm-qwen-sudoku-seed0` (0.0 puzzle / 0.191 cell),
`sudoku-mlp-seed0` (0.7456). Header row.

**Missing rows that we have data for:**

- `sudoku-mlp-seed1` (from-scratch, `22aagzdz`, partial)
- `sudoku-mlp-seed2` (from-scratch, `hdky5fnw`, complete)
- `sudoku-mlp-hfinit-seed1` (`sa6b3d2y`)
- `sudoku-mlp-hfinit-seed3` (`dhz4ksox`)

The `scripts/aggregate.py` pipeline (per `src/evaluation/aggregate.py`)
walks `experiments/` and `$TRM_EXPERIMENT_DIR`; running it should refresh
the CSV automatically once the runs are catalogued.

### 2.4 Maze runs

Per memory note `project_trm_maze_trained_3_seeds.md`, maze has been
fine-tuned from HF weights on machines FCM/FDK/FFN across 3 seeds, but
the data is in **wandb cloud** under `shamykyzer/TRM`, not on this
laptop's disk. The local `experiments/maze` and `experiments/maze-official`
dirs may have CSVs from an earlier rig; not inspected this round.

---

## 3. LLM-fleet config audit (corrected)

12 YAMLs in `configs/`: 10 LLM + 2 new TRM-finetune configs that
appeared in the 14-commit batch I pulled before writing this section.

### LLM configs vs. local artifacts (across `results/runs/`,
`results/summary.csv`, root analyses)

| Config | Status | Source of evidence |
|---|---|---|
| `llm_qwen.yaml` (sudoku) | ✅ run | `summary.csv: llm-qwen-sudoku-seed0` (0 % puzzle / 19.07 % cell) |
| `llm_qwen_maze.yaml` | ✅ run | `results/runs/qwen2.5-0.5b-maze-seed0-2026-04-22/` |
| `llm_llama.yaml` | ❌ no artifact | — |
| `llm_llama_maze.yaml` | ❌ no artifact | — |
| `llm_smollm.yaml` | ❌ no artifact | — |
| `llm_smollm_maze.yaml` | ❌ no artifact | — |
| `llm_deepseek.yaml` | ❌ no artifact | — |
| `llm_deepseek_maze.yaml` | ❌ no artifact | — |
| `llm_gpt2_maze.yaml` | ❌ no artifact | — |
| `llm_config.yaml` | n/a | generic template, not a runnable target |

Plus distilled-LLM rows (separate from base-LLM fleet but same axis):

| Run | Status | Source |
|---|---|---|
| Distill student on sudoku | ✅ run | `training_run_analysis.md` (rig 2, run #5, ~25.8 % cell at 4 min / 0.011 kWh) |
| Distill student on maze | ✅ run | `analysis_2026-04-23_distill_maze.md` (`results/runs/distill-maze-seed0-2026-04-23/`) |

### New TRM finetune configs (post-pull)

| Config | Notes |
|---|---|
| `trm_official_sudoku_mlp_finetune.yaml` | Specifically a fine-tune-from-HF recipe; suggests the team is consolidating the warm-start workflow into a dedicated YAML rather than the `--init-weights` CLI flag. |
| `trm_official_maze_finetune.yaml` | Same pattern for maze. |

These two YAMLs were committed by a teammate (commits `5ef357c` and
neighbours). My latest run (`dhz4ksox`) used the *old* method
(`trm_official_sudoku_mlp.yaml` + `--init-weights`), so it does not
exercise these new YAMLs.

### Coverage summary for the report

The "LLM" column of the comparison table can have **2 base-LLM cells
filled (Qwen-sudoku, Qwen-maze) and 2 distill cells (distill-sudoku,
distill-maze)** — a 2×2 LLM block, not 1×1 as I claimed in the TL;DR
of the previous draft. The 6 missing base-LLM configs (Llama / SmolLM /
DeepSeek + the GPT-2 maze) are likely deliberately deferred under the
3-rig iso-time budget, not waiting to be picked up locally.

---

## 4. Recommendations for the remaining 6 days

Ranked by coursework-deadline impact, not by interestingness.

### Priority 0 — today (2026-04-25)

1. **Stop `dhz4ksox`** cleanly. `best.pt` is locked at epoch 0 (HF
   init); further training is purely overfitting. Free the GPU.
2. **Decide on seed-1 in-flight runs.** `sa6b3d2y` (HF-init s=1) and
   `22aagzdz` (from-scratch s=1) are still alive on disk. If they're
   actually still progressing, leaving them is cheap — they fill cells
   in the table. If the host is asleep / process dead, restart only
   `22aagzdz` (the from-scratch needs more epochs; HF-init s=1 already
   has a usable 0.8483 at epoch 45).
3. **Sync wandb cloud** — pull project `shamykyzer/TRM` and
   `shamykyzer/TRM-LLM` runs lists. Maze rows and most LLM rows likely
   live there from teammate rigs.

### Priority 1 — days 1–2

4. Re-run `python -m src.evaluation.aggregate` (or
   `scripts/publish_wandb_metrics_report.py`) to refresh
   `results/summary.csv` with the seeds listed in §2.3.
5. **Decide whether more LLM cells are needed.** With Qwen-sudoku,
   Qwen-maze, distill-sudoku, distill-maze already filled (a 2×2
   LLM/distill block), the comparison is more defensible than I
   initially claimed. Adding a Llama or SmolLM cell only matters if the
   report explicitly compares *across LLM scales*. If it just needs
   "one fine-tuned LLM, one distilled student, both tasks" then the
   block is complete — spend the time on writing instead.

### Priority 2 — days 3–6

6. Report writing. Figures already exist (`results/figures/
   sudoku_mlp_peak_and_overfit.png`,
   `sudoku_att_rise_and_collapse.png`). What's missing is prose:
   Methods (paper-faithful re-impl, halt-RNG variance, energy logging
   via `co2_per_correct_puzzle`), Results (tables + the two figures),
   Discussion (overfit story already framed in README), Limitations
   (single-GPU / single-rig training, only two LLM datapoints if §1.5
   isn't filled, halt-RNG stochasticity in eval).
7. Final pre-submission audit: confirm every claim in the report has a
   row in `results/summary.csv` or a wandb URL.

### Things NOT to do

- Do **not** continue `dhz4ksox` past epoch 500 — sunk cost.
- Do **not** try to beat 84.80 % by re-tuning lr/wd in 6 days — every
  prior run says 84.80 % is the ceiling for this config. Reproducing it
  is the contribution.
- Do **not** rely on local files for maze data — they live in wandb.
- Do **not** force-push or rewrite history; this repo's git index has
  stale cache-tree entries from past gc/worktree activity (visible via
  `git fsck`). New commits work fine but `git gc` / `git filter-repo`
  could surface the corruption.

---

## 5. Open questions for the next session

- Is `dhz4ksox` actually still running, or is the host asleep? (Check
  with `tasklist | findstr python` and `wandb run state`.)
- Do `experiments/maze` / `experiments/maze-official` / `experiments/sudoku-mlp`
  dirs contain CSVs we haven't aggregated yet? (Not inspected this round.)
- Which LLM rows already exist in wandb cloud under teammates' rigs?
  Need to query `shamykyzer/TRM-LLM` API directly.

---

*Generated by analysing local wandb run dirs, `results/summary.csv`,
`configs/`, and `best.pt` at 2026-04-25. Frozen snapshot — current state
will diverge as runs continue or are stopped.*

---

## Postscript — 2026-04-25 evening: why `dhz4ksox` actually froze

Investigation after the snapshot above revealed the run was not just
"stalled" but **wedged** — Python alive at 1.6 GB RSS, CUDA context
dead, last activity 2026-04-23 16:06. Two-day silence with the laptop
intermittently waking (wandb heartbeat thread updating the run file)
but the training thread never resuming.

**Root cause:** Windows laptop default power policy. The host
(`STU-CZC5277FGD`) was sleeping/hibernating during long jobs, even on
AC. When CUDA contexts go away in a sleep transition, PyTorch only
finds out the next time the *training thread* makes a CUDA call — and
if it's blocked in `DataLoader.__next__` or waiting on a CUDA result,
it never gets there. The process becomes a zombie.

Earlier corroborating evidence in the same CSV: `elapsed_min` jumped
138 → 258 between epochs 50 and 60, then 397 → 534, and so on —
multi-hour mid-run gaps that match suspend/resume cycles. The freeze
on 2026-04-23 was just the suspend cycle that didn't recover.

**Fix applied (this session):**

```cmd
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change disk-timeout-ac 0
```

Verified with `powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE` —
the AC indices now read `0x00000000` (= never). Battery (DC) policy
left alone. Lid-close action not changed; close the lid and it'll
still sleep, so future long runs need the lid open.

**Operational note for the team's other rigs (FCM / FDK / FFN-style
hostnames in memory):** apply the same three commands on each rig
before launching anything that runs longer than the default 30-min
idle timeout. Combined with `TRM_ROLLING_CHECKPOINT_DIR` (already set
to `C:\ml-trm-work`), even a hard hang costs at most one
`save_interval=20` epochs of work because `--checkpoint
<latest.pt>` resumes cleanly.

**What this does NOT fix:** OneDrive sync races on `.git/` (still a
risk — the `--refetch` recipe in §4 is the recovery path). Lid-close
sleep (close lid, still sleeps).
