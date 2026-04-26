# M1 / M3 recovery sprint — 2026-04-26

You are an autonomous Claude Code agent on **M1 or M3** of the ML-TRM Group Project sprint.

Repo: `https://github.com/shamykyzer/ML-TRM` — standard local clone **outside OneDrive** (`.git` corrupts under OneDrive sync).
Today: 2026-04-26. Submission: 2026-05-01 17:00 BST.
Hardware: RTX 5070 (Blackwell, 12 GB, sm_120, CUDA 12.8). ~30 h GPU budget on this box.

Your assignment depends on which physical box you're on:
- **M1 → seed 0**
- **M3 → seed 2**

If you cannot determine which box you're on, ask the user before proceeding. **Do not guess. The seed is the only knob you must not get wrong.**

---

## Why this prompt exists (read once)

M2 ran the original sprint plan (TRM-Att Maze fine-tune, seed 1) on the AdamW config and **collapsed at the first eval** to `val_puzzle_acc = 0.000` — same end state as M3's seed 2 on adam_atan2. AdamW didn't save it.

Forensic comparison against `hf_checkpoints/Maze-Hard/all_config.yaml` and `losses.py` (Sanjin's actual training artifacts) identified the root cause: **`q_loss_weight: 0.0` in `configs/trm_official_maze_finetune.yaml` zeros out the Q-halt gradient signal entirely.** Combined with `halt_exploration_prob: 0.0`, the halt head has no learning signal at all — it trivially settles on "always continue", `avg_steps` pins at 16, no puzzle ever halts → `val_puzzle_acc = 0` regardless of how good the cell-prediction layers stay.

Sanjin trained with `q_loss_weight = 0.5` (hardcoded in `losses.py:148` — `lm_loss + 0.5 * (q_halt_loss + q_continue_loss)`).

Full M2 post-mortem: see the conversation thread that produced this prompt; eventually saved as `analysis_run_1u5fesvh.md`.

So the original sprint plan is **dead on arrival** until that config is fixed. This prompt does two things instead, both producing real paper artifacts:

1. **K-vote the HF checkpoint as-is.** Sanjin's released checkpoint is the maze baseline (`val_puzzle_acc = 0.7960`). K-voting it gives a real puzzle-acc-vs-K curve at the published baseline. ~30 min. Honest framing: "we evaluated the released checkpoint with our K-vote scheme; here's the Pareto."
2. **Short validation run with the loss fix.** Edit *only* `q_loss_weight: 0.0 → 0.5` in the fine-tune config, run for 10 epochs, eval at epochs 5 and 10. ~3 h. If `val_puzzle_acc` recovers toward 0.7960, the diagnosis is confirmed empirically and we have a *method contribution* for the paper.

Total wall-clock per machine: ~3.5 h. Leaves ~26 h for paper writing or extra experiments.

---

## Hard rules

- **Never push to remote unless the user explicitly says "push".** Pulls and local commits are fine.
- **Don't change your seed.** M1 = 0, M3 = 2. If your seed fails, that's a finding to report.
- **One config edit allowed (Phase 2 only):** `q_loss_weight: 0.0 → 0.5` in `configs/trm_official_maze_finetune.yaml`. Don't touch anything else. Revert before committing if asked.
- **Don't commit `wandb_api.txt`, `.env`, `wandb/`, `hf_checkpoints/`** — all gitignored, never override.
- **No `--no-verify`, no force-pushes, no destructive git ops** without explicit permission.
- **Use `python start.py` for Phase 2 launch** (option 8). It now works — M2 fixed the `menus.py:626` TypeError and pushed (`dc09f0c fix(cli): use subprocess.run directly in option-8 launcher`). Pull before launching.
- **If a phase fails, log the exact error and tell the user.** Do not improvise fixes that involve editing the trainer or models.
- **Report honestly, including failures.**

---

## Phase 0 — Pre-flight (~10 min, blocking)

### 0.1 — Process check

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*main.py*--mode*train*' -or $_.CommandLine -like '*run_novelty_k_vote*' } |
  Select-Object ProcessId, CommandLine | Format-List
```

If a previous run is still going, decide with the user whether to wait, kill, or let it finish. Don't interleave runs.

### 0.2 — Repo + setup

```powershell
cd <repo root, outside OneDrive>
git pull
& <venv>\Scripts\python.exe start.py status
```

Expected:
```
[✓] venv     [✓] sync     [✓] env
[✓] wandb    [✓] transfer [✓] data
```

If any stage is `[ ]`, run `& <venv>\Scripts\python.exe start.py` (advances one stage; re-run until all ✓). Wandb setup uses `wandb_api.txt` at repo root if needed.

**Critical pull check:** `git log --oneline -1` must show `dc09f0c` (the option-8 fix) or newer. If not, `git pull` again.

### 0.3 — Inventory

```powershell
& <venv>\Scripts\python.exe scripts\check_local_checkpoints.py --machine M1   # or M3
```

If your machine's row reads STARTED or DONE from a prior run, move the dir aside (don't delete):
```powershell
Move-Item -Path "C:\ml-trm-work\trm-att-maze-50ep-seed$seed" -Destination "C:\ml-trm-work\trm-att-maze-50ep-seed$seed.preempted-2026-04-26" -Force
```

---

## Phase 1 — K-vote the HF baseline (~30 min)

This produces the K-vote Pareto curve at `puzzle_acc ≈ 0.7960` baseline. **Highest-priority deliverable** — do this first.

### 1.1 — Bridge the HF checkpoint to K-vote's expected path

```powershell
$seed = 0    # M1
# $seed = 2  # M3
New-Item -ItemType Directory -Path "C:\ml-trm-work\novelty-trm-att-maze-seed$seed" -Force | Out-Null
Copy-Item "<repo root>\hf_checkpoints\Maze-Hard\remapped_for_local.pt" `
          "C:\ml-trm-work\novelty-trm-att-maze-seed$seed\latest.pt"
```

### 1.2 — Run K-vote scoped to this seed only

```powershell
& <venv>\Scripts\python.exe scripts\run_novelty_k_vote.py `
    --seed $seed `
    --work-dir C:/ml-trm-work `
    --k-values 1,2,4 `
    --skip-labels trm-mlp-sudoku,qwen-sudoku,qwen-maze,distill-sudoku,distill-maze
```

(`--seed $seed` matches your assigned seed; the skip list excludes the other 5 RUNS slots so you K-vote only the trm-att-maze entry.)

### 1.3 — Outputs

- `results/novelty/k_vote_runs/trm-att-maze/<seed>/predictions.csv` — per-puzzle predictions per K
- `results/novelty/k_vote_results-rig{N}.csv` — accuracy + kWh per K

**Sanity check:** `puzzle_acc` at K=1 should be near 0.7960 (Sanjin's baseline). If it's near 0, the HF checkpoint didn't load — stop and report.

---

## Phase 2 — Validation: q_loss_weight = 0.5 short fine-tune (~3 h)

Tests whether restoring Sanjin's hardcoded `q_loss_weight = 0.5` fixes the seed-1 collapse. **Cross-seed reproduction is the value-add** — if M1 (seed 0), M2 (seed 1, post-hoc), and M3 (seed 2) all recover toward baseline, the fix is confirmed across seeds and becomes a robust paper finding.

### 2.1 — Edit the config (one line)

Open `configs/trm_official_maze_finetune.yaml`. Find the `q_loss_weight` line:

```yaml
  q_loss_weight: 0.0
```

Change to:

```yaml
  q_loss_weight: 0.5
```

Don't touch anything else. **Don't commit this edit yet** — leave it as a working-tree change so the team can review before merging. (If asked to commit, message: `experiment(maze-finetune): q_loss_weight=0.5 to test halt-head signal recovery`.)

### 2.2 — Also bump epochs cap to 10 for this short test

In the same file, find:
```yaml
  epochs: 100
```
Change to:
```yaml
  epochs: 10
```

(Eval still fires at epoch 5 and 10 because eval_interval=20 — wait, the new config has eval_interval=20, which means no eval inside 10 epochs. **Override this experiment with eval_interval=2** so we get evals at epochs 2, 4, 6, 8, 10:)
```yaml
  eval_interval: 2
  log_interval: 2
  save_interval: 2
```

These are temporary — revert all four lines (`q_loss_weight`, `epochs`, three intervals) before any commit.

### 2.3 — Launch via start.py

```powershell
cd <repo root>
& <venv>\Scripts\python.exe start.py
# Pick option 8
# Enter seed: 0 on M1 / 2 on M3
# Confirm: y
```

### 2.4 — Sanity gate (first 30 sec of stdout)

Must see, in order:
```
[GPU Config] Detected: NVIDIA GeForce RTX 5070
[Seed] 0   (or 2)
TRM-Official params: 6,823,938
[Optimizer] Using AdamW
Loading init weights from hf_checkpoints/Maze-Hard/remapped_for_local.pt
  loaded:     24/24 keys from checkpoint
  missing   (3): inner.rotary_emb.{inv_freq,cos_cached,sin_cached}
[EMA] shadow reseeded from model params (post-init-weights, fp32)
```

If anything is missing or wrong, stop and tell the user.

### 2.5 — Watch the eval CSV at `<work_dir>\trm-att-maze-50ep-seed$seed\trm_official_maze_train_log.csv`

| `val_puzzle_acc` at epoch 2 | Interpretation |
|---|---|
| ≥ 0.78 | The fix works immediately — `q_loss_weight=0` was the bug, full stop. |
| 0.5 – 0.78 | Drifting but recoverable; let it run to epoch 10. |
| 0.1 – 0.5 | Marginal recovery; let it run to epoch 10 and report trajectory. |
| < 0.1 | Fix didn't take. There's something else wrong (LR schedule? batch noise?). Report and stop. |

### 2.6 — Outputs

- Best checkpoint: `<work_dir>\trm-att-maze-50ep-seed$seed\best.pt`
- CSV with per-eval rows
- emissions.csv with energy used

---

## Phase 3 — Inventory + handoff (~5 min)

```powershell
& <venv>\Scripts\python.exe scripts\check_local_checkpoints.py --machine M1   # or M3
```

Report to the user (in this order):

1. **Phase 1 K-vote curve at your seed:** puzzle_acc at K=1, K=2, K=4 (from `results/novelty/k_vote_results-rig{N}.csv`).
2. **Phase 2 best `val_puzzle_acc` and the epoch it occurred at.**
3. **Phase 2 final eval (epoch 10).**
4. **Phase 2 wall-clock + total CO₂** (last row of `emissions.csv`).
5. **The single most important number for the report:** Did the `q_loss_weight=0.5` fix recover `val_puzzle_acc` from 0 to non-zero at epoch 2 eval? **Yes/No, with the exact value.** This is what the team is comparing across seeds (M1 seed 0 / M2 seed 1 / M3 seed 2) to confirm the diagnosis.
6. The full path to `best.pt` (Phase 2) and `latest.pt` (Phase 1 source) so the team can sync for cross-seed aggregation.

---

## After handoff

GPU is free. Use the remaining ~26 h for paper writing or, if you want to push further, ask the user about:
- Longer fine-tune at the corrected config (50–100 epochs)
- LR/batch-size sweep (the 16 vs 4608 batch gap is the next-most-suspect remaining gap)
- Sanity-evaluating the HF checkpoint without K-vote (option 3 in `start.py`) to ground-truth the 0.7960 baseline on this hardware

Don't escalate scope without explicit user approval.

---

## Reading order if you have 10 min before launch

1. `analysis_run_8kaoy99b.md` — M3 seed-2 post-mortem (the original "AdamW will fix it" hypothesis).
2. `analysis_run_1u5fesvh.md` — M2 seed-1 post-mortem (this prompt's motivation; refutes the AdamW hypothesis, identifies `q_loss_weight=0.0` as root cause).
3. `hf_checkpoints/Maze-Hard/all_config.yaml` — Sanjin's actual training config (compare to ours).
4. `hf_checkpoints/Maze-Hard/losses.py` — line 148: `lm_loss + 0.5 * (q_halt_loss + q_continue_loss)`. The 0.5 is what we need to restore.
5. `src/models/losses_official.py:144` — our equivalent line. Identical structure, but `q_loss_weight` is config-driven.

Good luck. Numbers matter more than narrative — report what happened, including failures.
