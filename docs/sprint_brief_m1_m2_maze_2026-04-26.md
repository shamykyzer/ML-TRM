# Claude Code agent prompt — TRM-Att Maze fine-tune (M1 seed 0 / M2 seed 1)

**Paste everything below this line into Claude Code on M1 or M2.** The
agent will identify which box it's on, wait for any in-flight run to
finish, then drive Phase 0 → Phase 2 autonomously.

---

You are an autonomous Claude Code agent on **either M1 or M2** of the
ML-TRM Group Project sprint (2026-04-26 → 2026-05-01).

- Repo: `https://github.com/shamykyzer/ML-TRM`. Standard local clone
  **outside OneDrive** — `.git` corrupts under OneDrive sync.
- Today: 2026-04-26. Submission: 2026-05-01 17:00 BST.
- Hardware: RTX 5070 (Blackwell, 12.8 GB, sm_120, CUDA 12.8). 30 h GPU
  budget on this box.
- Your assignment depends on your hostname / which physical box you're
  on:
  - **M1 → TRM-Att Maze fine-tune, seed 0**
  - **M2 → TRM-Att Maze fine-tune, seed 1**
  - (M3 already ran seed 2 and produced a meaningful failure finding —
   see `analysis_run_8kaoy99b.md`.)
  
  If you cannot determine which box you're on, **ask the user** before
  proceeding. Do **not** guess. The seed is the only knob you must not
  get wrong.

Total wall-clock budget: ~18 h training + ~30 min K-vote, leaving
~10 h+ for paper writing.

DO ALL OF THE FOLLOWING IN ORDER. Each phase is independent — if one
fails, fix and resume from there.

## Hard rules (read these once, apply throughout)

- **Use `python start.py` for every launch**, not direct `python main.py`
  invocations. Bypassing the launcher dropped the wandb-auth path
  silently in the M3 sprint and almost cost a 23 h run with no cloud
  logging.
- **Never push to remote unless the user explicitly says "push".** Pulls
  are fine. Local commits are fine.
- **Don't change your seed.** M1 is seed 0, M2 is seed 1. If your seed
  fails, that's a finding to report, not a reason to swap.
- **Don't edit configs.** The config you'll use
  (`configs/trm_official_maze_finetune.yaml`) was updated on `main` with
  the M3 post-mortem fixes; pull before launch, then leave it alone.
- **Don't commit `wandb_api.txt`, `.env`, or anything under `wandb/` /
  `hf_checkpoints/`.** All gitignored. Never override.
- **No `--no-verify`, no force-pushes, no destructive git ops without
  asking.** Fix root causes, don't bypass guardrails.
- **If a phase fails, log the exact error and tell the user.** Do not
  improvise fixes that involve editing the trainer or models. Config
  changes already live on `main` are the only changes you should rely on.
- **Report honestly.** Including failures. The M3 seed-2 collapse is a
  paper-relevant finding, not an embarrassment.

## What the M3 post-mortem changed (tl;dr)

Seed 2 of TRM-Att Maze with the original config collapsed
`val_puzzle_acc` from 0.7960 → 0.000 inside one epoch (~100 batches).
Eval-only of the HF init alone confirmed the checkpoint is valid
(0.7960 puzzle, 0.9754 cell). The diagnosed cause is **adam_atan2's
direction-only update doesn't damp at a converged optimum**: it
random-walks at ~`lr·π/2` per step regardless of gradient magnitude,
escaping the HF basin in ~100 steps. The fix:

| Field | Old | New | File |
|---|---|---|---|
| `optimizer` | `adam_atan2` | **`adamw`** | `configs/trm_official_maze_finetune.yaml` |
| `warmup_steps` | 200 | **500** | same |
| `log_interval` / `eval_interval` / `save_interval` | 5 | **20** | same |

These are already on `main`. Verify with `git pull` in Phase 0 and check
the first few stdout lines say `[Optimizer] Using AdamW`.

Realistic per-epoch cost on RTX 5070 at the GPU-profile-overridden
batch_size 16 is **~27 min/epoch**, not the original brief's ~7.4 min.
Plan for ~18 h to converge with the recommended early-stop, ~45 h if
you let it run all 100 epochs.

# Phase 0 — Pre-flight (~10 min, blocking)

## 0.1 — Wait for any in-flight run to finish

Before doing anything else, check whether the previous (pre-fix) config
is still training on this box. **You must not interleave runs** — that
corrupts the train_log.csv, the wandb run, and the checkpoint dir.

PowerShell check:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*main.py*--mode*train*' } |
  Select-Object ProcessId, CommandLine | Format-List
```

- **No matching process** → proceed to 0.2.
- **A matching process is running on the OLD config (no `optimizer:
  adamw` in its launch line)** → see §"Course-correction for in-flight
  old-config runs" at the bottom of this prompt. Decide with the user
  whether to wait it out, kill, or let it finish.
- **A matching process is running on the NEW config (i.e., this prompt
  was relaunched while a previous instance is still going)** → wait.
  Don't start a second one. Re-check status every ~30 min until it
  finishes; report the wait to the user. The Bash polling pattern:

  ```bash
  until [ -z "$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*main.py*--mode*train*' } | Select-Object -First 1 ProcessId | Out-String").trim()" ]; do sleep 60; done
  ```

  When it returns, proceed to 0.2.

## 0.2 — Repo + setup verification

```powershell
cd <repo root, outside OneDrive>
git pull
& <venv>\Scripts\python.exe start.py status
```

Expected:
```
[✓] venv     [✓] sync     [✓] env
[✓] wandb    [✓] transfer [✓] data
All blocking stages ready.
```

If any stage is `[ ]`:

```powershell
& <venv>\Scripts\python.exe start.py     # advances ONE missing stage; re-run until all ✓
```

The `wandb` stage uses `wandb_api.txt` at the repo root → `.env` /
`~/_netrc`. If it's still `[ ]` after re-running, drop the team's key
into `wandb_api.txt` and try again.

The `data` stage requires both `data/sudoku-extreme-full/train/all__inputs.npy`
and `data/maze-30x30-hard-1k-aug/train/all__inputs.npy`. If
`start.py` is rebuilding from scratch and you have a pre-built copy
elsewhere on the network, copying is faster than rebuilding.

## 0.3 — Final config sanity check

```powershell
Get-Content configs\trm_official_maze_finetune.yaml | Select-String -Pattern '^\s*(optimizer|warmup_steps|eval_interval|log_interval|save_interval):' -CaseSensitive
```

Expected (exact strings):
```
  optimizer: adamw
  warmup_steps: 500
  log_interval: 20
  eval_interval: 20
  save_interval: 20
```

If any disagrees → `git pull` again, you have a stale checkout. If
post-pull they still disagree → tell the user; do **not** edit the file
yourself.

## 0.4 — Inventory check

```powershell
& <venv>\Scripts\python.exe scripts\check_local_checkpoints.py --machine M1   # or M2
```

The single row for your machine should read `[--] NOT-STARTED`. If it
reads `STARTED` or `DONE`, your machine has a half-finished run from
before — clear it (move the dir aside, don't delete) and reset.

# Phase 1 — Training (~18 h with early-stop, ~45 h max)

```powershell
cd <repo root>
& <venv>\Scripts\python.exe start.py
# Pick option 8  (Group Project sprint — Maze 100-ep)
# Enter seed:    0   on M1   /   1   on M2
# Confirm:       y
```

The launcher will:
1. Run `scripts/bootstrap_hf_maze.py` (idempotent — skips if already
   bootstrapped).
2. Set `TRM_CHECKPOINT_DIR = TRM_EXPERIMENT_DIR =
   <work_dir>/trm-att-maze-50ep-seed{0|1}/`.
3. Launch `main.py` with the corrected config.

## 1.1 — First-30-second sanity gate

The first ~10 lines of the launcher's stdout MUST contain, in order:

```
[GPU Config] Detected: NVIDIA GeForce RTX 5070
[Seed] 0    (or 1)
TRM-Official params: 6,823,938
[Optimizer] Using AdamW                          ← critical
Loading init weights from hf_checkpoints/Maze-Hard/remapped_for_local.pt
  loaded:     24/24 keys from checkpoint
  missing   (3): inner.rotary_emb.{inv_freq,cos_cached,sin_cached}
[EMA] shadow reseeded from model params (post-init-weights, fp32)
```

Failure modes to catch immediately:

- `[Optimizer] Using AdamATan2` instead of AdamW → config not pulled.
  Stop, `git pull`, re-launch.
- `loaded: 0/24` or `missing: 27` → init checkpoint not loading. Stop,
  see `analysis_run_8kaoy99b.md` §"Verified: HF init is loading" for
  diagnostics. Don't continue training without init weights.
- Stdout ends after the codecarbon warning with no setup banner → the
  launcher is using buffered stdout under redirection. Add `-u` to the
  python invocation if you redirected output to a file. (start.py
  shouldn't redirect; if you've added `> file 2>&1`, you've reproduced
  the M3 visibility bug. Drop the redirect or add `-u`.)

If any of the above fails, stop and tell the user before re-attempting.

## 1.2 — KILL RULE (first eval lands at epoch 20, ~9 h in)

Watch wandb (live) or the on-disk CSV
(`<work_dir>/trm-att-maze-50ep-seed{0|1}/trm_official_maze_train_log.csv`).
Apply at the **first eval row** (epoch 20):

| First-eval val_puzzle_acc | Action |
|---|---|
| `≥ 0.78` (HF baseline range) | **Continue.** AdamW held the model in basin. Expect peak around epoch 50–80. |
| `0.5 ≤ val < 0.78` | **Continue, flag.** Drifted but didn't collapse; K-vote may still help. |
| `< 0.5` | **Kill.** AdamW didn't save this seed either. Tell the team — implies the failure is *not* optimizer-driven. **Do not auto-restart.** Next hypothesis to investigate (off this box) is train-set distribution shift. |

The current trainer doesn't auto-halt on regression at this cadence, so
the kill is a manual `Ctrl+C` (or `Stop-Process -Force` on the python
PID — both work; `Ctrl+C` is graceful and lets the optimizer flush the
last checkpoint).

## 1.3 — Recommended early-stop (saves ~25 h)

The dz3tkge9 (sudoku) post-mortem and the M3 (maze) run both showed that
fine-tuning a converged HF checkpoint plateaus quickly — there's no
research value in 100 epochs if 50 already at the peak. **If
`val_puzzle_acc` has not improved over the previous eval for two
consecutive evals (= 40 epochs without progress), `Ctrl+C` and proceed
to Phase 2 with the latest `best.pt`.** Roughly an 18 h cap instead of
45 h.

## 1.4 — Hard wall-clock budget

If the run is still going after **20 h** with no `best.pt` better than
the HF baseline (0.7960), kill and report. Better to spend remaining
GPU on a useful artifact than on a long tail.

## 1.5 — Note `best.pt` path on completion

It will be at `<work_dir>/trm-att-maze-50ep-seed{0|1}/best.pt`. The
launcher reuses the `trm-att-maze-50ep-seed{N}` folder name even though
the config is now 100 ep — don't be confused by the `50ep` substring.

# Phase 2 — K-vote on the resulting checkpoint (~30 min)

When Phase 1 produces a usable `best.pt`:

## 2.1 — Bridge the path

`scripts/run_novelty_k_vote.py` looks for checkpoints at
`<work_dir>/novelty-trm-att-maze-seed{N}/latest.pt`, not at the
launcher's `trm-att-maze-50ep-seed{N}/`. Copy:

```powershell
$seed = 0    # or 1 on M2
New-Item -ItemType Directory -Path "C:\ml-trm-work\novelty-trm-att-maze-seed$seed" -Force | Out-Null
Copy-Item "C:\ml-trm-work\trm-att-maze-50ep-seed$seed\best.pt" `
          "C:\ml-trm-work\novelty-trm-att-maze-seed$seed\latest.pt"
```

## 2.2 — Run the K-vote sweep, scoped to this seed only

```powershell
& <venv>\Scripts\python.exe scripts\run_novelty_k_vote.py `
    --seed 0 `       # or 1 on M2
    --work-dir C:/ml-trm-work `
    --k-values 1,2,4 `
    --skip-labels trm-mlp-sudoku,qwen-sudoku,qwen-maze,distill-sudoku,distill-maze
```

(`--seed N` matches your assigned seed.) The skip list excludes the
other 5 RUNS slots in the matrix so this box only K-votes its own
checkpoint.

Outputs:
- `results/novelty/k_vote_runs/trm-att-maze/` — per-K predictions
- `results/novelty/k_vote_results-rig0.csv` (or `-rigN.csv`) — accuracy + kWh per K

# Phase 3 — Inventory + handoff (~5 min)

```powershell
& <venv>\Scripts\python.exe scripts\check_local_checkpoints.py --machine M1   # or M2
```

Both rows for your machine (training + K-vote) should now read **DONE**.

Tell the user, in this order:

1. Phase 1 best `val_puzzle_acc` and the epoch it occurred at
2. Phase 1 final eval (last logged row, in case it doesn't match best)
3. Phase 1 wall-clock and total CO₂ (last row of `emissions.csv`)
4. Phase 2 K-vote curve numbers (`puzzle_acc` at K=1, 2, 4)
5. The full path to `best.pt` so the team can sync it for the cross-seed
   aggregation on Sunday/Monday
6. **Whether the AdamW swap reproduced the HF baseline** — i.e., did the
   first eval land at ≥0.78? This is the single most important number
   for the report; the team is comparing it to M3's 0.000.

# Course-correction for in-flight old-config runs

If 0.1's process check returned a python.exe running on the **previous
adam_atan2 + warmup=200 + every-5-eval config** (you can tell because
the launcher's stdout will say `[Optimizer] Using AdamATan2` and the
config diff vs `main` will show the old values):

- **Has it crossed epoch 5?** Look at the train_log.csv. If the row at
  epoch 5 has `val_puzzle_acc < 0.5`, it's reproducing M3's seed-2
  collapse — `Ctrl+C` and re-launch on the new config.
  If `val_puzzle_acc ≥ 0.5`, your seed has tolerated adam_atan2 better
  than seed 2 did; let it finish on the old config and **report the
  cross-config comparison as a finding** (this seed's threshold for
  surviving adam_atan2 vs seed 2's collapse is paper-relevant).
- **Has it not yet hit epoch 5?** Either is reasonable — kill and
  re-launch on the new config (faster, expected to work), or let it
  finish to add the data point on the old config. Tell the user which
  you chose and why.
- **Either way: don't half-step.** Don't pull the new config mid-run —
  that gives a confused trajectory that's neither old nor new.
  Either fully restart on the new config or fully finish on the old.

# Reading order if you have 10 minutes before launching

1. `analysis_run_8kaoy99b.md` — M3 seed-2 post-mortem; the failure-stack
   hypothesis (1)–(5).
2. `analysis_run_dz3tkge9.md` — earlier sudoku-mlp post-mortem; explains
   why the original config was tuned the way it was.
3. `configs/trm_official_maze_finetune.yaml` — current config; inline
   comments document each non-default value's rationale.
4. `docs/setup-guide.txt` — `start.py` stages reference.
5. `docs/weave_setup.md` — Weave regression-alert mechanism (optional
   instrumentation).

Good luck. Numbers matter more than narrative — report what happened,
including failures.
