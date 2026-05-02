# Claude Code agent prompt — Final layer (30-min redundancy + metric realism)

**Created:** 2026-04-28
**Audience:** Any Claude Code session running on M1–M6, *layered after*
the machine-specific master prompt. This is the **final** layer — it
supersedes `sprint_layer_metric_realism_2026-04-28.md`. Paste the
section between BEGIN LAYER and END LAYER as the *next* user message
after the per-machine prompt.
**Submission deadline:** 2026-05-01 17:00 BST.

This layer encodes two contracts that every run in the sprint must
honour:

- **A. 30-minute checkpoint redundancy** — every active training run
  copies its latest checkpoint + train log + emissions CSV to a
  fleet-uniform location every 30 minutes, capping the worst-case crash
  loss at ~30 min of compute.
- **B. Metric realism monitoring** — a run only ships to the report if
  its training and evaluation metrics look like *realistic slow
  learning*, not like the saturation artifacts the
  `mask_non_path: true` bug produced. If a metric pattern violates the
  contract, the agent stops the run, marks the checkpoint suspect, and
  logs the diagnosis in `findings.md` §5.

---

## BEGIN LAYER

You have already read the machine-specific instructions above. Now read
this layer and apply all three contracts on top — they govern every
training run and every re-evaluation in this sprint.

# Contract 0 — Bootstrap via `start.py` (do FIRST, every session)

Before any other action — before any training launch, before any
re-eval, before any contamination check — run the project's
stage-aware bootstrap once. It verifies (and where possible repairs)
the venv, the `.env` file, wandb auth, the requirements lockstep, and
the dataset downloads. The whole sprint depends on these stages being
ready.

```bash
# Quick check: are all stages ready?
.venv/Scripts/python.exe start.py status

# Expected output for a healthy machine:
#   [✓] venv     Python venv with CUDA torch
#   [✓] sync     Venv matches requirements.txt
#   [✓] env      Machine-local .env file
#   [✓] wandb    wandb logged in
#   [✓] transfer HF reference remapped + verified
#   [✓] data     Sudoku + Maze datasets
#   All blocking stages ready.
```

If any stage shows `[ ]` or `[✗]`, run the bootstrap (no args) to
advance the next missing stage; re-run until all stages are green:

```bash
.venv/Scripts/python.exe start.py            # advances next missing stage and exits
```

`start.py` is the canonical setup path — installing requirements,
setting up wandb auth from `.env`, downloading the Sudoku-Extreme + Maze
datasets, and verifying the HF reference checkpoints all happen here.
Do **not** reinvent these checks. If `start.py status` says everything
is ready, every script in the sprint (training, re-eval, watchdog,
contamination check) is safe to run; if it doesn't, **stop, fix the
missing stage via `start.py`, and only then proceed**.

If `start.py` itself fails (e.g. CUDA mismatch, no GPU, network error
during data download), log the failure to `findings.md` §5 with
`bootstrap failed on machineN` and stop — this is the only blocker that
cannot be worked around in-session.

# Contract A — 30-minute checkpoint redundancy save

## A.1 Target path

```
C:/ml-trm-work/checkpoints to use/machine{N}/
```

`{N}` is your machine digit (M1 → `machine1`, M2 → `machine2`, …, M6 →
`machine6`). The folder name has a literal space in `checkpoints to
use`. Create the directory at the start of any training run if it does
not already exist:

```bash
mkdir -p "C:/ml-trm-work/checkpoints to use/machine1"   # M1 example
```

## A.2 What to copy every 30 minutes

For each in-flight training run, copy the following files (do not move,
do not symlink — copy, so the canonical run dir stays intact):

- `<run-dir>/<model>_latest.pt` → the live checkpoint
- `<run-dir>/<model>_train_log.csv` → the per-epoch metric log
- `<run-dir>/emissions.csv` → the CodeCarbon trace
- `<run-dir>/<model>_training_results.json` if it exists

Where `<run-dir>` is the trainer's `checkpoint_dir` (e.g.
`C:/ml-trm-work/llm-smollm-sudoku-seed0/`).

## A.3 Naming convention in the redundancy folder

Append an ISO-ish timestamp suffix so older snapshots are not
overwritten:

```
machine1/llm-smollm-sudoku-seed0__2026-04-28T1830__smollm2_360m_sudoku_latest.pt
machine1/llm-smollm-sudoku-seed0__2026-04-28T1830__train_log.csv
machine1/llm-smollm-sudoku-seed0__2026-04-28T1830__emissions.csv
```

Pattern: `{run_name}__{YYYY-MM-DDTHHMM}__{original_filename}`. The
double underscore `__` is the separator the aggregator can split on.

## A.4 Watchdog script — minimal and crash-isolated

Save as `scripts/checkpoint_redundancy_watchdog.sh` (Git Bash on
Windows) or as inline. The watchdog runs as a *separate* background
process from the trainer so a trainer crash cannot take it down:

```bash
#!/usr/bin/env bash
# Usage: bash scripts/checkpoint_redundancy_watchdog.sh <machine_n> <run_dir> <run_name>
set -u
MACHINE_N="${1:?machine number required, e.g. 1}"
RUN_DIR="${2:?run dir required, e.g. C:/ml-trm-work/llm-smollm-sudoku-seed0}"
RUN_NAME="${3:?run name required, e.g. llm-smollm-sudoku-seed0}"
DEST="C:/ml-trm-work/checkpoints to use/machine${MACHINE_N}"
mkdir -p "$DEST"

while true; do
  ts=$(date +%Y-%m-%dT%H%M)
  for src in "$RUN_DIR"/*.pt "$RUN_DIR"/*.csv "$RUN_DIR"/*.json; do
    [ -e "$src" ] || continue
    base=$(basename "$src")
    cp -p "$src" "$DEST/${RUN_NAME}__${ts}__${base}" 2>/dev/null || true
  done
  echo "[redundancy] $(date -Is) snapshot to $DEST"
  sleep 1800
done
```

Launch in background BEFORE starting the trainer:

```bash
mkdir -p "C:/ml-trm-work/checkpoints to use/machine1"
bash scripts/checkpoint_redundancy_watchdog.sh 1 \
  "C:/ml-trm-work/llm-smollm-sudoku-seed0" \
  "llm-smollm-sudoku-seed0" > /tmp/watchdog-machine1.log 2>&1 &
WATCHDOG_PID=$!

# Now launch the trainer
.venv/Scripts/python.exe main.py --mode train --config configs/llm_smollm.yaml

# When trainer exits, do one final snapshot then stop the watchdog
ts=$(date +%Y-%m-%dT%H%M)
for src in C:/ml-trm-work/llm-smollm-sudoku-seed0/*.{pt,csv,json}; do
  [ -e "$src" ] && cp -p "$src" "C:/ml-trm-work/checkpoints to use/machine1/llm-smollm-sudoku-seed0__${ts}__$(basename "$src")"
done
kill "$WATCHDOG_PID" 2>/dev/null || true
```

## A.5 Re-eval runs are exempt

Re-evaluations (no training, just inference) finish in 5–30 min, well
inside the 30-minute window — running the watchdog around them is not
required. Re-eval outputs land in `results/eval_fixed/` directly; for
fleet visibility, copy that single set of files to
`C:/ml-trm-work/checkpoints to use/machine{N}/eval_fixed/` ONCE at the
end of the re-eval run.

---

# Contract B — Metric realism monitoring

## B.1 Why this contract exists

The previous Maze evaluation reported **1.000 puzzle accuracy** for
every LLM and distilled student. That was a metric artifact (the
`mask_non_path: true` bug — the eval graded only path cells, ~10–15 %
of the 900-cell grid; a model that spams the path marker `o` at every
cell scored a fake 100 %). After the fix
(`mask_non_path: false`, scores all 900 cells), Qwen Maze re-evaluated
as **0/1000 puzzle, 12.52 % cell** and Distill-Qwen Maze as
**0/1000 puzzle, 12.50 % cell** — close to the chance-on-path-cells
floor and far below TRM's 79.60 % HF baseline.

The thesis the report needs to defend is *not* "LLMs are bad at
puzzles" in absolute terms — it is "LLMs **learn slowly and
inefficiently** at structured reasoning, while TRM converges with
orders of magnitude less compute". To support that thesis we need to
**see the slow learning curve in `val_cell_accuracy`**, with
**`val_puzzle_accuracy` pinned at zero**, throughout LLM/distill
training. Anything else is either a bug or an overfit/generalisation
failure that disqualifies the run from the report.

The TRM-Att Maze fine-tune training is heavy and out of our
hardware budget — that is settled. **TRM-Att Maze headline = HF
baseline 79.60 %** (Sanjin's published checkpoint). No TRM-Att Maze
retraining in this sprint.

## B.2 Realism contract — expected metric behaviour

| Class | Task | Expected `val_puzzle_acc` | Expected `val_cell_acc` |
|---|---|---|---|
| TRM-MLP HF eval | Sudoku | ~84.74 % | ~91.55 % |
| TRM-MLP from-scratch (3 seeds) | Sudoku | 74.25 ± 0.63 % | ~85.3 % |
| TRM-Att HF eval | Maze | ~79.60 % | ~99.30 % |
| TRM-Att fine-tune | Maze | **out of scope this sprint** | (heavy compute, not our hardware) |
| Any LLM + LoRA | Sudoku | **0.00 % across all epochs** | rises slowly from chance (1/11 ≈ 9.1 %) toward ~13–20 % |
| Any LLM + LoRA | Maze | **0.00 % across all epochs** | rises slowly from chance toward ~12–20 % |
| Distilled student | Sudoku | **0.00 % across all epochs** | rises slowly toward ~25–36 % |
| Distilled student | Maze | **0.00 % across all epochs** | rises slowly toward ~12–20 % |

The "rise" the contract expects is small per epoch — typically
**+0.5–2 percentage points per 10 epochs of cell accuracy**. A
monotonic, modest climb in cell accuracy with puzzle accuracy pinned at
zero is the signature of a model that is genuinely learning per-token
statistics but cannot compose them into a globally consistent solution.
**That is the report's thesis, observed.**

If you see `val_cell_accuracy` rising steadily — that is good news,
log it as evidence the model is learning even though it can never
solve a full puzzle.

If you see `val_puzzle_accuracy` rising on an LLM or distill — that is
**bad news**. Either (a) the `mask_non_path: false` fix is not active
on this run, (b) the dataset has train/test contamination not caught
by `scripts/check_maze_split_contamination.py`, (c) the model is
overfitting to a tiny validation split, or (d) there is a bug in the
eval comparison (e.g. shift mismatch). **Stop and investigate; do not
report the number.**

## B.3 Red flags — STOP the run, mark suspect, log

If any condition fires during training or re-evaluation, stop the run
cleanly at the next checkpoint boundary, mark the resulting artifact as
suspect, and log to `findings.md` §5 under a `metric realism
violation` entry. Do not include suspect runs in the report's headline
tables until a human investigates.

| Flag | Threshold | Likely cause |
|---|---|---|
| `val_puzzle_acc` ≥ 0.99 on LLM or distill, any epoch | one batch | `mask_non_path: true` not applied — config or eval script reading buggy default |
| `val_puzzle_acc` > 0.05 on LLM or distill | sustained ≥ 2 evals | mask bug, dataset contamination, or genuine overfit on tiny train |
| `val_cell_acc` flat at ~chance ≥ 10 epochs | LLM Sudoku ≤ 9.5 %, LLM Maze ≤ 17 % | model not learning — loss-mask bug, optimizer broken, wrong tokenizer |
| Training loss NaN or Inf | one batch | numerical issue — stop immediately, do not save |
| Training loss flat ≥ 5 epochs | abs(loss[t] − loss[t−5]) < 1e-3 | LR broken, gradient flow blocked, grad-accum misconfigured |
| `val_cell_acc` decreases monotonically ≥ 3 evaluations | running min on cell_acc | overfit on training set |
| TRM `avg_act_steps` > 14 throughout training | epoch ≥ 10 | halt head not learning — Q-loss or halt config broken |
| TRM `avg_act_steps` < 2 from epoch 0 | one epoch | halt head over-confident / collapsed — usually a `q_loss_weight` regression |

## B.4 Green flags — continue, the run is producing report-worthy data

| Flag | What it shows |
|---|---|
| `val_puzzle_acc` = 0.000 across all epochs (LLM/distill) | Composition failure — the report's central LLM finding |
| `val_cell_acc` rises slowly and monotonically | Per-token statistics being learned — supports the "slow learning" half of the thesis |
| Final `val_cell_acc` ∈ [chance, 2 × chance] | Learning is happening but inefficient — perfect Pareto-axis data point |
| TRM `avg_act_steps` falls from ~16 → ~5 over training | ACT halt head is learning when to stop |
| Training loss descends monotonically | Optimization is working |

## B.5 Pre-launch sanity check (for retrains, before the first epoch)

```python
from src.utils.config import load_config
cfg = load_config(<config_path>)
assert cfg.data.dataset in {"sudoku", "maze"}
if cfg.data.dataset == "maze":
    assert getattr(cfg.data, "mask_non_path", True) is False, \
        "Maze training MUST set mask_non_path: false"
print("[Sanity] config OK; mask_non_path =",
      getattr(cfg.data, "mask_non_path", "n/a"))
```

If the assertion fails, **stop, log to `findings.md` §5, do not
launch**.

## B.6 Mid-run monitoring (during training)

At every `eval_interval` (or `save_interval` if no eval interval is
defined), the trainer writes a row to `<model>_train_log.csv` and emits
a wandb step. After the row lands, run the realism checks above
against the latest row + the last 5 rows. If a red flag fires, send
SIGINT to the trainer's PID:

```bash
kill -INT <trainer_pid>
```

`trainer_llm.py` catches it and saves a final checkpoint + emissions
before exit. Then in `findings.md` §5 add: run name / wandb run ID,
epoch + metric values that triggered the stop, which red-flag rule
fired, what to investigate next.

## B.7 Post-run viability gate (after training completes)

For a run to be eligible for the report's headline tables, all of these
must hold:

1. Final `val_puzzle_acc` and `val_cell_acc` are within the §B.2
   ranges (or are explicitly in scope as a TRM headline).
2. Training loss reached at least its first plateau.
3. Emissions CSV exists with non-zero `energy_consumed` row.
4. Train-log CSV has one row per epoch (no gaps, no NaN columns).
5. No red flag from §B.3 fired during the run.

If all five hold, append the row to `results/summary_fixed.csv` and a
**`viability gate passed`** note to `findings.md` §5.

If any fail, mark the run as **superseded** in the summary CSV (do not
delete; preserve the audit trail) and post a one-line diagnosis in
`findings.md` §5 with the explicit string **`metric realism
violation`**.

## B.8 What to do for *existing* checkpoints (re-evals only)

For Track A maze re-evals, the same contract applies, with two
adaptations:

- The contract is checked against the *single* `puzzle_acc` /
  `cell_acc` from the re-eval, not a curve.
- If `puzzle_acc` is still ≥ 0.05 *after* `mask_non_path: false` was
  applied, that is the §B.3 dataset-contamination red flag. Verify
  the dataset version (run
  `scripts/check_maze_split_contamination.py`), log to `findings.md`
  §5, and route the issue. **Do not retrain** Qwen Maze, Llama Maze,
  or any model the per-machine prompt marked "re-eval only".

## B.9 Calibration anchor — what a known-good post-fix re-eval looks like

The first two re-evals to land under this contract were on M1:

```
Qwen Maze:           0/1000 puzzle, cell_acc = 0.125154   -> green flag
Distill-Qwen Maze:   0/1000 puzzle, cell_acc = 0.125021   -> green flag
```

Both within 0.02 percentage points of each other — the student inherits
the teacher's degenerate "spam path marker" strategy, and both score
~12.5 % cell accuracy, just above the chance-on-path floor.

Use this as the calibration anchor. If a Maze re-eval comes back with
puzzle_acc meaningfully above 0 or cell_acc meaningfully above 0.20,
**investigate before reporting**.

## B.10 Coordination tags

Any red-flag entry in `findings.md` §5 must include the explicit string
**`metric realism violation`** so M4 (the report aggregator) can
grep-find them when compiling the supplementary ZIP. Likewise any
viability-gate-passed entry must include **`viability gate passed`**.
Any 30-min redundancy snapshot can be tagged
**`redundancy snapshot machineN`** in case it needs to be referenced
in a recovery scenario.

## END LAYER

That is everything. Apply both contracts to every run for the rest of
the sprint. The Qwen / Distill-Qwen Maze post-fix results in §B.9 are
the calibration anchor — when in doubt, ask: *does this run look more
like the calibration anchor, or more like the pre-fix 1.000 saturation*?
If the second, stop and log. And remember: every 30 minutes during
training, the redundancy watchdog should already be running so a crash
costs at most ~30 minutes.
