# TRM Project Plan

**Module:** UFCFAS-15-2 Machine Learning | **Deadline:** 1 May 2026, 17:00 BST
**Team:** Ahmed AlShamy, Armin Ghaseminejad, Nickolas Greiner
**Today:** 11 April 2026 | **Days remaining:** 20

---

## Windows bootstrap handoff (2026-04-19 — paused; resume when ready)

Separate stream from the coursework work below. A session designed a one-liner Windows setup (`irm | iex`) so new machines can reach "ready to train ML-TRM" without SSH-key setup. The design and plan are committed; the L1 script is written, reviewed, and committed locally; no pushes have happened.

**Where to find everything:**
- Design spec: `docs/superpowers/specs/2026-04-19-windows-bootstrap-setup-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-19-windows-bootstrap-setup.md`
- Feature branch (for ML-TRM changes): `feat/windows-bootstrap` (currently 0 commits on this branch beyond main — Dispatches B & C not yet run)
- New sibling repo (local only): `C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ml-machine-setup/` — 4 commits, no remote yet.

**State of each task (plan task numbers):**
- Tasks 1–3 (Dispatch A) — **DONE locally.** `ml-machine-setup/` has `.gitignore`, `setup.ps1` (160 lines, syntax-checked, 5 code-review fixes applied), `README.md`. Commits: `5cca4cf` scaffold → `9135ee4` setup.ps1 → `6e31ddd` README → `29a7754` robustness fixes.
- Task 4 (user gate) — **paused here.** Next action below.
- Tasks 5–7 (Dispatch B & C) — not started. Will run once Task 4 is complete so `setup.ps1` can reference the live raw URL confidently.
- Task 8 (make ML-TRM public + push) — not started.
- Task 9 (Windows Sandbox verification) — not started.

**To resume — exact next action:**

1. Create the public GitHub repo (web UI, since `gh` CLI is not yet installed):
   - Visit <https://github.com/new>
   - Owner: `shamykyzer`, Name: `ml-machine-setup`, Visibility: **Public**
   - Do NOT add README / .gitignore / license (already have them locally).
2. From a terminal:
   ```bash
   cd "C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ml-machine-setup"
   git remote add origin https://github.com/shamykyzer/ml-machine-setup.git
   git branch -M main
   git push -u origin main
   ```
3. Verify: `curl -sI https://raw.githubusercontent.com/shamykyzer/ml-machine-setup/main/setup.ps1 | head -1` should return `HTTP/2 200`.
4. Open a new Claude Code session, point it at this file + the plan, and say "continue bootstrap at Dispatch B". It will write ML-TRM's `bootstrap.ps1`, untrack the `papers/*.pdf`, update the README, and stop at the next user gate.

**Absolute rules in force** (from user memory):
- No push to any remote without user literally saying "push".
- No destructive actions (no `git reset --hard`, no `branch -D`, no `worktree remove`).
- Commits on `feat/windows-bootstrap` branch only for ML-TRM changes — `main` stays untouched until the user merges.

---

## Session Handoff (read this first)

A Claude Code session just finished auditing the project against a 7-section coursework checklist and started executing the fixes. Training of `trm_official_sudoku` is **live** (epoch ≥ 1 of 500) — do **not** touch any file used by the running process.

**Mandate for the next session:** "all that you can do now before we run the model for days" + "make sure all is complete for the group coursework, polish all the checkpoints audits to-do list". Execute the checklist in §A below, in order, following TDD where applicable.

### What's done before the handoff

- Full audit of data pipeline, model architecture, training loop, energy logging, experiment management, eval.
- Nine-item task list created (see §A) ranked by coursework-deadline impact.
- **Task 1 is mid-TDD**:
  - ✅ `tests/__init__.py` created (empty).
  - ✅ `tests/test_encoding.py` created — 20 test functions, pytest-compatible + stdlib `_run_all()` runner at the bottom so `python tests/test_encoding.py` works without adding pytest to requirements.
  - ❌ `src/data/encoding.py` does **not** exist yet — tests currently fail at `ImportError` (this is the intended RED phase).

### Exact next action

From the repo root:

```bash
# 1. Verify the RED phase (should show ImportError on src.data.encoding)
.venv/Scripts/python.exe tests/test_encoding.py

# 2. Implement src/data/encoding.py minimally (see §A Task 1 spec below)

# 3. Verify GREEN
.venv/Scripts/python.exe tests/test_encoding.py

# 4. Proceed through Tasks 2 → 9 in order.
```

The venv python is `.venv/Scripts/python.exe` — the system `python` on PATH is Python 3.13 from `C:\Program Files\Python313` and does **not** have numpy/torch/etc. Always run the repo code via the venv.

### Hard constraints — DO NOT TOUCH during execution

Files/dirs with a **running** training process attached. Editing them risks corrupting the run, the checkpoint, or the wandb/CSV logs.

| Path | Reason |
|------|--------|
| `src/training/trainer_official.py` | Active training loop — already correctly sets up wandb `define_metric`. |
| `src/models/trm_official.py` | Active model — imported by the running process. |
| `src/data/sudoku_dataset.py` | **SAFE for docstring-only edits** (Task 8). Python caches imports, so source edits cannot affect the running interpreter. Still: do not alter class signatures, function bodies, or remove code. |
| `src/data/maze_dataset.py` | Same as above — safe for docstring-only edits. Not currently training, but keep symmetric with sudoku. |
| `configs/trm_official_sudoku.yaml` | Active config — do not edit. `configs/trm_official_sudoku_mlp.yaml` and `trm_official_maze.yaml` may be edited if needed. |
| `C:/TRM checkpoints/sudoku-att/*.pt` | Live checkpoints. Never rm, never overwrite. |
| `experiments/sudoku-att/*` | Live training logs + emissions CSVs being appended. Read-only. |

Safe to create, edit, or delete: anything under `tests/`, `scripts/`, `src/data/encoding.py` (new), `src/evaluation/inspection.py` (new), `src/training/wandb_utils.py` (append-only), the three non-official trainers (`trainer_trm.py`, `trainer_llm.py`, `trainer_distill.py`), `results/`, `plan.md`, `README.md`.

---

## §A — Pre-training execution checklist

The TaskList in the live Claude Code session already holds these nine tasks (IDs 1–9). Re-create them in the new session with `TaskCreate` (or rely on the TaskList carrying over). Work them in ID order — earlier tasks unblock later ones.

### Task 1 — Build encode/decode + validity helpers with TDD  **(in progress)**

**Why:** The coursework rubric asks for "encoders/decoders as functions" and "correctness check functions for outputs". Currently the encoding is baked inline into `data/build_sudoku_dataset.py` (line 72: `np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0')`) with no reusable helper, and `src/evaluation/metrics.py` only computes cell/puzzle argmax accuracy — it never verifies sudoku constraints or maze path connectivity.

**Files already written:**
- `tests/__init__.py` (empty)
- `tests/test_encoding.py` (20 test functions, see the file for exact assertions)

**File to create:** `src/data/encoding.py` — minimal implementation to make all 20 tests pass. Required public API:

```python
def encode_sudoku(board: str) -> np.ndarray:
    """81-char string -> (81,) int64. '.'/'0' -> 1 (blank), '1'..'9' -> 2..10.
    Raises ValueError on length != 81 or any character outside {'.', '0'-'9'}.
    """

def decode_sudoku(tokens: np.ndarray) -> str:
    """Inverse of encode_sudoku. Blanks (stored 1) become '.' (canonical form)."""

def encode_maze(maze: str) -> np.ndarray:
    """Multi-line string -> flattened (H*W,) int64.
    Mapping (from data/build_maze_dataset.py CHARSET = '# SGo'):
      '#' -> 1, ' ' -> 2, 'S' -> 3, 'G' -> 4, 'o' -> 5. Raises ValueError
    on unknown characters.
    """

def decode_maze(tokens: np.ndarray, n_rows: int) -> str:
    """Inverse of encode_maze. Uses '\\n' row separators, no trailing newline."""

def is_valid_sudoku_solution(tokens: np.ndarray) -> bool:
    """True iff tokens is a complete, valid 9x9 Sudoku.
    - Input is the stored-token form (1-10); must contain no blanks (no token==1).
    - Checks: each row, each column, each 3x3 box contains each of 1..9 exactly once.
    """

def is_valid_maze_path(tokens: np.ndarray, grid_shape: tuple[int, int]) -> bool:
    """True iff the 'o' chain (with 'G' as terminus) forms a 4-connected path
    from the unique 'S' cell to the unique 'G' cell.
    - Rejects boards with != 1 'S', != 1 'G', or no connected path.
    - Does NOT require the path to be shortest; a valid overlong path is OK.
    BFS: start at S, enqueue 4-neighbours whose token is in {5 ('o'), 4 ('G')},
    return True when the G cell is popped.
    """
```

**Minimal is_valid_sudoku_solution approach:**

```python
raw = tokens.reshape(9, 9) - 1  # stored 1..10 -> raw 0..9 (0 = blank)
if (raw == 0).any():
    return False
for axis in (0, 1):  # rows then cols
    for i in range(9):
        line = raw[i] if axis == 0 else raw[:, i]
        if len(set(line.tolist())) != 9:
            return False
for br in range(3):
    for bc in range(3):
        box = raw[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten().tolist()
        if len(set(box)) != 9:
            return False
return True
```

**Minimal is_valid_maze_path approach:**

```python
from collections import deque
grid = tokens.reshape(*grid_shape)
# Find S and G
s_coords = np.argwhere(grid == 3)
g_coords = np.argwhere(grid == 4)
if len(s_coords) != 1 or len(g_coords) != 1:
    return False
start = tuple(s_coords[0])
goal = tuple(g_coords[0])
visited = {start}
q = deque([start])
while q:
    r, c = q.popleft()
    if (r, c) == goal:
        return True
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]:
            if grid[nr, nc] in (5, 4) and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
return False
```

**Verification:** `.venv/Scripts/python.exe tests/test_encoding.py` → 20/20 PASS.

### Task 2 — Build failure inspection renderer

**Why:** The audit found no way to visually inspect model errors. The report needs "example puzzle visualizations — input / prediction / ground truth" (see Phase 2 figure 5 below).

**File to create:** `src/evaluation/inspection.py`.

```python
from src.data.encoding import decode_sudoku, decode_maze

def render_sudoku_board(tokens: np.ndarray, title: str = "") -> str:
    """9x9 pretty grid with 3x3 box dividers and a title line. Reuses decode_sudoku."""

def render_maze(tokens: np.ndarray, grid_shape: tuple[int, int], title: str = "") -> str:
    """Grid with newlines. Reuses decode_maze."""

def inspect_failures(
    model, dataset, task_type: str,
    n_samples: int = 10,
    out_path: str = "results/failure_inspection.txt",
    device: str = "cpu",
) -> None:
    """Run the model on `dataset`, find the first `n_samples` failing puzzles,
    write side-by-side input / prediction / truth to `out_path`. task_type in
    ('sudoku', 'maze') picks which renderer + grid_shape to use.
    """
```

**Unit tests:** add `tests/test_inspection.py` that mocks a model returning the ground truth + one corruption, asserts the output file contains both a PASS and FAIL render.

### Task 3 — Experiment metrics aggregator

**Why:** The report needs one CSV row per model with {best_val_acc, train_time_min, energy_kwh, co2_kg, param_count}. Currently this has to be assembled by hand from four different file types scattered across `experiments/*/` and `results/`.

**File to create:** `scripts/aggregate_metrics.py`.

- Walk `experiments/*/` for `*_train_log.csv` and `emissions.csv` (note: some hostname-tagged variants like `experiments/sudoku-att/trm_official_sudoku_train_log-STU-CZC5277FGD.csv` — glob `*train_log*.csv`).
- Walk `results/*_eval.json` for `cell_accuracy`, `puzzle_accuracy`, `avg_act_steps`, `inference_emissions`.
- Output: `results/summary.csv` with columns:
  `model, task, params, best_val_puzzle_acc, final_train_loss, train_time_min, train_energy_kwh, train_co2_kg, eval_puzzle_acc, eval_cell_acc, avg_act_steps, inference_energy_kwh`
- The `params` column comes from loading the checkpoint's config and calling `model.param_count()` — or, cheaper, hard-code a dict from the paper numbers in plan.md.

**Smoke-test data available right now:**
- `experiments/sudoku-att/trm_official_sudoku_train_log.csv` — real epochs 5, 10 logged (epoch=5: lm_loss=29.5474 acc=0.3404; epoch=10: lm_loss=20.2280 acc=0.6945).
- `experiments/sudoku-att/emissions.csv`
- `experiments/sudoku-mlp/emissions.csv`
- `results/trm_official_sudoku_eval.json` — cell_acc 0.9155, puzzle_acc 0.8474, avg_act_steps 16.0, inference CO2 0.4787 kg.

### Task 4 — Results plotting script

**Why:** The report needs the 5 figures listed in Phase 2 §"Figures to Generate". Currently no plotting code exists anywhere in `src/` or `scripts/` (confirmed by grep for `matplotlib|plt\.|savefig`). `matplotlib>=3.8.0` and `seaborn>=0.13.0` are already in `requirements.txt` (lines 15–16), no dependency install needed.

**File to create:** `scripts/plot_results.py` writing PNGs into `results/figures/`.

1. `accuracy_vs_epoch.png` — line plot, one line per model, x=epoch, y=val_puzzle_acc. Reads per-run `*_train_log.csv`.
2. `model_accuracy_bars.png` — bar chart across all 7+ models on the Sudoku task (TRM-MLP, TRM-Att, TRM-Official-Sudoku, GPT-2, SmolLM2, Qwen2.5, Llama-3.2, Distilled). Reads `results/summary.csv`.
3. `carbon_footprint_bars.png` — bars of train_co2_kg per model.
4. `params_vs_accuracy.png` — scatter, log-scale x (params), y=puzzle_acc.
5. `act_convergence.png` — average ACT steps over epochs (TRM runs only — the `avg_steps` column in `trm_official_sudoku_train_log.csv`).

Use `seaborn.set_theme(context="paper")` for consistent typography. Save at `dpi=150`. Every figure must have axis labels + title + legend.

### Task 5 — Smoke-test the aggregator + plotting pipeline

Run `scripts/aggregate_metrics.py` and `scripts/plot_results.py` on the existing data from Task 3. Confirm `results/summary.csv` has at least one non-header row and `results/figures/*.png` contain actual data (not empty axes). This proves the pipeline works before the big runs finish.

### Task 6 — Add `define_common_metrics` helper to wandb_utils

**Why:** The user's mid-audit feedback was: *"the metrics has to be within their own panels as well to be viewed in the wandb dashboard"*. Only `trainer_official.py` (lines 273–358) currently uses `wandb.define_metric` — the other three trainers dump metrics at the default global root, which in the wandb UI shows up as an unsorted pile. The fix is a shared helper so every trainer registers the same panel structure.

**File to edit:** `src/training/wandb_utils.py` (append a new function, do not modify `init_wandb`).

```python
def define_common_metrics(
    use_wandb: bool,
    namespaces: tuple[str, ...] = ("train", "val", "carbon", "system"),
    summaries: dict[str, str] | None = None,
) -> None:
    """Register the wandb panel structure shared by all TRM trainers.

    - `epoch` is the hidden step metric.
    - Every metric under `<namespace>/*` is x-axised against `epoch`.
    - `summaries` lets callers override the default summary aggregation per
      metric glob. Defaults (opinionated but sensible):
        val/*_acc   -> max
        */loss      -> min      (matches ce_loss, q_loss, lm_loss, loss)
        carbon/*    -> last
        system/*    -> max
        train/lr    -> last
        */_sec      -> mean
    """
    if not use_wandb:
        return
    import wandb
    wandb.define_metric("epoch", hidden=True)
    for ns in namespaces:
        wandb.define_metric(f"{ns}/*", step_metric="epoch")
    default_summaries = {
        "val/*_acc": "max",
        "*/loss": "min",
        "carbon/*": "last",
        "system/*": "max",
        "train/lr": "last",
        "*/_sec": "mean",
    }
    if summaries:
        default_summaries.update(summaries)
    for pattern, agg in default_summaries.items():
        wandb.define_metric(pattern, summary=agg)
```

The existing `init_wandb(config)` call site stays unchanged.

### Task 7 — Wire the helper into the three non-official trainers

**File edits (existing files, minimal diff):**

#### `src/training/trainer_trm.py`
After line 104 (`self.use_wandb = init_wandb(config)`) add:
```python
from src.training.wandb_utils import define_common_metrics
define_common_metrics(self.use_wandb)
```
The existing `wandb.log({**train, **val}, step=epoch+1)` at lines 170–177 already uses the correct `train/` and `val/` prefixes — no log-site change needed.

#### `src/training/trainer_llm.py`
1. Same `define_common_metrics(self.use_wandb)` call after `self.use_wandb = init_wandb(config)` (line 61).
2. **Fix orphaned `elapsed_min`** at line 100. Current:
   ```python
   wandb.log({
       "train/loss": metrics["loss"],
       "val/puzzle_acc": val_metrics["puzzle_acc"],
       "elapsed_min": elapsed,      # <-- no prefix, escapes panels
   }, step=epoch + 1)
   ```
   Change to `"train/elapsed_min": elapsed`.

#### `src/training/trainer_distill.py`
1. Same `define_common_metrics` call after `self.use_wandb = init_wandb(config)` (line 102).
2. Same `elapsed_min` → `train/elapsed_min` fix at line 141.

**Do NOT touch `trainer_official.py`** — it already does this correctly and it is the active training loop.

### Task 8 — Schema docstrings at the top of dataset files

**Why:** The audit found that the token schema (sudoku 1–10 with blank=1; maze 1–5 from CHARSET "# SGo") is only documented inside method bodies and inside `data/build_*_dataset.py`. A report reader opening `src/data/sudoku_dataset.py` cold should see the schema at the top of the file.

**Edits:** prepend a module-level docstring to each.

#### `src/data/sudoku_dataset.py`
```python
"""PyTorch Dataset for preprocessed Sudoku-Extreme data.

Token schema (authoritative):
    stored token | meaning       | raw CSV char
    -------------|---------------|-------------
        0        | pad (unused)  | —
        1        | blank         | '.' or '0'
        2..10    | digits 1..9   | '1'..'9'

The +1 shift is applied by data/build_sudoku_dataset.py:_seq_to_numpy
(line 118) so that vocab_size = 11 leaves 0 free as the ignore_label_id.

Labels are masked: pre-filled positions (where inputs[i] == labels[i]) are
replaced with 0 so the cross-entropy loss ignores them — the model is only
graded on the blank cells it has to predict.
"""
```

#### `src/data/maze_dataset.py`
```python
"""PyTorch Dataset for preprocessed Maze-Hard data.

Token schema (from data/build_maze_dataset.py CHARSET = '# SGo'):
    stored token | char | meaning
    -------------|------|------------
        0        |  —   | pad
        1        |  #   | wall
        2        | ' '  | open cell
        3        |  S   | start
        4        |  G   | goal
        5        |  o   | path marker (the solution the model must output)

Labels are masked: where inputs[i] == labels[i] (walls, empty cells, S, G),
the label is set to 0 so the loss only applies to the 'o' path cells the
model has to predict.
"""
```

**Safety note:** Doc-only edits to these files do NOT affect the running training process. Python caches the imported module bytecode; editing the source file after import does not trigger a reload. Verified: `sys.modules['src.data.sudoku_dataset']` is already bound in the training process and will not re-read the file. Still, make the edit via `Edit` (single replacement) rather than `Write` (full rewrite) to minimize risk.

### Task 9 — Dataset + TRM forward-pass sanity check

**Why:** Before launching multi-day runs, verify end-to-end that both datasets load, the tokens histogram matches the expected schema (0–10 for sudoku, 0–5 for maze), and a tiny TRM model can do one forward pass without NaNs.

**File to create:** `scripts/sanity_check.py`.

- Import `get_sudoku_loaders` from `src.data.sudoku_dataset`, load a 4-example batch.
- Same for maze.
- Print token histograms: `np.unique(batch, return_counts=True)`.
- Use `src.evaluation.inspection.render_sudoku_board` to pretty-print one example.
- Build `TRMSudoku(vocab_size=11, seq_len=81, d_model=64, ff_hidden=128)` — **tiny**, CPU only, just to exercise the forward path. **Do NOT instantiate TRMOfficial** (it requires CUDA + bf16).
- Run `model.embedding(inputs); model.block(...); model.output_head(...)`; assert no NaN/Inf.
- Exit 0 on success, non-zero on any failure.

This script is safe to run at any time — it does not touch `experiments/` or `C:/TRM checkpoints/`.

---

## §B — Deferred tasks (do NOT attempt during live training)

### Task 10 — val/test split fix
The current code calls both splits `test_loader` — `val_loader` is just an alias. Carving a real validation slice mid-training would invalidate the active checkpoint's validation history. **Action:** document this limitation in the Methods section of the report, and cite it as a known constraint. Actually implement the split only after all training runs finish.

### Task 11 — Per-run experiment directories
`experiments/sudoku-att/` is shared across all runs of the same config — restarts append to the same CSV. Moving to per-run dirs (`experiments/sudoku-att/run_<timestamp>/`) is a breaking change to the checkpoint-resume path and would require touching `trainer_official.py`. **Action:** defer until after all runs finish.

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Codebase | Done | Models, trainers, eval, data pipelines, AMP, resume support |
| Official TRM port | Done | Q-learning ACT, StableMax CE, AdamATan2, task-type embeddings, native bf16 forward, EMA fp32 (gotcha fixed), SwiGLU Llama-rounding patch for mlp_t variant |
| Data preprocessing | Done | Sudoku: 1K train / 423K test. Maze: 1K train / 1K test |
| Configs (paper-faithful) | Done | `ff_hidden: 1536` (Llama-rounded), `weight_decay: 1.0`, `task_emb_lr: 0.0001`, `no_ACT_continue: true`, flat LR after warmup. Three configs: `trm_official_sudoku.yaml` (att, 77.70% target), `trm_official_sudoku_mlp.yaml` (mlp, 84.80% headline), `trm_official_maze.yaml` (78.70% target) |
| LLM baselines | Done | 4 models: GPT-2, Qwen2.5-0.5B, SmolLM2-360M, Llama-3.2-1B |
| Training logs | Done | CSV logs + W&B + Weave + auto-push script + GitHub Action notifications |
| Resume support | Done | `--resume` flag for crash recovery; mutually exclusive with `--init-weights` |
| Mixed precision | Done | Native bf16 forward (no GradScaler needed); EMA shadow stored in fp32 |
| HF checkpoint reproduction | Done | Three Sanjin2024 reference checkpoints set up under `hf_checkpoints/` (ARC/, Sudoku-Extreme-mlp/, Maze-Hard/) with per-source remap scripts and verifiers. Each verifier confirms 0 unexpected keys, only `rotary_emb` buffers missing (deterministic), and weight-change checks. |
| Eval from init weights | Done | `--mode eval --checkpoint <remapped.pt>` works directly on remapped HF checkpoints (`evaluate.py:load_and_evaluate` uses `strict=False` + load report). Verified: `results/trm_official_sudoku_eval.json` = cell 0.9155, puzzle 0.8474, 16 ACT steps |
| Encode/decode helpers | **In progress** | `tests/test_encoding.py` written (RED). `src/data/encoding.py` not yet created — see §A Task 1 |
| Failure inspection renderer | Not started | See §A Task 2 |
| Metrics aggregator | Not started | See §A Task 3 |
| Results plotting | Not started | See §A Task 4 |
| wandb panel helper (non-official trainers) | Not started | See §A Task 6/7 |
| Dataset schema docstrings | Not started | See §A Task 8 — doc-only, safe during training |
| Sanity check script | Not started | See §A Task 9 |
| Training | In progress | Sudoku attention variant epoch 1 of 500. Sudoku-MLP and maze runs not started yet. |
| Report | Not started | 6-page conference paper due May 1 |

---

## Checkpoint reproduction methodology

The paper's published Sudoku-Extreme MLP (84.80%) and Maze-Hard (78.70%) results
were trained on 8×H200 with global batch 4608 for 100K epochs — not reproducible
on a single RTX 3070 in the available time budget. Instead we **start from the
published Sanjin2024 checkpoints as `--init-weights`** and either eval directly
or fine-tune from them:

```
hf_checkpoints/
├── ARC/                       # arcprize/trm_arc_prize_verification (617M params, ARC-AGI-2)
│   ├── step_723914
│   └── remapped_for_local.pt
├── Sudoku-Extreme-mlp/        # Sanjin2024 — 84.80% headline result
│   ├── step_16275
│   └── remapped_for_local.pt
└── Maze-Hard/                 # Sanjin2024 — 78.70% headline result
    ├── step_9765
    └── remapped_for_local.pt
```

Per-source remap scripts (`scripts/remap_*.py`) translate the reference
state_dict layout (`_orig_mod.model.inner.*` prefix, fused QKV projections,
fused gate_up SwiGLU, single-row puzzle_emb) into our local TRMOfficial layout
(stripped prefix, split q/k/v, split w1/w3, two-row task_emb with the unused
row zero-padded, dual-registered CastedLinear weights). Each remap asserts
source tensor shapes and fails loudly on mismatch. Per-source verifiers
(`scripts/verify_*_loads.py`) construct a fresh TRMOfficial, load the remap
with `strict=False`, and assert: zero unexpected keys, only `rotary_emb`
buffers in the missing list, weight values actually changed from fresh init,
and the correct task_emb row was populated.

**The SwiGLU patch.** To make the MLP variant load, the local SwiGLU's
`ff_hidden = int(hidden_size * expansion)` formula had to be extended with an
optional `ff_hidden` kwarg, used by the `mlp_t` branch in `TRMBlock.__init__`
to pass an explicit Llama-rounded value (`llama_rounded_ff(97, expansion=4) = 512`).
The simple formula cannot produce both 1536 (regular FFN at hidden=512) and 512
(token-mixer at hidden=97) under any single expansion value. The patch is
backwards-compatible: the regular FFN keeps the simple formula, only `mlp_t`
uses the new path. Without the patch, loading the Sanjin2024 sudoku-mlp
checkpoint fails with a shape mismatch on `mlp_t.w1.weight`.

**Reproduction targets per variant:**

| Config | Source | Target | Method |
|--------|--------|--------|--------|
| `trm_official_sudoku_mlp.yaml` | `Sudoku-Extreme-mlp/remapped_for_local.pt` | 84.80% (paper headline) | eval-only or short fine-tune |
| `trm_official_maze.yaml` | `Maze-Hard/remapped_for_local.pt` | 78.70% (paper headline) | eval-only or short fine-tune |
| `trm_official_sudoku.yaml` | (in-progress from-scratch run) | 77.70% (att variant ablation) | full training |

This is more honest and more paper-faithful than attempting a from-scratch
reproduction at 1/72 the global batch size on 1/8 the GPU count.

## Methodology contributions

1. **EMA fp32 fix.** The trainer initially stored the EMA shadow in the model's
   forward dtype (bf16). bf16's 7-bit mantissa cannot resolve the per-step
   `1 - decay = 0.001` delta — the shadow froze at init values and val accuracy
   was stuck at the random baseline (~0.1233 for sudoku) for 95+ epochs while
   the trained weights themselves were learning correctly. Diagnosed via
   `scripts/diagnose_real_weights.py` (a one-shot script that evaluates raw
   model weights bypassing the EMA swap). Fixed by storing the shadow in fp32
   (`src/training/ema.py:31-46`) and explicitly casting the bf16 param to fp32
   inside the update fn.

2. **SwiGLU Llama-rounding patch** (described above) to enable loading the
   Sanjin2024 MLP-token-mixer checkpoint into our local model.

3. **Per-source remap + verify pipeline** with assertion-based shape checks
   and weight-change diagnostics. Catches silent breakage on any future model
   code change.

4. **TDD-backed encoding + validity helpers** (planned §A Task 1). `tests/test_encoding.py` pins the token schemas and both task validity checks, so any future schema drift is caught by a 20-test failure instead of silent mis-training.

---

## Phase 1: Lab Training (Days 1-5)

### 3-Machine Setup

| Machine | Terminal 1 | Terminal 2 | Est. Time |
|---------|-----------|-----------|-----------|
| **A** | `python run.py setup-cuda && python run.py train-sudoku` | `bash scripts/auto_push.sh` | ~12-14 hrs |
| **B** | `python run.py setup-cuda && python run.py train-maze` | `bash scripts/auto_push.sh` | ~3-4 days |
| **C** | `python run.py setup-cuda && python run.py train-llm-all && python run.py train-distill` then `python run.py data-maze-aug && python run.py train-maze-fast` | `bash scripts/auto_push.sh` | ~1-2 days |

### What Each Machine Trains

**Machine A -- TRM-Sudoku (core result):**
- TRM-MLP, 6.4M params, 5000 epochs on 1K puzzles
- Expected: ~87% puzzle accuracy
- Checkpoints: `models/best.pt`, `models/latest.pt`
- Also run: `python main.py --config configs/trm_official_sudoku.yaml` (official arch variant)

**Machine B -- TRM-Maze (core result):**
- TRM-Att, 8.4M params, 5000 epochs on 1K mazes
- Expected: ~85% puzzle accuracy
- Longest training -- start first
- Also run: `python main.py --config configs/trm_official_maze.yaml` (official arch variant)

**Machine C -- All LLM baselines + fast maze:**
1. GPT-2 (124M) + LoRA → ~30-45 min
2. Qwen2.5-0.5B (494M) + LoRA → ~45-60 min
3. SmolLM2-360M (360M) + LoRA → ~30-45 min
4. Llama-3.2-1B (1.2B) + LoRA → ~60-90 min
5. Distillation from GPT-2 → ~30 min
6. TRM-Maze fast (augmented, 2000 epochs) → ~1-2 days

All LLMs expected to score ~0%. That's the point.

### If Training Crashes

```bash
python run.py resume-sudoku    # Resumes from models/latest.pt
python run.py resume-maze      # Resumes from models/latest.pt
```

### Remote Monitoring

- Training CSV logs auto-pushed to GitHub every hour
- GitHub Action posts stats to an issue -- subscribe for phone notifications
- Check: `experiments/trm_sudoku_train_log.csv`, `experiments/trm_maze_train_log.csv`

---

## Phase 2: Evaluation (Days 5-7)

```bash
# TRM models
python run.py eval-sudoku
python run.py eval-maze

# LLM baselines
python run.py eval-llm
python run.py eval-llm-qwen
python run.py eval-llm-smollm
python run.py eval-llm-llama
```

### Metrics to Collect

| Metric | Source | Report Section |
|--------|--------|----------------|
| Puzzle accuracy (%) | eval output | Experiments |
| Cell accuracy (%) | eval output | Experiments |
| Avg ACT steps | eval output (TRM only) | Experiments |
| Parameter count | model.param_count() | Methods |
| Training energy (kWh) | experiments/*_emissions.csv | Experiments |
| Training CO2 (kg) | experiments/*_emissions.csv | Experiments |
| Inference energy | eval CodeCarbon output | Experiments |
| Training time | experiments/*_train_log.csv (elapsed_min) | Experiments |

### Figures to Generate

1. **Accuracy comparison table** -- 7 models on Sudoku, 2 on Maze
2. **Parameter efficiency plot** -- Accuracy vs. param count (log scale)
3. **Carbon footprint bar chart** -- Training energy per model
4. **Training curves** -- CE loss and val accuracy over epochs
5. **Example puzzle visualizations** -- Input / prediction / ground truth
6. **ACT convergence** -- Average halting steps over training (TRM only)

All six are produced by `scripts/plot_results.py` once §A Task 4 is done.

---

## Phase 3: Report Writing (Days 7-24)

**Format:** 6 pages max, conference-style, using provided Word template.

| Section | Weight | Key Content |
|---------|--------|-------------|
| **Abstract** | -- | Problem, approach (TRM vs 4 LLMs), key result, energy savings. Max 200 words. |
| **Introduction** | 10% | LLMs fail at structured reasoning. TRM thesis: recursive weight-sharing enables tiny models to outperform massive ones. |
| **Related Work** | 10% | Jolicoeur-Martineau (2025) TRM; Dillon (2026) WTA; McGovern (2025) adaptation; LLM puzzle-solving failures. |
| **Data** | 10% | Sudoku-Extreme (1K/423K), Maze-Hard (1K/1K), encoding, augmentation. |
| **Methods** | 30% | TRM architecture + 4 LLM baselines + distillation. Architecture diagrams. Ethical discussion: energy cost. |
| **Experiments** | 30% | All tables/figures. 7-model accuracy comparison. Carbon footprint. Limitations. |
| **Conclusion** | 5% | TRM outperforms LLMs at fraction of energy. Future: WTA, test-time adaptation. |
| **Writing** | 5% | Follow template, clear figures, consistent references. |

### Models Summary Table (for report)

| Model | Type | Total Params | Trainable | Year | Expected Sudoku |
|-------|------|-------------|-----------|------|-----------------|
| TRM-MLP | Recursive | 6.4M | 6.4M | 2025 | ~87% |
| TRM-Att | Recursive | 8.4M | 8.4M | 2025 | -- (Maze ~85%) |
| TRM-Official-Sudoku | Recursive (official) | ~8.4M | ~8.4M | 2025 | TBD |
| TRM-Official-Maze | Recursive (official) | ~8.4M | ~8.4M | 2025 | TBD |
| GPT-2 + LoRA | Fine-tuned LLM | 124M | ~0.8M | 2019 | ~0% |
| SmolLM2-360M + LoRA | Fine-tuned LLM | 360M | ~1.5M | 2024 | ~0% |
| Qwen2.5-0.5B + LoRA | Fine-tuned LLM | 494M | ~2M | 2024 | ~0% |
| Llama-3.2-1B + LoRA | Fine-tuned LLM | 1.2B | ~3M | 2024 | ~0% |
| Distilled (from GPT-2) | Small transformer | 2.4M | 2.4M | -- | ~0% |

---

## Phase 4: Final Assembly (Days 24-25)

1. Proofread report (all members)
2. Verify all figures render correctly in PDF
3. Package code as supplementary ZIP (Jupyter notebook or Python files)
4. Submit on Blackboard before 17:00, 1 May 2026

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Training crashes overnight | Resume support: `make resume-sudoku` / `make resume-maze` |
| Maze takes too long | Machine C runs fast variant with augmented data as backup |
| GPU OOM on Llama 1B | Config uses batch_size=8 + grad_accum=2; can enable QLoRA if needed |
| Results differ from paper | Report honestly; discuss why (different ff_hidden was fixed, training scale) |
| Report too long | Start with skeleton, 6-page limit enforced by template |
| Encoding schema drift | `tests/test_encoding.py` pins all four token mappings — any future change has to update the tests first (TDD). |
| Plots missing before deadline | `scripts/plot_results.py` (Task 4) runs at any point with partial data; re-run after each model finishes. |
| wandb panels unreadable | `define_common_metrics` helper (Task 6) registers the same structure for all trainers. |
