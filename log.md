# Change Log — Multi-Device Checkpoint Strategy

**Date:** 2026-04-10
**Scope:** Prepare repo for 1C-tasks workflow — sudoku on machine A, maze on machine B, independent parallel runs — with thesis-grade milestone snapshots mirrored to HF Hub and wandb for cross-machine durability.
**Predecessor commit:** `952569b` (rolling + milestone + wandb-artifact checkpointing)

---

## 1. Context

The previous commit landed the checkpoint scaffolding (rolling backups, milestone snapshots, wandb Artifact for best.pt). The follow-up need surfaced once we decided to run the two task trainings on different physical machines:

- **Sudoku on Machine A, maze on Machine B, simultaneously.** Gradient descent is sequential in time, so two disconnected machines cannot split a single run. But the two *tasks* are independent — running them in parallel gives a real ~2× wall-clock speedup for the full thesis sweep.
- **Thesis milestones must survive machine failure and be fetchable from either machine** without manual file copying.
- **Storage budget is 500 GB under `C:/TRM checkpoints`**, wandb is free-tier (100 GB), HF Hub is free-tier (1 TB private).

The change below is the minimum set of edits to meet those constraints without touching `trainer_trm.py` (deferred refactor) or `src/utils/config.py` (env-override semantics already correct).

### Nine final decisions

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Workflow scope | **1C-tasks** — sudoku on A, maze on B | Real parallelism (independent tasks), not pseudo-parallelism |
| 2 | Shared HF repo | **2B** — both tasks push to the same `$TRM_HF_REPO_ID` | One repo = one source of truth; subpaths separate tasks |
| 3 | Milestone durability | **3D** — HF Hub + wandb Artifact, under `snapshots_for_thesis/` | Two independent backups + local; survives single-machine wipe |
| 4 | Rolling policy | **4A** — local only (existing `rolling_checkpoint_dir`) | Crash recovery is per-machine; cross-machine is HF Hub's job |
| 5 | Maze `save_interval` | **5C** — 500 → 50 | Handoff granularity: 50-epoch resumes on a 3-4 day maze run, not 500-epoch resumes |
| 6 | Payload size | **6D** — `best.pt` + milestones = model+EMA only (no optimizer) | ~28 MB vs ~85 MB. Milestones are for analysis/inference, not resume |
| 7 | Wandb best artifact | **Disabled** | HF Hub is the durable `best.pt` store. Avoids burning wandb free-tier quota on frequent best updates |
| 8 | Local pruning | **Removed** | 500 GB budget means full `epoch_N.pt` history fits with room for 20+ reruns |
| 9 | Local checkpoint root | **`C:/TRM checkpoints/<task>-official`** | Dedicated space outside OneDrive sync path |

---

## 2. Files modified

Three files touched. `src/training/trainer_trm.py`, `src/utils/config.py`, `main.py`, and `.env.example` are intentionally unchanged.

### 2.1 `configs/trm_official_sudoku.yaml` (2 line changes)

```diff
-  wandb_best_artifact: true
+  wandb_best_artifact: false
...
-checkpoint_dir: models/sudoku-official
+checkpoint_dir: "C:/TRM checkpoints/sudoku-official"
```

### 2.2 `configs/trm_official_maze.yaml` (4 line changes)

```diff
-  save_interval: 500
+  save_interval: 50
+  hf_repo_id: ""                      # set via TRM_HF_REPO_ID in .env
...
-  wandb_best_artifact: true
+  wandb_best_artifact: false
...
-checkpoint_dir: models/maze-official
+checkpoint_dir: "C:/TRM checkpoints/maze-official"
```

The `hf_repo_id: ""` line is added for parity with the sudoku config. The real value is still supplied by `.env` via `_apply_env_overrides()` in `src/utils/config.py`, so both machines share the same repo without committing any secret.

### 2.3 `src/training/trainer_official.py` (3 code changes)

All changes are additive or localized substitutions. No new imports.

**(a) `_save_checkpoint` — auto-create subdirectories**

One line added so `snapshots_for_thesis/...` can land in a subfolder:

```python
def _save_checkpoint(self, filename: str, payload: dict) -> None:
    path = os.path.join(self.config.checkpoint_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)   # NEW
    if not self._safe_torch_save(payload, path, "checkpoint"):
        return
    # ... existing HF upload block unchanged ...
```

Safe for top-level filenames too: `os.path.dirname("C:/TRM checkpoints/sudoku-official/best.pt")` already exists from `__init__`, and `exist_ok=True` makes the call idempotent.

**(b) `_save_milestone_checkpoint` — rewrite with snapshots subdir + wandb mirror**

Old signature `(self, pct, payload)` → new `(self, pct, payload_slim)`. Old filename `milestone_NNpct.pt` → new `snapshots_for_thesis/{dataset}_milestone_NNpct_epoch{N}.pt`. New behavior:
- Writes under `snapshots_for_thesis/` using the dataset name (`sudoku` / `maze`) and epoch number so both machines' milestones coexist in the shared HF repo
- HF Hub upload is piggy-backed via `_save_checkpoint` (which already handles the `hf_api.upload_file` call)
- Wandb Artifact mirror: named `{dataset}-milestone-NNpct` with alias `milestone_NNpct` and metadata (puzzle_acc, epoch, global_step, pct)
- Idempotent `isfile` guard keeps resume-past-a-milestone a no-op — no duplicate HF commits, no duplicate wandb artifact versions

**(c) Training loop — slim payload cache**

The loop now maintains two lazy caches instead of one:

```python
for epoch in range(self.start_epoch, self.tc.epochs):
    payload: dict | None = None
    slim: dict | None = None  # model+EMA only — for best.pt + milestones
    ...
```

`slim` is derived from `payload` by a shallow dict comprehension `{k: v for k, v in payload.items() if k != "optimizer_state_dict"}` — shared tensor references, zero extra `state_dict()` calls, zero extra memory. Best save (log_interval branch) and milestone save now pass `slim`; `epoch_N.pt` and `latest.pt` still pass full `payload` (resume-capable).

---

## 3. New on-disk layout

```
C:/TRM checkpoints/sudoku-official/
├── best.pt                                          ← slim (~28 MB), overwritten
├── latest.pt                                        ← full (~85 MB), end of run
├── epoch_25.pt, epoch_50.pt, ..., epoch_500.pt      ← full (~85 MB each), every save_interval
└── snapshots_for_thesis/
    ├── sudoku_milestone_10pct_epoch50.pt            ← slim (~28 MB)
    ├── sudoku_milestone_25pct_epoch125.pt
    ├── sudoku_milestone_50pct_epoch250.pt
    └── sudoku_milestone_75pct_epoch375.pt

C:/TRM checkpoints/maze-official/
├── best.pt
├── latest.pt
├── epoch_50.pt, epoch_100.pt, ..., epoch_5000.pt
└── snapshots_for_thesis/
    ├── maze_milestone_10pct_epoch500.pt
    ├── maze_milestone_25pct_epoch1250.pt
    ├── maze_milestone_50pct_epoch2500.pt
    └── maze_milestone_75pct_epoch3750.pt
```

The `snapshots_for_thesis/` folder is dataset-keyed in the filename, so when both tasks sync to the same HF repo, the milestones from A and B live side-by-side without collision.

---

## 4. Storage budgets

**Local (`C:/TRM checkpoints`, 500 GB budget)**

| File | Payload | Size | Count per run | Sudoku run | Maze run |
|---|---|---|---|---|---|
| `best.pt` | slim | ~28 MB | 1 (overwrite) | 28 MB | 28 MB |
| `latest.pt` | full | ~85 MB | 1 (end of training) | 85 MB | 85 MB |
| `epoch_N.pt` | full | ~85 MB | 20 / 100 | 1.7 GB | 8.5 GB |
| milestones | slim | ~28 MB | 4 / 4 | 112 MB | 112 MB |
| **Per run** | | | | **~1.9 GB** | **~8.7 GB** |

20 full reruns of each task ≈ ~210 GB — inside the 500 GB budget. Rolling backups (if `TRM_ROLLING_CHECKPOINT_DIR` set) continue to land on a separate drive untouched by this change.

**HF Hub (1 TB private)**: ~2 GB per sudoku run, ~9 GB per maze run. Dozens of reruns fit easily.

**Wandb (100 GB free)**: ~112 MB per run (milestones only, since `wandb_best_artifact=false`). Effectively unbounded. Per-step metrics logging unchanged.

---

## 5. Verification evidence

All three verification steps from the plan passed:

**5.1 Module import (catches syntax errors)**
```
$ .venv/Scripts/python.exe -c "import src.training.trainer_official; print('OK import')"
OK import
```

**5.2 Method signatures**
```
$ .venv/Scripts/python.exe -c "
import inspect
from src.training.trainer_official import OfficialTRMTrainer
for m in ['_save_checkpoint', '_save_milestone_checkpoint', '_checkpoint_payload']:
    print(f'{m}{inspect.signature(getattr(OfficialTRMTrainer, m))}')
"
_save_checkpoint(self, filename: str, payload: dict) -> None
_save_milestone_checkpoint(self, pct: int, payload_slim: dict) -> None
_checkpoint_payload(self, epoch: int) -> dict
```
New `payload_slim` parameter name on the milestone method confirms the rewrite landed. Others unchanged.

**5.3 Config parse (new paths + flags take effect)**
```
$ .venv/Scripts/python.exe -c "
from src.utils.config import load_config
for p in ['configs/trm_official_sudoku.yaml', 'configs/trm_official_maze.yaml']:
    c = load_config(p)
    print(p, '|', c.checkpoint_dir, '|', c.training.save_interval, '|', c.training.wandb_best_artifact)
"
configs/trm_official_sudoku.yaml | C:/TRM checkpoints/sudoku-official | 25 | False
configs/trm_official_maze.yaml | C:/TRM checkpoints/maze-official | 50 | False
```
Both configs round-trip through pydantic with the new absolute Windows paths, maze `save_interval=50`, and `wandb_best_artifact=False`.

---

## 6. Workflow — running 1C-tasks

No coordination between machines needed. Set `TRM_HF_REPO_ID` in `.env` on both:

**Machine A (sudoku):**
```
python main.py --mode train --config configs/trm_official_sudoku.yaml
```

**Machine B (maze), simultaneously:**
```
python main.py --mode train --config configs/trm_official_maze.yaml
```

Different `checkpoint_dir`, different wandb runs, different HF Hub subpaths — no collisions. Wall-clock time for the full sweep drops from `sudoku + maze` to `max(sudoku, maze)`.

---

## 7. Known trade-off (flagged, not fixing in this commit)

`_save_checkpoint` builds the HF `path_in_repo` as `f"{self.config.checkpoint_dir}/{filename}"`. With the new absolute Windows path, HF Hub commits land under paths like `C:/TRM checkpoints/sudoku-official/best.pt` inside the repo. That works (HF treats the string as a key) but produces ugly paths in the repo browser. Cross-device sync still works with the ugly paths — fixing this would require a separate `hf_prefix` config field and is deferred.

---

## 8. Non-goals (deliberately not done)

- **Distributed training / DDP** — requires networked machines; infeasible for a disconnected laptop + desktop pair.
- **Stacking best checkpoints across runs** — gradient descent is path-dependent; merging trajectories either duplicates work or degrades the model. (Federated learning is the real-world version of this, but that's a privacy technique, not a speed technique.)
- **Sharing checkpoint logic between `trainer_trm.py` and `trainer_official.py`** — cross-file refactor, deferred.
- **Pretty HF Hub paths** — see section 7.
- **Cleaning up old top-level `milestone_NNpct.pt` files** from the pre-952569b layout — one-time manual delete, not worth code.
