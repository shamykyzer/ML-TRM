# Claude Code agent prompt — Fleet sprint (Maze eval fix + Sudoku Fix-B retrain) — v2

**Created:** 2026-04-27 (v2 — supersedes the maze-only v1 of this file)
**Author:** Ahmed
**Audience:** A fresh Claude Code session, on whichever machine has its
assigned checkpoints in scope. Per-machine routing in §3.
**Submission deadline:** 2026-05-01 17:00 BST (+48 h grace).
**Companion plan file (other agent):** `critical-discovery-first-before-enchanted-parasol.md`
— this file should stay consistent with that one.

**Paste everything below this line into Claude Code on the chosen
machine.** The agent will resolve the M5 pull conflict first (Step 0),
re-eval before retraining wherever possible, and stay inside the
machine-specific workload defined in §3.

---

You are an autonomous Claude Code agent continuing the UFCFAS-15-2 Group
Project sprint (TRM vs LLMs on Sudoku and Maze). Today is 2026-04-27.
The final report is due 2026-05-01 17:00 BST. You have ~4 days. Read
sections 1, 2, 3 in full before touching any tool.

## 1. Critical context

Two independent issues are being fixed in this sprint:

**Track A — Maze evaluation bug.** All LLM/distilled Maze checkpoints
report 1.000 puzzle accuracy. Two causes:

- *Cause A1 — eval-mask bug (`mask_non_path: true`).* Only path cells
  (~10–15 % of the 900-cell grid) are scored; LLMs trivially output
  reasonable values for path cells after fine-tune so the metric pegs.
  Fix in config.
- *Cause A2 — possible dataset contamination.* The dataset name
  `maze-30x30-hard-1k-aug` may have applied dihedral augmentation
  *before* the train/test split. Verify; if contaminated, retrain on
  the non-augmented `maze-30x30-hard-1k`.

**Track B — Sudoku Fix-B retrain.** Several Sudoku LLM checkpoints
were trained under a pre-Fix-B regime; only their *eval-time* shift
correction was Fix-B, the *train-time* objective was Fix-A.
GPT-2 + Distill-GPT-2 Sudoku have already been retrained on M4 with
Fix-B applied at training time. Qwen, Distill-Qwen, SmolLM, and Llama
Sudoku still need a Fix-B retrain. See §3 for per-machine assignment.

**Settled and out of scope:**

- *TRM training: nothing in this sprint touches TRM training.* M2's
  Phase 1/2b/2c established that TRM-Att Maze fine-tune collapses at
  `batch_size=16` regardless of optimizer / halt knobs. Headline
  TRM-Att Maze number is therefore the **HF baseline 79.60 %** plus
  the K-vote curve. The M5 `trm-att-maze-seed1` `best.pt` captures
  a 27.50 % epoch-50 peak (avg_steps dropped 16 → 5.4, ACT was
  working before the overfit) — preserve it as a secondary observation
  in the report, but do not retrain or extend it.
- *TRM-MLP Sudoku headline = 74.25 ± 0.63 %* (3-seed mean from rigs
  FDK/FFN), already in `results/summary.csv`.
- *GPT-2 + Distill-GPT-2 Sudoku Fix-B retrains* — done on M4. Skip.

## 2. Step 0 — Pull conflict resolution (M5 only, ~5 min)

If you are on **M5**, before pulling main you will hit a conflict on
the K-vote artifact files because both M5 and another machine wrote
files of the same name. Resolution:

1. Rename M5's K-vote artifacts with a machine suffix:

```bash
cd "$REPO_ROOT/results/novelty/k_vote_runs"
for f in *.csv *.png *.log; do
  [ -e "$f" ] && mv "$f" "${f%.*}-m5.${f##*.}"
done
```

2. Stage the renames as a separate commit so the audit trail is clear:

```bash
git add results/novelty/k_vote_runs/*-m5.*
git commit -m "chore(novelty): suffix M5 k-vote artifacts to avoid collision"
```

3. Then pull main:

```bash
git pull --rebase origin main
```

If the rebase still has conflicts in non-K-vote files, **stop and ask
the user**. Do not auto-resolve `findings.md`, `results/summary.csv`,
or any `report*.md` file — those are likely live edits from another
machine.

## 3. Per-machine workload routing — 6-machine split

All 6 machines have explicit assignments. If you cannot determine which
box you are on, **ask the user before proceeding**. Coordinate via
`findings.md` §5 — log when each task lands so other machines know
what they can pull.

Wall-clock targets (all inside ~9 h per-machine budget):
**M1** ~3–3.5 h · **M2** ~3–4 h · **M3** ~6–8 h ·
**M4** ~3–4 h ongoing · **M5** ~3.5–5 h · **M6** ~5–7 h

### M1 — SmolLM Fix-B + fleet-wide one-offs + Folder 08 maze re-evals (~3–3.5 h)

*Fleet-wide one-time tasks — do these first; everyone else depends on them:*
- **Step A.1 dataset contamination check** (~30 min, see §4.A.1). Post the result in `findings.md` §5 so M3, M5, and any A.5 retrain branch know whether to use `maze-30x30-hard-1k-aug` or `maze-30x30-hard-1k`.
- **Step A.2 write `configs/llm_maze_fixed.yaml`** (~5 min) — `mask_non_path: false`, `eval_on_full_grid: true`, `eval_metric: exact_match`. Commit, push, signal in `findings.md` so other machines pull.

*Sudoku Fix-B retrain:*
- **SmolLM-360M Sudoku Fix-B** (~1.5–2 h, RTX 5070 Blackwell, checkpoint already on M1).

*Folder 08 maze re-evals — download checkpoints from Drive once, run inference locally:*
- **Qwen Maze re-eval** (~30 min, Folder 08 `08_qwen-distill-maze-saturated/qwen-maze/`).
- **Distill-Qwen Maze re-eval** (~5 min, Folder 08 distill-maze subdir; note: original distill train log was empty (QLoRA bug); if re-eval produces sane numbers, keep — otherwise retrain in ~10 min).

### M2 — Plot factory + K-vote on post-Fix-B Sudoku (~3–4 h)

M2 has the codebase set up from earlier Phase 1/2b/2c work. Pivot to figure generation + novelty extension as Track B retrains land.

- *Wait for ≥ 3 of the 4 Track B retrains to land (M4 will signal in `findings.md`), then:*
  - **Figure 1 — Sudoku puzzle accuracy bar chart** (TRM-MLP HF, TRM-MLP from-scratch, all six LLM families post-Fix-B, both distill students). Source: `results/summary_fixed.csv`.
  - **Figure 3 — CO₂-per-correct-puzzle bar chart** (log-scale y-axis to handle ∞ for LLMs/distill that score 0).
  - **Figure 4 — Sudoku-MLP fine-tune curve** showing the HF-init epoch-10 peak + subsequent decay. Source: `results/history_sudoku-mlp_best.csv`.
- **K-vote sweep** on the post-Fix-B Sudoku checkpoints (K ∈ {1, 2, 4} on Qwen-Fix-B once it lands). Adds inference-side novelty data point. Use the existing `scripts/run_novelty_k_vote.py` loop.
- **Cross-check the M5 K-vote artifacts** renamed in Step 0 — verify they survived the rebase and are referenced consistently in any plotting code.

### M3 — Llama Maze re-eval + Llama Sudoku Fix-B (joint with M6) (~6–8 h)

- *Step 0 if applicable:* handle any local pull conflicts (likely smaller scope than M5's K-vote collision).
- **Llama Maze re-eval** (~30 min) using `configs/llm_maze_fixed.yaml` once M1 commits it. **Do NOT retrain Llama Maze (~22 h won't fit).**
- **Llama Sudoku Fix-B retrain** (~5–7 h) — checkpoint locally on M3 (and in Drive at 1.60 GB). Coordinate with M6 at the 4 h mark — see M6 below.

### M4 — Report editor + artifact aggregator (ongoing, ~3–4 h spread over 24 h)

M4 already finished GPT-2 + Distill-GPT-2 Sudoku Fix-B retrains. Pivot to central editor / aggregator.

- *As each Track A re-eval lands*, append to `findings.md` §5 (eval-mask bug entry: before/after).
- *As each Track B retrain lands*, append to `findings.md` §5 (Fix-A vs Fix-B entry: before/after).
- *Update* `docs/report_methods_experiments_draft.md`:
  - §5.1 (Sudoku table): post-Fix-B numbers for the four newly retrained models.
  - §5.3 (Maze-Hard): replace saturation paragraph with the corrected re-eval table; reference the M5 27.50 % TRM-Att seed-1 peak as a secondary observation; reaffirm 79.60 % HF baseline as headline TRM-Att Maze number.
- *Maintain* `results/summary_fixed.csv` (preserve old `summary.csv` for the audit trail).
- *Compile* the final supplementary ZIP for Blackboard. Pull in M2's figures, all `results/eval_fixed/*-emissions.csv`, the Drive top-level `README.md`, and a Jupyter notebook (build a minimal one if absent).
- *Deliver Drive top-level `README.md`* mapping the tree to report sections.
- *M4 reference checkpoints (already done):*
  - `models/llm/gpt2_sudoku_latest.pt` (Drive: `05_GPT-2_Sudoku_LoRA-FixB-ep30.pt`) → 0.00 % / 13.28 % cell
  - `models/distill_gpt2_sudoku/distill_sudoku_latest.pt` (Drive: `07_Distill-GPT-2_Sudoku-ep30.pt`) → 0.00 % / 36.43 % cell

### M5 — Qwen track + GPT-2 Maze cleanup + TRM-Att seed-1 upload (~3.5–5 h)

- **Step 0: pull conflict resolution** (see §2) — rename M5 K-vote artifacts with `-m5` suffix, commit, then pull. ~5 min.
- *Sudoku Fix-B retrains:*
  - **Qwen-0.5B** (~2–3 h, checkpoint on M5).
  - **Distill-Qwen** (~30 min – 1 h, checkpoint on M5).
- *Maze re-evals:*
  - **GPT-2 Maze** (~30 min, checkpoint on M5).
  - **Distill-GPT-2 Maze** (~5 min, checkpoint on M5).
- *Drive upload:* M5's `runs/trm-att-maze-seed1/best.pt` + train log + emissions CSV to `TRM-ML .chk` under `runs/trm-att-maze-seed1/` (Drive currently has seed 2; seed 1 is the one with the 27.50 % peak) — ~30 min.

### M6 — Llama backup OR 3-seed variance on Qwen Sudoku Fix-B (~5–7 h)

M6 acts as M3's joint partner for the Apr-26 sprint. Two routing options based on M3's load — coordinate at the 4 h mark:

- **Option A (default — backup):** if M3's Llama Sudoku Fix-B is going slowly (e.g. > 4 h elapsed without an epoch-15 checkpoint), **M6 takes over Llama Sudoku Fix-B retrain**. M3 keeps the maze re-eval and helps M4 with documentation. ~5–7 h on M6.
- **Option B (parallel — variance bars):** if M3's Llama retrain is on track, **M6 runs 3-seed variance on Qwen Sudoku Fix-B** (seeds 1 + 2; M5 produces seed 0). Result: error bars on the Qwen Sudoku Fix-B row, mirroring the TRM-MLP 3-seed protocol from `findings.md`. Strengthens the "fully justified methodology" rubric. ~6 h total for two seeds.

Default to Option B unless M3 explicitly hands off Llama. Whichever option fires, log the decision in `findings.md` §5 so M4 can reference the right artifacts in the report draft.

## 4. Track A — Maze evaluation fix details

### Step A.1 — Verify dataset split contamination (≤30 min, do once for the fleet)

```bash
grep -rn "maze-30x30" configs/ scripts/ src/ | head -20
```

If the dataset name is `maze-30x30-hard-1k-aug`, run a contamination
check that hashes the puzzle grids in train and test:

```python
# scripts/check_maze_split_contamination.py
import hashlib
from datasets import load_dataset

def h(p):
    return hashlib.sha256(repr(p).encode()).hexdigest()[:16]

ds_name = "maze-30x30-hard-1k-aug"  # adjust to actual loader path
train = load_dataset(ds_name, split="train")
test  = load_dataset(ds_name, split="test")
train_h = {h(s["puzzle"]) for s in train}
test_h  = {h(s["puzzle"]) for s in test}
overlap = train_h & test_h
print(f"Train: {len(train_h)} unique  Test: {len(test_h)} unique  "
      f"Overlap: {len(overlap)}")
```

Decision rule:
- *overlap > 0* → contaminated. Switch all subsequent LLM Maze
  retraining/eval to **`maze-30x30-hard-1k`** (non-augmented).
- *overlap == 0* → augmented split is clean; only Cause A1 needs
  fixing.

### Step A.2 — Fix the eval config (5 min)

```yaml
# configs/llm_maze_fixed.yaml
mask_non_path: false        # THE FIX — score all 900 cells
eval_on_full_grid: true
eval_metric: exact_match
```

If `mask_non_path` is hard-coded in `scripts/eval_llm_checkpoint.py`,
patch it and add a CLI flag.

### Step A.3 — Re-eval the maze checkpoints (per machine assignment in §3)

| Model | Checkpoint path | Re-eval time | Machine |
|---|---|---|---|
| GPT-2 Maze | `runs/gpt2-maze-seed0/gpt2_maze_latest.pt` | ~30 min | M5 |
| Distilled GPT-2 Maze | `runs/distill-gpt2-maze-seed0/latest.pt` | ~5 min | M5 |
| Qwen Maze | `08_qwen-distill-maze-saturated/qwen-maze/latest.pt` | ~30 min | unassigned (§3) |
| Distilled Qwen Maze | `08_qwen-distill-maze-saturated/distill-maze/latest.pt` | ~5 min | unassigned (§3) |
| Llama-3.2-1B Maze | `runs/llm-llama-maze-seed0/llama_3.2_1b_maze_latest.pt` | ~30 min | M3 |

```bash
# Generic invocation pattern
python scripts/eval_llm_checkpoint.py \
  configs/llm_maze_fixed.yaml \
  <CHECKPOINT_PATH> 50 \
  --emissions-out results/eval_fixed/<RUN_NAME>-emissions.csv \
  --eval-metric exact_match
```

Save outputs under `results/eval_fixed/`. Use file names matching the
run identifiers (e.g. `gpt2-maze-seed0-emissions.csv`) plus a sibling
JSON containing puzzle/cell accuracy.

**Expected outcome:** all five drop from 1.000 to ~0 % puzzle
accuracy with cell accuracy in the 10–20 % range, mirroring the
Sudoku result. If they drop, the report's central thesis is reinforced.
**No retraining required in this branch.**

### Step A.4 — Decision point

| Re-eval result | Action |
|---|---|
| All drop to ~0 % | ✅ Done. Update report tables (§6). Skip A.5. |
| GPT-2 still > ~5 % after `mask_non_path: false` | Confirms Cause A2. Retrain GPT-2 + Distill-GPT-2 Maze on `maze-30x30-hard-1k` (Step A.5). **Do not retrain Qwen or Llama Maze** — time budget. |
| Some checkpoints fail to load | Diagnose load errors. If keys are mismatched, check whether a `remapped_for_local.pt` artifact exists (the convention used in `hf_baselines/Sudoku-Extreme-mlp/` and `hf_baselines/Maze-Hard/`). Re-eval after remap. |

### Step A.5 — Retrain GPT-2 + Distill-GPT-2 Maze, only if A.4 says so (~5–7 h)

```yaml
# configs/llm_maze_clean.yaml
model: gpt2
dataset: maze-30x30-hard-1k        # non-augmented, clean split
train_split: train                  # 1,000 mazes
eval_split: test                    # 1,000 mazes, verified non-overlapping
lora_r: 8
lora_alpha: 16
batch_size: 2
grad_accum_steps: 8
epochs: 30
lr: 5.0e-5
mask_non_path: false                # THE FIX
eval_metric: exact_match
emissions_project_name: gpt2_maze_train_fixed
```

Distilled student inherits the freshly retrained clean GPT-2 teacher;
same dataset and eval config; `emissions_project_name:
distill_gpt2_maze_fixed`. Save under `runs/gpt2-maze-seed0-fixed/` and
`runs/distill-gpt2-maze-seed0-fixed/`. CodeCarbon throughout.

## 5. Track B — Sudoku Fix-B retrain details

GPT-2 + Distill-GPT-2 Sudoku Fix-B are already done on M4. The four
remaining models (per §3 routing) use the following Fix-B retrain
config:

```yaml
# configs/llm_sudoku_fixb.yaml
model: <qwen | distill_qwen | smollm | llama>
dataset: sudoku-extreme-1k          # 1k train / 423k test, no augmentation
train_split: train
eval_split: test
fixb_shift_correction: true         # apply at train time, not just eval
lora_r: 8
lora_alpha: 16
batch_size: 2
grad_accum_steps: 8
epochs: 30
lr: 5.0e-5
eval_metric: exact_match
emissions_project_name: <model>_sudoku_train_fixb
```

Approximate budgets:
- **Qwen-0.5B Sudoku Fix-B (M5):** ~2–3 h
- **Distill-Qwen Sudoku Fix-B (M5):** ~30 min – 1 h
- **SmolLM-360M Sudoku Fix-B (M1):** ~1.5–2 h
- **Llama-3.2-1B Sudoku Fix-B (M3 or M6):** ~5–7 h

Save under `runs/<model>-sudoku-seed0-fixb/`. CodeCarbon required
throughout (the report's CO2-per-correct-puzzle column needs the new
energy numbers).

## 6. Audit tables — what's clean vs what's pending

### Sudoku — model status post-Fix-B

| Model | Pre-Fix-B result | Fix-B retrained? | Where | Post-Fix-B result |
|---|---|---|---|---|
| TRM-MLP from-scratch (3 seeds) | 74.25 ± 0.63 % | n/a (TRM not affected by Fix-B) | FDK/FFN | unchanged |
| TRM-MLP HF eval | 84.74 % | n/a | — | unchanged |
| GPT-2 + LoRA | 0.00 % / 13.18 % cell | ✅ Done (M4, 30 ep) | M4 + Drive `05_…` | 0.00 % / 13.28 % cell |
| Distill-GPT-2 | 0.00 % / 25.80 % cell | ✅ Done (M4, 30 ep) | M4 + Drive `07_…` | 0.00 % / 36.43 % cell |
| Qwen-0.5B + LoRA | 0.00 % / 19.07 % cell | ❌ Pending | M5 | TBD |
| Distill-Qwen | 0.00 % / 35.87 % cell | ❌ Pending | M5 | TBD |
| SmolLM-360M + LoRA | 0.00 % / 14.11 % cell | ❌ Pending | M1 | TBD |
| Llama-3.2-1B + LoRA | 0.00 % / 19.74 % cell | ❌ Pending | M3 | TBD |

Naming convention reminder: always disambiguate **`sudoku-mlp`**
(mlp_t=true, ~84.80 %) vs **`sudoku-att`** (mlp_t=false, ~77.70 %).
Never use bare "sudoku" or "sudoku-official".

### Maze — model status

| Model | Checkpoint | Action |
|---|---|---|
| TRM-Att HF baseline | (HF download) | ✅ Headline = 79.60 % under all-cell metric |
| TRM-Att fine-tune seed 1 (M5) | `runs/trm-att-maze-seed1/best.pt` | ✅ Secondary observation — 27.50 % epoch-50 peak. Preserve in report. **No further training.** |
| GPT-2 Maze (M5) | `runs/gpt2-maze-seed0/...` | ⚠️ Re-eval first (M5). Retrain only if A.4 triggers. |
| Distilled GPT-2 Maze (M5) | `runs/distill-gpt2-maze-seed0/...` | ⚠️ Re-eval first (M5). Retrain only if A.4 triggers. |
| Qwen Maze (Folder 08) | `08_qwen-distill-maze-saturated/qwen-maze/` | ⚠️ Re-eval only — **unassigned**. Do NOT retrain. |
| Distilled Qwen Maze (Folder 08) | `08_qwen-distill-maze-saturated/distill-maze/` | ⚠️ Re-eval only — **unassigned**. Original distill train log was empty (QLoRA bug); if re-eval produces sane numbers, keep — otherwise retrain in ~10 min. |
| Llama-3.2-1B Maze (M3) | `runs/llm-llama-maze-seed0/...` | ⚠️ Re-eval only on M3 — Do NOT retrain (~22 h won't fit). |
| SmolLM-360M Maze | — | n/a (never trained) |

## 7. What NOT to do

- ❌ **Do not run any TRM training in this sprint** — TRM-Att Maze is
  settled; TRM-MLP Sudoku headline is settled; everything else is
  out-of-scope per direction.
- ❌ **Do not retrain Llama Maze** — ~22 h won't fit before the
  deadline.
- ❌ **Do not retrain Qwen Maze or Distill-Qwen Maze** — re-eval only.
- ❌ **Do not retrain GPT-2 or Distill-GPT-2 Sudoku** — already done by
  M4 with Fix-B at training time; check Drive `05_…` and `07_…`.
- ❌ **Do not apply paper-faithful TRM hyperparameters to a TRM
  fine-tune** (`lr=1e-4, weight_decay=1.0, q_loss_weight=0.5,
  halt_exploration_prob=0.1`). Run `dz3tkge9` documents a 12-pp
  regression of the HF init. Use `configs/trm_official_sudoku_mlp_finetune.yaml`
  if for any reason TRM fine-tuning re-enters scope.
- ❌ **Do not push to GitHub remote** unless the user literally types
  "push" (per global rules).
- ❌ **Do not run destructive git operations** (`reset --hard`,
  `branch -D`, `push --delete`, `worktree remove`) without per-action
  approval.
- ❌ **Do not skip pre-commit hooks** with `--no-verify` — fix the
  underlying issue.
- ❌ **Do not auto-resolve conflicts in `findings.md`,
  `results/summary.csv`, or any `report*.md`** during the M5 pull —
  those are live edits from other machines.

## 8. Update artifacts after each track lands (≤30 min per track)

1. Add a row to `findings.md` §5 documenting:
   - For Track A: the eval-mask bug name (`mask_non_path: true`), the
     fix, the contamination-check outcome, and the before-vs-after
     numbers per checkpoint.
   - For Track B: the Fix-A vs Fix-B distinction, which models were
     re-trained, and the before-vs-after Sudoku table.
2. Update `docs/report_methods_experiments_draft.md`:
   - §5.1 (Sudoku table): swap in the post-Fix-B numbers for the four
     newly retrained Sudoku models.
   - §5.3 (Maze-Hard): replace the saturation paragraph with the
     corrected re-eval table; reference the M5 27.50 % TRM-Att seed-1
     peak as a secondary observation; reaffirm 79.60 % HF baseline as
     the headline TRM-Att Maze number.
3. Append the new emissions CSVs and re-eval JSONs to the supplementary
   Drive folder
   (`https://drive.google.com/drive/folders/1pjIrzzAOflwB385KJKfaXq0ctUmAovd9`,
   `TRM-ML .chk`) under `results/eval_fixed/` (mirror existing
   `runs/.../emissions/` pattern).
4. Add a top-level `README.md` to the Drive folder mapping the tree
   to the report sections — flag `eval_fixed/` as the post-bug
   numbers, and the original `runs/.../` numbers as superseded but
   preserved for the audit trail.
5. Update `results/summary.csv` (or write `results/summary_fixed.csv`
   to preserve the audit trail) with the corrected rows.
6. Upload M5's `runs/trm-att-maze-seed1/best.pt` + train log +
   emissions CSV to Drive under `runs/trm-att-maze-seed1/` (Drive
   currently has seed 2; seed 1 is the one with the 27.50 % peak).

## 9. Verification before claiming "done"

Track A (maze re-eval / fix):
- [ ] All 5 LLM/distilled maze checkpoints have a re-eval emissions CSV in `results/eval_fixed/`.
- [ ] Each re-eval has a sibling JSON or CSV row with the new puzzle/cell accuracies.
- [ ] `findings.md` has a §5.x entry naming the bug as `mask_non_path: true` and listing the before/after per checkpoint.
- [ ] Report draft §5.3 (Maze-Hard) reflects the corrected story.
- [ ] M5's TRM-Att seed-1 `best.pt` + train log + emissions CSV are uploaded to Drive under `runs/trm-att-maze-seed1/`.
- [ ] Drive top-level `README.md` exists and points at `eval_fixed/`.

Track B (Sudoku Fix-B retrain):
- [ ] Qwen, Distill-Qwen, SmolLM, Llama Sudoku each have a `runs/<model>-sudoku-seed0-fixb/` folder with checkpoint, train log, emissions CSV.
- [ ] CodeCarbon `project_name` matches `<model>_sudoku_train_fixb`.
- [ ] Old (pre-Fix-B) numbers are clearly marked as superseded in `results/summary.csv`, not deleted.
- [ ] Report draft §5.1 (Sudoku) tables reflect post-Fix-B numbers.

If retraining was triggered for Track A (Step A.5):
- [ ] New `gpt2-maze-seed0-fixed/` and `distill-gpt2-maze-seed0-fixed/` folders contain checkpoint, train log, emissions CSV.

## 10. Fleet bookkeeping

- Wandb project: `shamykyzer/TRM`. Multi-machine seed coverage lives
  there, not on local disk.
- Machines:
  - **M1** (RTX 5070 Blackwell, 12 GB) — SmolLM Sudoku Fix-B retrain.
  - **M3** — Llama Sudoku Fix-B retrain + Llama Maze re-eval. Joint
    with **M6** for the Apr-26 sprint.
  - **M4** — already done (GPT-2 + Distill-GPT-2 Sudoku Fix-B). Holds
    the report draft `docs/report_methods_experiments_draft.md` with
    Tables 1 + 2 populated.
  - **M5** — Qwen + Distill-Qwen Sudoku Fix-B retrain + GPT-2 +
    Distill-GPT-2 Maze re-eval (after Step 0 pull resolution). Holds
    the TRM-Att seed-1 27.5 % run.
  - **Folder 08** — original Qwen + Distill-Qwen Maze archive; the
    two re-eval jobs there are unassigned and need a machine.
- Drive supplementary archive: `TRM-ML .chk` (ID
  `1pjIrzzAOflwB385KJKfaXq0ctUmAovd9`).
- Local repo lives under
  `C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ML-TRM/`.
  Note: `.git` corrupts under OneDrive sync — the canonical clone for
  any commit work lives outside OneDrive (see `docs/setup-guide.txt`).

---

End of agent prompt.
