# ML-TRM coursework checkpoint manifest

This document is the canonical map from `best.pt` filenames in our GitHub
releases to the runs they came from, the metrics they achieved, and the
config that produced them. The report's Table 1 cites these checkpoints
directly, so any number quoted there should be traceable here.

## Releases

| Release | Status | Contents | URL |
|---|---|---|---|
| `v1.0-checkpoints` | live | 3 TRM-MLP-Sudoku FT seeds (existing) | https://github.com/shamykyzer/ML-TRM/releases/tag/v1.0-checkpoints |
| `v1.1-sprint` | pending | Sprint outputs (TRM-Att-Maze × 3, GPT-2 × 2, Qwen-Maze re-eval, 4 distillations) | TBD after Mon 27 Apr training wave |

---

## v1.0-checkpoints — pre-sprint baseline

These three runs are referenced by the report's headline TRM-MLP-Sudoku
3-seed mean (`0.7425 ± 0.006`). All trained from-scratch on a single RTX 5070
following the paper-faithful regime, peaks captured around epochs 500–900.

Each run folder is uploaded as a 3-part split zip (Drive auto-splits >2 GB
folders; each part is independently extractable). Each folder contains
`best.pt`, every `epoch_<N>.pt` milestone, `emissions.csv`,
`<config>_train_log.csv`, and the wandb run dir for full reproducibility.

| Run | Assets (in release) | Total folder size | Seed | Wandb run | Peak val_puzzle_acc | Peak val_cell_acc | Peak epoch | Train time |
|---|---|---|---|---|---|---|---|---|
| sudoku-mlp seed 0 | `sudoku-mlp-seed0-part1.zip` + `-part2.zip` + `-part3.zip` | ~3.6 GB | 0 | `ihj6hpsn` | **0.7456** | 0.8584 | 900 | 20.1 h |
| sudoku-mlp seed 1 | `sudoku-mlp-seed1-part1.zip` + `-part2.zip` + `-part3.zip` | ~3.6 GB | 1 | `c5kt8l2i` | **0.7420** | 0.8585 | 650 | 8.7 h |
| sudoku-mlp seed 2 | `sudoku-mlp-seed2-part1.zip` + `-part2.zip` + `-part3.zip` | ~3.7 GB | 2 | `8hncpi2x` | **0.7486** | 0.8613 | 700 | 10.8 h |

To extract a complete seed folder, download all 3 parts and unzip each
in turn into the same target directory; the zips have non-overlapping
contents (Drive's multi-part split is by file boundary).

**Architecture:** TRM-MLP (token-mixer = MLP, no self-attention). 6.4M params.
**Config:** `configs/trm_official_sudoku_mlp.yaml` (paper-faithful from-scratch
regime — `lr=1e-4`, `weight_decay=1.0`, `q_loss_weight=0.5`,
`warmup_steps=2000`).
**Init:** random (no HF init for these from-scratch runs).
**Dataset:** Sudoku-Extreme (1k train / 423k test, 17-clue 9×9 boards).

### NOT included in v1.0 (and why)

| Excluded | Reason |
|---|---|
| `sudoku-mlp-seed3` | Not validated in findings.md run inventory |
| `sudoku-mlp-seed4` (run `dz3tkge9`) | Regressed HF init by 12pp — paper-faithful hparams applied to a converged checkpoint corrupt the weights. See `analysis_run_dz3tkge9.md`. |
| `sudoku-mlp-seed5` | Not validated |
| `novelty-trm-mlp-sudoku-seed0` (any) | iso-time experiment (different scope; 84.84% peak at epoch 10), not the canonical from-scratch FT |
| `sudoku-att` (any) | Collapsed run — val peaked 18% at epoch 100, decayed to 0% by ep 500. Documented as a "when not to retrain" finding in the report's §5.4. |
| `maze-seed0/1/2` (existing) | Corrupted by `q_loss_weight=0.5` regime (per findings.md §5.7). Sprint produces fresh maze checkpoints under the corrected `q_loss_weight=0.0` config. |
| All `epoch_<N>.pt` intermediates | Superseded by `best.pt` |
| Older duplicate folders (`(1)` `(2)` `(3)`) | Latest mtime + highest val_acc selected per seed |
| `gpt2_sudoku_latest.pt` (501 MB) | Older GPT-2 sudoku run — M4 produces a fresh one this sprint |
| LLMs out of scope: `llm-deepseek-*`, `llm-llama-*`, `llm-smollm-*` | Not used in the report's cross-architecture comparison |
| `novelty-distill-*` (3 versions) | Superseded by this sprint's distillations from M1/M4/M5 |

---

## v1.1-sprint — pending

Will be uploaded by each lab machine as it completes its training phase (Mon 27 Apr
afternoon → evening). Expected contents:

| Asset (planned) | Producing machine | Phase | Wall-clock |
|---|---|---|---|
| `trm-att-maze-seed0-best.pt` | M1 | Phase 1 | ~12.4 h |
| `trm-att-maze-seed1-best.pt` | M2 | Phase 1 | ~12.4 h |
| `trm-att-maze-seed2-best.pt` | M3 | Phase 1 | ~12.4 h |
| `qwen-sudoku-existing-eval-override.json` | M1 (re-uploaded) | (no train) | ~5 min |
| `qwen-maze-eval-override.json` | M1 (Phase 3 re-eval) | (no train) | ~15 min |
| `qwen2.5_0.5b_sudoku_latest.pt` | M1 | (existing, sourced from M1 local disk) | (no train) |
| `qwen2.5_0.5b_maze_latest.pt` | M1 | (existing, sourced from M1 local disk) | (no train) |
| `distill-qwen-sudoku-best.pt` | M1 Phase 4 | distill | ~1 h |
| `distill-qwen-maze-best.pt` | M1 Phase 5 | distill | ~1 h |
| `gpt2-sudoku-best.pt` | M4 Phase 1 | LoRA | ~1 h |
| `gpt2-maze-best.pt` | M5 Phase 1 | LoRA | ~2 h |
| `distill-gpt2-sudoku-best.pt` | M4 Phase 2 | distill | ~1 h |
| `distill-gpt2-maze-best.pt` | M5 Phase 2 | distill | ~1 h |
| `kvote-results.csv` (TRM-MLP-Sudoku K∈{1,2,4}) | M4 Phase 3 | aggregate | ~6 h |
| 9 individual K-vote prediction CSVs | M1/M2/M3/M4/M5 | post-train | ~3 h each |

Each `best.pt` will be paired with its own `eval_override.json` (Fix-B
shift-corrected test-set metrics) once `scripts/finalize_results.py`
completes the backfill on Sunday-evening.

---

## How to use a checkpoint

1. Download from the release URL above (e.g.
   `gh release download v1.0-checkpoints --repo shamykyzer/ML-TRM`)
2. Place under `hf_checkpoints/<task>/` or `models/<task>/` per the
   trainer's expected layout (see the matching config YAML's
   `checkpoint_dir`).
3. Evaluate via:

   ```bash
   python main.py --mode eval \
       --config configs/trm_official_sudoku_mlp.yaml \
       --checkpoint <local-path-to-best.pt>
   ```

4. Or fine-tune from this checkpoint using `--init-weights` (NOT `--resume`)
   on a new run with the corrected fine-tune hparams in
   `configs/trm_official_sudoku_mlp_finetune.yaml` (see the dz3tkge9
   post-mortem in `analysis_run_dz3tkge9.md` for why this matters).

---

## Provenance and integrity

- All checkpoints in v1.0 were trained on University of the West of England
  Bristol consumer-grade hardware (RTX 5070, 12 GB) following the regime
  documented in findings.md.
- `best.pt` files are saved directly from `OfficialTRMTrainer` at the
  validation peak under the EMA shadow weights (per `src/training/ema.py`).
- All configs that produced these checkpoints are pinned at the commit
  hash of the run; check `wandb.config` on the linked wandb run for the
  exact YAML snapshot.

---

*Last updated: 2026-04-26. Maintained by the ML-TRM Group Project team
(Ahmed AlShamy, Armin Ghaseminejad, Nickolas Greiner). UWE UFCFAS-15-2.*
