# TRM Project Plan

**Module:** UFCFAS-15-2 Machine Learning | **Deadline:** 1 May 2026, 17:00 BST
**Team:** Ahmed AlShamy, Armin Ghaseminejad, Nickolas Greiner
**Today:** 11 April 2026 | **Days remaining:** 20

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
| Eval from init weights | Done | `--mode eval --checkpoint <remapped.pt>` works directly on remapped HF checkpoints (`evaluate.py:load_and_evaluate` uses `strict=False` + load report) |
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
