# TRM Project Plan

**Module:** UFCFAS-15-2 Machine Learning | **Deadline:** 1 May 2026, 17:00 BST
**Team:** Ahmed AlShamy, Armin Ghaseminejad, Nickolas Greiner
**Today:** 6 April 2026 | **Days remaining:** 25

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Codebase | Done | Models, trainers, eval, data pipelines, AMP, resume support |
| Data preprocessing | Done | Sudoku: 1K train / 423K test. Maze: 1K train / 1K test |
| Configs fixed | Done | ff_hidden=2048, maze vocab=6, matching official paper |
| LLM baselines | Done | 4 models: GPT-2, Qwen2.5-0.5B, SmolLM2-360M, Llama-3.2-1B |
| Training logs | Done | CSV logs + auto-push script + GitHub Action notifications |
| Resume support | Done | `--resume` flag for crash recovery |
| Mixed precision | Done | AMP auto-enabled on CUDA (~1.5-2x speedup) |
| Checkpoints | None | No pretrained weights available anywhere -- must train from scratch |
| Training | Not started | Ready to run at the lab |
| Report | Not started | 6-page conference paper due May 1 |

---

## Phase 1: Lab Training (Days 1-5)

### 3-Machine Setup

| Machine | Terminal 1 | Terminal 2 | Est. Time |
|---------|-----------|-----------|-----------|
| **A** | `make setup-cuda && make train-sudoku` | `bash scripts/auto_push.sh` | ~12-14 hrs |
| **B** | `make setup-cuda && make train-maze` | `bash scripts/auto_push.sh` | ~3-4 days |
| **C** | `make setup-cuda && make train-llm-all && make train-distill` then `make data-maze-aug && make train-maze-fast` | `bash scripts/auto_push.sh` | ~1-2 days |

### What Each Machine Trains

**Machine A -- TRM-Sudoku (core result):**
- TRM-MLP, 6.4M params, 5000 epochs on 1K puzzles
- Expected: ~87% puzzle accuracy
- Checkpoints: `models/best.pt`, `models/latest.pt`

**Machine B -- TRM-Maze (core result):**
- TRM-Att, 8.4M params, 5000 epochs on 1K mazes
- Expected: ~85% puzzle accuracy
- Longest training -- start first

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
make resume-sudoku    # Resumes from models/latest.pt
make resume-maze      # Resumes from models/latest.pt
```

### Remote Monitoring

- Training CSV logs auto-pushed to GitHub every hour
- GitHub Action posts stats to an issue -- subscribe for phone notifications
- Check: `experiments/trm_sudoku_train_log.csv`, `experiments/trm_maze_train_log.csv`

---

## Phase 2: Evaluation (Days 5-7)

```bash
# TRM models
make eval-sudoku
make eval-maze

# LLM baselines
make eval-llm
make eval-llm-qwen
make eval-llm-smollm
make eval-llm-llama
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
