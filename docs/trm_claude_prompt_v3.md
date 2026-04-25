# TRM ML Group Project — Claude Code Agent Prompt

## CONTEXT: Who You Are Helping
You are assisting **Ahmed AlShamy**, a Software Engineering Intern at Trevolution, working on a university ML group project at UWE Bristol (Module: UFCFAS-15-2, Machine Learning, 70% coursework).

**Group:** Ahmed AlShamy, Armin Ghaseminejad, Nickolas Greiner  
**Deadline:** 1 May 2026, 17:00 BST  
**Repo:** https://github.com/shamykyzer/ML-TRM  
**Hardware:** 6 × NVIDIA RTX 5070 (12 GB GDDR7, Blackwell sm_120, ~30.87 TFLOPS FP32)

---

## PROJECT GOAL
Fine-tune from **HuggingFace TRM checkpoints** (already at 84.74% Sudoku / 79.6% Maze) and extend with SmolLM2-360M + distillation.

| Task | Dataset | Train | Test | HF Checkpoint Baseline |
|------|---------|-------|------|------------------------|
| Sudoku | Sudoku-Extreme | 1,000 | 423,000 | 84.74% (TRM-MLP) |
| Maze | Maze-Hard | 1,000 | 1,000 | 79.6% (TRM-Att) |

**Three models to evaluate:**
1. **TRM** — fine-tuned FROM HF checkpoint (not scratch), 5–7M params
2. **SmolLM2-360M** — fine-tuned with LoRA (rank=16) as the "baseline LLM"
3. **Distilled Student** — compact student trained via knowledge distillation from TRM teacher

**Extra metric:** CodeCarbon energy (kWh) + CO₂ (gCO₂eq) — required for MO4 ethics mark.

---

## HARDWARE CRITICAL WARNING — READ FIRST

```
RTX 5070 = Blackwell architecture = CUDA sm_120
Standard PyTorch < 2.7 DOES NOT SUPPORT sm_120.
You will get: "CUDA capability sm_120 is not compatible" errors.

FIX — Run this on EVERY machine before anything else:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Verify with:
python -c "import torch; print(torch.cuda.get_device_name(0), torch.version.cuda)"
Expected output: NVIDIA GeForce RTX 5070  12.x
```

---

## FINE-TUNING vs SCRATCH — KEY DIFFERENCE

Since you are **fine-tuning from the HF checkpoint**, NOT training from scratch:

| Param | Scratch Training | Fine-tune from HF ✅ |
|---|---|---|
| Epochs | 60,000 | **2,000–5,000** |
| Learning rate | 1e-4 | **1e-5** (10× lower) |
| Warmup steps | 2,000 | **100–200** |
| Weight decay | 1.0 | **0.01–0.1** |
| Est. time per seed (RTX 5070) | ~12–18h | **~30 min – 2h** |
| First eval checkpoint | epoch 5,000 | **epoch 500** |
| All 3 seeds total | ~36–54h | **~2–6h** |

This means all 3 seeds fit on a **single machine** — freeing other machines for Maze and SmolLM2.

---

## CONFIRMED FINE-TUNING HYPERPARAMETERS

```python
# TRM Fine-tune from HF Checkpoint
epochs          = 5_000         # NOT 60,000 — starting from pretrained
eval_interval   = 500           # NOT 5,000 — finer monitoring
lr              = 1e-5          # 10x lower than scratch
puzzle_emb_lr   = 1e-5
weight_decay    = 0.1           # much lower than scratch (was 1.0)
warmup_steps    = 100           # short warmup for fine-tune
ema_rate        = 0.999         # keep EMA
optimizer       = "AdamW"
beta1, beta2    = 0.9, 0.95

# TRM-MLP (Sudoku) — load from HF
pretrained_ckpt = "wtfmahe/Samsung-TRM"   # or your group's HF repo
arch.mlp_t      = True
arch.pos_encodings = "none"
arch.L_layers   = 2
arch.H_cycles   = 3
arch.L_cycles   = 6

# TRM-Att (Maze) — load from HF
arch.mlp_t      = False
arch.L_cycles   = 4

# SmolLM2-360M LoRA (unchanged — always fine-tuned)
lora_rank       = 16
lora_alpha      = 32
lora_target_modules = ["q_proj", "v_proj"]
smollm_lr       = 5e-4      # Sudoku
smollm_maze_lr  = 3e-4      # Maze
smollm_batch    = 8         # Sudoku (reduce to 4 if OOM)
smollm_maze_batch = 4       # Maze (reduce to 2 if OOM)
```

---

## 6-MACHINE ASSIGNMENT (UPDATED)

| Machine | Job | Model | Est. Time | Priority |
|---------|-----|-------|-----------|----------|
| M1 | TRM-MLP Sudoku FT seeds 0+1+2 | TRM 5M, 5K epochs from HF | ~2–6h total | 🔴 P1 Critical |
| M2 | TRM-Att Maze FT (q_loss fix) | TRM 7M, 5K epochs from HF | ~2–4h or KILL | 🔴 P1 Critical |
| M3 | SmolLM2-360M Sudoku LoRA ×3 seeds | 360M, LoRA r=16, 30 epochs | ~8–12h total | 🟣 SmolLM2 |
| M4 | Distilled Student Sudoku (1 seed) | Student from TRM teacher | ~1–2h total | 🟡 High |
| M5 | SmolLM2-360M Maze LoRA ×2 seeds | 360M, LoRA r=16, 30 epochs | ~8–14h total | 🟣 SmolLM2 |
| M6 | TRM-Att Maze FT seed=1 (backup) OR spare for paper writing | TRM 7M | ~2–4h | 🟢 Backup |

---

## EXACT TRAINING COMMANDS

### M1 — TRM-MLP Sudoku Fine-tune (3 seeds sequentially, ~2h total)
```bash
for seed in 0 1 2; do
  python pretrain.py \
    arch=trm \
    data_paths="[data/sudoku-extreme-1k-aug-1000]" \
    evaluators="[]" \
    epochs=5000 eval_interval=500 \
    lr=1e-5 puzzle_emb_lr=1e-5 \
    weight_decay=0.1 puzzle_emb_weight_decay=0.1 \
    warmup_steps=100 \
    arch.mlp_t=True arch.pos_encodings=none \
    arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
    pretrained_ckpt="wtfmahe/Samsung-TRM" \
    run_name="trm_mlp_sudoku_ft_seed${seed}" \
    ema=True seed=$seed
done
```

### M2 — TRM-Att Maze Fine-tune (q_loss=0.01 fix — VERIFY BEFORE LAUNCH)
```bash
# FIRST: confirm q_loss weight = 0.01 in losses.py
grep -n "q_loss" losses.py   # must show 0.01, NOT 1.0

python pretrain.py \
  arch=trm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  epochs=5000 eval_interval=500 \
  lr=1e-5 puzzle_emb_lr=1e-5 \
  weight_decay=0.1 puzzle_emb_weight_decay=0.1 \
  warmup_steps=100 \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  pretrained_ckpt="wtfmahe/Samsung-TRM" \
  run_name="trm_att_maze_ft_seed0" \
  ema=True seed=0
```

**KILL RULE:** val_acc < 50% at epoch 1,000 (since you're starting from 79.6%, a drop below 50% means the fix didn't work) → kill → lock in HF 79.6% → repurpose M2.

### M3 — SmolLM2-360M Sudoku LoRA (3 seeds sequentially)
```bash
for seed in 0 1 2; do
  python finetune_llm.py \
    --model HuggingFaceTB/SmolLM2-360M \
    --task sudoku \
    --data data/sudoku-extreme-1k-aug-1000 \
    --lora_rank 16 --lora_alpha 32 \
    --lora_target_modules "q_proj,v_proj" \
    --epochs 30 --lr 5e-4 --batch_size 8 \
    --run_name "smollm2_360m_sudoku_lora_seed${seed}" \
    --seed $seed
done
```

### M4 — Distilled Student Sudoku (3 seeds sequentially)
```bash
python distill.py \
    --teacher checkpoints/trm_mlp_sudoku_ft_seed0 \
    --task sudoku \
    --data data/sudoku-extreme-1k-aug-1000 \
    --student_layers 1 \
    --epochs 30 --lr 1e-3 --batch_size 32 \
    --run_name "distilled_student_sudoku_seed0" \
    --seed 0
```

### M5 — SmolLM2-360M Maze LoRA (2 seeds sequentially)
```bash
for seed in 0 1; do
  python finetune_llm.py \
    --model HuggingFaceTB/SmolLM2-360M \
    --task maze \
    --data data/maze-30x30-hard-1k \
    --lora_rank 16 --lora_alpha 32 \
    --lora_target_modules "q_proj,v_proj" \
    --epochs 30 --lr 3e-4 --batch_size 4 \
    --run_name "smollm2_360m_maze_lora_seed${seed}" \
    --seed $seed
done
```

### M6 — TRM-Att Maze Seed 1 (backup) OR paper writing machine
```bash
# Only launch if M2 shows healthy val_acc > 50% at epoch 1,000
python pretrain.py \
  arch=trm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  epochs=5000 eval_interval=500 \
  lr=1e-5 puzzle_emb_lr=1e-5 \
  weight_decay=0.1 \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  pretrained_ckpt="wtfmahe/Samsung-TRM" \
  run_name="trm_att_maze_ft_seed1" \
  ema=True seed=1
```

---

## KNOWN BUGS & FIXES

### Bug 1: q_loss weight too high (root cause of Maze ~11% accuracy)
```python
# In losses.py:
# WRONG:  loss += 1.0 * q_loss
# RIGHT:  loss += 0.01 * q_loss
```

### Bug 2: RTX 5070 CUDA sm_120 compatibility
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Bug 3: OOM on Maze (30×30 self-attention)
```bash
# Reduce batch size for Maze fine-tuning if OOM:
batch_size=256   # instead of 768
```

### Bug 4: SmolLM2-360M VRAM OOM
```bash
--batch_size 4 --gradient_accumulation_steps 2   # Sudoku
--batch_size 2 --gradient_accumulation_steps 4   # Maze
```

---

## EVAL CHECKPOINTS — EXPECTED PROGRESSION

```
TRM-MLP Sudoku fine-tune (starting from 84.74%):
  Epoch 500:   val_acc ~82-85%   (slight dip during adaptation is normal)
  Epoch 1,000: val_acc ~84-86%   (back to or above baseline)
  Epoch 3,000: val_acc ~85-88%   (target — improvement over HF)
  Epoch 5,000: val_acc ~85-88%   (converged)

TRM-Att Maze fine-tune KILL RULE (starting from 79.6%):
  Epoch 500:   val_acc > 50%     (if below 50%, q_loss fix didn't work — KILL)
  Epoch 1,000: val_acc > 70%     (recovering toward baseline)
  Epoch 3,000: val_acc > 79%     (at or above HF baseline)

SmolLM2-360M (always fails constraint satisfaction):
  Sudoku puzzle_acc: 0-5%        (expected — proves TRM advantage)
  Maze puzzle_acc:   0-10%       (expected — proves TRM advantage)
```

---

## CODECARBON SETUP (Required for MO4 Ethics Mark)

```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(
    project_name="TRM-finetuning",
    output_dir="./codecarbon_logs",
    log_level="WARNING"
)
tracker.start()
# ... training loop ...
emissions = tracker.stop()   # returns kg CO2eq
print(f"CO2: {emissions:.6f} kg CO2eq")
```

---

## RESULTS TABLE TEMPLATE

```markdown
| Model | Task | Test Acc (%) | kWh | gCO₂eq | Params |
|-------|------|-------------|-----|--------|--------|
| TRM-MLP (HF baseline) | Sudoku | 84.74 | - | - | 5M |
| TRM-Att (HF baseline) | Maze | 79.6 | - | - | 7M |
| TRM-MLP FT (ours, mean±std) | Sudoku | [FILL] | [FILL] | [FILL] | 5M |
| TRM-Att FT (ours, fixed) | Maze | [FILL] | [FILL] | [FILL] | 7M |
| SmolLM2-360M LoRA | Sudoku | [FILL] | [FILL] | [FILL] | 360M |
| SmolLM2-360M LoRA | Maze | [FILL] | [FILL] | [FILL] | 360M |
| Distilled Student | Sudoku | [FILL] | [FILL] | [FILL] | ~1M |
| Qwen2.5-7B (0-shot) | Sudoku | 0.0 | - | - | 7B |
```

---

## PAPER STRUCTURE (6-page limit, 10pt, double column)

| Section | Mark % | Key Content |
|---------|--------|-------------|
| Abstract | - | Problem + TRM FT vs SmolLM2 + best acc + CO₂ |
| Introduction | 10% | Constraint puzzles, LLM failure, TRM motivation |
| Related Work | 10% | TRM paper, HRM, SmolLM2, LoRA, distillation |
| Data | 10% | Sudoku-Extreme + Maze-Hard, augmentation |
| **Methods** | **30%** | TRM fine-tuning setup, LoRA config, distillation |
| **Experiments** | **30%** | Results table, CO₂ chart, HF vs FT comparison |
| Conclusion | 5% | TRM wins, LLM limits, future work |
| Writing/Format | 5% | Template, citations, clarity |

**Methods + Experiments = 60% of marks.**

---

## THINGS CLAUDE CODE CAN HELP WITH

1. `Fix q_loss bug` — change weight from 1.0 to 0.01 in losses.py
2. `Wire up HF checkpoint loading` — add pretrained_ckpt arg to pretrain.py
3. `Add CodeCarbon logging` — wrap training loop in EmissionsTracker
4. `Write finetune_llm.py` — LoRA fine-tuning script for SmolLM2-360M
5. `Write distill.py` — knowledge distillation from TRM teacher
6. `Write evaluate.py` — outputs puzzle_acc and cell_acc
7. `Build results_table.csv` — parse all checkpoint logs
8. `Generate CO₂ bar chart` — matplotlib comparing kWh per model
9. `Monitor GPU health` — nvidia-smi script with temp alert >85°C
10. `Verify PyTorch/CUDA` — check all 6 machines on PyTorch 2.7+ cu128

---

## DEADLINE COUNTDOWN

```
SAT 26 APR  — Launch all 6 machines (M1 done in ~2h, M2 kill/continue by noon)
SUN 27 APR  — All TRM fine-tunes done, SmolLM2 results in, start paper
MON 28 APR  — Distillation done, full results table complete
TUE 29 APR  — Methods + Experiments drafted
WED 30 APR  — Full paper draft, group review, code cleanup
THU 1 MAY   — SUBMIT BY 17:00 BST (PDF report + ZIP notebook)
```

---

*Generated: 25 April 2026 | UWE ML UFCFAS-15-2 | Repo: github.com/shamykyzer/ML-TRM*
