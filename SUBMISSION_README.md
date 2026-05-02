# Tiny Recursive Models vs Fine-Tuned LLMs for Structured Reasoning

**Authors:** Ahmed AlShamy, Armin Ghaseminejad, Nickolas Greiner
**Module:** UFCFAS-15-2 Machine Learning (2025-26)

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build datasets (downloads from HuggingFace)
python data/build_sudoku_dataset.py
python data/build_maze_dataset.py

# 4. Train a model
python main.py --config configs/trm_official_sudoku_mlp.yaml --mode train --seed 42

# 5. Evaluate a checkpoint
python main.py --config configs/trm_official_sudoku_mlp.yaml --mode eval \
    --checkpoint models/trm_mlp/best.pt
```

## Reproducing Paper Results

### TRM-MLP on Sudoku-Extreme (3 seeds)
```bash
python main.py --config configs/trm_official_sudoku_mlp.yaml --mode train --seed 0
python main.py --config configs/trm_official_sudoku_mlp.yaml --mode train --seed 1
python main.py --config configs/trm_official_sudoku_mlp.yaml --mode train --seed 2
```

### LLM Baselines (LoRA fine-tuning)
```bash
python main.py --config configs/llm_qwen.yaml --mode train      # Qwen2.5-0.5B
python main.py --config configs/llm_smollm.yaml --mode train     # SmolLM2-360M
python main.py --config configs/llm_llama.yaml --mode train      # Llama-3.2-1B
```

### Knowledge Distillation
```bash
python main.py --config configs/distill_qwen_sudoku.yaml --mode distill \
    --checkpoint <path-to-qwen-teacher-best.pt>
```

### Generate Figures
```bash
python scripts/gen_training_curve.py    # Fig 1: training trajectory
python scripts/gen_energy_scatter.py    # Fig 2: energy vs accuracy
```

## Project Structure

| Directory | Contents |
|-----------|----------|
| `src/models/` | TRM, LLM baseline, distilled student model definitions |
| `src/training/` | Training loops + CodeCarbon carbon tracking |
| `src/data/` | Dataset loading, tokenisation, augmentation |
| `src/evaluation/` | Metrics (cell/puzzle accuracy), K-vote ensemble |
| `src/utils/` | Config schema, GPU setup, seeding |
| `configs/` | 24 YAML experiment configurations |
| `data/` | Dataset build scripts + metadata |
| `results/` | Consolidated results (summary_fixed.csv) and figures |
| `scripts/` | Reproduction helpers, figure generation, WandB aggregation |
| `tests/` | Unit tests |

## Hardware Requirements

- GPU: NVIDIA with 12+ GB VRAM (RTX 3060 or better)
- Training time: TRM-MLP ~25 hours/seed, LLMs ~1-8 hours each
- Carbon tracking via CodeCarbon (emissions logged per run)

## Key Files

- `main.py` -- Entry point for train/eval/distill
- `ML_TRM_Walkthrough.ipynb` -- Guided notebook demonstrating the pipeline
- `results/summary_fixed.csv` -- All experimental results
- `docs/wandb_metrics_glossary.md` -- WandB metric definitions
