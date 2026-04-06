#!/usr/bin/env bash
# TRM Project Task Runner (Windows/Git Bash compatible)
# Usage: bash run.sh <target> [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect system python (for venv creation)
if command -v python3 &>/dev/null; then
    SYS_PYTHON="python3"
elif command -v python &>/dev/null; then
    SYS_PYTHON="python"
elif command -v py &>/dev/null; then
    SYS_PYTHON="py -3"
else
    echo "Error: No python found. Install Python 3 and add it to PATH."
    exit 1
fi

# Detect venv python/pip (for running project code)
if [[ -f .venv/Scripts/python.exe ]]; then
    PYTHON=".venv/Scripts/python.exe"
    PIP=".venv/Scripts/pip.exe"
elif [[ -f .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
    PIP=".venv/bin/pip"
else
    PYTHON="$SYS_PYTHON"
    PIP="$SYS_PYTHON -m pip"
fi

# ============================================================
# Setup
# ============================================================

setup() {
    echo "==> Creating venv and installing dependencies (CPU)..."
    $SYS_PYTHON -m venv .venv
    $PYTHON -m pip install --upgrade pip
    $PIP install -r requirements.txt
    echo "==> Setup complete."
}

setup-cuda() {
    echo "==> Creating venv and installing dependencies (CUDA 12.4)..."
    $SYS_PYTHON -m venv .venv
    $PYTHON -m pip install --upgrade pip
    $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    $PIP install -r requirements.txt
    $PYTHON -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
    echo "==> CUDA setup complete."
}

setup-dev() {
    setup
    echo "==> Installing dev tools..."
    $PIP install pytest black isort flake8
    echo "==> Dev setup complete."
}

# ============================================================
# Data Preprocessing
# ============================================================

data-sudoku() {
    echo "==> Building Sudoku-Extreme dataset..."
    (cd data && ../$PYTHON build_sudoku_dataset.py --output-dir sudoku-extreme-full)
}

data-sudoku-aug() {
    echo "==> Building Sudoku-Extreme dataset (1000x augmentation)..."
    (cd data && ../$PYTHON build_sudoku_dataset.py --output-dir sudoku-extreme-full --num-aug 1000)
}

data-sudoku-small() {
    echo "==> Building Sudoku-Extreme small subset (100 samples)..."
    (cd data && ../$PYTHON build_sudoku_dataset.py --output-dir sudoku-extreme-full --subsample-size 100)
}

data-maze() {
    echo "==> Building Maze-Hard dataset..."
    (cd data && ../$PYTHON build_maze_dataset.py --output-dir maze-30x30-hard-1k)
}

data-maze-aug() {
    echo "==> Building Maze-Hard dataset (dihedral augmentation)..."
    (cd data && ../$PYTHON build_maze_dataset.py --output-dir maze-30x30-hard-1k --aug)
}

data-all() {
    data-sudoku
    data-maze
}

# ============================================================
# Training
# ============================================================

train-sudoku() {
    echo "==> Training TRM-MLP on Sudoku-Extreme..."
    $PYTHON main.py --mode train --config configs/trm_sudoku.yaml
}

train-maze() {
    echo "==> Training TRM-Att on Maze-Hard..."
    $PYTHON main.py --mode train --config configs/trm_maze.yaml
}

train-maze-fast() {
    echo "==> Training TRM-Maze (augmented data, fewer epochs)..."
    $PYTHON main.py --mode train --config configs/trm_maze_fast.yaml
}

train-llm() {
    echo "==> Fine-tuning GPT-2 with LoRA..."
    $PYTHON main.py --mode train --config configs/llm_config.yaml
}

train-llm-qwen() {
    echo "==> Fine-tuning Qwen2.5-0.5B with LoRA..."
    $PYTHON main.py --mode train --config configs/llm_qwen.yaml
}

train-llm-smollm() {
    echo "==> Fine-tuning SmolLM2-360M with LoRA..."
    $PYTHON main.py --mode train --config configs/llm_smollm.yaml
}

train-llm-llama() {
    echo "==> Fine-tuning Llama-3.2-1B with LoRA..."
    $PYTHON main.py --mode train --config configs/llm_llama.yaml
}

train-llm-all() {
    train-llm
    train-llm-qwen
    train-llm-smollm
    train-llm-llama
}

train-distill() {
    echo "==> Knowledge distillation..."
    $PYTHON main.py --mode distill --config configs/llm_config.yaml --checkpoint models/llm/gpt2_latest.pt
}

# ============================================================
# Resume Training
# ============================================================

resume-sudoku() {
    echo "==> Resuming TRM-Sudoku training..."
    $PYTHON main.py --mode train --config configs/trm_sudoku.yaml --resume models/sudoku/latest.pt
}

resume-maze() {
    echo "==> Resuming TRM-Maze training..."
    $PYTHON main.py --mode train --config configs/trm_maze.yaml --resume models/maze/latest.pt
}

# ============================================================
# Evaluation
# ============================================================

eval-sudoku() {
    echo "==> Evaluating TRM-Sudoku..."
    $PYTHON main.py --mode eval --config configs/trm_sudoku.yaml --checkpoint models/sudoku/best.pt
}

eval-maze() {
    echo "==> Evaluating TRM-Maze..."
    $PYTHON main.py --mode eval --config configs/trm_maze.yaml --checkpoint models/maze/best.pt
}

eval-llm() {
    echo "==> Evaluating GPT-2..."
    $PYTHON main.py --mode eval --config configs/llm_config.yaml --checkpoint models/llm/gpt2_latest.pt
}

eval-llm-qwen() {
    echo "==> Evaluating Qwen2.5-0.5B..."
    $PYTHON main.py --mode eval --config configs/llm_qwen.yaml --checkpoint models/llm/qwen2.5_0.5b_latest.pt
}

eval-llm-smollm() {
    echo "==> Evaluating SmolLM2-360M..."
    $PYTHON main.py --mode eval --config configs/llm_smollm.yaml --checkpoint models/llm/smollm2_360m_latest.pt
}

eval-llm-llama() {
    echo "==> Evaluating Llama-3.2-1B..."
    $PYTHON main.py --mode eval --config configs/llm_llama.yaml --checkpoint models/llm/llama_3.2_1b_latest.pt
}

# ============================================================
# Verification & Testing
# ============================================================

verify() {
    echo "==> Running import and forward pass verification..."
    $PYTHON -c "
import torch
from src.utils.config import load_config
from src.models.trm_sudoku import TRMSudoku, TRMMaze
from src.models.recursion import deep_recursion
from src.models.distilled_llm import DistilledLLM
print('=== Import Check ===')
config = load_config('configs/trm_sudoku.yaml')
print('Config OK:', config.model.model_type.value)
ms = TRMSudoku()
print(f'TRM-Sudoku: {ms.param_count():,} params')
mm = TRMMaze()
print(f'TRM-Maze:   {mm.param_count():,} params')
ds = DistilledLLM()
print(f'DistilledLLM: {ds.param_count():,} params')
print('=== Forward Pass ===')
x = torch.randint(1, 11, (2, 81))
emb = ms.embedding(x)
y = ms.y_init.expand(2,-1,-1).clone()
z = ms.z_init.expand(2,-1,-1).clone()
(y,z), logits, q = deep_recursion(ms.block, ms.output_head, ms.q_head, emb, y, z, n=2, T=2)
print(f'logits: {logits.shape} | q: {q.shape}')
print('=== All OK ===')
"
}

verify-data() {
    echo "==> Verifying data scripts..."
    (cd data && ../$PYTHON -c "from common import PuzzleDatasetMetadata, dihedral_transform; print('data/common.py OK')")
    (cd data && ../$PYTHON build_sudoku_dataset.py --help > /dev/null && echo "build_sudoku_dataset.py OK")
    (cd data && ../$PYTHON build_maze_dataset.py --help > /dev/null && echo "build_maze_dataset.py OK")
}

smoke-test() {
    echo "==> Running smoke test (small subset + 2 batch train)..."
    data-sudoku-small
    $PYTHON -c "
import torch
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.models.trm_sudoku import TRMSudoku
from src.models.layers import StableMaxCrossEntropy
from src.models.recursion import deep_supervision_step
from src.data.sudoku_dataset import get_sudoku_loaders
set_seed(42)
config = load_config('configs/trm_sudoku.yaml')
model = TRMSudoku()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1.0)
loss_fn = StableMaxCrossEntropy(ignore_index=0)
train_loader, _ = get_sudoku_loaders(config.data.data_dir, batch_size=4, num_workers=0)
print('Training 2 batches...')
for i, (inp, lab) in enumerate(train_loader):
    metrics = deep_supervision_step(model, inp, lab, loss_fn, optimizer, n=2, T=2, N_sup=2)
    print(f'  batch {i+1}: ce={metrics[\"ce_loss\"]:.4f} q={metrics[\"q_mean\"]:.3f}')
    if i >= 1: break
print('Smoke test PASSED')
"
}

# ============================================================
# Utilities
# ============================================================

clean() {
    echo "==> Cleaning generated data, checkpoints, and experiments..."
    rm -rf data/sudoku-extreme-full data/maze-30x30-hard-1k
    rm -rf models/sudoku models/maze models/llm
    rm -rf experiments/sudoku experiments/maze experiments/llm
    rm -rf results/*.json
    echo "==> Clean complete."
}

clean-all() {
    clean
    echo "==> Removing venv..."
    rm -rf .venv
    echo "==> Full clean complete."
}

lint() {
    echo "==> Linting..."
    $PYTHON -m flake8 src/ main.py --max-line-length 120 --ignore E501,W503
}

format() {
    echo "==> Formatting..."
    $PYTHON -m black src/ main.py --line-length 120
    $PYTHON -m isort src/ main.py
}

# ============================================================
# Pipelines
# ============================================================

pipeline() {
    echo "==> Full pipeline: setup CUDA + all data + all training..."
    setup-cuda
    data-all
    train-sudoku
    train-maze
    train-llm-all
    echo ""
    echo "=== PIPELINE COMPLETE ==="
}

pipeline-sudoku() {
    echo "==> Sudoku pipeline: setup CUDA + sudoku data + train..."
    setup-cuda
    data-sudoku
    train-sudoku
    echo ""
    echo "=== SUDOKU PIPELINE COMPLETE ==="
}

pipeline-maze() {
    echo "==> Maze pipeline: setup CUDA + maze data + train..."
    setup-cuda
    data-maze
    train-maze
    echo ""
    echo "=== MAZE PIPELINE COMPLETE ==="
}

pipeline-llm() {
    echo "==> LLM pipeline: setup CUDA + sudoku data + train all LLMs..."
    setup-cuda
    data-sudoku
    train-llm-all
    echo ""
    echo "=== LLM PIPELINE COMPLETE ==="
}

help() {
    cat <<'HELP'
Usage: bash run.sh <target>
       bash run.sh              (interactive menu)

Setup:
  setup             Create venv and install all dependencies (CPU)
  setup-cuda        Create venv and install with CUDA 12.4 GPU support
  setup-dev         Setup + install dev tools

Data Preprocessing:
  data-sudoku       Download and preprocess full Sudoku-Extreme dataset
  data-sudoku-aug   Preprocess Sudoku with 1000x augmentation (full training)
  data-sudoku-small Preprocess small 100-sample Sudoku subset (for testing)
  data-maze         Download and preprocess full Maze-Hard dataset
  data-maze-aug     Preprocess Maze with dihedral augmentation
  data-all          Preprocess both datasets (no augmentation)

Training:
  train-sudoku      Train TRM-MLP on Sudoku-Extreme
  train-maze        Train TRM-Att on Maze-Hard
  train-maze-fast   Train TRM-Maze with augmented data + fewer epochs
  train-llm         Fine-tune GPT-2 baseline with LoRA
  train-llm-qwen    Fine-tune Qwen2.5-0.5B with LoRA
  train-llm-smollm  Fine-tune SmolLM2-360M with LoRA
  train-llm-llama   Fine-tune Llama-3.2-1B with LoRA
  train-llm-all     Fine-tune all 4 LLM baselines sequentially
  train-distill     Knowledge distillation (requires trained teacher)

Resume:
  resume-sudoku     Resume TRM-Sudoku training from last checkpoint
  resume-maze       Resume TRM-Maze training from last checkpoint

Evaluation:
  eval-sudoku       Evaluate best TRM-Sudoku checkpoint
  eval-maze         Evaluate best TRM-Maze checkpoint
  eval-llm          Evaluate GPT-2 baseline
  eval-llm-qwen     Evaluate Qwen2.5-0.5B baseline
  eval-llm-smollm   Evaluate SmolLM2-360M baseline
  eval-llm-llama    Evaluate Llama-3.2-1B baseline

Verification:
  verify            Run full import and forward pass verification
  verify-data       Verify data scripts import correctly
  smoke-test        Quick end-to-end: small subset + 2 batch train

Pipelines (fire-and-forget):
  pipeline          Full: setup CUDA + all data + all training
  pipeline-sudoku   Setup CUDA + sudoku data + train sudoku
  pipeline-maze     Setup CUDA + maze data + train maze
  pipeline-llm      Setup CUDA + sudoku data + train all 4 LLMs

Utilities:
  clean             Remove generated data, checkpoints, and experiments
  clean-all         Clean everything including venv
  lint              Run linting
  format            Auto-format code
  help              Show this help
HELP
}

# ============================================================
# Interactive Menu
# ============================================================

menu() {
    echo ""
    echo "=========================================="
    echo "   TRM Project"
    echo "=========================================="
    echo ""
    echo "  1) Setup (CUDA)          5) Verify installation"
    echo "  2) Pipeline: Sudoku      6) Smoke test"
    echo "  3) Pipeline: Maze        7) Clean"
    echo "  4) Pipeline: LLM         8) Help (all commands)"
    echo ""
    echo "  0) Quit"
    echo ""

    while true; do
        read -rp "Select [0-8]: " choice
        case "$choice" in
            0) echo "Bye!"; exit 0 ;;
            1) setup-cuda ;;
            2) pipeline-sudoku ;;
            3) pipeline-maze ;;
            4) pipeline-llm ;;
            5) verify ;;
            6) smoke-test ;;
            7) clean ;;
            8) help; exit 0 ;;
            *) echo "Invalid selection."; continue ;;
        esac
        break
    done
}

# ============================================================
# Dispatch
# ============================================================

TARGET="${1:-menu}"

# Normalize: accept underscores or hyphens
TARGET="${TARGET//_/-}"

case "$TARGET" in
    menu)            menu ;;
    setup)           setup ;;
    setup-cuda)      setup-cuda ;;
    setup-dev)       setup-dev ;;
    data-sudoku)     data-sudoku ;;
    data-sudoku-aug) data-sudoku-aug ;;
    data-sudoku-small) data-sudoku-small ;;
    data-maze)       data-maze ;;
    data-maze-aug)   data-maze-aug ;;
    data-all)        data-all ;;
    train-sudoku)    train-sudoku ;;
    train-maze)      train-maze ;;
    train-maze-fast) train-maze-fast ;;
    train-llm)       train-llm ;;
    train-llm-qwen)  train-llm-qwen ;;
    train-llm-smollm) train-llm-smollm ;;
    train-llm-llama) train-llm-llama ;;
    train-llm-all)   train-llm-all ;;
    train-distill)   train-distill ;;
    resume-sudoku)   resume-sudoku ;;
    resume-maze)     resume-maze ;;
    eval-sudoku)     eval-sudoku ;;
    eval-maze)       eval-maze ;;
    eval-llm)        eval-llm ;;
    eval-llm-qwen)   eval-llm-qwen ;;
    eval-llm-smollm) eval-llm-smollm ;;
    eval-llm-llama)  eval-llm-llama ;;
    verify)          verify ;;
    verify-data)     verify-data ;;
    smoke-test)      smoke-test ;;
    pipeline)        pipeline ;;
    pipeline-sudoku) pipeline-sudoku ;;
    pipeline-maze)   pipeline-maze ;;
    pipeline-llm)    pipeline-llm ;;
    clean)           clean ;;
    clean-all)       clean-all ;;
    lint)            lint ;;
    format)          format ;;
    help)            help ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Run 'bash run.sh help' for available targets."
        exit 1
        ;;
esac
