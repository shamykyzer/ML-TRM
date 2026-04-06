# TRM Project Makefile
# Usage: make <target>

PYTHON = .venv/Scripts/python.exe
PIP = .venv/Scripts/pip.exe

# Detect OS for venv activation
ifeq ($(OS),Windows_NT)
	PYTHON = .venv/Scripts/python.exe
	PIP = .venv/Scripts/pip.exe
else
	PYTHON = .venv/bin/python
	PIP = .venv/bin/pip
endif

# ============================================================
# Setup
# ============================================================

.PHONY: setup
setup: ## Create venv and install all dependencies (CPU)
	python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: setup-cuda
setup-cuda: ## Create venv and install with CUDA 12.4 GPU support
	python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu124
	$(PIP) install -r requirements.txt
	$(PYTHON) -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

.PHONY: setup-dev
setup-dev: setup ## Setup + install dev tools
	$(PIP) install pytest black isort flake8

# ============================================================
# Data Preprocessing
# ============================================================

# NOTE: Build scripts use `from common import ...` so must run from data/.
# Default output_dir in scripts is "data/sudoku-extreme-full" which would
# create data/data/... when run from data/. We override to write directly
# into data/<dataset> so configs can find it at data/<dataset> from project root.

.PHONY: data-sudoku
data-sudoku: ## Download and preprocess full Sudoku-Extreme dataset
	cd data && ../$(PYTHON) build_sudoku_dataset.py --output-dir sudoku-extreme-full

.PHONY: data-sudoku-aug
data-sudoku-aug: ## Preprocess Sudoku with 1000x augmentation (full training)
	cd data && ../$(PYTHON) build_sudoku_dataset.py --output-dir sudoku-extreme-full --num-aug 1000

.PHONY: data-sudoku-small
data-sudoku-small: ## Preprocess small 100-sample Sudoku subset (for testing)
	cd data && ../$(PYTHON) build_sudoku_dataset.py --output-dir sudoku-extreme-full --subsample-size 100

.PHONY: data-maze
data-maze: ## Download and preprocess full Maze-Hard dataset
	cd data && ../$(PYTHON) build_maze_dataset.py --output-dir maze-30x30-hard-1k

.PHONY: data-maze-aug
data-maze-aug: ## Preprocess Maze with dihedral augmentation
	cd data && ../$(PYTHON) build_maze_dataset.py --output-dir maze-30x30-hard-1k --aug

.PHONY: data-all
data-all: data-sudoku data-maze ## Preprocess both datasets (no augmentation)

# ============================================================
# Training
# ============================================================

.PHONY: train-sudoku
train-sudoku: ## Train TRM-MLP on Sudoku-Extreme
	$(PYTHON) main.py --mode train --config configs/trm_sudoku.yaml

.PHONY: train-maze
train-maze: ## Train TRM-Att on Maze-Hard
	$(PYTHON) main.py --mode train --config configs/trm_maze.yaml

.PHONY: train-llm
train-llm: ## Fine-tune GPT-2 baseline with LoRA
	$(PYTHON) main.py --mode train --config configs/llm_config.yaml

.PHONY: train-llm-qwen
train-llm-qwen: ## Fine-tune Qwen2.5-0.5B with LoRA
	$(PYTHON) main.py --mode train --config configs/llm_qwen.yaml

.PHONY: train-llm-smollm
train-llm-smollm: ## Fine-tune SmolLM2-360M with LoRA
	$(PYTHON) main.py --mode train --config configs/llm_smollm.yaml

.PHONY: train-llm-llama
train-llm-llama: ## Fine-tune Llama-3.2-1B with LoRA
	$(PYTHON) main.py --mode train --config configs/llm_llama.yaml

.PHONY: train-llm-all
train-llm-all: train-llm train-llm-qwen train-llm-smollm train-llm-llama ## Fine-tune all 4 LLM baselines sequentially

.PHONY: train-distill
train-distill: ## Knowledge distillation (requires trained teacher)
	$(PYTHON) main.py --mode distill --config configs/llm_config.yaml --checkpoint models/gpt2_latest.pt

.PHONY: train-maze-fast
train-maze-fast: ## Train TRM-Maze with augmented data + fewer epochs (~1-2 days)
	$(PYTHON) main.py --mode train --config configs/trm_maze_fast.yaml

.PHONY: resume-sudoku
resume-sudoku: ## Resume TRM-Sudoku training from last checkpoint
	$(PYTHON) main.py --mode train --config configs/trm_sudoku.yaml --resume models/latest.pt

.PHONY: resume-maze
resume-maze: ## Resume TRM-Maze training from last checkpoint
	$(PYTHON) main.py --mode train --config configs/trm_maze.yaml --resume models/latest.pt

# ============================================================
# Evaluation
# ============================================================

.PHONY: eval-sudoku
eval-sudoku: ## Evaluate best TRM-Sudoku checkpoint
	$(PYTHON) main.py --mode eval --config configs/trm_sudoku.yaml --checkpoint models/best.pt

.PHONY: eval-maze
eval-maze: ## Evaluate best TRM-Maze checkpoint
	$(PYTHON) main.py --mode eval --config configs/trm_maze.yaml --checkpoint models/best.pt

.PHONY: eval-llm
eval-llm: ## Evaluate GPT-2 baseline
	$(PYTHON) main.py --mode eval --config configs/llm_config.yaml --checkpoint models/gpt2_latest.pt

.PHONY: eval-llm-qwen
eval-llm-qwen: ## Evaluate Qwen2.5-0.5B baseline
	$(PYTHON) main.py --mode eval --config configs/llm_qwen.yaml --checkpoint models/qwen2.5_0.5b_latest.pt

.PHONY: eval-llm-smollm
eval-llm-smollm: ## Evaluate SmolLM2-360M baseline
	$(PYTHON) main.py --mode eval --config configs/llm_smollm.yaml --checkpoint models/smollm2_360m_latest.pt

.PHONY: eval-llm-llama
eval-llm-llama: ## Evaluate Llama-3.2-1B baseline
	$(PYTHON) main.py --mode eval --config configs/llm_llama.yaml --checkpoint models/llama_3.2_1b_latest.pt

# ============================================================
# Verification & Testing
# ============================================================

.PHONY: verify
verify: ## Run full import and forward pass verification
	$(PYTHON) -c "\
	import torch; \
	from src.utils.config import load_config; \
	from src.models.trm_sudoku import TRMSudoku, TRMMaze; \
	from src.models.recursion import deep_recursion; \
	from src.models.distilled_llm import DistilledLLM; \
	print('=== Import Check ==='); \
	config = load_config('configs/trm_sudoku.yaml'); \
	print('Config OK:', config.model.model_type.value); \
	ms = TRMSudoku(); \
	print(f'TRM-Sudoku: {ms.param_count():,} params'); \
	mm = TRMMaze(); \
	print(f'TRM-Maze:   {mm.param_count():,} params'); \
	ds = DistilledLLM(); \
	print(f'DistilledLLM: {ds.param_count():,} params'); \
	print('=== Forward Pass ==='); \
	x = torch.randint(1, 11, (2, 81)); \
	emb = ms.embedding(x); \
	y = ms.y_init.expand(2,-1,-1).clone(); \
	z = ms.z_init.expand(2,-1,-1).clone(); \
	(y,z), logits, q = deep_recursion(ms.block, ms.output_head, ms.q_head, emb, y, z, n=2, T=2); \
	print(f'logits: {logits.shape} | q: {q.shape}'); \
	print('=== All OK ==='); \
	"

.PHONY: verify-data
verify-data: ## Verify data scripts import correctly
	cd data && ../$(PYTHON) -c "from common import PuzzleDatasetMetadata, dihedral_transform; print('data/common.py OK')"
	cd data && ../$(PYTHON) build_sudoku_dataset.py --help > /dev/null && echo "build_sudoku_dataset.py OK"
	cd data && ../$(PYTHON) build_maze_dataset.py --help > /dev/null && echo "build_maze_dataset.py OK"

.PHONY: smoke-test
smoke-test: data-sudoku-small ## Quick end-to-end: preprocess small subset + 2 epoch train
	$(PYTHON) -c "\
	import torch; \
	from src.utils.config import load_config; \
	from src.utils.seed import set_seed; \
	from src.models.trm_sudoku import TRMSudoku; \
	from src.models.layers import StableMaxCrossEntropy; \
	from src.models.recursion import deep_supervision_step; \
	from src.data.sudoku_dataset import get_sudoku_loaders; \
	set_seed(42); \
	config = load_config('configs/trm_sudoku.yaml'); \
	model = TRMSudoku(); \
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1.0); \
	loss_fn = StableMaxCrossEntropy(ignore_index=0); \
	train_loader, _ = get_sudoku_loaders(config.data.data_dir, batch_size=4, num_workers=0); \
	print('Training 2 batches...'); \
	for i, (inp, lab) in enumerate(train_loader): \
	    metrics = deep_supervision_step(model, inp, lab, loss_fn, optimizer, n=2, T=2, N_sup=2); \
	    print(f'  batch {i+1}: ce={metrics[\"ce_loss\"]:.4f} q={metrics[\"q_mean\"]:.3f}'); \
	    if i >= 1: break; \
	print('Smoke test PASSED'); \
	"

# ============================================================
# Utilities
# ============================================================

.PHONY: clean
clean: ## Remove generated data, checkpoints, and experiments
	rm -rf data/sudoku-extreme-full data/maze-30x30-hard-1k
	rm -rf models/*.pt experiments/*.csv experiments/*.json
	rm -rf results/*.json

.PHONY: clean-all
clean-all: clean ## Clean everything including venv
	rm -rf .venv

.PHONY: lint
lint: ## Run linting
	$(PYTHON) -m flake8 src/ main.py --max-line-length 120 --ignore E501,W503

.PHONY: format
format: ## Auto-format code
	$(PYTHON) -m black src/ main.py --line-length 120
	$(PYTHON) -m isort src/ main.py

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
