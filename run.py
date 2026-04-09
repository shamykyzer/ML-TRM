"""
TRM Project Task Runner (cross-platform: Windows + Linux)
Usage: python run.py <target>
       python run.py help
       python run.py pipeline          # setup + data + train everything
       python run.py pipeline-sudoku   # setup + data + train sudoku only
"""

import subprocess
import sys
import os
import platform

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# Use a shorter venv path on Windows to avoid MAX_PATH issues during pip install.
if platform.system() == "Windows":
    VENV_DIR = os.environ.get("TRM_VENV_DIR", os.path.join(os.path.expanduser("~"), ".venvs", "ml-trm"))
else:
    VENV_DIR = os.environ.get("TRM_VENV_DIR", os.path.join(ROOT, ".venv"))

# --- Detect venv python/pip ---
if platform.system() == "Windows":
    PYTHON = os.path.join(VENV_DIR, "Scripts", "python.exe")
    PIP = os.path.join(VENV_DIR, "Scripts", "pip.exe")
else:
    PYTHON = os.path.join(VENV_DIR, "bin", "python")
    PIP = os.path.join(VENV_DIR, "bin", "pip")


def run(cmd, cwd=None):
    """Run a command, stream output, abort on failure."""
    if platform.system() == "Windows":
        # Quote known executables so shell=True works with spaces in paths.
        for exe in (sys.executable, PYTHON, PIP):
            quoted = f'"{exe}"'
            if exe in cmd and quoted not in cmd:
                cmd = cmd.replace(exe, quoted)
    print(f"\n>>> {cmd}\n")
    result = subprocess.run(cmd, shell=True, cwd=cwd or ROOT)
    if result.returncode != 0:
        print(f"\n!!! Command failed (exit {result.returncode}): {cmd}")
        sys.exit(result.returncode)


# ============================================================
# Setup
# ============================================================

def setup():
    """Create venv and install all dependencies (CPU)."""
    os.makedirs(os.path.dirname(VENV_DIR), exist_ok=True)
    run(f"{sys.executable} -m venv {VENV_DIR}")
    run(f"{PYTHON} -m pip install --upgrade pip")
    run(f"{PIP} install -r requirements.txt")

def setup_cuda():
    """Create venv and install with CUDA 12.8 GPU support (supports RTX 50-series Blackwell)."""
    os.makedirs(os.path.dirname(VENV_DIR), exist_ok=True)
    run(f"{sys.executable} -m venv {VENV_DIR}")
    run(f"{PYTHON} -m pip install --upgrade pip")
    run(f"{PIP} install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    run(f"{PIP} install -r requirements.txt")
    run(f'{PYTHON} -c "import torch; print(f\'CUDA: {{torch.cuda.is_available()}}, GPU: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}}\')"')

def setup_dev():
    """Setup + install dev tools."""
    setup()
    run(f"{PIP} install pytest black isort flake8")


# ============================================================
# Data Preprocessing
# ============================================================

def data_sudoku():
    """Download and preprocess Sudoku-Extreme (1K train, augmentation is on-the-fly)."""
    run(f"{PYTHON} build_sudoku_dataset.py --output-dir sudoku-extreme-full --subsample-size 1000", cwd=os.path.join(ROOT, "data"))

def data_sudoku_aug():
    """Preprocess Sudoku with 1000x pre-generated augmentation (not needed, on-the-fly is used)."""
    run(f"{PYTHON} build_sudoku_dataset.py --output-dir sudoku-extreme-full --num-aug 1000", cwd=os.path.join(ROOT, "data"))

def data_sudoku_small():
    """Preprocess small 100-sample Sudoku subset (for quick testing only)."""
    run(f"{PYTHON} build_sudoku_dataset.py --output-dir sudoku-extreme-full --subsample-size 100", cwd=os.path.join(ROOT, "data"))

def data_maze():
    """Download and preprocess full Maze-Hard dataset."""
    run(f"{PYTHON} build_maze_dataset.py --output-dir maze-30x30-hard-1k", cwd=os.path.join(ROOT, "data"))

def data_maze_aug():
    """Preprocess Maze with dihedral augmentation."""
    run(f"{PYTHON} build_maze_dataset.py --output-dir maze-30x30-hard-1k --aug", cwd=os.path.join(ROOT, "data"))

def data_all():
    """Preprocess both datasets (no augmentation)."""
    data_sudoku()
    data_maze()


# ============================================================
# Training
# ============================================================

def train_sudoku():
    """Train TRM-MLP on Sudoku-Extreme."""
    run(f"{PYTHON} main.py --mode train --config configs/trm_sudoku.yaml")

def train_maze():
    """Train TRM-Att on Maze-Hard."""
    run(f"{PYTHON} main.py --mode train --config configs/trm_maze.yaml")

def train_maze_fast():
    """Train TRM-Maze with augmented data + fewer epochs."""
    run(f"{PYTHON} main.py --mode train --config configs/trm_maze_fast.yaml")

def train_llm():
    """Fine-tune GPT-2 baseline with LoRA."""
    run(f"{PYTHON} main.py --mode train --config configs/llm_config.yaml")

def train_llm_qwen():
    """Fine-tune Qwen2.5-0.5B with LoRA."""
    run(f"{PYTHON} main.py --mode train --config configs/llm_qwen.yaml")

def train_llm_smollm():
    """Fine-tune SmolLM2-360M with LoRA."""
    run(f"{PYTHON} main.py --mode train --config configs/llm_smollm.yaml")

def train_llm_llama():
    """Fine-tune Llama-3.2-1B with LoRA."""
    run(f"{PYTHON} main.py --mode train --config configs/llm_llama.yaml")

def train_llm_all():
    """Fine-tune all 4 LLM baselines sequentially."""
    train_llm()
    train_llm_qwen()
    train_llm_smollm()
    train_llm_llama()

def train_distill():
    """Knowledge distillation (requires trained teacher)."""
    run(f"{PYTHON} main.py --mode distill --config configs/llm_config.yaml --checkpoint models/llm/gpt2_latest.pt")


# ============================================================
# Resume Training
# ============================================================

def resume_sudoku():
    """Resume TRM-Sudoku training from last checkpoint."""
    run(f"{PYTHON} main.py --mode train --config configs/trm_sudoku.yaml --resume models/sudoku/latest.pt")

def resume_maze():
    """Resume TRM-Maze training from last checkpoint."""
    run(f"{PYTHON} main.py --mode train --config configs/trm_maze.yaml --resume models/maze/latest.pt")


# ============================================================
# Evaluation
# ============================================================

def eval_sudoku():
    """Evaluate best TRM-Sudoku checkpoint."""
    run(f"{PYTHON} main.py --mode eval --config configs/trm_sudoku.yaml --checkpoint models/sudoku/best.pt")

def eval_maze():
    """Evaluate best TRM-Maze checkpoint."""
    run(f"{PYTHON} main.py --mode eval --config configs/trm_maze.yaml --checkpoint models/maze/best.pt")

def eval_llm():
    """Evaluate GPT-2 baseline."""
    run(f"{PYTHON} main.py --mode eval --config configs/llm_config.yaml --checkpoint models/llm/gpt2_latest.pt")

def eval_llm_qwen():
    """Evaluate Qwen2.5-0.5B baseline."""
    run(f"{PYTHON} main.py --mode eval --config configs/llm_qwen.yaml --checkpoint models/llm/qwen2.5_0.5b_latest.pt")

def eval_llm_smollm():
    """Evaluate SmolLM2-360M baseline."""
    run(f"{PYTHON} main.py --mode eval --config configs/llm_smollm.yaml --checkpoint models/llm/smollm2_360m_latest.pt")

def eval_llm_llama():
    """Evaluate Llama-3.2-1B baseline."""
    run(f"{PYTHON} main.py --mode eval --config configs/llm_llama.yaml --checkpoint models/llm/llama_3.2_1b_latest.pt")


# ============================================================
# Verification & Testing
# ============================================================

def verify():
    """Run full import and forward pass verification."""
    run(f'{PYTHON} -c "'
        "import torch; "
        "from src.utils.config import load_config; "
        "from src.models.trm_sudoku import TRMSudoku, TRMMaze; "
        "from src.models.recursion import deep_recursion; "
        "from src.models.distilled_llm import DistilledLLM; "
        "print('=== Import Check ==='); "
        "config = load_config('configs/trm_sudoku.yaml'); "
        "print('Config OK:', config.model.model_type.value); "
        "ms = TRMSudoku(); "
        "print(f'TRM-Sudoku: {ms.param_count():,} params'); "
        "mm = TRMMaze(); "
        "print(f'TRM-Maze:   {mm.param_count():,} params'); "
        "ds = DistilledLLM(); "
        "print(f'DistilledLLM: {ds.param_count():,} params'); "
        "print('=== Forward Pass ==='); "
        "x = torch.randint(1, 11, (2, 81)); "
        "emb = ms.embedding(x); "
        "y = ms.y_init.expand(2,-1,-1).clone(); "
        "z = ms.z_init.expand(2,-1,-1).clone(); "
        "(y,z), logits, q, _q_logits = deep_recursion(ms.block, ms.output_head, ms.q_head, emb, y, z, n=2, T=2); "
        "print(f'logits: {logits.shape} | q: {q.shape}'); "
        "print('=== All OK ==='); "
        '"')

def verify_data():
    """Verify data scripts import correctly."""
    run(f'{PYTHON} -c "from common import PuzzleDatasetMetadata, dihedral_transform; print(\'data/common.py OK\')"',
        cwd=os.path.join(ROOT, "data"))
    run(f"{PYTHON} build_sudoku_dataset.py --help", cwd=os.path.join(ROOT, "data"))
    run(f"{PYTHON} build_maze_dataset.py --help", cwd=os.path.join(ROOT, "data"))

def smoke_test():
    """Quick end-to-end: preprocess small subset + 2 batch train."""
    data_sudoku_small()
    run(f'{PYTHON} -c "'
        "import torch; "
        "from src.utils.config import load_config; "
        "from src.utils.seed import set_seed; "
        "from src.models.trm_sudoku import TRMSudoku; "
        "from src.models.layers import StableMaxCrossEntropy; "
        "from src.models.recursion import deep_supervision_step; "
        "from src.data.sudoku_dataset import get_sudoku_loaders; "
        "set_seed(42); "
        "config = load_config('configs/trm_sudoku.yaml'); "
        "model = TRMSudoku(); "
        "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1.0); "
        "loss_fn = StableMaxCrossEntropy(ignore_index=0); "
        "train_loader, _ = get_sudoku_loaders(config.data.data_dir, batch_size=4, num_workers=0); "
        "print('Training 2 batches...'); "
        "[None for i, (inp, lab) in enumerate(train_loader) if i < 2 for metrics in [deep_supervision_step(model, inp, lab, loss_fn, optimizer, n=2, T=2, N_sup=2)] for _ in [print(f'  batch {i+1}: ce={metrics[\"ce_loss\"]:.4f} q={metrics[\"q_mean\"]:.3f}')]]; "
        "print('Smoke test PASSED'); "
        '"')


# ============================================================
# Utilities
# ============================================================

def clean():
    """Remove generated data, checkpoints, and experiments."""
    import shutil
    import glob
    for d in ["data/sudoku-extreme-full", "data/maze-30x30-hard-1k",
              "models/sudoku", "models/maze", "models/llm",
              "experiments/sudoku", "experiments/maze", "experiments/llm"]:
        p = os.path.join(ROOT, d)
        if os.path.exists(p):
            shutil.rmtree(p)
            print(f"Removed {d}")
    for pattern in ["results/*.json"]:
        for f in glob.glob(os.path.join(ROOT, pattern)):
            os.remove(f)
            print(f"Removed {os.path.relpath(f, ROOT)}")
    print("Clean complete.")

def clean_all():
    """Clean everything including venv."""
    import shutil
    clean()
    if os.path.exists(VENV_DIR):
        shutil.rmtree(VENV_DIR)
        print(f"Removed venv at {VENV_DIR}")

def lint():
    """Run linting."""
    run(f"{PYTHON} -m flake8 src/ main.py --max-line-length 120 --ignore E501,W503")

def fmt():
    """Auto-format code."""
    run(f"{PYTHON} -m black src/ main.py --line-length 120")
    run(f"{PYTHON} -m isort src/ main.py")


# ============================================================
# Pipelines (fire-and-forget)
# ============================================================

def pipeline():
    """Full pipeline: setup CUDA + all data + all training."""
    setup_cuda()
    data_all()
    train_sudoku()
    train_maze()
    train_llm_all()
    print("\n=== PIPELINE COMPLETE ===")

def pipeline_sudoku():
    """Sudoku pipeline: setup CUDA + sudoku data + train sudoku."""
    setup_cuda()
    data_sudoku()
    train_sudoku()
    print("\n=== SUDOKU PIPELINE COMPLETE ===")

def pipeline_maze():
    """Maze pipeline: setup CUDA + maze data + train maze."""
    setup_cuda()
    data_maze()
    train_maze()
    print("\n=== MAZE PIPELINE COMPLETE ===")

def pipeline_llm():
    """LLM pipeline: setup CUDA + sudoku data + train all LLMs."""
    setup_cuda()
    data_sudoku()
    train_llm_all()
    print("\n=== LLM PIPELINE COMPLETE ===")


# ============================================================
# Dispatch
# ============================================================

TARGETS = {
    # Setup
    "setup":            setup,
    "setup-cuda":       setup_cuda,
    "setup-dev":        setup_dev,
    # Data
    "data-sudoku":      data_sudoku,
    "data-sudoku-aug":  data_sudoku_aug,
    "data-sudoku-small": data_sudoku_small,
    "data-maze":        data_maze,
    "data-maze-aug":    data_maze_aug,
    "data-all":         data_all,
    # Training
    "train-sudoku":     train_sudoku,
    "train-maze":       train_maze,
    "train-maze-fast":  train_maze_fast,
    "train-llm":        train_llm,
    "train-llm-qwen":   train_llm_qwen,
    "train-llm-smollm": train_llm_smollm,
    "train-llm-llama":  train_llm_llama,
    "train-llm-all":    train_llm_all,
    "train-distill":    train_distill,
    # Resume
    "resume-sudoku":    resume_sudoku,
    "resume-maze":      resume_maze,
    # Eval
    "eval-sudoku":      eval_sudoku,
    "eval-maze":        eval_maze,
    "eval-llm":         eval_llm,
    "eval-llm-qwen":    eval_llm_qwen,
    "eval-llm-smollm":  eval_llm_smollm,
    "eval-llm-llama":   eval_llm_llama,
    # Verification
    "verify":           verify,
    "verify-data":      verify_data,
    "smoke-test":       smoke_test,
    # Utilities
    "clean":            clean,
    "clean-all":        clean_all,
    "lint":             lint,
    "format":           fmt,
    # Pipelines
    "pipeline":         pipeline,
    "pipeline-sudoku":  pipeline_sudoku,
    "pipeline-maze":    pipeline_maze,
    "pipeline-llm":     pipeline_llm,
}

def show_help():
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    print(f"""
{BOLD}{CYAN}===============================================================
                   TRM Project Task Runner
==============================================================={RESET}

  {DIM}Usage:{RESET}  python run.py {CYAN}<command>{RESET}
""")

    groups = [
        ("Setup", [
            ("setup-cuda", "Create venv + install with CUDA GPU support"),
            ("setup", "Create venv + install (CPU only)"),
        ]),
        ("Data", [
            ("data-sudoku", "Download Sudoku-Extreme (1K train / 423K test)"),
            ("data-maze", "Download Maze-Hard (1K train / 1K test)"),
            ("data-all", "Download both datasets"),
        ]),
        ("Train", [
            ("train-sudoku", "TRM-MLP on Sudoku  (60K epochs, ~24h on 3070)"),
            ("train-maze", "TRM-Att on Maze    (5K epochs,  ~5d on 3070)"),
            ("train-llm-all", "All 4 LLM baselines sequentially (~4-6h)"),
            ("train-distill", "Knowledge distillation (needs trained GPT-2)"),
        ]),
        ("Resume", [
            ("resume-sudoku", "Resume sudoku from last checkpoint"),
            ("resume-maze", "Resume maze from last checkpoint"),
        ]),
        ("Evaluate", [
            ("eval-sudoku", "Evaluate best TRM-Sudoku checkpoint"),
            ("eval-maze", "Evaluate best TRM-Maze checkpoint"),
            ("eval-llm", "Evaluate GPT-2 baseline"),
        ]),
        ("Pipelines", [
            ("pipeline", "Full: setup + data + train everything"),
            ("pipeline-sudoku", "Setup + data + train sudoku only"),
            ("pipeline-maze", "Setup + data + train maze only"),
        ]),
        ("Utils", [
            ("verify", "Check imports + forward pass"),
            ("smoke-test", "Quick 2-batch end-to-end test"),
            ("clean", "Remove data, checkpoints, experiments"),
        ]),
    ]

    for group_name, items in groups:
        print(f"  {BOLD}{CYAN}{group_name}{RESET}")
        for cmd, desc in items:
            print(f"    {CYAN}{cmd:<20s}{RESET} {desc}")
        print()

    print(f"  {DIM}Tip: run 'bash scripts/auto_push.sh' in a separate terminal")
    print(f"  to auto-push training stats to GitHub every hour.{RESET}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("help", "--help", "-h"):
        show_help()
        sys.exit(0)

    target = sys.argv[1].replace("_", "-")

    if target not in TARGETS:
        print(f"Unknown target: {target}")
        print("Run 'python run.py help' for available targets.")
        sys.exit(1)

    TARGETS[target]()
