"""Path constants for ML-TRM. Stdlib-only, zero dependencies.

Resolves ROOT (project root), venv paths, and HF checkpoint locations.
Also performs os.chdir(ROOT) at import time so the rest of the code can
use relative paths without caring where it was invoked from. This
mirrors the behaviour of the original start.py module-level chdir.
"""
import os
import platform

# ROOT: the project root. paths.py is at <ROOT>/src/cli/paths.py (three
# levels up from __file__). Byte-identical to what start.py computed
# pre-refactor (where start.py lived at <ROOT>/start.py, one level up).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

# --- Venv path detection ---
# Windows uses a shorter path to avoid MAX_PATH issues during pip install.
if platform.system() == "Windows":
    VENV_DIR = os.environ.get(
        "TRM_VENV_DIR",
        os.path.join(os.path.expanduser("~"), ".venvs", "ml-trm"),
    )
    PYTHON = os.path.join(VENV_DIR, "Scripts", "python.exe")
    PIP = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    WANDB = os.path.join(VENV_DIR, "Scripts", "wandb.exe")
    ACTIVATE_HINT = f'"{os.path.join(VENV_DIR, "Scripts", "activate")}"'
else:
    VENV_DIR = os.environ.get("TRM_VENV_DIR", os.path.join(ROOT, ".venv"))
    PYTHON = os.path.join(VENV_DIR, "bin", "python")
    PIP = os.path.join(VENV_DIR, "bin", "pip")
    WANDB = os.path.join(VENV_DIR, "bin", "wandb")
    ACTIVATE_HINT = f"source {os.path.join(VENV_DIR, 'bin', 'activate')}"

# Marker file written into the venv after a successful sync, containing the
# SHA-1 of requirements.txt at sync time. If the file's current hash differs
# (or the marker is absent), the `sync` stage re-runs pip install.
REQUIREMENTS_HASH_FILE = os.path.join(VENV_DIR, ".trm_requirements_hash")
REQUIREMENTS_TXT = os.path.join(ROOT, "requirements.txt")

# Transfer-learning artifacts: the HF reference TRM checkpoint (trained on
# ARC-AGI-2) can be remapped into our local TRMOfficial shape and used as
# `--init-weights` to transfer the ~99.8% reasoning core. This whole pipeline
# is optional — only runs when the source file is present on disk. See
# scripts/remap_hf_checkpoint.py for the full rationale.
HF_SOURCE_CKPT = os.path.join(ROOT, "hf_checkpoints", "ARC", "step_723914")
HF_REMAPPED_CKPT = os.path.join(ROOT, "hf_checkpoints", "ARC", "remapped_for_local.pt")
REMAP_SCRIPT = os.path.join(ROOT, "scripts", "remap_hf_checkpoint.py")
VERIFY_SCRIPT = os.path.join(ROOT, "scripts", "verify_remap_loads.py")

# Per-task remapped HF checkpoints used by the seed-variance fine-tune plan.
# Each of these is a paper-faithful reference checkpoint that the trainer
# loads via --init-weights to start close to the published accuracy rather
# than training from scratch on RTX-5070 compute (which would take months).
HF_REMAPPED_SUDOKU_MLP = os.path.join(ROOT, "hf_checkpoints", "Sudoku-Extreme-mlp", "remapped_for_local.pt")
HF_REMAPPED_SUDOKU_ATT = os.path.join(ROOT, "hf_checkpoints", "Sudoku-Extreme-att", "remapped_for_local.pt")
HF_REMAPPED_MAZE = os.path.join(ROOT, "hf_checkpoints", "Maze-Hard", "remapped_for_local.pt")
