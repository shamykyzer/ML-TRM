#!/usr/bin/env python
"""Stage-aware onboarding for ML-TRM. Runs on system Python, no third-party deps.

One command, one stage at a time. Each invocation runs the next missing
setup stage, prints "done" + what to run next, and exits. When every
stage is satisfied, prints a menu of training commands to copy-paste.

Usage:
    python start.py               # run the next missing setup stage
    python start.py status        # show stage status without running anything
    python start.py --skip-wandb  # skip the wandb auth stage (and continue)
"""
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# Path constants + os.chdir(ROOT) now live in src/cli/paths.py so they can
# be reused by the dashboard, tests, and any future cli module without
# pulling in the rest of bootstrap.py. Imported values are byte-identical
# to what this file computed pre-extraction.
from src.cli.paths import (
    ROOT,
    VENV_DIR, PYTHON, PIP, WANDB, ACTIVATE_HINT,
    REQUIREMENTS_HASH_FILE, REQUIREMENTS_TXT,
    HF_SOURCE_CKPT, HF_REMAPPED_CKPT, REMAP_SCRIPT, VERIFY_SCRIPT,
    HF_REMAPPED_SUDOKU_MLP, HF_REMAPPED_SUDOKU_ATT, HF_REMAPPED_MAZE,
)

# ANSI colors + stdout reconfigure moved to src/cli/console.py. Importing
# console runs reconfigure_stdout() at import time (same side-effect timing
# as before), so the unicode glyphs below print correctly on Windows cp1252.
from src.cli.console import CYAN, GREEN, YELLOW, DIM, BOLD, RESET  # noqa: F401,E402

# Training launchers moved to src/cli/launchers.py.
from src.cli.launchers import (  # noqa: E402,F401
    _run,
    _run_training_subprocess,
    _dispatch_training,
)

# Checkpoint discovery + run-dir introspection moved to src/cli/checkpoints.py.
from src.cli.checkpoints import (  # noqa: E402,F401
    RESUME_CONFIG_BY_PREFIX,
    _scan_for_checkpoints,
    _config_for_run_dir,
    _seed_for_run_dir,
)

# The six-seed fleet plan: machine index (1..6) -> (task, seed).
# Machines 1-3 run sudoku-att seeds 0-2; machines 4-6 run maze seeds 3-5.
# Three seeds per task gives mean ± std for the sudoku vs maze comparison.
# Each machine owns one (task, seed) pair so per-seed output dirs
# (<task>-seed<N>) are unambiguous.
FLEET_PLAN: List[Tuple[int, str, int]] = [
    (1, "sudoku-att", 0),
    (2, "sudoku-att", 1),
    (3, "sudoku-att", 2),
    (4, "maze", 3),
    (5, "maze", 4),
    (6, "maze", 5),
]

# _default_work_dir() and _resolve_work_dir() moved to src/cli/workdir.py
# along with _DEFAULT_WORK_DIR_CACHE. Imported below.
from src.cli.workdir import _default_work_dir  # noqa: F401,E402

# Task dispatch table — single source of truth for every fine-tuneable task.
# Used by both the interactive launcher and the printed copy-paste commands.
# An empty init string means "random init" (the LLM tasks load their own HF
# weights via transformers; only the TRM tasks accept a remapped HF init).
# sudoku-att and sudoku-mlp share one model type enum but different YAMLs —
# trm_official_sudoku.yaml is the attention variant (mlp_t=false),
# trm_official_sudoku_mlp.yaml is MLP-t.
#
# LLM coverage: every LLM × {sudoku, maze} so the coursework's three-way
# comparison (TRM vs fine-tuned LLM vs distilled student) has a full scaling
# sweep on both reasoning tasks. Sudoku and maze configs differ in vocab_size
# (11 vs 6), seq_len (81 vs 900), and per-LLM batch_size + grad_accum_steps.
TASK_DISPATCH = {
    "sudoku-mlp": (
        "configs/trm_official_sudoku_mlp.yaml",
        HF_REMAPPED_SUDOKU_MLP,
        "Sudoku-Extreme MLP-t (paper 84.80%)",
    ),
    "sudoku-att": (
        "configs/trm_official_sudoku.yaml",
        HF_REMAPPED_SUDOKU_ATT,
        "Sudoku-Extreme attention (paper 77.70%)",
    ),
    "maze": (
        "configs/trm_official_maze.yaml",
        HF_REMAPPED_MAZE,
        "Maze-Hard 30x30 (paper 85.30%)",
    ),
    "llm-gpt2-sudoku":   ("configs/llm_config.yaml",        "", "GPT-2 (124M) LoRA on Sudoku"),
    "llm-smollm-sudoku": ("configs/llm_smollm.yaml",        "", "SmolLM2-360M LoRA on Sudoku"),
    "llm-qwen-sudoku":   ("configs/llm_qwen.yaml",          "", "Qwen2.5-0.5B LoRA on Sudoku"),
    "llm-llama-sudoku":  ("configs/llm_llama.yaml",         "", "Llama-3.2-1B LoRA on Sudoku"),
    "llm-gpt2-maze":     ("configs/llm_gpt2_maze.yaml",     "", "GPT-2 (124M) LoRA on Maze"),
    "llm-smollm-maze":   ("configs/llm_smollm_maze.yaml",   "", "SmolLM2-360M LoRA on Maze"),
    "llm-qwen-maze":     ("configs/llm_qwen_maze.yaml",     "", "Qwen2.5-0.5B LoRA on Maze"),
    "llm-llama-maze":    ("configs/llm_llama_maze.yaml",    "", "Llama-3.2-1B LoRA on Maze"),
    "llm-deepseek-sudoku": ("configs/llm_deepseek.yaml",      "", "DeepSeek-R1-Distill-Qwen-1.5B LoRA on Sudoku"),
    "llm-deepseek-maze":   ("configs/llm_deepseek_maze.yaml", "", "DeepSeek-R1-Distill-Qwen-1.5B LoRA on Maze"),
}


# Per-rig LLM fleet assignments (Option B: model-family ownership).
# Each rig runs its assigned LLM family/families in longest-first order
# so the riskiest job (longest maze run) starts while the operator is
# fresh and can catch early-epoch OOM. Total wall-clock per rig is
# ~15-17 hr on RTX 5070 at 30 epochs. Qwen sudoku is intentionally
# absent — that baseline already completed in an earlier session.
#
# rig 1 -> Llama owner         (17 hr)
# rig 2 -> DeepSeek owner      (16 hr)
# rig 3 -> Small-fleet owner   (17 hr — GPT-2, SmolLM, Qwen-maze)
RIG_FLEET_PLAN: dict = {
    1: ["llm-llama-maze", "llm-llama-sudoku"],
    2: ["llm-deepseek-maze", "llm-deepseek-sudoku"],
    3: [
        "llm-qwen-maze",
        "llm-smollm-maze",
        "llm-gpt2-maze",
        "llm-smollm-sudoku",
        "llm-gpt2-sudoku",
    ],
}


def _resolve_rig() -> int:
    """Read TRM_RIG from env; prompt once if unset and persist to .env.

    Returns 1, 2, or 3 — the machine's fleet-assignment key per
    RIG_FLEET_PLAN. On first call (TRM_RIG unset) this prompts the
    operator and appends a ``TRM_RIG=<n>`` line to ``.env`` so the
    prompt fires at most once per machine. Exits with status 2 on
    invalid input rather than guessing.
    """
    raw = os.environ.get("TRM_RIG", "").strip()
    if raw:
        try:
            rig = int(raw)
        except ValueError:
            print(f"{YELLOW}!!! TRM_RIG='{raw}' is not an integer.{RESET}")
            sys.exit(2)
    else:
        print(f"\n{BOLD}Which rig is this?{RESET}  {DIM}(1=Llama, 2=DeepSeek, 3=small-fleet){RESET}")
        try:
            reply = input(f"{CYAN}TRM_RIG [1-3]: {RESET}").strip()
        except EOFError:
            print(f"{YELLOW}!!! No input received.{RESET}")
            sys.exit(2)
        try:
            rig = int(reply)
        except ValueError:
            print(f"{YELLOW}!!! Need an integer 1-3, got '{reply}'.{RESET}")
            sys.exit(2)
        # Persist so the prompt fires at most once per machine. Best-effort:
        # don't fail the launch just because .env was read-only or absent.
        env_path = os.path.join(ROOT, ".env")
        try:
            with open(env_path, "a", encoding="utf-8") as fh:
                fh.write(f"\n# Auto-added by start.py on first llm-fleet launch\n")
                fh.write(f"TRM_RIG={rig}\n")
            os.environ["TRM_RIG"] = str(rig)
        except OSError as exc:
            print(f"{YELLOW}!!! Could not write TRM_RIG to .env: {exc}{RESET}")

    if rig not in (1, 2, 3):
        print(f"{YELLOW}!!! TRM_RIG must be 1, 2, or 3 — got {rig}.{RESET}")
        sys.exit(2)

    return rig


# ANSI color constants now live in src/cli/console.py (imported at the
# top of this file). Kept this marker so the section ordering in this
# module mirrors the original start.py layout for git-blame continuity.

# ============================================================
# Direct-launch preflight (for `python start.py <task> <seed>`)
# Moved to src/cli/preflight.py. Re-imported here so main() below can
# keep calling the names unchanged. _run, _dispatch_training, etc. are
# already imported from launchers at the top of this module.
# ============================================================
from src.cli.preflight import (  # noqa: E402,F401
    _kill_training_processes,
    _preflight_relaunch,
)


# ============================================================
# Stage actions — what to do when a stage is not ready
# ============================================================

def _setup_venv() -> None:
    """Create venv and install CUDA torch + requirements."""
    os.makedirs(os.path.dirname(VENV_DIR), exist_ok=True)
    _run([sys.executable, "-m", "venv", VENV_DIR])
    _run([PYTHON, "-m", "pip", "install", "--upgrade", "pip"])
    _run([
        PIP, "install", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    ])
    _run([PIP, "install", "-r", "requirements.txt"])
    # Record the requirements.txt hash so the `sync` stage starts green —
    # otherwise we'd immediately re-run pip install after a fresh venv build.
    _write_requirements_hash()
    _run([
        PYTHON, "-c",
        "import torch; "
        "print(f'CUDA: {torch.cuda.is_available()}, "
        "GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')",
    ])


def _sync_venv() -> None:
    """Install/update packages so the venv matches requirements.txt."""
    print(f"{CYAN}requirements.txt changed (or venv was built externally) — syncing...{RESET}")
    _run([PIP, "install", "-r", "requirements.txt"])
    _write_requirements_hash()
    print(f"\n{GREEN}✓ Venv synced to requirements.txt{RESET}")


def _bootstrap_env() -> None:
    """Copy .env.example to .env and tell the user to edit it."""
    src = os.path.join(ROOT, ".env.example")
    dst = os.path.join(ROOT, ".env")
    if not os.path.exists(src):
        print(f"{YELLOW}!!! .env.example not found — can't bootstrap .env{RESET}")
        sys.exit(1)
    shutil.copy(src, dst)
    print(f"\n{GREEN}✓ Created .env from .env.example{RESET}")
    print(f"\n{BOLD}→ Edit .env to set at least:{RESET}")
    print(f"   {CYAN}WANDB_API_KEY{RESET}   (from https://wandb.ai/authorize)")
    print(f"\n{DIM}Then re-run:{RESET} python start.py")


# _bootstrap_wandb_from_file() and _wandb_instructions() moved to
# src/cli/wandb_bootstrap.py. Imported below.
from src.cli.wandb_bootstrap import (  # noqa: E402
    _bootstrap_wandb_from_file,
    _wandb_instructions,
)


# _setup_transfer() moved to src/cli/transfer.py.
from src.cli.transfer import _setup_transfer  # noqa: E402,F401


def _bootstrap_data() -> None:
    """Download whichever datasets are missing."""
    data_dir = os.path.join(ROOT, "data")
    sudoku_ok = os.path.exists(os.path.join(data_dir, "sudoku-extreme-full/train/all__inputs.npy"))
    maze_ok = os.path.exists(os.path.join(data_dir, "maze-30x30-hard-1k-aug/train/all__inputs.npy"))

    if not sudoku_ok:
        print(f"{CYAN}Downloading Sudoku dataset...{RESET}")
        _run([
            PYTHON, "build_sudoku_dataset.py",
            "--output-dir", "sudoku-extreme-full",
            "--subsample-size", "1000",
        ], cwd=data_dir)

    if not maze_ok:
        # `--aug` enables 8× dihedral augmentation on the train split (1000
        # → 8000 samples). Without it the maze train set saturates train
        # accuracy in a few hundred epochs and the learning curve is flat
        # for the rest of training. Test split is unaffected (aug only
        # applies to train in build_maze_dataset.py).
        print(f"{CYAN}Downloading Maze dataset (with 8x dihedral aug)...{RESET}")
        _run([
            PYTHON, "build_maze_dataset.py",
            "--output-dir", "maze-30x30-hard-1k-aug",
            "--aug",
        ], cwd=data_dir)


# ============================================================
# Stage checks — pure os-level probes, no third-party imports
# ============================================================

def _venv_ready() -> bool:
    """Venv exists AND its python can import torch."""
    if not os.path.exists(PYTHON):
        return False
    result = subprocess.run(
        [PYTHON, "-c", "import torch"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _requirements_hash() -> str:
    """SHA-1 of requirements.txt, or '' if the file is missing."""
    if not os.path.exists(REQUIREMENTS_TXT):
        return ""
    with open(REQUIREMENTS_TXT, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def _write_requirements_hash() -> None:
    """Write the current requirements.txt hash into the venv marker file."""
    os.makedirs(os.path.dirname(REQUIREMENTS_HASH_FILE), exist_ok=True)
    with open(REQUIREMENTS_HASH_FILE, "w") as f:
        f.write(_requirements_hash())


def _sync_ready() -> bool:
    """Venv's recorded requirements.txt hash matches the current file.

    The marker file is written whenever we successfully run pip install on
    requirements.txt. If the file gets edited (new package added, version
    bumped), the stored hash no longer matches and the sync stage fires.
    """
    if not os.path.exists(PYTHON):
        return False  # venv stage will handle this first
    if not os.path.exists(REQUIREMENTS_HASH_FILE):
        return False
    try:
        with open(REQUIREMENTS_HASH_FILE) as f:
            stored = f.read().strip()
    except OSError:
        return False
    return stored == _requirements_hash()


def _env_ready() -> bool:
    return os.path.exists(os.path.join(ROOT, ".env"))


# _WANDB_API_FILE, _read_wandb_api_file, and _wandb_ready moved to
# src/cli/wandb_bootstrap.py. Re-imported so the rest of this module
# (STAGES list in particular) can reference _wandb_ready unchanged.
from src.cli.wandb_bootstrap import (  # noqa: E402
    _WANDB_API_FILE,  # noqa: F401
    _read_wandb_api_file,  # noqa: F401
    _wandb_ready,
)


def _transfer_ready() -> bool:
    """Remapped HF checkpoint exists AND verifies cleanly — or source isn't present.

    Three outcomes:
      1. No source checkpoint on disk → N/A, return ready (most users).
      2. Source present, remapped file missing → not ready (build it).
      3. Both present → delegate to scripts/verify_remap_loads.py via the venv
         python. Exit code 0 means load report matched exactly: zero unexpected
         keys, and the only "missing" keys are the intentionally-skipped
         embed/task_emb/lm_head/rotary buffers. Anything else → not ready,
         force a rebuild.

    Stays silent (captures output) because this runs during the status
    display. The loud re-run happens in _setup_transfer if this returns False.
    """
    if not os.path.exists(HF_SOURCE_CKPT):
        return True  # N/A — user isn't doing transfer learning
    if not os.path.exists(HF_REMAPPED_CKPT):
        return False
    if not os.path.exists(PYTHON):
        return False  # venv stage handles this first; we'll re-check next run
    result = subprocess.run(
        [PYTHON, VERIFY_SCRIPT],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _data_ready() -> bool:
    return (
        os.path.exists(os.path.join(ROOT, "data/sudoku-extreme-full/train/all__inputs.npy"))
        and os.path.exists(os.path.join(ROOT, "data/maze-30x30-hard-1k-aug/train/all__inputs.npy"))
    )


# ============================================================
# Stage definition
# ============================================================

@dataclass
class Stage:
    key: str
    label: str
    check: Callable[[], bool]
    action: Callable[[], None]
    blocking: bool = True


STAGES: List[Stage] = [
    Stage("venv",     "Python venv with CUDA torch",    _venv_ready,     _setup_venv),
    Stage("sync",     "Venv matches requirements.txt",  _sync_ready,     _sync_venv),
    Stage("env",      "Machine-local .env file",        _env_ready,      _bootstrap_env),
    Stage("wandb",    "wandb logged in",                _wandb_ready,    _wandb_instructions, blocking=False),
    # Non-blocking because most users won't have the 2.47 GB source file —
    # _transfer_ready() returns True when the source is absent, so this stage
    # silently passes for anyone not doing ARC→local transfer learning.
    Stage("transfer", "HF reference remapped + verified", _transfer_ready, _setup_transfer, blocking=False),
    Stage("data",     "Sudoku + Maze datasets",         _data_ready,     _bootstrap_data),
]


# ============================================================
# Output helpers
# ============================================================

def _print_stage_status(results: List[Tuple[Stage, bool]], skip_wandb: bool) -> None:
    print(f"\n{BOLD}=== ML-TRM Setup Status ==={RESET}\n")
    for stage, done in results:
        if stage.key == "wandb" and skip_wandb and not done:
            mark, color, suffix = "~", DIM, " (skipped)"
        elif done:
            mark, color, suffix = "✓", GREEN, ""
        else:
            mark, color, suffix = " ", "", ""
        print(f"  [{color}{mark}{RESET}] {stage.key:<7s}  {stage.label}{suffix}")
    print()


# All of the interactive menu functions (_prompt, _prompt_task_and_seed,
# _resume_training_picker, _prompt_fresh_target_and_seed, _confirm_wipe_run_dir,
# _fresh_start_launcher, _interactive_launcher, _print_copy_paste_commands,
# _print_training_menu) plus the LLM_FAMILIES constant moved to src/cli/menus.py.
# Re-imported here so main() can call _print_training_menu unchanged.
from src.cli.menus import _print_training_menu  # noqa: E402,F401


# ============================================================
# Entry points
# ============================================================

def main() -> None:
    args = sys.argv[1:]
    skip_wandb = "--skip-wandb" in args
    args = [a for a in args if a != "--skip-wandb"]

    # Direct launch: `python start.py <task> <seed>` — run preflight (pull,
    # kill existing, back up best.pt) then hand off to _dispatch_training.
    # Bypasses stage checks and the interactive menu — assumes the machine
    # has already been through setup once.
    if len(args) >= 2 and args[0] in TASK_DISPATCH:
        task = args[0]
        try:
            seed = int(args[1])
            if seed < 0:
                raise ValueError
        except ValueError:
            print(f"{YELLOW}!!! seed must be a non-negative int, got '{args[1]}'.{RESET}")
            sys.exit(2)
        _preflight_relaunch(task, seed)
        _dispatch_training(task, seed)  # sys.exit's on completion
        return

    if args and args[0] == "status":
        results = [(s, s.check()) for s in STAGES]
        _print_stage_status(results, skip_wandb)
        missing = [s for s, done in results if not done and s.blocking]
        if not missing:
            print(f"{GREEN}All blocking stages ready.{RESET} Run: python start.py")
        else:
            print(f"{BOLD}Next stage:{RESET} {missing[0].key}. Run: python start.py")
        return

    results = [(s, s.check()) for s in STAGES]
    _print_stage_status(results, skip_wandb)

    for stage, done in results:
        if done:
            continue
        if stage.key == "wandb" and skip_wandb:
            continue
        print(f"{BOLD}[{stage.key}] {stage.label}{RESET} — not ready. Running...")
        stage.action()
        if stage.check():
            print(f"\n{GREEN}✓ [{stage.key}] ready.{RESET}")
            print(f"\n{BOLD}→ Next:{RESET} python start.py")
        return

    # All stages done
    _print_training_menu()


if __name__ == "__main__":
    main()
