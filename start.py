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
from typing import Callable, List, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
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

# The six-seed-per-task fleet plan: machine index (1..6) -> (task, seed).
# Machines 1-2 run sudoku-mlp seeds 0-1, machines 3-4 run maze seeds 0-1,
# machine 5 runs the Qwen LLM baseline for the proposal's three-way
# comparison, machine 6 is the spare (or a second LLM run). Each machine
# takes one task AND one seed so per-seed output dirs are unambiguous.
FLEET_PLAN: List[Tuple[int, str, int]] = [
    (1, "sudoku-mlp", 0),
    (2, "sudoku-mlp", 1),
    (3, "maze", 0),
    (4, "maze", 1),
    (5, "llm-sudoku", 0),
    (6, "sudoku-mlp", 2),
]

# Where per-seed outputs land. MUST be a local non-OneDrive path: the 6-box
# fleet shares one OneDrive tree for code + data + HF weights (good), but
# parallel training writes inside the sync folder would corrupt checkpoints
# during upload. USERPROFILE / $HOME is outside OneDrive on a standard
# install, so that's the default. Override with the TRM_WORK_DIR env var.
DEFAULT_WORK_DIR = os.path.join(
    os.environ.get("USERPROFILE") or os.path.expanduser("~"),
    "ml-trm-work",
)

# Task dispatch table — single source of truth for the 4 fine-tuneable tasks.
# Used by both the interactive launcher and the printed copy-paste commands.
# An empty init string means "random init" (only llm-sudoku, which loads its
# own HF weights via transformers). sudoku-att and sudoku-mlp share one model
# type enum but different YAML configs — trm_official_sudoku.yaml is the
# attention variant (mlp_t=false), trm_official_sudoku_mlp.yaml is MLP-t.
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
    "llm-sudoku": (
        "configs/llm_qwen.yaml",
        "",
        "Qwen2.5-0.5B LoRA on Sudoku",
    ),
}


# ANSI color codes (best-effort; Windows terminals >= Win10 support these)
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _run(cmd: List[str], cwd: str = None) -> None:
    """Run a subprocess, stream output, abort on failure."""
    print(f"\n{DIM}>>> {' '.join(cmd)}{RESET}\n")
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    if result.returncode != 0:
        print(f"\n{YELLOW}!!! Command failed (exit {result.returncode}){RESET}")
        sys.exit(result.returncode)


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


def _wandb_instructions() -> None:
    """Print wandb login instructions and exit (user takes action manually).

    If the user's current shell has the venv activated, `wandb` will be on
    PATH and we show the short form. Otherwise we hand them the full path
    to the venv's wandb.exe — that works from any shell without activation,
    which is the common stumble point on Windows/PowerShell.
    """
    print(f"\n{YELLOW}⚠  wandb not authed.{RESET} Your configs have use_wandb=true.")
    print(f"{DIM}   (Training will still work — wandb tracking just gets disabled.){RESET}")

    on_path = shutil.which("wandb") is not None
    venv_wandb_exists = os.path.exists(WANDB)

    print(f"\n{BOLD}→ To enable wandb:{RESET}")
    if on_path:
        print(f"   {CYAN}wandb login{RESET}           # paste your API key when prompted")
    elif venv_wandb_exists:
        print(f"   {DIM}(venv not activated in this shell — `wandb` isn't on PATH){RESET}")
        print(f"   {CYAN}\"{WANDB}\" login{RESET}")
        print(f"   {DIM}…or, equivalently (works from any shell):{RESET}")
        print(f"   {CYAN}\"{PYTHON}\" -m wandb login{RESET}")
        print(f"   {DIM}…or activate the venv first, then use the short form:{RESET}")
        print(f"   {CYAN}{ACTIVATE_HINT}{RESET}")
        print(f"   {CYAN}wandb login{RESET}")
    else:
        print(f"   {CYAN}wandb login{RESET}           # paste your API key when prompted")
        print(f"   {DIM}(make sure your venv is activated first){RESET}")
    print(f"   {CYAN}python start.py{RESET}       # continue")

    print(f"\n{BOLD}→ To skip wandb and continue anyway:{RESET}")
    print(f"   {CYAN}python start.py --skip-wandb{RESET}")


def _setup_transfer() -> None:
    """Remap the HF reference checkpoint and verify it loads cleanly.

    Idempotent: always regenerates the remapped file when this stage fires,
    on the theory that if we got here something was either missing or broken,
    and rebuilding from source is cheaper than trying to diagnose which.
    Both sub-scripts are streamed so the user sees the full transfer report
    and the pass/fail verdict inline — that's the whole point of automating
    this (no more eyeballing trainer startup logs to catch a silent breakage).
    """
    if not os.path.exists(HF_SOURCE_CKPT):
        # User isn't doing transfer learning — nothing to do.
        print(f"{DIM}No source checkpoint at {HF_SOURCE_CKPT} — skipping.{RESET}")
        return
    print(f"{CYAN}Remapping HF reference checkpoint → local TRMOfficial shape...{RESET}")
    _run([PYTHON, REMAP_SCRIPT])
    print(f"\n{CYAN}Verifying remapped checkpoint loads into a fresh TRMOfficial...{RESET}")
    _run([PYTHON, VERIFY_SCRIPT])
    print(f"\n{GREEN}✓ Transfer-learning init weights verified and ready.{RESET}")


def _bootstrap_data() -> None:
    """Download whichever datasets are missing."""
    data_dir = os.path.join(ROOT, "data")
    sudoku_ok = os.path.exists(os.path.join(data_dir, "sudoku-extreme-full/train/all__inputs.npy"))
    maze_ok = os.path.exists(os.path.join(data_dir, "maze-30x30-hard-1k/train/all__inputs.npy"))

    if not sudoku_ok:
        print(f"{CYAN}Downloading Sudoku dataset...{RESET}")
        _run([
            PYTHON, "build_sudoku_dataset.py",
            "--output-dir", "sudoku-extreme-full",
            "--subsample-size", "1000",
        ], cwd=data_dir)

    if not maze_ok:
        print(f"{CYAN}Downloading Maze dataset...{RESET}")
        _run([
            PYTHON, "build_maze_dataset.py",
            "--output-dir", "maze-30x30-hard-1k",
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


def _wandb_ready() -> bool:
    """WANDB_API_KEY env var set, OR a netrc file has wandb credentials.

    Checks both `~/.netrc` (POSIX convention) and `~/_netrc` (Windows
    convention — what the wandb CLI actually writes on Windows). Either
    one counts as ready.
    """
    if os.getenv("WANDB_API_KEY"):
        return True
    home = os.path.expanduser("~")
    for name in (".netrc", "_netrc"):
        path = os.path.join(home, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                if "api.wandb.ai" in f.read():
                    return True
        except OSError:
            continue
    return False


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
        and os.path.exists(os.path.join(ROOT, "data/maze-30x30-hard-1k/train/all__inputs.npy"))
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


def _resolve_work_dir() -> str:
    """Resolve TRM_WORK_DIR, exit loudly if it points inside OneDrive.

    The single most important hygiene check in the 6-box fleet. Every machine
    syncs code + data + HF weights via OneDrive, but writing training outputs
    into the sync folder would corrupt checkpoints mid-run as OneDrive uploads
    partial tensor files. The warning in src/utils/config.py is a belt; this
    is the suspenders — we refuse to launch if the path looks OneDrive-ish.
    """
    work_dir = os.environ.get("TRM_WORK_DIR") or DEFAULT_WORK_DIR
    if "onedrive" in work_dir.lower():
        print(f"\n{YELLOW}!!! TRM_WORK_DIR='{work_dir}' looks like a OneDrive path.{RESET}")
        print(f"{YELLOW}    Parallel training on shared OneDrive will corrupt checkpoints.{RESET}")
        print(f"{DIM}    Pick a local path and re-run:{RESET}")
        if platform.system() == "Windows":
            print(f"      {CYAN}$env:TRM_WORK_DIR = 'C:/ml-trm-work'{RESET}")
        else:
            print(f"      {CYAN}export TRM_WORK_DIR=$HOME/ml-trm-work{RESET}")
        sys.exit(3)
    return work_dir


def _prompt(msg: str, default: str = "") -> str:
    """Prompt user on stdin, return default on empty input or EOF."""
    suffix = f" [{default}]" if default else ""
    try:
        reply = input(f"{CYAN}{msg}{suffix}: {RESET}").strip()
    except EOFError:
        reply = ""
    return reply or default


def _prompt_task_and_seed() -> Tuple[str, int]:
    """Interactive picker for task label and seed int.

    Shows the 4 tasks as a numbered menu, flags options whose HF init file is
    missing (such runs still work but start from random init, so they are
    exploratory rather than paper-faithful). Defaults to seed 0 — the first
    row of FLEET_PLAN, a safe starter on any machine.
    """
    tasks = list(TASK_DISPATCH.keys())
    print(f"\n{BOLD}Which task?{RESET}")
    for i, t in enumerate(tasks, 1):
        _, init, desc = TASK_DISPATCH[t]
        if init and not os.path.exists(init):
            suffix = f"  {YELLOW}(HF init missing — will use random init){RESET}"
        else:
            suffix = ""
        print(f"  {CYAN}{i}{RESET}) {t:<12s}  {DIM}{desc}{RESET}{suffix}")

    choice = _prompt("Pick 1-4", default="1")
    try:
        task = tasks[int(choice) - 1]
    except (ValueError, IndexError):
        print(f"{YELLOW}!!! Invalid task choice '{choice}'.{RESET}")
        sys.exit(2)

    seed_str = _prompt("Seed (non-negative int)", default="0")
    try:
        seed = int(seed_str)
        if seed < 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}!!! Seed must be a non-negative integer, got '{seed_str}'.{RESET}")
        sys.exit(2)

    return task, seed


def _dispatch_training(task: str, seed: int, dry_run: bool = False) -> None:
    """Build main.py argv and exec, routing checkpoints to a per-seed local dir.

    Python twin of `scripts/run_seed.sh <task> <seed>`. Sets TRM_CHECKPOINT_DIR
    and TRM_EXPERIMENT_DIR in the child env so the trainer writes to
    <TRM_WORK_DIR>/<task>-seed<N>/, never into the OneDrive-synced repo. Shell
    launchers are kept for automation (cron, CI); this path is for users who
    drive everything from start.py interactively.
    """
    config, init, description = TASK_DISPATCH[task]

    work_dir = _resolve_work_dir()
    task_dir = os.path.join(work_dir, f"{task}-seed{seed}")
    os.makedirs(task_dir, exist_ok=True)

    env = os.environ.copy()
    env["TRM_CHECKPOINT_DIR"] = task_dir
    env["TRM_EXPERIMENT_DIR"] = task_dir

    args = [PYTHON, "main.py", "--mode", "train", "--config", config, "--seed", str(seed)]
    if init and os.path.exists(init):
        args.extend(["--init-weights", init])
    if dry_run:
        args.extend(["--epochs", "5"])

    bar = "=" * 64
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"  task               : {CYAN}{task}{RESET}  {DIM}({description}){RESET}")
    print(f"  seed               : {CYAN}{seed}{RESET}")
    print(f"  config             : {config}")
    init_label = init if (init and os.path.exists(init)) else "<none, random init>"
    print(f"  init_weights       : {init_label}")
    print(f"  TRM_CHECKPOINT_DIR : {task_dir}")
    print(f"  TRM_EXPERIMENT_DIR : {task_dir}")
    print(f"  dry-run            : {'YES (5 epochs)' if dry_run else 'no'}")
    print(f"{BOLD}{bar}{RESET}\n")

    result = subprocess.run(args, env=env)
    sys.exit(result.returncode)


def _interactive_launcher() -> None:
    """Prompt the user to pick an action after all setup stages are ready.

    Main entry for the 6-box workflow: re-run `python start.py` per seed,
    pick a task + seed from the menu, train. Option 3 is the Regime A
    one-shot (no training); option 4 dumps the full copy-paste command
    list for scripted workflows.
    """
    print(f"\n{BOLD}What do you want to run?{RESET}")
    print(f"  {CYAN}1{RESET}) Dry run         {DIM}(5-epoch pipeline smoke test — always do this first){RESET}")
    print(f"  {CYAN}2{RESET}) Seed-variance   {DIM}(full fine-tune from HF init — the 6-machine plan){RESET}")
    print(f"  {CYAN}3{RESET}) Evaluate HF     {DIM}(Regime A — all 3 paper checkpoints, no training){RESET}")
    print(f"  {CYAN}4{RESET}) Show commands   {DIM}(print copy-paste commands and exit){RESET}")
    print(f"  {CYAN}Q{RESET}) Quit")

    choice = _prompt("Pick", default="Q").upper()

    if choice in ("Q", ""):
        print(f"\n{DIM}Nothing launched. Re-run `python start.py` when ready.{RESET}\n")
        return

    if choice == "3":
        _run([PYTHON, os.path.join("scripts", "eval_hf_checkpoints.py")])
        return

    if choice == "4":
        _print_copy_paste_commands()
        return

    if choice in ("1", "2"):
        task, seed = _prompt_task_and_seed()
        _dispatch_training(task, seed, dry_run=(choice == "1"))
        return

    print(f"{YELLOW}!!! Unknown choice '{choice}'.{RESET}")


def _print_copy_paste_commands() -> None:
    """Print all manual training commands. Fallback for scripted workflows."""
    py_quoted = f'"{PYTHON}"'
    is_powershell = (
        platform.system() == "Windows"
        and os.environ.get("PSModulePath") is not None
    )
    py = f"& {py_quoted}" if is_powershell else py_quoted

    print(f"\n{BOLD}Regime A — Evaluate paper checkpoints (no training):{RESET}")
    print(f"  {CYAN}{py} scripts/eval_hf_checkpoints.py{RESET}\n")

    print(f"{BOLD}Regime B — Seed-variance fine-tune:{RESET} {DIM}(direct main.py){RESET}")
    for task, (config, init, desc) in TASK_DISPATCH.items():
        init_arg = f" --init-weights {init}" if init and os.path.exists(init) else ""
        print(f"  {CYAN}{py} main.py --mode train --config {config}{init_arg} --seed 0{RESET} {DIM}# {task}{RESET}")
    print()

    print(f"{BOLD}...or the shell launchers (they auto-set TRM_CHECKPOINT_DIR per seed):{RESET}")
    if platform.system() == "Windows":
        print(f"  {CYAN}scripts/run_seed.ps1 -Task sudoku-mlp -Seed 0{RESET}")
        print(f"  {CYAN}scripts/run_seed.ps1 -Task sudoku-mlp -Seed 0 -DryRun{RESET}  {DIM}# 5-epoch smoke{RESET}")
    else:
        print(f"  {CYAN}scripts/run_seed.sh sudoku-mlp 0{RESET}")
        print(f"  {CYAN}scripts/run_seed.sh sudoku-mlp 0 --dry-run{RESET}  {DIM}# 5-epoch smoke{RESET}")
    print()

    print(f"{BOLD}Other (non-fleet) TRM and LLM configs:{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/trm_sudoku.yaml --seed 0{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/trm_maze.yaml --seed 0{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_config.yaml --seed 0{RESET}  {DIM}# GPT-2{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_smollm.yaml --seed 0{RESET}  {DIM}# SmolLM2-360M{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_llama.yaml --seed 0{RESET}   {DIM}# Llama-3.2-1B{RESET}\n")

    print(f"{BOLD}Evaluate (after training):{RESET}")
    print(f"  {CYAN}{py} main.py --mode eval --config configs/<name>.yaml --checkpoint models/<name>/best.pt{RESET}\n")

    print(f"{BOLD}Resume from last checkpoint:{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/<name>.yaml --resume models/<name>/latest.pt --seed 0{RESET}\n")

    print(f"{BOLD}Reproducibility — seed convention:{RESET}")
    print(f"  {DIM}Every command above passes {RESET}{CYAN}--seed N{RESET}{DIM} explicitly so the seed shows up in{RESET}")
    print(f"  {DIM}shell history AND the wandb run name. Omitting --seed inherits the{RESET}")
    print(f"  {DIM}{RESET}{CYAN}seed:{RESET}{DIM} field from the YAML (default 42 — reproducible). Set {RESET}{CYAN}seed: -1{RESET}{DIM}{RESET}")
    print(f"  {DIM}in the YAML for wall-clock seeding.{RESET}\n")

    print(f"{DIM}Tip: run `python start.py status` any time to re-check stages.{RESET}")
    print(f"{DIM}Tip: Weave traces appear at wandb.ai/<entity>/<project>/weave/monitors{RESET}\n")


def _print_training_menu() -> None:
    print(f"\n{GREEN}{BOLD}✓ All setup complete!{RESET}\n")

    # 6-machine fleet assignment table. Each operator picks the row matching
    # their box — seeds 0..5 across tasks give the full seed-variance set the
    # coursework report needs for mean ± std on the three-way model comparison.
    print(f"{BOLD}6-Machine Fleet Plan:{RESET} {DIM}(pick the row for this box){RESET}")
    print(f"  {DIM}machine   task          seed{RESET}")
    for idx, task, seed in FLEET_PLAN:
        print(f"  {CYAN}{idx:<9}{RESET} {task:<13s} {CYAN}{seed}{RESET}")
    print()

    # TRM_WORK_DIR status + reminder. The launcher will exit(3) later if this
    # resolves inside OneDrive; this printout is the friendly early warning.
    print(f"{BOLD}Work dir for training outputs:{RESET}")
    current_work = os.environ.get("TRM_WORK_DIR")
    if current_work:
        work_label = f"{CYAN}{current_work}{RESET}"
    else:
        work_label = f"{CYAN}{DEFAULT_WORK_DIR}{RESET}  {DIM}(default — TRM_WORK_DIR not set){RESET}"
    print(f"  TRM_WORK_DIR = {work_label}")
    if platform.system() == "Windows":
        print(f"  {DIM}To set for this shell: {RESET}{CYAN}$env:TRM_WORK_DIR = 'C:/ml-trm-work'{RESET}")
    else:
        print(f"  {DIM}To set for this shell: {RESET}{CYAN}export TRM_WORK_DIR=$HOME/ml-trm-work{RESET}")
    print(f"  {DIM}(MUST be a local non-OneDrive path — parallel runs on the shared{RESET}")
    print(f"  {DIM} OneDrive would corrupt each machine's checkpoints during upload.){RESET}\n")

    # wandb reminder — re-check here even though the setup stage covered it,
    # because it's common to lose the netrc entry when switching machines.
    if _wandb_ready():
        print(f"{BOLD}wandb:{RESET} {GREEN}✓ logged in{RESET}  {DIM}(runs will track to your wandb project){RESET}\n")
    else:
        print(f"{BOLD}wandb:{RESET} {YELLOW}not logged in{RESET}  {DIM}(runs still train, just without cloud tracking){RESET}")
        if os.path.exists(WANDB):
            print(f"  {CYAN}\"{WANDB}\" login{RESET}  {DIM}# paste your key from wandb.ai/authorize{RESET}\n")
        else:
            print(f"  {CYAN}wandb login{RESET}  {DIM}# paste your key from wandb.ai/authorize{RESET}\n")

    # Hand off to the interactive launcher. If the user picks a task, it
    # dispatches via subprocess.run + sys.exit — we never return here. If
    # they quit or pick "show commands", we fall through and exit naturally.
    _interactive_launcher()


# ============================================================
# Entry points
# ============================================================

def main() -> None:
    args = sys.argv[1:]
    skip_wandb = "--skip-wandb" in args
    args = [a for a in args if a != "--skip-wandb"]

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
