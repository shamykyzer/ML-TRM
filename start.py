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
    Stage("venv",  "Python venv with CUDA torch",   _venv_ready,  _setup_venv),
    Stage("sync",  "Venv matches requirements.txt", _sync_ready,  _sync_venv),
    Stage("env",   "Machine-local .env file",       _env_ready,   _bootstrap_env),
    Stage("wandb", "wandb logged in",               _wandb_ready, _wandb_instructions, blocking=False),
    Stage("data",  "Sudoku + Maze datasets",        _data_ready,  _bootstrap_data),
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


def _print_training_menu() -> None:
    print(f"\n{GREEN}{BOLD}✓ All setup complete!{RESET}\n")

    # Build a copy-pasteable "python" command that uses the venv's python
    # by full path, so `main.py` finds all its deps without the user having
    # to activate the venv first. On PowerShell, we need a `&` call-operator
    # prefix so the quoted path is interpreted as a command, not a string.
    py_quoted = f'"{PYTHON}"'
    is_powershell = (
        platform.system() == "Windows"
        and os.environ.get("PSModulePath") is not None
    )
    py = f"& {py_quoted}" if is_powershell else py_quoted

    print(f"{BOLD}Train:{RESET} {DIM}(copy-paste as-is — no venv activation needed){RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/trm_official_sudoku.yaml{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/trm_official_maze.yaml{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_config.yaml{RESET}     {DIM}# GPT-2{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_qwen.yaml{RESET}       {DIM}# Qwen2.5-0.5B{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_smollm.yaml{RESET}     {DIM}# SmolLM2-360M{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_llama.yaml{RESET}      {DIM}# Llama-3.2-1B{RESET}\n")

    print(f"{BOLD}Evaluate (after training):{RESET}")
    print(f"  {CYAN}{py} main.py --mode eval --config configs/<name>.yaml --checkpoint models/<name>/best.pt{RESET}\n")

    print(f"{BOLD}Resume from last checkpoint:{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/<name>.yaml --resume models/<name>/latest.pt{RESET}\n")

    print(f"{DIM}Why the full path? Your system `python` isn't the venv's python, so a bare{RESET}")
    print(f"{DIM}`python main.py` hits ModuleNotFoundError. The full-path form dodges that{RESET}")
    print(f"{DIM}and works identically whether or not the venv is activated.{RESET}\n")

    print(f"{DIM}Prefer activating the venv instead? {RESET}{CYAN}{ACTIVATE_HINT}{RESET}{DIM} — then{RESET}")
    print(f"{DIM}you can drop the full path and use bare `python main.py ...`{RESET}\n")

    print(f"{DIM}Tip: run `python start.py status` any time to re-check stages.{RESET}")
    print(f"{DIM}Tip: Weave traces for monitors appear at wandb.ai/<entity>/<project>/weave/monitors{RESET}\n")


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
