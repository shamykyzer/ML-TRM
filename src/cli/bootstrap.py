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
}


# ANSI color constants now live in src/cli/console.py (imported at the
# top of this file). Kept this marker so the section ordering in this
# module mirrors the original start.py layout for git-blame continuity.


def _run(cmd: List[str], cwd: str = None) -> None:
    """Run a subprocess, stream output, abort on failure."""
    print(f"\n{DIM}>>> {' '.join(cmd)}{RESET}\n")
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    if result.returncode != 0:
        print(f"\n{YELLOW}!!! Command failed (exit {result.returncode}){RESET}")
        sys.exit(result.returncode)


# ============================================================
# Direct-launch preflight (for `python start.py <task> <seed>`)
# ============================================================

def _kill_training_processes(config_path: str) -> List[int]:
    """Kill python processes running `main.py --mode train --config <config_path>`.

    Cross-platform: uses psutil if available; falls back to platform-specific
    subprocess (wmic/taskkill on Windows, pgrep/kill elsewhere).

    Config_path match is tolerant — checks both full path and basename so it
    hits processes regardless of whether they were launched via relative or
    absolute path.
    """
    cfg_base = os.path.basename(config_path)
    killed: List[int] = []

    # Primary path: psutil gives us clean process iteration with cmdline access.
    try:
        import psutil  # type: ignore
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                joined = " ".join(cmdline)
                if "main.py" not in joined or "--mode" not in joined:
                    continue
                if config_path not in joined and cfg_base not in joined:
                    continue
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    proc.kill()
                killed.append(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return killed
    except ImportError:
        pass  # fall through to platform-specific

    # Fallback without psutil.
    if platform.system() == "Windows":
        try:
            # wmic gives pid + full commandline so we can filter on the config.
            result = subprocess.run(
                ["wmic", "process", "where", "name='python.exe'",
                 "get", "processid,commandline", "/format:csv"],
                capture_output=True, text=True, check=False,
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if "main.py" not in line:
                    continue
                if config_path not in line and cfg_base not in line:
                    continue
                # CSV last column is PID
                parts = line.rsplit(",", 1)
                if len(parts) == 2 and parts[1].strip().isdigit():
                    pid = int(parts[1].strip())
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True,
                    )
                    killed.append(pid)
        except FileNotFoundError:
            pass
    else:
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"main.py.*--config.*{cfg_base}"],
                capture_output=True, text=True, check=False,
            )
            for pid_str in result.stdout.split():
                if pid_str.isdigit():
                    pid = int(pid_str)
                    subprocess.run(["kill", str(pid)], capture_output=True)
                    killed.append(pid)
        except FileNotFoundError:
            pass
    return killed


def _preflight_relaunch(task: str, seed: int) -> None:
    """Pull latest code, kill any existing training for this task, back up best.pt.

    Called from the direct-launch path (`python start.py <task> <seed>`) so
    one command covers the full relaunch cycle: pull new code, stop stale
    process, preserve existing best.pt as insurance, then fall through to
    _dispatch_training. Skips gracefully on steps that aren't applicable
    (no running process, no existing best.pt).
    """
    config, _init, _desc = TASK_DISPATCH[task]

    # 1. Fast-forward pull only — refuse if local has diverged from upstream.
    print(f"\n{BOLD}[preflight 1/3] git pull --ff-only{RESET}")
    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            print(f"{YELLOW}!!! git pull failed:{RESET}")
            print((result.stdout or "") + (result.stderr or ""))
            print(f"{DIM}    Resolve manually, then re-run `python start.py {task} {seed}`.{RESET}")
            sys.exit(result.returncode)
        print(f"{DIM}{(result.stdout or 'Already up to date.').strip()}{RESET}")
    except FileNotFoundError:
        print(f"{YELLOW}!!! git not on PATH — skipping pull. Code may be stale.{RESET}")

    # 2. Kill any existing train subprocess using this config. This is the
    # step that saves the user a manual `ps | grep | kill`.
    print(f"\n{BOLD}[preflight 2/3] kill existing training for {config}{RESET}")
    killed = _kill_training_processes(config)
    if killed:
        print(f"{DIM}Killed PIDs: {killed}{RESET}")
    else:
        print(f"{DIM}No existing training process found.{RESET}")

    # 3. Back up best.pt if present. Non-destructive: adds a timestamped .bak
    # alongside. Important here because the previous maze runs wrote a
    # corrupted best.pt we don't want clobbered by --resume logic downstream.
    print(f"\n{BOLD}[preflight 3/3] back up existing best.pt{RESET}")
    work_dir = _resolve_work_dir()
    task_dir = os.path.join(work_dir, f"{task}-seed{seed}")
    best_pt = os.path.join(task_dir, "best.pt")
    if os.path.exists(best_pt):
        import time
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = f"{best_pt}.{ts}.bak"
        try:
            shutil.copy2(best_pt, backup)
            print(f"{DIM}Copied {best_pt} -> {os.path.basename(backup)}{RESET}")
        except OSError as exc:
            print(f"{YELLOW}!!! backup failed: {exc} (continuing anyway){RESET}")
    else:
        print(f"{DIM}No existing best.pt at {best_pt}.{RESET}")

    print(f"\n{GREEN}[preflight] complete — handing off to training launcher.{RESET}")


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


# _resolve_work_dir() now lives in src/cli/workdir.py. Re-imported here so
# the rest of this module can keep referring to it by the same name.
from src.cli.workdir import _resolve_work_dir  # noqa: F401,E402


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

    Shows every task in TASK_DISPATCH as a numbered menu, grouped TRM-first
    then LLM, so the operator can scan by family. Tasks whose HF init file is
    missing get a yellow flag (the run still works — it just starts from
    random init, exploratory rather than paper-faithful). Defaults to seed 0
    (first row of FLEET_PLAN, a safe starter on any machine).
    """
    tasks = list(TASK_DISPATCH.keys())
    trm_tasks = [t for t in tasks if not t.startswith("llm-")]
    llm_tasks = [t for t in tasks if t.startswith("llm-")]

    print(f"\n{BOLD}Which task?{RESET}")
    print(f"  {DIM}-- TRM (paper architectures) --{RESET}")
    for t in trm_tasks:
        i = tasks.index(t) + 1
        _, init, desc = TASK_DISPATCH[t]
        suffix = (
            f"  {YELLOW}(HF init missing — will use random init){RESET}"
            if init and not os.path.exists(init) else ""
        )
        print(f"  {CYAN}{i:>2}{RESET}) {t:<20s}  {DIM}{desc}{RESET}{suffix}")
    print(f"  {DIM}-- LLM baselines (LoRA fine-tune) --{RESET}")
    for t in llm_tasks:
        i = tasks.index(t) + 1
        _, _, desc = TASK_DISPATCH[t]
        print(f"  {CYAN}{i:>2}{RESET}) {t:<20s}  {DIM}{desc}{RESET}")

    choice = _prompt(f"Pick 1-{len(tasks)}", default="1")
    try:
        task = tasks[int(choice) - 1]
        if int(choice) < 1:
            raise IndexError
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


def _run_training_subprocess(
    task: str,
    seed: int,
    dry_run: bool = False,
    epochs: Optional[int] = None,
) -> int:
    """Build main.py argv and run a training subprocess. Return its exit code.

    Sets TRM_CHECKPOINT_DIR and TRM_EXPERIMENT_DIR in the child env so the
    trainer writes to <TRM_WORK_DIR>/<task>-seed<N>/, never into the
    OneDrive-synced repo.

    Used by both `_dispatch_training` (single-task path that sys.exit's on
    completion) and `_llm_sweep_launcher` (loop that needs the exit code so it
    can chain follow-up runs).
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
        epochs_label = "5 (dry run — pipeline smoke test)"
    elif epochs is not None:
        args.extend(["--epochs", str(epochs)])
        epochs_label = f"{epochs} (overridden via --epochs)"
    else:
        epochs_label = "from YAML config.training.epochs"

    bar = "=" * 64
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"  task               : {CYAN}{task}{RESET}  {DIM}({description}){RESET}")
    print(f"  seed               : {CYAN}{seed}{RESET}")
    print(f"  config             : {config}")
    init_label = init if (init and os.path.exists(init)) else "<none, random init>"
    print(f"  init_weights       : {init_label}")
    print(f"  TRM_CHECKPOINT_DIR : {task_dir}")
    print(f"  TRM_EXPERIMENT_DIR : {task_dir}")
    print(f"  epochs             : {epochs_label}")
    print(f"{BOLD}{bar}{RESET}\n")

    result = subprocess.run(args, env=env)
    return result.returncode


def _dispatch_training(
    task: str,
    seed: int,
    dry_run: bool = False,
    epochs: Optional[int] = None,
) -> None:
    """Single-task launcher: run one (task, seed) and exit with its return code.

    Python twin of `scripts/run_seed.sh <task> <seed>`. Shell launchers are
    kept for automation (cron, CI); this path is for users who drive
    everything from start.py interactively.

    When `epochs` is passed, it forwards as --epochs N to main.py (overriding
    YAML's training.epochs for this run only). dry_run wins if both are set.
    """
    rc = _run_training_subprocess(task, seed, dry_run=dry_run, epochs=epochs)
    sys.exit(rc)


# ============================================================
# Resume / extend training
# ============================================================

# Map a checkpoint dir (or its task-prefix) to the YAML config that produced
# it. The fleet dirs use `<task>-seed<N>` naming where <task> is one of the
# TASK_DISPATCH keys; legacy / non-fleet runs (older `models/sudoku/`,
# `models/maze/`) use the simple-TRM configs. New entries can be added here
# without touching anything else.
RESUME_CONFIG_BY_PREFIX: dict = {
    "sudoku-mlp":        "configs/trm_official_sudoku_mlp.yaml",
    "sudoku-att":        "configs/trm_official_sudoku.yaml",
    "sudoku-official":   "configs/trm_official_sudoku.yaml",  # legacy alias
    "maze":              "configs/trm_official_maze.yaml",
    "llm-sudoku":        "configs/llm_qwen.yaml",             # legacy alias
    "llm-gpt2-sudoku":   "configs/llm_config.yaml",
    "llm-smollm-sudoku": "configs/llm_smollm.yaml",
    "llm-qwen-sudoku":   "configs/llm_qwen.yaml",
    "llm-llama-sudoku":  "configs/llm_llama.yaml",
    "llm-gpt2-maze":     "configs/llm_gpt2_maze.yaml",
    "llm-smollm-maze":   "configs/llm_smollm_maze.yaml",
    "llm-qwen-maze":     "configs/llm_qwen_maze.yaml",
    "llm-llama-maze":    "configs/llm_llama_maze.yaml",
    # Legacy simple-TRM dirs (trainer_trm.py, not trainer_official.py):
    "sudoku":            "configs/trm_sudoku.yaml",
}


def _scan_for_checkpoints(root: str) -> List[Tuple[str, int, str]]:
    """Walk one root for subdirs containing epoch_<N>.pt files.

    Returns a list of (dir_path, max_epoch_int, latest_pt_path) tuples,
    one per discovered run dir, sorted by mtime descending so the most
    recent run shows first in the picker. Skips any dir with no
    epoch_*.pt files. We deliberately accept BOTH a `latest.pt` (only
    written by trainer_trm at end-of-run) and the highest `epoch_N.pt`
    (the granular crash-recovery point) — the picker prefers epoch_N.pt
    because it's available mid-run after Ctrl+C, which `latest.pt` is not.
    """
    import re
    if not os.path.isdir(root):
        return []
    results: List[Tuple[str, int, str]] = []
    epoch_re = re.compile(r"epoch_(\d+)\.pt$")
    for entry in sorted(os.listdir(root)):
        run_dir = os.path.join(root, entry)
        if not os.path.isdir(run_dir):
            continue
        max_ep = -1
        max_path = ""
        try:
            files = os.listdir(run_dir)
        except OSError:
            continue
        for fname in files:
            m = epoch_re.match(fname)
            if not m:
                continue
            ep = int(m.group(1))
            if ep > max_ep:
                max_ep = ep
                max_path = os.path.join(run_dir, fname)
        if max_ep >= 0:
            results.append((run_dir, max_ep, max_path))
    # Sort newest-mtime first (the run the user just Ctrl+C'd should be
    # at the top so they don't have to scroll).
    results.sort(key=lambda r: os.path.getmtime(r[2]), reverse=True)
    return results


def _config_for_run_dir(run_dir: str) -> str:
    """Best-effort: derive the YAML config from the run dir's basename.

    Strips a trailing `-seed<N>` suffix and looks the prefix up in
    RESUME_CONFIG_BY_PREFIX. Returns "" when the prefix is unknown — the
    caller prompts for a manual config path in that case.
    """
    import re
    base = os.path.basename(run_dir.rstrip(os.sep).rstrip("/"))
    base = re.sub(r"-seed\d+$", "", base)
    return RESUME_CONFIG_BY_PREFIX.get(base, "")


def _seed_for_run_dir(run_dir: str) -> int:
    """Parse the seed N out of a `<task>-seed<N>` directory name.

    Returns 0 when the dir doesn't follow the convention (legacy runs
    written before the seed-variance launcher existed).
    """
    import re
    m = re.search(r"-seed(\d+)$", os.path.basename(run_dir.rstrip(os.sep).rstrip("/")))
    return int(m.group(1)) if m else 0


def _resume_training_picker() -> None:
    """Interactive: pick a finished/interrupted run and extend it by N epochs.

    Discovery scans both:
      • $TRM_WORK_DIR (fleet runs from `start.py` dispatch / run_seed.sh)
      • <repo>/models/  (legacy simple-TRM runs that wrote inside the repo)
    Each candidate run is shown with its highest epoch_<N>.pt checkpoint —
    NOT `latest.pt`, because latest.pt is only written when a run completes
    cleanly, but Ctrl+C recovery has to start from the most recent
    epoch_N.pt that the save_interval cadence wrote to disk.

    The user picks a run, the picker resolves the right config via the dir
    name, asks how many ADDITIONAL epochs to run, then dispatches main.py
    with --resume <ckpt> --epochs <max_ep + extra>. main.py's _run_train
    treats --epochs as the new total target — the trainer loop runs from
    the resumed start_epoch through to the new total.
    """
    work_dir = os.environ.get("TRM_WORK_DIR") or _default_work_dir()
    candidates = (
        _scan_for_checkpoints(work_dir)
        + _scan_for_checkpoints(os.path.join(ROOT, "models"))
    )

    if not candidates:
        print(f"\n{YELLOW}!!! No resumable runs found.{RESET}")
        print(f"{DIM}    Looked in:{RESET}")
        print(f"{DIM}      • {work_dir}{RESET}")
        print(f"{DIM}      • {os.path.join(ROOT, 'models')}{RESET}")
        print(f"{DIM}    A run is resumable when its directory contains at least one{RESET}")
        print(f"{DIM}    epoch_<N>.pt file (written every save_interval epochs).{RESET}\n")
        return

    print(f"\n{BOLD}Resumable runs:{RESET}  {DIM}(newest first){RESET}")
    for i, (run_dir, max_ep, ckpt_path) in enumerate(candidates, 1):
        cfg = _config_for_run_dir(run_dir) or f"{YELLOW}<unknown — will prompt>{RESET}"
        seed = _seed_for_run_dir(run_dir)
        size_mb = os.path.getsize(ckpt_path) / 1e6
        print(
            f"  {CYAN}{i:>2}{RESET}) {os.path.basename(run_dir):<28s}"
            f"  epoch={CYAN}{max_ep:>4}{RESET}"
            f"  seed={seed}"
            f"  ({size_mb:.0f} MB)"
        )
        print(f"      {DIM}{ckpt_path}{RESET}")
        print(f"      {DIM}config: {cfg}{RESET}")

    raw = _prompt(f"\nPick run 1-{len(candidates)}", default="1")
    try:
        idx = int(raw) - 1
        run_dir, max_ep, ckpt_path = candidates[idx]
    except (ValueError, IndexError):
        print(f"{YELLOW}!!! Invalid choice '{raw}'.{RESET}")
        return

    config = _config_for_run_dir(run_dir)
    if not config:
        config = _prompt(
            f"Config YAML for this run (relative to repo root)",
            default="configs/trm_sudoku.yaml",
        )
        if not os.path.exists(os.path.join(ROOT, config)):
            print(f"{YELLOW}!!! Config not found: {config}{RESET}")
            return

    # The "how many MORE epochs?" framing is the natural one for the user
    # ("I just hit 500, give me 200 more"); we convert to the absolute
    # total that main.py / the trainer loop expect via --epochs.
    extra_str = _prompt(
        f"Run for how many MORE epochs? (current = {max_ep})",
        default="100",
    )
    try:
        extra = int(extra_str)
        if extra <= 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}!!! Need a positive integer, got '{extra_str}'.{RESET}")
        return
    new_total = max_ep + extra
    seed = _seed_for_run_dir(run_dir)

    # Pin checkpoint/experiment dirs back to the SAME run dir so the new
    # epoch_<N>.pt files, train_log.csv, and emissions.csv append to the
    # existing artifacts instead of starting a fresh dir somewhere else.
    env = os.environ.copy()
    env["TRM_CHECKPOINT_DIR"] = run_dir
    env["TRM_EXPERIMENT_DIR"] = run_dir

    args = [
        PYTHON, "main.py",
        "--mode", "train",
        "--config", config,
        "--resume", ckpt_path,
        "--epochs", str(new_total),
        "--seed", str(seed),
    ]

    bar = "=" * 64
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"  resume from        : {CYAN}epoch_{max_ep}.pt{RESET}  {DIM}({ckpt_path}){RESET}")
    print(f"  config             : {config}")
    print(f"  current epoch      : {max_ep}")
    print(f"  extra epochs       : {CYAN}{extra}{RESET}")
    print(f"  new total epochs   : {CYAN}{new_total}{RESET}")
    print(f"  seed               : {seed}")
    print(f"  TRM_CHECKPOINT_DIR : {run_dir}")
    print(f"  wandb              : "
          f"{GREEN}✓ ready{RESET}" if _wandb_ready() else f"{YELLOW}not configured (will train without){RESET}")
    print(f"{BOLD}{bar}{RESET}\n")

    confirm = _prompt(f"Launch? [y/N]", default="N").lower()
    if confirm not in ("y", "yes"):
        print(f"{DIM}Aborted — nothing launched.{RESET}\n")
        return

    result = subprocess.run(args, env=env, cwd=ROOT)
    sys.exit(result.returncode)


# ============================================================
# Fresh-start (overwrite) launcher
# ============================================================

# Fresh-start menu collapses the 8 per-task LLM entries into 4 family combos.
# Picking a family runs sudoku FIRST then maze, each as its own subprocess
# (and thus its own wandb run with the enriched name in wandb_utils.py).
# Tuple shape: (family_label, description, sudoku_task_key, maze_task_key).
# The two task keys must exist in TASK_DISPATCH or the picker errors loudly
# at launch — single source of truth for "which configs go with which family".
LLM_FAMILIES: List[Tuple[str, str, str, str]] = [
    ("llm-gpt2",   "GPT-2 (124M) LoRA",     "llm-gpt2-sudoku",   "llm-gpt2-maze"),
    ("llm-smollm", "SmolLM2-360M LoRA",     "llm-smollm-sudoku", "llm-smollm-maze"),
    ("llm-qwen",   "Qwen2.5-0.5B LoRA",     "llm-qwen-sudoku",   "llm-qwen-maze"),
    ("llm-llama",  "Llama-3.2-1B LoRA",     "llm-llama-sudoku",  "llm-llama-maze"),
]


def _prompt_fresh_target_and_seed() -> Tuple[str, List[str], int]:
    """Show TRM tasks + 4 LLM families; return (label, [task, ...], seed).

    TRM picks return a single-element task list; LLM family picks return
    [sudoku_task, maze_task] so the caller can wipe + run both back-to-back
    with one prompt instead of forcing the user through option 6 twice per
    family. Keeps the menu at 7 entries (3 TRM + 4 LLM families) instead of
    the 11 entries the per-task picker would show.
    """
    trm_tasks = [t for t in TASK_DISPATCH if not t.startswith("llm-")]
    entries: List[Tuple[str, str, List[str]]] = []
    for t in trm_tasks:
        _, init, desc = TASK_DISPATCH[t]
        suffix = (
            "  (HF init missing — random init)"
            if init and not os.path.exists(init) else ""
        )
        entries.append((t, desc + suffix, [t]))
    for family, desc, sudoku_task, maze_task in LLM_FAMILIES:
        entries.append((family, f"{desc}  (sudoku → maze)", [sudoku_task, maze_task]))

    print(f"\n{BOLD}Which target?{RESET}")
    print(f"  {DIM}-- TRM (paper architectures) --{RESET}")
    n_trm = len(trm_tasks)
    for i, (label, desc, _) in enumerate(entries[:n_trm], 1):
        print(f"  {CYAN}{i:>2}{RESET}) {label:<13s}  {DIM}{desc}{RESET}")
    print(f"  {DIM}-- LLM families (each runs sudoku then maze) --{RESET}")
    for i, (label, desc, _) in enumerate(entries[n_trm:], n_trm + 1):
        print(f"  {CYAN}{i:>2}{RESET}) {label:<13s}  {DIM}{desc}{RESET}")

    choice = _prompt(f"Pick 1-{len(entries)}", default="1")
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(entries):
            raise IndexError
    except (ValueError, IndexError):
        print(f"{YELLOW}!!! Invalid target choice '{choice}'.{RESET}")
        sys.exit(2)
    label, _, tasks = entries[idx]

    seed_str = _prompt("Seed (non-negative int)", default="0")
    try:
        seed = int(seed_str)
        if seed < 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}!!! Seed must be a non-negative integer, got '{seed_str}'.{RESET}")
        sys.exit(2)

    return label, tasks, seed


def _confirm_wipe_run_dir(task_dir: str) -> bool:
    """List the contents of task_dir, prompt y/N to delete, then rmtree.

    Returns True if the dir was wiped (or was already empty/absent), False if
    the user declined or the rmtree failed. Bails after printing instead of
    raising — callers usually want to abort the whole launcher cleanly.
    """
    if not os.path.isdir(task_dir):
        print(f"{DIM}[fresh-start] no existing dir at {task_dir} — clean launch.{RESET}")
        return True
    try:
        existing = sorted(os.listdir(task_dir))
    except OSError:
        existing = []
    if not existing:
        return True

    print(f"\n{YELLOW}⚠  Existing run directory will be WIPED:{RESET}")
    print(f"   {task_dir}")
    print(f"{DIM}   Contains {len(existing)} item(s):{RESET}")
    for name in existing[:10]:
        full = os.path.join(task_dir, name)
        if os.path.isfile(full):
            size_mb = os.path.getsize(full) / 1e6
            size_label = f"  ({size_mb:.1f} MB)" if size_mb >= 0.1 else ""
            print(f"{DIM}     • {name}{size_label}{RESET}")
        else:
            print(f"{DIM}     • {name}/{RESET}")
    if len(existing) > 10:
        print(f"{DIM}     ... and {len(existing) - 10} more{RESET}")

    confirm = _prompt(
        f"\n{BOLD}Delete these and start fresh?{RESET} [y/N]",
        default="N",
    ).lower()
    if confirm not in ("y", "yes"):
        return False
    try:
        shutil.rmtree(task_dir)
    except OSError as exc:
        print(f"{YELLOW}!!! Could not remove {task_dir}: {exc}{RESET}")
        return False
    print(f"{DIM}[fresh-start] removed {task_dir}{RESET}")
    return True


def _fresh_start_launcher() -> None:
    """Launch NEW runs for a chosen target+seed, wiping existing run dir(s).

    Counterpart to the resume picker: instead of continuing from the latest
    epoch_<N>.pt, this path guarantees epoch 0 by deleting everything under
    <TRM_WORK_DIR>/<task>-seed<N>/ first.

    Two modes, both driven by `_prompt_fresh_target_and_seed`:
      • TRM pick → wipe + run one task with the chosen epochs (default 2000).
      • LLM family pick → wipe BOTH the sudoku and maze seed dirs for that
        family, then run sudoku then maze sequentially as two distinct wandb
        runs, each capped at the chosen epochs (default 30 — the scaling-
        comparison cadence).

    A failure in run 1 (sudoku) of an LLM family does NOT abort run 2 (maze):
    we collect the exit codes and print a summary so the operator can retry
    only what failed.
    """
    label, tasks, seed = _prompt_fresh_target_and_seed()
    is_family = len(tasks) > 1

    # 30 epochs is the LLM scaling default the user is comparing across the 4
    # families on both datasets. TRM tasks still default to the long 2000-epoch
    # cadence the paper-faithful runs need to converge.
    default_epochs = "30" if is_family else "2000"
    epochs_str = _prompt("Epochs per task (overrides YAML)", default=default_epochs)
    try:
        epochs = int(epochs_str)
        if epochs <= 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}!!! Need a positive integer, got '{epochs_str}'.{RESET}")
        return

    work_dir = _resolve_work_dir()

    # Per-task wipe pass — confirm once per dir, abort the whole launcher if
    # the user declines any single confirmation (safer than a partial sweep
    # against half-stale dirs that would silently resume).
    for task in tasks:
        task_dir = os.path.join(work_dir, f"{task}-seed{seed}")
        if not _confirm_wipe_run_dir(task_dir):
            print(f"{DIM}Aborted — existing run preserved, nothing launched.{RESET}\n")
            return

    if not is_family:
        _dispatch_training(tasks[0], seed, epochs=epochs)
        return  # unreachable — _dispatch_training calls sys.exit

    bar = "=" * 64
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"  {BOLD}LLM family run — {label}{RESET}")
    print(f"  seed   : {CYAN}{seed}{RESET}")
    print(f"  epochs : {CYAN}{epochs}{RESET} per task")
    print(f"  order  : {DIM}{' → '.join(tasks)}{RESET}")
    print(f"{BOLD}{bar}{RESET}\n")

    results: List[Tuple[str, int]] = []
    for i, task in enumerate(tasks, 1):
        print(f"\n{BOLD}>>> [{i}/{len(tasks)}] {task}{RESET}")
        rc = _run_training_subprocess(task, seed, dry_run=False, epochs=epochs)
        results.append((task, rc))
        if rc != 0:
            print(f"{YELLOW}!!! [{task}] exited {rc} — continuing.{RESET}")

    print(f"\n{BOLD}{bar}{RESET}")
    print(f"  {BOLD}{label} sweep complete:{RESET}")
    n_ok = 0
    for task, rc in results:
        if rc == 0:
            mark, color = "✓", GREEN
            n_ok += 1
        else:
            mark, color = "✗", YELLOW
        print(f"  [{color}{mark}{RESET}] {task:<22s} rc={rc}")
    print(f"{BOLD}{bar}{RESET}")
    print(f"  {n_ok}/{len(results)} tasks succeeded.\n")


# ============================================================
# Interactive launcher
# ============================================================

def _interactive_launcher() -> None:
    """Prompt the user to pick an action after all setup stages are ready.

    Main entry for the 6-box workflow: re-run `python start.py` per seed,
    pick a task + seed from the menu, train. Option 3 is the Regime A
    one-shot (no training); option 4 dumps the full copy-paste command
    list for scripted workflows. Option 5 is the resume/extend path —
    finds the latest epoch_<N>.pt for any prior run (including ones the
    user Ctrl+C'd) and extends training by a chosen number of epochs.
    Option 6 is the fresh-start path; for LLM family picks it sequences
    sudoku → maze as two distinct wandb runs.
    """
    print(f"\n{BOLD}What do you want to run?{RESET}")
    print(f"  {CYAN}1{RESET}) Dry run         {DIM}(5-epoch pipeline smoke test — always do this first){RESET}")
    print(f"  {CYAN}2{RESET}) Seed-variance   {DIM}(full fine-tune from HF init — the 6-machine plan){RESET}")
    print(f"  {CYAN}3{RESET}) Evaluate HF     {DIM}(Regime A — all 3 paper checkpoints, no training){RESET}")
    print(f"  {CYAN}4{RESET}) Show commands   {DIM}(print copy-paste commands and exit){RESET}")
    print(f"  {CYAN}5{RESET}) Resume/extend   {DIM}(continue a finished or Ctrl+C'd run for N more epochs){RESET}")
    print(f"  {CYAN}6{RESET}) Fresh start     {DIM}(new run; LLM family picks run sudoku → maze back-to-back){RESET}")
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

    if choice == "5":
        _resume_training_picker()
        return

    if choice == "6":
        _fresh_start_launcher()
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

    print(f"{BOLD}Other (non-fleet) TRM configs:{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/trm_sudoku.yaml --seed 0{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/trm_maze.yaml --seed 0{RESET}\n")

    print(f"{BOLD}LLM baselines on Sudoku (LoRA fine-tune):{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_config.yaml --seed 0{RESET}  {DIM}# GPT-2 (124M){RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_smollm.yaml --seed 0{RESET}  {DIM}# SmolLM2-360M{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_qwen.yaml --seed 0{RESET}    {DIM}# Qwen2.5-0.5B{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_llama.yaml --seed 0{RESET}   {DIM}# Llama-3.2-1B{RESET}\n")

    print(f"{BOLD}LLM baselines on Maze (LoRA fine-tune):{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_gpt2_maze.yaml --seed 0{RESET}    {DIM}# GPT-2 (124M){RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_smollm_maze.yaml --seed 0{RESET}  {DIM}# SmolLM2-360M{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_qwen_maze.yaml --seed 0{RESET}    {DIM}# Qwen2.5-0.5B{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_llama_maze.yaml --seed 0{RESET}   {DIM}# Llama-3.2-1B{RESET}\n")

    print(f"{BOLD}Distill a fine-tuned LLM into a small student:{RESET}")
    print(f"  {CYAN}{py} main.py --mode distill --config configs/llm_qwen.yaml --checkpoint models/llm/qwen2.5_0.5b_sudoku_latest.pt{RESET}\n")

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
        work_label = f"{CYAN}{_default_work_dir()}{RESET}  {DIM}(auto-picked — TRM_WORK_DIR not set){RESET}"
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
