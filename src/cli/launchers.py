"""Subprocess launchers for ML-TRM.

Three helpers used across the CLI:
  - _run: generic subprocess runner with streamed output and abort-on-fail
  - _run_training_subprocess: builds main.py argv and runs training
  - _dispatch_training: single-task launcher used by the direct-launch path
"""
import os
import subprocess
import sys
from typing import List, Optional

from src.cli.console import BOLD, CYAN, DIM, RESET, YELLOW
from src.cli.paths import PYTHON, ROOT
from src.cli.workdir import _resolve_work_dir


def _run(cmd: List[str], cwd: str = None) -> None:
    """Run a subprocess, stream output, abort on failure."""
    print(f"\n{DIM}>>> {' '.join(cmd)}{RESET}\n")
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    if result.returncode != 0:
        print(f"\n{YELLOW}!!! Command failed (exit {result.returncode}){RESET}")
        sys.exit(result.returncode)


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
    # Late import of TASK_DISPATCH to avoid circular dependency with
    # bootstrap.py (which imports this module at its top level).
    from src.cli.bootstrap import TASK_DISPATCH

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
        epochs_label = "5 (dry run \u2014 pipeline smoke test)"
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
