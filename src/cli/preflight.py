"""Direct-launch preflight for `python start.py <task> <seed>`.

Wraps the relaunch cycle into one command for the 6-machine fleet:

    python start.py maze 0      # on FDK
    python start.py maze 1      # on FCM
    python start.py maze 2      # on FFN

Runs three steps in order: git pull --ff-only, kill existing training
for this config, back up best.pt. Bypasses stage checks and the
interactive menu; assumes the machine has already been through setup.

Kept stdlib-only so start.py can import this module before the venv is
built. psutil is opportunistically imported inside _kill_training_processes
with a wmic/taskkill/pgrep fallback for the common case.
"""
import os
import platform
import shutil
import subprocess
import sys
from typing import List

from src.cli.console import BOLD, DIM, GREEN, RESET, YELLOW
from src.cli.paths import ROOT
from src.cli.workdir import _resolve_work_dir


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
    # Late import of TASK_DISPATCH to avoid a circular dependency with
    # bootstrap (which imports this module at its top level).
    from src.cli.bootstrap import TASK_DISPATCH

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
        print(f"{YELLOW}!!! git not on PATH \u2014 skipping pull. Code may be stale.{RESET}")

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

    print(f"\n{GREEN}[preflight] complete \u2014 handing off to training launcher.{RESET}")
