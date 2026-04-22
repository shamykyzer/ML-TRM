"""Interactive submenu for the novelty experiment (iso-time + K-vote).

Wired into the main launcher as option 7. Dispatches to the two scripts
in `scripts/run_novelty_*.py`; this module owns only the prompts and
subprocess hand-off, never the experiment logic itself.

The three options mirror the README's experiment decomposition:
  1) Iso-time training sweep only — produces checkpoints + first plot set
  2) K-vote inference sweep only — requires checkpoints from (1) to exist
  3) Both, sequenced — the end-to-end novelty run
"""
import os
import subprocess
import sys
from typing import List

from src.cli.console import BOLD, CYAN, DIM, GREEN, RESET, YELLOW
from src.cli.paths import PYTHON, ROOT
from src.cli.workdir import _resolve_work_dir


# Default budget = 2.5 hr/run × 6 runs = 15 hr. Matches the README-justified
# 14-hr-per-machine window with an extra cushion for the K-vote sweep.
DEFAULT_MAX_TRAIN_SECONDS = 9000  # 2.5 hr
DEFAULT_K_VALUES = "1,2,4,8,16"
DEFAULT_TEMPERATURE = "0.7"


def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        reply = input(f"{CYAN}{msg}{suffix}: {RESET}").strip()
    except EOFError:
        reply = ""
    return reply or default


def _prompt_seed() -> int:
    raw = _prompt("Seed (non-negative int)", default="0")
    try:
        seed = int(raw)
        if seed < 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}!!! Seed must be a non-negative integer, got '{raw}'.{RESET}")
        sys.exit(2)
    return seed


def _prompt_max_train_seconds() -> int:
    raw = _prompt(
        "Wall-clock cap per run in seconds (2.5hr = 9000)",
        default=str(DEFAULT_MAX_TRAIN_SECONDS),
    )
    try:
        seconds = int(raw)
        if seconds <= 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}!!! Need a positive integer, got '{raw}'.{RESET}")
        sys.exit(2)
    return seconds


def _confirm(msg: str) -> bool:
    reply = _prompt(f"{msg} [y/N]", default="N").lower()
    return reply in ("y", "yes")


def _print_banner(title: str, lines: List[str]) -> None:
    bar = "=" * 64
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"  {BOLD}{title}{RESET}")
    for line in lines:
        print(f"  {line}")
    print(f"{BOLD}{bar}{RESET}\n")


def _run_script(argv: List[str], env_extras: dict[str, str]) -> int:
    """Invoke a novelty script as a subprocess. Returns exit code."""
    env = os.environ.copy()
    env.update(env_extras)
    # Ensure TRM_WORK_DIR is exported so the script's internal resolve sees
    # the same value we printed in the confirmation banner. Without this,
    # an operator who picked the work dir interactively here (not via env)
    # would have the subprocess fall back to the default.
    work_dir = env.get("TRM_WORK_DIR") or _resolve_work_dir()
    env["TRM_WORK_DIR"] = work_dir
    return subprocess.run(argv, env=env, cwd=ROOT).returncode


def _run_iso_time(seed: int, max_seconds: int) -> int:
    argv = [
        PYTHON,
        os.path.join("scripts", "run_novelty_iso_time.py"),
        "--seed", str(seed),
        "--max-train-seconds", str(max_seconds),
    ]
    return _run_script(argv, env_extras={})


def _run_k_vote(seed: int, k_values: str, temperature: str) -> int:
    argv = [
        PYTHON,
        os.path.join("scripts", "run_novelty_k_vote.py"),
        "--seed", str(seed),
        "--k-values", k_values,
        "--temperature", temperature,
    ]
    return _run_script(argv, env_extras={})


def _novelty_iso_time_only() -> None:
    seed = _prompt_seed()
    max_seconds = _prompt_max_train_seconds()
    work_dir = _resolve_work_dir()

    _print_banner(
        "Novelty: iso-time training sweep",
        [
            f"seed              : {CYAN}{seed}{RESET}",
            f"wall-clock cap    : {CYAN}{max_seconds}{RESET} sec "
            f"{DIM}(~{max_seconds / 3600:.2f} hr/run){RESET}",
            f"total budget      : {CYAN}{max_seconds * 6 / 3600:.1f}{RESET} hr "
            f"{DIM}(6 runs back-to-back){RESET}",
            f"heavy artifacts   : {DIM}{work_dir}/novelty-*-seed{seed}/{RESET}",
            f"aggregated output : {DIM}results/novelty/iso_time_results.csv + plots{RESET}",
        ],
    )
    if not _confirm("Launch the 6-run sweep?"):
        print(f"{DIM}Aborted.{RESET}\n")
        return
    rc = _run_iso_time(seed, max_seconds)
    sys.exit(rc)


def _novelty_k_vote_only() -> None:
    seed = _prompt_seed()
    k_values = _prompt("K values (comma-separated)", default=DEFAULT_K_VALUES)
    temperature = _prompt("LLM sampling temperature", default=DEFAULT_TEMPERATURE)
    work_dir = _resolve_work_dir()

    _print_banner(
        "Novelty: K-vote inference sweep",
        [
            f"seed              : {CYAN}{seed}{RESET}",
            f"K values          : {CYAN}{k_values}{RESET}",
            f"LLM temperature   : {CYAN}{temperature}{RESET}",
            f"reads checkpoints : {DIM}{work_dir}/novelty-*-seed{seed}/*latest.pt{RESET}",
            f"aggregated output : {DIM}results/novelty/k_vote_results.csv + plots{RESET}",
        ],
    )
    missing = _find_missing_checkpoints(work_dir, seed)
    if missing:
        print(f"{YELLOW}!!! Missing checkpoints for: {', '.join(missing)}{RESET}")
        print(f"{DIM}    Run the iso-time sweep (option 1) first, or skip these{RESET}")
        print(f"{DIM}    labels at run time with --skip-labels.{RESET}")
        if not _confirm("Continue anyway?"):
            print(f"{DIM}Aborted.{RESET}\n")
            return
    elif not _confirm("Launch the K-vote sweep?"):
        print(f"{DIM}Aborted.{RESET}\n")
        return
    rc = _run_k_vote(seed, k_values, temperature)
    sys.exit(rc)


def _novelty_both() -> None:
    seed = _prompt_seed()
    max_seconds = _prompt_max_train_seconds()
    k_values = _prompt("K values (comma-separated)", default=DEFAULT_K_VALUES)
    temperature = _prompt("LLM sampling temperature", default=DEFAULT_TEMPERATURE)
    work_dir = _resolve_work_dir()

    _print_banner(
        "Novelty: iso-time sweep -> K-vote sweep",
        [
            f"seed              : {CYAN}{seed}{RESET}",
            f"wall-clock cap    : {CYAN}{max_seconds}{RESET} sec "
            f"{DIM}(~{max_seconds / 3600:.2f} hr/run){RESET}",
            f"K values          : {CYAN}{k_values}{RESET}",
            f"LLM temperature   : {CYAN}{temperature}{RESET}",
            f"total budget      : {CYAN}~{max_seconds * 6 / 3600 + 2:.1f}{RESET} hr "
            f"{DIM}(training + ~2 hr K-vote){RESET}",
            f"heavy artifacts   : {DIM}{work_dir}/novelty-*-seed{seed}/{RESET}",
        ],
    )
    if not _confirm("Launch both phases back-to-back?"):
        print(f"{DIM}Aborted.{RESET}\n")
        return

    rc_iso = _run_iso_time(seed, max_seconds)
    if rc_iso != 0:
        print(f"{YELLOW}!!! iso-time sweep exited {rc_iso} — K-vote phase skipped.{RESET}")
        sys.exit(rc_iso)

    print(f"\n{GREEN}{BOLD}iso-time sweep complete.{RESET} Launching K-vote sweep...\n")
    rc_kvote = _run_k_vote(seed, k_values, temperature)
    sys.exit(rc_kvote)


# The six labels the iso-time orchestrator writes. Keep in lockstep with
# RUNS in scripts/run_novelty_iso_time.py — duplicating the label list
# here instead of importing keeps this module import-light (no torch).
_NOVELTY_LABELS = (
    "trm-mlp-sudoku",
    "trm-att-maze",
    "qwen-sudoku",
    "qwen-maze",
    "distill-sudoku",
    "distill-maze",
)


def _find_missing_checkpoints(work_dir: str, seed: int) -> List[str]:
    """Return labels whose artifact dir has no *latest.pt file."""
    missing: List[str] = []
    for label in _NOVELTY_LABELS:
        dir_path = os.path.join(work_dir, f"novelty-{label}-seed{seed}")
        if not os.path.isdir(dir_path):
            missing.append(label)
            continue
        has_ckpt = any(
            name.endswith("latest.pt") for name in os.listdir(dir_path)
        )
        if not has_ckpt:
            missing.append(label)
    return missing


def novelty_launcher() -> None:
    """Entry point wired into the main launcher as option 7."""
    print(f"\n{BOLD}Novelty experiments{RESET}  {DIM}(iso-time + K-vote){RESET}")
    print(f"  {DIM}See results/novelty/README.md for design rationale.{RESET}\n")
    print(f"  {CYAN}1{RESET}) Iso-time training sweep only   {DIM}(6 runs, ~15 hr){RESET}")
    print(f"  {CYAN}2{RESET}) K-vote inference sweep only    {DIM}(requires checkpoints from option 1){RESET}")
    print(f"  {CYAN}3{RESET}) Both, sequenced                {DIM}(~17 hr end-to-end){RESET}")
    print(f"  {CYAN}Q{RESET}) Back")

    choice = _prompt("Pick", default="Q").upper()

    if choice in ("Q", ""):
        return
    if choice == "1":
        _novelty_iso_time_only()
    elif choice == "2":
        _novelty_k_vote_only()
    elif choice == "3":
        _novelty_both()
    else:
        print(f"{YELLOW}!!! Unknown choice '{choice}'.{RESET}")
