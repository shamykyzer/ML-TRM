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
import re
import subprocess
import sys
import time
from typing import List, Tuple

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


def _run_iso_time(seed: int, max_seconds: int, rig: int = 0) -> int:
    argv = [
        PYTHON,
        os.path.join("scripts", "run_novelty_iso_time.py"),
        "--seed", str(seed),
    ]
    # In rig mode, omit --max-train-seconds so the orchestrator picks the
    # rig's NOVELTY_RIG_BUDGET_SEC default (7 hr / 6 hr per rig). Passing
    # the all-runs default of 9000 here would override that smarter default.
    if rig:
        argv += ["--rig", str(rig)]
    else:
        argv += ["--max-train-seconds", str(max_seconds)]
    return _run_script(argv, env_extras={})


def _run_k_vote(
    seed: int,
    k_values: str,
    temperature: str,
    rig: int = 0,
    skip_labels: List[str] | None = None,
    work_dir: str = "",
) -> int:
    argv = [
        PYTHON,
        os.path.join("scripts", "run_novelty_k_vote.py"),
        "--seed", str(seed),
        "--k-values", k_values,
        "--temperature", temperature,
    ]
    # --rig and --skip-labels are mutually exclusive in the underlying script;
    # caller is responsible for passing at most one. We assert here so a
    # programming bug surfaces as a clean error instead of an argparse failure
    # inside the subprocess.
    if rig and skip_labels:
        raise ValueError("pass rig OR skip_labels, not both")
    if rig:
        argv += ["--rig", str(rig)]
    if skip_labels:
        argv += ["--skip-labels", ",".join(skip_labels)]
    env_extras = {"TRM_WORK_DIR": work_dir} if work_dir else {}
    return _run_script(argv, env_extras=env_extras)


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

# Mirror of NOVELTY_RIG_PLAN in scripts/run_novelty_iso_time.py — used
# only for the menu's per-rig confirmation banner. Source of truth lives
# in the orchestrator; duplicating here keeps this module import-light.
_NOVELTY_RIG_LABELS: dict[int, tuple[str, ...]] = {
    1: ("trm-mlp-sudoku", "trm-att-maze"),
    2: ("qwen-sudoku", "distill-sudoku"),
    3: ("qwen-maze", "distill-maze"),
}
_NOVELTY_RIG_BUDGET_HR: dict[int, float] = {1: 7.0, 2: 6.0, 3: 6.0}


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


_NOVELTY_DIR_RE = re.compile(r"^novelty-(.+)-seed(\d+)$")


def _scan_novelty_checkpoints(
    work_dir: str,
) -> List[Tuple[str, int, str, float]]:
    """Return [(label, seed, ckpt_path, mtime), ...] for every novelty-*-seed*
    dir under work_dir that contains a `*latest.pt`.

    Used by the 'K-vote existing checkpoints' option so the operator doesn't
    have to remember which rig/seed/labels live in which work dir — useful
    after a Ctrl+C or when resuming on a fresh shell.
    """
    rows: List[Tuple[str, int, str, float]] = []
    if not os.path.isdir(work_dir):
        return rows
    valid_labels = set(_NOVELTY_LABELS)
    for name in os.listdir(work_dir):
        m = _NOVELTY_DIR_RE.match(name)
        if not m:
            continue
        label = m.group(1)
        if label not in valid_labels:
            continue
        try:
            seed = int(m.group(2))
        except ValueError:
            continue
        dir_path = os.path.join(work_dir, name)
        if not os.path.isdir(dir_path):
            continue
        # Prefer the newest `*latest.pt` so we pick up distill_*_latest.pt
        # over a stale qwen_*_latest.pt left over from a re-run.
        ckpts = [f for f in os.listdir(dir_path) if f.endswith("latest.pt")]
        if not ckpts:
            continue
        ckpts.sort(
            key=lambda f: os.path.getmtime(os.path.join(dir_path, f)),
            reverse=True,
        )
        ckpt_path = os.path.join(dir_path, ckpts[0])
        rows.append((label, seed, ckpt_path, os.path.getmtime(ckpt_path)))
    return rows


def _fmt_age(now: float, mtime: float) -> str:
    delta_hr = max(0.0, (now - mtime) / 3600.0)
    if delta_hr < 1.0:
        return f"{delta_hr * 60:.0f} min ago"
    if delta_hr < 48.0:
        return f"{delta_hr:.1f} hr ago"
    return f"{delta_hr / 24:.1f} d ago"


def _print_checkpoint_summary(
    work_dir: str, rows: List[Tuple[str, int, str, float]],
) -> None:
    now = time.time()
    print(f"  {DIM}work_dir: {work_dir}{RESET}")
    by_seed: dict[int, list[tuple[str, str, float]]] = {}
    for label, seed, ckpt, mtime in rows:
        by_seed.setdefault(seed, []).append((label, ckpt, mtime))
    for seed in sorted(by_seed):
        print(f"  {BOLD}seed={seed}{RESET}")
        for label, ckpt, mtime in sorted(by_seed[seed]):
            age = _fmt_age(now, mtime)
            print(
                f"    {CYAN}{label:<18}{RESET}"
                f" {DIM}{age:<14}{RESET}"
                f" {DIM}{os.path.basename(ckpt)}{RESET}"
            )


def _novelty_kvote_existing() -> None:
    """K-vote whatever checkpoints already exist under the work dir.

    Differs from option 2 in that it auto-discovers (label, seed) pairs
    rather than assuming the full 6-run set — the intended workflow is
    'I trained rig N, Ctrl+C'd the K-vote, now re-run K-vote on what
    actually got trained' without remembering --skip-labels flags.

    A custom work_dir prompt covers the case where checkpoints live on a
    different drive (e.g. a spare C:/ml-trm-work from a previous machine).
    """
    default_work_dir = _resolve_work_dir()
    rows = _scan_novelty_checkpoints(default_work_dir)

    if rows:
        print(f"\n{BOLD}Detected novelty checkpoints:{RESET}")
        _print_checkpoint_summary(default_work_dir, rows)
        alt = _prompt(
            "Use a different work_dir? (blank = keep the one above)",
            default="",
        )
        if alt:
            alt_rows = _scan_novelty_checkpoints(alt)
            if not alt_rows:
                print(f"{YELLOW}!!! No novelty checkpoints under {alt}{RESET}")
                return
            rows = alt_rows
            work_dir = alt
            print(f"\n{BOLD}Detected novelty checkpoints:{RESET}")
            _print_checkpoint_summary(work_dir, rows)
        else:
            work_dir = default_work_dir
    else:
        print(f"{YELLOW}!!! No novelty checkpoints under {default_work_dir}{RESET}")
        alt = _prompt(
            "Path to an alternate work_dir (blank to abort)",
            default="",
        )
        if not alt:
            print(f"{DIM}Aborted.{RESET}\n")
            return
        rows = _scan_novelty_checkpoints(alt)
        if not rows:
            print(f"{YELLOW}!!! No novelty checkpoints under {alt} either{RESET}")
            return
        work_dir = alt
        print(f"\n{BOLD}Detected novelty checkpoints:{RESET}")
        _print_checkpoint_summary(work_dir, rows)

    seeds_available = sorted({s for _, s, _, _ in rows})
    if len(seeds_available) == 1:
        seed = seeds_available[0]
        print(f"  {DIM}(only seed {seed} detected; using it){RESET}")
    else:
        raw = _prompt(
            f"Seed to K-vote (available: {', '.join(map(str, seeds_available))})",
            default=str(seeds_available[0]),
        )
        try:
            seed = int(raw)
        except ValueError:
            print(f"{YELLOW}!!! Invalid seed: {raw}{RESET}")
            return
        if seed not in seeds_available:
            print(f"{YELLOW}!!! Seed {seed} has no checkpoints under {work_dir}{RESET}")
            return

    available_labels = sorted({label for label, s, _, _ in rows if s == seed})
    skip_labels = sorted(set(_NOVELTY_LABELS) - set(available_labels))

    k_values = _prompt("K values (comma-separated)", default=DEFAULT_K_VALUES)
    temperature = _prompt("LLM sampling temperature", default=DEFAULT_TEMPERATURE)

    _print_banner(
        "Novelty: K-vote existing checkpoints",
        [
            f"work_dir    : {CYAN}{work_dir}{RESET}",
            f"seed        : {CYAN}{seed}{RESET}",
            f"labels      : {CYAN}{', '.join(available_labels)}{RESET}",
            f"skipping    : {DIM}{', '.join(skip_labels) if skip_labels else '(none)'}{RESET}",
            f"K values    : {CYAN}{k_values}{RESET}",
            f"temperature : {CYAN}{temperature}{RESET}",
            f"output      : {DIM}results/novelty/k_vote_results.csv + plots{RESET}",
        ],
    )
    if not _confirm("Launch K-vote?"):
        print(f"{DIM}Aborted.{RESET}\n")
        return

    rc = _run_k_vote(
        seed, k_values, temperature,
        skip_labels=skip_labels, work_dir=work_dir,
    )
    sys.exit(rc)


def _novelty_this_rig() -> None:
    """Run this rig's slice of the 3-rig split (training + K-vote).

    Resolves TRM_RIG via the existing helper in src.cli.bootstrap, which
    reads from env or prompts and persists to .env. Each rig gets only
    its 2 runs so the per-run wall-clock budget can be 2.4-2.8x longer
    than the 1-rig mode (7 hr or 6 hr/run vs 2.5 hr/run).
    """
    from src.cli.bootstrap import _resolve_rig

    rig = _resolve_rig()
    seed = _prompt_seed()
    k_values = _prompt("K values (comma-separated)", default=DEFAULT_K_VALUES)
    temperature = _prompt("LLM sampling temperature", default=DEFAULT_TEMPERATURE)
    work_dir = _resolve_work_dir()

    labels = _NOVELTY_RIG_LABELS[rig]
    per_run_hr = _NOVELTY_RIG_BUDGET_HR[rig]
    total_train_hr = per_run_hr * len(labels)

    _print_banner(
        f"Novelty: rig {rig} slice — iso-time -> K-vote",
        [
            f"rig                : {CYAN}{rig}{RESET}  {DIM}(TRM_RIG){RESET}",
            f"seed               : {CYAN}{seed}{RESET}",
            f"runs on this rig   : {CYAN}{', '.join(labels)}{RESET}",
            f"wall-clock per run : {CYAN}{per_run_hr:.1f}{RESET} hr "
            f"{DIM}(rig default){RESET}",
            f"training budget    : {CYAN}{total_train_hr:.1f}{RESET} hr "
            f"{DIM}({len(labels)} runs){RESET}",
            f"K values           : {CYAN}{k_values}{RESET}",
            f"LLM temperature    : {CYAN}{temperature}{RESET}",
            f"heavy artifacts    : {DIM}{work_dir}/novelty-*-seed{seed}/{RESET}",
            f"partial CSV/plots  : {DIM}results/novelty/ (merge via aggregator){RESET}",
        ],
    )
    print(
        f"{DIM}    After all 3 rigs finish, run "
        f"`python scripts/run_novelty_aggregate.py` on any machine to{RESET}"
    )
    print(f"{DIM}    merge the three partial CSVs + plots into the final figure set.{RESET}")
    if not _confirm("Launch this rig's slice?"):
        print(f"{DIM}Aborted.{RESET}\n")
        return

    rc_iso = _run_iso_time(seed, max_seconds=0, rig=rig)
    if rc_iso != 0:
        print(f"{YELLOW}!!! iso-time slice exited {rc_iso} — K-vote skipped.{RESET}")
        sys.exit(rc_iso)

    print(f"\n{GREEN}{BOLD}Rig {rig} training complete.{RESET} Launching K-vote...\n")
    rc_kvote = _run_k_vote(seed, k_values, temperature, rig=rig)
    sys.exit(rc_kvote)


def novelty_launcher() -> None:
    """Entry point wired into the main launcher as option 7."""
    print(f"\n{BOLD}Novelty experiments{RESET}  {DIM}(iso-time + K-vote){RESET}")
    print(f"  {DIM}See results/novelty/README.md for design rationale.{RESET}\n")
    print(f"  {DIM}-- Single-rig (~17 hr end-to-end) --{RESET}")
    print(f"  {CYAN}1{RESET}) Iso-time training sweep only   {DIM}(6 runs, ~15 hr){RESET}")
    print(f"  {CYAN}2{RESET}) K-vote inference sweep only    {DIM}(requires checkpoints from option 1){RESET}")
    print(f"  {CYAN}3{RESET}) Both, sequenced                {DIM}(~17 hr end-to-end){RESET}")
    print(f"  {DIM}-- 3-rig split (each rig runs ~12-14 hr) --{RESET}")
    print(f"  {CYAN}4{RESET}) This rig's slice               {DIM}(TRM_RIG-scoped; training + K-vote){RESET}")
    print(f"  {DIM}-- Existing checkpoints --{RESET}")
    print(f"  {CYAN}5{RESET}) K-vote existing checkpoints    {DIM}(auto-discover; resume after Ctrl+C){RESET}")
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
    elif choice == "4":
        _novelty_this_rig()
    elif choice == "5":
        _novelty_kvote_existing()
    else:
        print(f"{YELLOW}!!! Unknown choice '{choice}'.{RESET}")
