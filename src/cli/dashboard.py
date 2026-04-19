"""Plain-text project state dashboard for ML-TRM.

Renders bordered text panels summarising what's happened in this repo so
far — per-task best accuracy, seed-run states, CO2 per correct puzzle,
open decisions parsed from findings.md, queued next actions, and current
environment readiness.

Stdlib-only: uses csv, re, os, datetime. Post-venv (can assume Python
3.10+). Every data source is read with a (rows, warnings) tuple so a
missing CSV or truncated file renders a `(no data)` hint instead of
crashing. Action is read-only: dashboard NEVER dispatches training.

sudoku-mlp (mlp_t=true) and sudoku-att (mlp_t=false) are always shown as
two separate rows. The experiments-panel dedup key is (dataset, mlp_t),
never just dataset.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from src.cli.console import BOLD, CYAN, DIM, GREEN, RESET, YELLOW
from src.cli.paths import (
    HF_REMAPPED_MAZE,
    HF_REMAPPED_SUDOKU_ATT,
    HF_REMAPPED_SUDOKU_MLP,
    ROOT,
    WANDB,
)

_SUMMARY_CSV = os.path.join(ROOT, "results", "summary.csv")
_RUNS_OVERVIEW_CSV = os.path.join(ROOT, "results", "trm_runs_overview.csv")
_FINDINGS_MD = os.path.join(ROOT, "findings.md")


# ---------------------------------------------------------------------------
# Data loaders — all fallback-safe
# ---------------------------------------------------------------------------

def _load_csv_rows(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read a CSV into a list of dicts. Returns (rows, warnings).

    Missing file -> empty list + warning. Parse errors on individual rows
    are tolerated (the bad row is skipped and appended to warnings).
    """
    if not os.path.exists(path):
        return [], [f"{os.path.relpath(path, ROOT)} not found"]
    try:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows: List[Dict[str, str]] = []
            for i, row in enumerate(reader, 1):
                try:
                    rows.append(row)
                except Exception as exc:  # noqa: BLE001
                    return rows, [f"{os.path.relpath(path, ROOT)} row {i}: {exc}"]
        return rows, []
    except OSError as exc:
        return [], [f"{os.path.relpath(path, ROOT)}: {exc}"]


def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _fmt_float(x: Optional[float], digits: int = 3) -> str:
    return f"{x:.{digits}f}" if x is not None else "-"


def _fmt_sci(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:.2e}"


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def _panel_header(title: str, width: int = 76) -> None:
    bar = "=" * width
    print(f"{BOLD}{bar}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{bar}{RESET}")


def _section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")


def _panel_experiments() -> None:
    """Per-task best accuracy, current state, CO2/correct."""
    _section("EXPERIMENTS (per-task best)")
    summary_rows, summary_warn = _load_csv_rows(_SUMMARY_CSV)
    runs_rows, runs_warn = _load_csv_rows(_RUNS_OVERVIEW_CSV)

    if not summary_rows and not runs_rows:
        print(f"  {DIM}(no data yet \u2014 run: python scripts/aggregate_metrics.py{RESET}")
        print(f"  {DIM}                 and: python scripts/aggregate_wandb_runs.py){RESET}")
        return

    # Index runs by (dataset, mlp_t) so sudoku-mlp and sudoku-att stay distinct.
    runs_by_task: Dict[Tuple[str, Optional[bool]], List[Dict[str, str]]] = {}
    for row in runs_rows:
        dataset = (row.get("dataset") or "").strip()
        mlp_t_raw = (row.get("mlp_t") or "").strip()
        mlp_t: Optional[bool]
        if mlp_t_raw.lower() in {"true", "1"}:
            mlp_t = True
        elif mlp_t_raw.lower() in {"false", "0"}:
            mlp_t = False
        else:
            mlp_t = None
        runs_by_task.setdefault((dataset, mlp_t), []).append(row)

    header = f"  {'task':<16} {'best':>6} {'state':>10} {'seeds':>7} {'CO2/correct':>13}"
    print(f"  {DIM}{'-' * (len(header) - 2)}{RESET}")
    print(f"  {BOLD}{'task':<16} {'best':>6} {'state':>10} {'seeds':>7} {'CO2/correct':>13}{RESET}")

    # summary.csv drives the row order (one row per canonical task).
    # Row names in summary.csv sometimes carry a -seedN suffix (mixed
    # conventions over the project's history). Strip it so sudoku-mlp-seed0
    # and sudoku-mlp collapse to the same canonical row — then keep the
    # BEST numbers per canonical key.
    seed_suffix = re.compile(r"-seed\d+$")
    canonical: Dict[str, Dict[str, str]] = {}
    for row in summary_rows:
        raw_task = (row.get("task") or "").strip()
        if not raw_task:
            continue
        task = seed_suffix.sub("", raw_task)
        existing = canonical.get(task)
        best_new = _to_float(row.get("best_val_puzzle_acc")) or -1.0
        best_old = _to_float(existing.get("best_val_puzzle_acc") if existing else None) or -1.0
        if existing is None or best_new > best_old:
            canonical[task] = row

    for task, row in canonical.items():
        best = _to_float(row.get("best_val_puzzle_acc"))
        co2_per_correct = _to_float(row.get("co2_per_correct_puzzle"))

        # Match runs by dataset + mlp_t. summary.csv uses tasks like
        # "sudoku-mlp", "sudoku-att", "maze" which we split back to
        # (dataset, mlp_t) for the runs index.
        if task == "sudoku-mlp":
            runs = runs_by_task.get(("sudoku", True), [])
        elif task == "sudoku-att":
            runs = runs_by_task.get(("sudoku", False), [])
        elif task == "maze":
            runs = (
                runs_by_task.get(("maze", False), [])
                + runs_by_task.get(("maze", None), [])
            )
        elif task == "sudoku":
            # Legacy simple-TRM rows in summary.csv are labelled "sudoku"
            # (not sudoku-mlp / sudoku-att). No wandb match; show them
            # in a muted state.
            runs = []
        else:
            runs = []

        states = {(r.get("state") or "unknown").strip().lower() for r in runs}
        if "running" in states:
            state = "running"
            state_colored = f"{YELLOW}{state:>10}{RESET}"
        elif "finished" in states:
            state = "finished"
            state_colored = f"{GREEN}{state:>10}{RESET}"
        elif "crashed" in states or "failed" in states or "killed" in states:
            state = next(s for s in ("crashed", "failed", "killed") if s in states)
            state_colored = f"{YELLOW}{state:>10}{RESET}"
        else:
            state = "-"
            state_colored = f"{DIM}{state:>10}{RESET}"

        seeds = len({(r.get("seed") or "").strip() for r in runs if r.get("seed")})
        seed_label = f"{seeds}/3" if task in {"sudoku-mlp", "sudoku-att", "maze"} else f"{seeds}/1"

        print(
            f"  {task:<16} "
            f"{_fmt_float(best):>6} "
            f"{state_colored} "
            f"{seed_label:>7} "
            f"{_fmt_sci(co2_per_correct):>13}"
        )

    if summary_warn or runs_warn:
        for w in summary_warn + runs_warn:
            print(f"  {DIM}warn: {w}{RESET}")


def _panel_decisions() -> None:
    """Open decisions + next actions from findings.md."""
    _section("OPEN DECISIONS (findings.md)")
    if not os.path.exists(_FINDINGS_MD):
        print(f"  {DIM}findings.md not found{RESET}")
        return
    try:
        with open(_FINDINGS_MD, encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        print(f"  {DIM}findings.md unreadable: {exc}{RESET}")
        return

    # Pull out the first few bullets under each of a handful of section
    # headings that the user actively maintains. This is intentionally
    # permissive — findings.md evolves, so we pick up whichever of these
    # headings currently exist.
    wanted_headings = [
        r"^##\s*TL;DR",
        r"^##\s*\d*\.?\s*Decision",
        r"^##\s*\d*\.?\s*What\s+I\s+want",
        r"^##\s*\d*\.?\s*Next",
        r"^##\s*\d*\.?\s*Things\s+that\s+are\s+NOT",
    ]
    lines = text.splitlines()
    any_shown = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if any(re.match(pat, line) for pat in wanted_headings):
            any_shown = True
            heading = line.strip("# ").strip()
            print(f"  {CYAN}\u25b8 {heading}{RESET}")
            # print up to 6 bullet lines under this heading
            shown = 0
            j = i + 1
            while j < len(lines) and shown < 6:
                lj = lines[j].rstrip()
                if lj.startswith("## "):
                    break
                if lj.lstrip().startswith(("- ", "* ", "1.", "2.", "3.", "4.")):
                    # trim to ~90 chars for terminal width
                    trimmed = lj if len(lj) <= 90 else lj[:87] + "..."
                    print(f"    {DIM}{trimmed}{RESET}")
                    shown += 1
                j += 1
            i = j
        else:
            i += 1

    if not any_shown:
        print(f"  {DIM}(no standard sections matched \u2014 open findings.md to read in full){RESET}")


def _panel_transfer_status() -> None:
    """Which HF remapped checkpoints are present on disk."""
    _section("HF REFERENCE CHECKPOINTS")
    for label, path in [
        ("sudoku-mlp", HF_REMAPPED_SUDOKU_MLP),
        ("sudoku-att", HF_REMAPPED_SUDOKU_ATT),
        ("maze",       HF_REMAPPED_MAZE),
    ]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"  {GREEN}\u2713{RESET} {label:<12} {DIM}{os.path.basename(os.path.dirname(path))}/remapped_for_local.pt  ({size_mb:.0f} MB){RESET}")
        else:
            print(f"  {YELLOW}\u2717{RESET} {label:<12} {DIM}missing \u2014 run: python start.py  (transfer stage){RESET}")


def _panel_env() -> None:
    """Venv, GPU, work dir, wandb readiness."""
    _section("ENVIRONMENT")
    work_dir = os.environ.get("TRM_WORK_DIR", "")
    if work_dir:
        wd_label = f"{CYAN}{work_dir}{RESET}"
    else:
        wd_label = f"{DIM}<unset \u2014 auto-picked at launch>{RESET}"
    print(f"  TRM_WORK_DIR : {wd_label}")

    gpu_label = _probe_gpu()
    print(f"  GPU          : {gpu_label}")

    wandb_label = (
        f"{GREEN}\u2713 logged in{RESET}"
        if _wandb_authed()
        else f"{YELLOW}not logged in{RESET}"
    )
    print(f"  wandb        : {wandb_label}")


def _probe_gpu() -> str:
    """Best-effort: run `nvidia-smi -L` and return the first GPU label.

    Returns a plain string on failure; the caller prints that to the panel.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=4, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return f"{DIM}nvidia-smi not available{RESET}"
    if r.returncode != 0 or not r.stdout.strip():
        return f"{DIM}no GPU detected{RESET}"
    first_line = r.stdout.strip().splitlines()[0]
    # typical: "GPU 0: NVIDIA GeForce RTX 5070 (UUID: GPU-xxxx)"
    m = re.search(r"GPU \d+:\s*(.+?)(?:\s*\(UUID|$)", first_line)
    label = m.group(1).strip() if m else first_line
    return f"{CYAN}{label}{RESET}"


def _wandb_authed() -> bool:
    """True when WANDB_API_KEY env, ~/.netrc, or ~/_netrc has wandb creds."""
    if os.getenv("WANDB_API_KEY"):
        return True
    home = os.path.expanduser("~")
    for name in (".netrc", "_netrc"):
        path = os.path.join(home, name)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    if "api.wandb.ai" in f.read():
                        return True
            except OSError:
                continue
    return False


def _panel_suggested_actions() -> None:
    """Tiny crib sheet of things to run next."""
    _section("RUN")
    print(f"  {CYAN}python start.py{RESET}                    {DIM}# interactive menu (setup/resume/train){RESET}")
    print(f"  {CYAN}python start.py <task> <seed>{RESET}      {DIM}# direct-launch with preflight{RESET}")
    print(f"  {CYAN}python start.py menu{RESET}               {DIM}# copy-paste command list{RESET}")
    print(f"  {CYAN}python scripts/eval_hf_checkpoints.py{RESET}  {DIM}# Regime A (paper checkpoints){RESET}")
    print(f"  {CYAN}python scripts/aggregate_metrics.py{RESET}     {DIM}# refresh results/summary.csv{RESET}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_dashboard(compact: bool = False) -> None:
    """Print the full dashboard to stdout.

    When `compact` is True, drops the trailing "suggested actions" block
    because the caller (main() in bootstrap) is about to print the full
    menu anyway.
    """
    _panel_header(f"ML-TRM Dashboard")
    _panel_env()
    _panel_transfer_status()
    _panel_experiments()
    _panel_decisions()
    if not compact:
        _panel_suggested_actions()
    print()
