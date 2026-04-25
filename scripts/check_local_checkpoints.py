"""Inventory C:/ml-trm-work/ (or local equivalent) and tell this machine what
to run next per the Apr-26 sprint plan.

Why: 6 machines run different jobs, each with several phases. This script
makes each machine self-aware — it scans the local work dir, sees which
checkpoints exist (and how stale they are), and prints the status of every
task in the sprint plan plus a "next action" suggestion. Avoids the
classic "I forgot which seed I started" hand-off problem.

Usage:
    python scripts/check_local_checkpoints.py            # report all tasks
    python scripts/check_local_checkpoints.py --machine M1   # narrow to that box
    python scripts/check_local_checkpoints.py --json     # machine-readable

Reads (no writes):
  TRM_WORK_DIR (env) or src.cli.workdir._default_work_dir() default
  -> typically C:/ml-trm-work on Windows, ~/ml-trm-work elsewhere

Each task row reports:
  status   = DONE / IN-PROGRESS / NOT-STARTED
  size     = best.pt size in MB
  modified = mtime of best.pt
  notes    = "missing best.pt" / "stale" warnings
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Sprint plan — one row per task expected on each machine.
# Edit this list when the plan changes; the script picks up the diff
# automatically. Folder names match what trainers write under TRM_WORK_DIR.
# ---------------------------------------------------------------------------

@dataclass
class Task:
    machine: str
    task_id: str
    folder_glob: list[str]   # candidate folder names under work_dir
    description: str
    teacher_for: str | None = None    # if non-None, this is a teacher needed by another task
    needs_teacher: str | None = None  # if non-None, task_id of the teacher we depend on
    notes: str = ""


PLAN: list[Task] = [
    # M1/M2/M3 — TRM-Att Maze fine-tune, 50 epochs, 1 seed each
    Task("M1", "trm-att-maze-seed0",
         ["trm-att-maze-50ep-seed0", "maze-seed0"],
         "TRM-Att Maze fine-tune (seed 0, 50 ep)"),
    Task("M2", "trm-att-maze-seed1",
         ["trm-att-maze-50ep-seed1", "maze-seed1"],
         "TRM-Att Maze fine-tune (seed 1, 50 ep)"),
    Task("M3", "trm-att-maze-seed2",
         ["trm-att-maze-50ep-seed2", "maze-seed2"],
         "TRM-Att Maze fine-tune (seed 2, 50 ep)"),

    # M4 — Sudoku LLM family + K-vote
    Task("M4", "gpt2-sudoku",
         ["llm-gpt2-sudoku-seed0", "gpt2-sudoku-seed0"],
         "GPT-2 Sudoku LoRA (30 ep)",
         teacher_for="distill-gpt2-sudoku"),
    Task("M4", "distill-gpt2-sudoku",
         ["distill-gpt2-sudoku-seed0", "novelty-distill-gpt2-sudoku-seed0"],
         "Distilled student from GPT-2 Sudoku",
         needs_teacher="gpt2-sudoku"),
    Task("M4", "distill-qwen-sudoku",
         ["distill-qwen-sudoku-seed0", "novelty-distill-sudoku-seed0"],
         "Distilled student from Qwen Sudoku (existing teacher)",
         notes="teacher: existing C:/ml-trm-work/llm-qwen-sudoku-seed0/qwen2.5_0.5b_sudoku_latest.pt"),
    Task("M4", "kvote-trm-mlp-sudoku",
         ["novelty/k_vote_runs/trm-mlp-sudoku", "k_vote_runs/trm-mlp-sudoku"],
         "K-vote sweep on TRM-MLP-Sudoku at K∈{1,2,4}"),

    # M5 — GPT-2 Maze family
    Task("M5", "gpt2-maze",
         ["llm-gpt2-maze-seed0", "gpt2-maze-seed0"],
         "GPT-2 Maze LoRA (30 ep)",
         teacher_for="distill-gpt2-maze"),
    Task("M5", "distill-gpt2-maze",
         ["distill-gpt2-maze-seed0", "novelty-distill-maze-seed0"],
         "Distilled student from GPT-2 Maze",
         needs_teacher="gpt2-maze"),

    # M6 — Qwen Maze family (NO RETRAIN — uses existing checkpoint)
    Task("M6", "qwen-maze-existing",
         ["llm-qwen-maze-seed0"],
         "Qwen-Maze EXISTING checkpoint (no retrain — re-eval first)",
         teacher_for="distill-qwen-maze"),
    Task("M6", "distill-qwen-maze",
         ["distill-qwen-maze-seed0"],
         "Distilled student from Qwen Maze",
         needs_teacher="qwen-maze-existing"),
]


# ---------------------------------------------------------------------------
# Workdir discovery
# ---------------------------------------------------------------------------

def resolve_workdir() -> Path:
    """Resolve TRM_WORK_DIR with the same logic as src.cli.workdir."""
    env = os.environ.get("TRM_WORK_DIR")
    if env:
        return Path(env)
    try:
        from src.cli.workdir import _default_work_dir  # type: ignore
        return Path(_default_work_dir())
    except Exception:
        # Fallback if the package can't be imported (e.g. on a stripped node).
        if sys.platform == "win32":
            return Path("C:/ml-trm-work")
        return Path.home() / "ml-trm-work"


# ---------------------------------------------------------------------------
# Scan logic
# ---------------------------------------------------------------------------

@dataclass
class TaskState:
    task: Task
    folder: Path | None = None
    best_pt: Path | None = None
    latest_pt: Path | None = None
    has_eval_override: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.best_pt and self.best_pt.exists():
            return "DONE"
        if self.latest_pt and self.latest_pt.exists():
            return "IN-PROGRESS"
        if self.folder and self.folder.exists():
            return "STARTED"
        return "NOT-STARTED"


def find_task_state(task: Task, work_dir: Path) -> TaskState:
    state = TaskState(task)
    for glob in task.folder_glob:
        candidate = work_dir / glob
        if candidate.exists():
            state.folder = candidate
            best = list(candidate.glob("best*.pt")) + list(candidate.glob("*best.pt"))
            if best:
                # Pick the largest file (typical convention: latest best)
                state.best_pt = max(best, key=lambda p: p.stat().st_size)
            latest = list(candidate.glob("*latest*.pt"))
            if latest:
                state.latest_pt = latest[0]
            if (candidate / "eval_override.json").exists():
                state.has_eval_override = True
            break

    if state.folder and not state.best_pt and not state.latest_pt:
        state.notes.append("folder exists but no best.pt or latest.pt — check status")

    if state.best_pt:
        age_days = (time.time() - state.best_pt.stat().st_mtime) / 86400
        if age_days > 14:
            state.notes.append(f"best.pt is {age_days:.0f} days old")

    return state


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt_size(path: Path | None) -> str:
    if not path or not path.exists():
        return "-"
    size_mb = path.stat().st_size / 1e6
    return f"{size_mb:>4.0f} MB"


def fmt_mtime(path: Path | None) -> str:
    if not path or not path.exists():
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))


STATUS_GLYPH = {
    "DONE": "[ok]",
    "IN-PROGRESS": "[..]",
    "STARTED": "[--]",
    "NOT-STARTED": "[  ]",
}


def render_text(states: list[TaskState], machine_filter: str | None) -> None:
    if machine_filter:
        states = [s for s in states if s.task.machine == machine_filter]

    by_machine: dict[str, list[TaskState]] = {}
    for s in states:
        by_machine.setdefault(s.task.machine, []).append(s)

    print()
    print(f"  {'task':<28s}  status        size       modified            notes")
    print("  " + "-" * 90)
    for machine in sorted(by_machine):
        print(f"\n  -- {machine} --")
        for s in by_machine[machine]:
            status = s.status
            glyph = STATUS_GLYPH[status]
            line = (
                f"  {s.task.task_id:<28s}  {glyph} {status:<11s} "
                f"{fmt_size(s.best_pt):<10s} {fmt_mtime(s.best_pt):<19s}"
            )
            extras = list(s.notes)
            if s.task.needs_teacher:
                extras.append(f"needs teacher: {s.task.needs_teacher}")
            if s.has_eval_override:
                extras.append("eval_override.json present")
            elif status == "DONE" and not s.task.task_id.startswith("kvote"):
                extras.append("no eval_override.json — consider Fix-B re-eval")
            if extras:
                line += "  " + " | ".join(extras)
            print(line)

    # Suggested next action per machine
    print()
    print("  Suggested next:")
    for machine in sorted(by_machine):
        next_task = _next_for_machine(by_machine[machine])
        if next_task:
            print(f"    {machine}: {next_task}")
        else:
            print(f"    {machine}: all tasks DONE — proceed to scripts/finalize_results.py")
    print()


def _next_for_machine(states: list[TaskState]) -> str | None:
    """First task on this machine that isn't DONE and whose teacher (if any) IS DONE."""
    state_by_id = {s.task.task_id: s for s in states}
    for s in states:
        if s.status == "DONE":
            continue
        if s.task.needs_teacher and state_by_id.get(s.task.needs_teacher, None):
            teacher = state_by_id[s.task.needs_teacher]
            if teacher.status != "DONE":
                continue  # teacher not ready yet
        return f"{s.task.task_id} ({s.task.description})"
    return None


def render_json(states: list[TaskState], machine_filter: str | None) -> None:
    if machine_filter:
        states = [s for s in states if s.task.machine == machine_filter]
    out = []
    for s in states:
        out.append({
            "machine": s.task.machine,
            "task_id": s.task.task_id,
            "description": s.task.description,
            "status": s.status,
            "folder": str(s.folder) if s.folder else None,
            "best_pt": str(s.best_pt) if s.best_pt else None,
            "best_pt_mb": (s.best_pt.stat().st_size / 1e6) if s.best_pt and s.best_pt.exists() else None,
            "best_pt_mtime": fmt_mtime(s.best_pt),
            "has_eval_override": s.has_eval_override,
            "notes": s.notes,
            "needs_teacher": s.task.needs_teacher,
            "teacher_for": s.task.teacher_for,
        })
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--machine", choices=["M1", "M2", "M3", "M4", "M5", "M6"],
                        help="Show only tasks for this machine")
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable JSON output instead of text table")
    args = parser.parse_args(list(argv) if argv is not None else None)

    work_dir = resolve_workdir()
    print(f"  scanning: {work_dir}")
    if not work_dir.exists():
        print(f"  WARNING: {work_dir} does not exist. Did you run a training job yet?")

    states = [find_task_state(t, work_dir) for t in PLAN]

    if args.json:
        render_json(states, args.machine)
    else:
        render_text(states, args.machine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
