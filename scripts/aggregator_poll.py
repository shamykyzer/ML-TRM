"""Aggregator polling — scans the work dir for new training artifacts.

When invoked, this script:
  1. Walks `TRM_WORK_DIR` (default `C:\\ml-trm-work\\`) for any
     `training_results.json` / `emissions.csv` newer than the last
     recorded scan timestamp in `.aggregator_state.json`.
  2. For each new artifact, parses the metrics it can and prints a
     proposed `results/summary_fixed.csv` row to stdout. Appending is
     done by the human or an outer agent — keeps this script
     deterministic and side-effect-free for the CSV.
  3. Applies Contract B realism gates (findings.md §5.14) to flag
     suspect runs as `metric realism violation` and clean runs as
     `viability gate passed`.
  4. Recognises Contract A redundancy snapshots
     (`{run_name}__{YYYY-MM-DDTHHMM}__{filename}` under
     `C:/ml-trm-work/checkpoints to use/machineN/`) and groups them
     by run so the operator sees what is being saved without
     drowning in per-snapshot rows.
  5. Refuses to mutate `report*.md` files automatically — reports any
     `<<<<<<< ` git conflict marker as `STOP: conflict <path>` to stderr.

Usage:
  python scripts/aggregator_poll.py                  # scan + print
  python scripts/aggregator_poll.py --since 1h       # only newer than 1h
  python scripts/aggregator_poll.py --reset-state    # forget last scan
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE_PATH = REPO / ".aggregator_state.json"
FINDINGS = REPO / "findings.md"
SUMMARY_FIXED = REPO / "results" / "summary_fixed.csv"
REPORT_DRAFT = REPO / "docs" / "report_methods_experiments_draft.md"
REPORT_MAIN = REPO / "docs" / "report.md"

# Contract B §B.9 calibration anchor (post-fix Maze).
# Any LLM/distill Maze re-eval whose puzzle_acc exceeds B_PUZZLE_RED_FLAG
# or whose cell_acc exceeds B_CELL_RED_FLAG is suspect — likely the
# `mask_non_path: true` bug or dataset contamination.
B_PUZZLE_RED_FLAG = 0.05
B_CELL_RED_FLAG = 0.20

# LLM/distill task substrings — matched against the parent dir name to
# decide whether realism gates apply (TRM rows are exempt; their headline
# range is captured separately in findings.md §B.2).
LLM_TASK_HINTS = ("llm-", "qwen", "smollm", "llama", "gpt2", "distill")
TRM_TASK_HINTS = ("trm-", "sudoku-mlp", "sudoku-att", "maze-seed",
                  "trm-att-maze", "trm-mlp-sudoku")


def workdir() -> Path:
    env = os.environ.get("TRM_WORK_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        return Path("C:/ml-trm-work")
    return Path.home() / "ml-trm-work"


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"last_scan_ts": 0, "seen": {}}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def parse_since(s: str | None) -> float:
    if not s:
        return 0.0
    n, unit = int(s[:-1]), s[-1]
    if unit == "h":
        return time.time() - n * 3600
    if unit == "m":
        return time.time() - n * 60
    if unit == "d":
        return time.time() - n * 86400
    raise SystemExit(f"unknown --since unit: {s!r}")


def check_report_conflicts() -> list[Path]:
    bad: list[Path] = []
    for p in (REPORT_DRAFT, REPORT_MAIN):
        if p.exists() and "<<<<<<< " in p.read_text(encoding="utf-8", errors="ignore"):
            bad.append(p)
    return bad


def scan_workdir(work: Path, since_ts: float, seen: dict) -> list[dict]:
    if not work.exists():
        print(f"[scan] work dir {work} does not exist", file=sys.stderr)
        return []
    out: list[dict] = []
    for results_json in work.rglob("*results*.json"):
        try:
            mtime = results_json.stat().st_mtime
        except OSError:
            continue
        if mtime < since_ts:
            continue
        key = str(results_json)
        if seen.get(key) == mtime:
            continue
        try:
            data = json.loads(results_json.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[scan] could not parse {results_json}: {exc}", file=sys.stderr)
            continue
        emissions = data.get("emissions", {}) or {}
        out.append({
            "path": results_json,
            "mtime": mtime,
            "best_puzzle_acc": data.get("best_puzzle_acc"),
            "best_cell_acc": data.get("best_cell_acc"),
            "energy_kwh": emissions.get("energy_kwh"),
            "co2_kg": emissions.get("emissions_kg"),
            "duration_s": emissions.get("duration_s"),
        })
        seen[key] = mtime
    return out


def realism_verdict(task_name: str, puzzle_acc: float | None,
                    cell_acc: float | None) -> tuple[str, str]:
    """Apply Contract B §B.3 / §B.7 to an LLM/distill row.

    Returns (tag, note). `tag` is one of:
        ""                           — TRM/uncategorised; no gate applied
        "viability gate passed"      — within §B.2 ranges, no red flag
        "metric realism violation"   — §B.3 red flag fired

    `note` is a short human-readable diagnosis suitable for the §5.14
    Track A / Track B sub-entry.
    """
    name = task_name.lower()
    is_llm = any(h in name for h in LLM_TASK_HINTS) and not any(
        h in name for h in ("trm-mlp", "trm-att", "trm-")
    )
    if not is_llm:
        return "", "TRM-family or unclassified — Contract B realism gate skipped (TRM rows have separate §B.2 ranges)"

    if puzzle_acc is None and cell_acc is None:
        return "", "no metrics in results JSON — manual classification required"

    if puzzle_acc is not None and puzzle_acc >= B_PUZZLE_RED_FLAG:
        return ("metric realism violation",
                f"puzzle_acc={puzzle_acc:.4f} >= {B_PUZZLE_RED_FLAG} on LLM/distill — likely mask_non_path bug or dataset contamination; investigate before reporting")

    if cell_acc is not None and cell_acc > B_CELL_RED_FLAG and "maze" in name:
        return ("metric realism violation",
                f"cell_acc={cell_acc:.4f} > {B_CELL_RED_FLAG} on Maze LLM/distill — exceeds §B.9 calibration anchor (~12.5%); investigate")

    return ("viability gate passed",
            f"within §B.2 ranges (puzzle={puzzle_acc!r}, cell={cell_acc!r}) — eligible for summary_fixed.csv")


_SNAPSHOT_PATTERN = re.compile(
    r"^(?P<run>[^_]+(?:[^_]|_(?!_))*)__(?P<ts>\d{4}-\d{2}-\d{2}T\d{4})__(?P<file>.+)$"
)


_MACHINE_DIR_PATTERN = re.compile(r"^machine\s?(\d+)$", re.IGNORECASE)


def detect_redundancy_snapshots(work: Path) -> dict[str, list[Path]]:
    """Walk `<work>/checkpoints to use/machineN/` and group snapshots by run.

    Accepts both naming variants the team has used:
      - `machineN` (Contract A.1 spec, no space, e.g. `machine1`)
      - `machine N` (existing team practice, with a space, e.g. `machine 4`)

    Snapshots inside either are matched. The space-form was introduced
    by M6's pre-Apr-28 curated artifact set and is preserved as legacy
    storage; new Contract A watchdog snapshots should land in the
    no-space form per the spec, but the aggregator is tolerant.

    Returns: {run_name: [paths]}, sorted newest-first inside each run.
    """
    out: dict[str, list[Path]] = {}
    base = work / "checkpoints to use"
    if not base.exists():
        return out
    for machine_dir in base.iterdir():
        if not machine_dir.is_dir():
            continue
        if not _MACHINE_DIR_PATTERN.match(machine_dir.name):
            continue
        for entry in machine_dir.iterdir():
            if not entry.is_file():
                continue
            m = _SNAPSHOT_PATTERN.match(entry.name)
            if not m:
                continue
            out.setdefault(m.group("run"), []).append(entry)
    for run in out:
        out[run].sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def render_proposed(rows: list[dict]) -> None:
    if not rows:
        print("[poll] no new artifacts since last scan.")
    else:
        print(f"[poll] {len(rows)} new artifacts:")
        for r in rows:
            rel = r["path"].relative_to(workdir()) if r["path"].is_relative_to(workdir()) else r["path"]
            when = datetime.fromtimestamp(r["mtime"]).isoformat(timespec="minutes")
            tag, note = realism_verdict(r["path"].parent.name,
                                        r["best_puzzle_acc"], r["best_cell_acc"])
            print(f"  - {rel}   ({when})")
            print(f"      best_puzzle_acc = {r['best_puzzle_acc']!r}")
            print(f"      best_cell_acc   = {r['best_cell_acc']!r}")
            print(f"      energy_kwh      = {r['energy_kwh']!r}")
            print(f"      co2_kg          = {r['co2_kg']!r}")
            if tag:
                print(f"      tag             = {tag}")
                print(f"      note            = {note}")
        print()
        print("Proposed summary_fixed.csv rows (one per artifact, paste/edit):")
        for r in rows:
            task = r["path"].parent.name
            ppa = r["best_puzzle_acc"] if r["best_puzzle_acc"] is not None else ""
            bca = r["best_cell_acc"] if r["best_cell_acc"] is not None else ""
            kwh = r["energy_kwh"] if r["energy_kwh"] is not None else ""
            co2 = r["co2_kg"] if r["co2_kg"] is not None else ""
            tag, _ = realism_verdict(task, r["best_puzzle_acc"], r["best_cell_acc"])
            note = tag if tag else "new artifact picked up by aggregator_poll.py"
            print(f'  {task},{ppa},{bca},,,,,,,,{kwh},{co2},,,,"{note}",')

    # Contract A redundancy-snapshot summary (always print, even if no new
    # results files — gives the operator a heartbeat that the watchdog is
    # actually saving things).
    snaps = detect_redundancy_snapshots(workdir())
    if snaps:
        print()
        print("[redundancy] snapshots present in 'checkpoints to use/machineN/':")
        for run, paths in sorted(snaps.items()):
            latest = paths[0]
            print(f"  {run}: {len(paths)} files; latest = {latest.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="only files newer than this (e.g. 1h, 30m, 2d)")
    ap.add_argument("--reset-state", action="store_true", help="forget the last-scan timestamp")
    args = ap.parse_args()

    conflicts = check_report_conflicts()
    if conflicts:
        for p in conflicts:
            print(f"STOP: conflict {p}", file=sys.stderr)
        return 2

    if args.reset_state and STATE_PATH.exists():
        STATE_PATH.unlink()

    state = load_state()
    since_ts = max(parse_since(args.since), state.get("last_scan_ts", 0))

    rows = scan_workdir(workdir(), since_ts, state["seen"])
    render_proposed(rows)

    state["last_scan_ts"] = time.time()
    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
