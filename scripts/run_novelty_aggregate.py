"""Merge per-rig novelty CSVs into one combined set and replot.

Each of the 3 rigs (see NOVELTY_RIG_PLAN in run_novelty_iso_time.py)
writes a partial CSV of its own 2 runs / 2*K k-vote rows to the shared
results/novelty/ folder. This script reads `iso_time_results-rig{1,2,3}.csv`
+ `k_vote_results-rig{1,2,3}.csv`, merges them into
`iso_time_results.csv` + `k_vote_results.csv`, and regenerates the 5
plots from the already-imported orchestrator plot functions.

Usage:
    python scripts/run_novelty_aggregate.py \
        [--seed 0] [--results-dir results/novelty] [--ignore-missing]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

# Reuse the orchestrator's dataclasses + plot functions so the merged
# output is bit-for-bit compatible with the single-rig path.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_novelty_iso_time import (  # noqa: E402
    CSV_COLUMNS as ISO_CSV_COLUMNS,
    NOVELTY_RIG_PLAN,
    RUNS,
    RunResult,
    RunSpec,
    _emit_plots as _emit_iso_plots,
    _write_csv as _write_iso_csv,
)
from run_novelty_k_vote import (  # noqa: E402
    CSV_COLUMNS as KVOTE_CSV_COLUMNS,
    KVoteRow,
    _emit_plots as _emit_kvote_plots,
    _write_csv as _write_kvote_csv,
)


# RUNS is a list; dict-indexed lookup by label / idx keeps merge O(N).
_SPEC_BY_LABEL: dict[str, RunSpec] = {s.label: s for s in RUNS}
_SPEC_BY_IDX: dict[int, RunSpec] = {s.idx: s for s in RUNS}


# --- CSV -> dataclass adapters --------------------------------------------

def _flt(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    if v in ("", None):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _intv(row: dict, key: str, default: int = 0) -> int:
    v = row.get(key, "")
    if v in ("", None):
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _row_to_run_result(row: dict) -> Optional[RunResult]:
    """Rehydrate an iso-time CSV row back into a RunResult.

    We re-lookup the RunSpec by label so model/task/config stay canonical
    even if a rig CSV somehow drifted. Unknown labels are dropped with a
    warning rather than fabricating a RunSpec on the fly.
    """
    label = (row.get("label") or "").strip()
    spec = _SPEC_BY_LABEL.get(label)
    if spec is None:
        print(f"  [warn] unknown label in iso-time CSV: {label!r} — dropped")
        return None
    r = RunResult(spec=spec, artifact_dir=Path(""))
    r.status = (row.get("status") or "").strip() or "ok"
    r.error = row.get("error") or ""
    r.wall_clock_sec = _flt(row, "wall_clock_sec")
    r.epochs_completed = _intv(row, "epochs_completed")
    r.energy_kwh = _flt(row, "kwh")
    r.emissions_kg = _flt(row, "emissions_kg")
    r.final_cell_acc = _flt(row, "final_cell_acc")
    r.final_puzzle_acc = _flt(row, "final_puzzle_acc")
    r.best_puzzle_acc = _flt(row, "best_puzzle_acc")
    r.checkpoint_path = row.get("checkpoint_path") or ""
    return r


def _row_to_kvote(row: dict) -> Optional[KVoteRow]:
    label = (row.get("label") or "").strip()
    spec = _SPEC_BY_LABEL.get(label)
    if spec is None:
        print(f"  [warn] unknown label in k-vote CSV: {label!r} — dropped")
        return None
    return KVoteRow(
        spec=spec,
        k=_intv(row, "k"),
        puzzle_acc=_flt(row, "puzzle_acc"),
        cell_acc=_flt(row, "cell_acc"),
        mean_latency_ms=_flt(row, "mean_latency_ms"),
        kwh_per_puzzle=_flt(row, "kwh_per_puzzle"),
        n_puzzles=_intv(row, "n_puzzles"),
        checkpoint_path=row.get("checkpoint_path") or "",
    )


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# --- Merge -----------------------------------------------------------------

def _gather_rigs(
    results_dir: Path, stem: str, ignore_missing: bool,
) -> tuple[list[Path], list[int], list[int]]:
    """Return (present_paths, present_rigs, missing_rigs) for a given stem.

    stem example: "iso_time_results" -> scans for "iso_time_results-rig{1,2,3}.csv".
    """
    present_paths: list[Path] = []
    present: list[int] = []
    missing: list[int] = []
    for rig in sorted(NOVELTY_RIG_PLAN):
        p = results_dir / f"{stem}-rig{rig}.csv"
        if p.is_file():
            present_paths.append(p)
            present.append(rig)
        else:
            missing.append(rig)
    if missing and not ignore_missing:
        raise FileNotFoundError(
            f"missing rig CSVs for {stem}: {missing}. "
            f"Pass --ignore-missing to merge the partial set."
        )
    return present_paths, present, missing


def _merge_iso_time(paths: list[Path]) -> list[RunResult]:
    """Dedupe by label (latest wins), sort by RunSpec.idx so plots keep the
    canonical run order regardless of which rig wrote first.
    """
    by_label: dict[str, RunResult] = {}
    for p in paths:
        for raw in _read_csv_rows(p):
            rr = _row_to_run_result(raw)
            if rr is not None:
                by_label[rr.spec.label] = rr
    ordered = sorted(by_label.values(), key=lambda r: r.spec.idx)
    return ordered


def _merge_kvote(paths: list[Path]) -> list[KVoteRow]:
    # Dedupe by (label, k); later rig wins. Sort by (run_idx, k) so the
    # accuracy curve / Pareto plots draw lines in the same order as runs.
    by_key: dict[tuple[str, int], KVoteRow] = {}
    for p in paths:
        for raw in _read_csv_rows(p):
            kr = _row_to_kvote(raw)
            if kr is not None:
                by_key[(kr.spec.label, kr.k)] = kr
    rows = sorted(by_key.values(), key=lambda r: (r.spec.idx, r.k))
    return rows


# --- Summary ---------------------------------------------------------------

def _print_section(
    title: str, present: list[int], missing: list[int], row_count: int,
) -> None:
    print(f"\n[{title}]")
    print(f"  rigs present : {present if present else '(none)'}")
    if missing:
        print(f"  rigs missing : {missing}")
    print(f"  merged rows  : {row_count}")


# --- Entry -----------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0,
                   help="recorded for traceability only; rig CSVs already "
                        "contain the metrics, no re-extraction needed")
    p.add_argument("--results-dir", type=str, default="results/novelty",
                   help="directory holding iso_time_results-rig*.csv and "
                        "k_vote_results-rig*.csv")
    p.add_argument("--ignore-missing", action="store_true",
                   help="merge whatever rigs are present instead of erroring")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (REPO_ROOT / results_dir).resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"results dir not found: {results_dir}")

    print(f"[aggregate] results_dir : {results_dir}")
    print(f"[aggregate] seed        : {args.seed}")

    iso_paths, iso_present, iso_missing = _gather_rigs(
        results_dir, "iso_time_results", args.ignore_missing,
    )
    kv_paths, kv_present, kv_missing = _gather_rigs(
        results_dir, "k_vote_results", args.ignore_missing,
    )

    iso_rows = _merge_iso_time(iso_paths)
    kv_rows = _merge_kvote(kv_paths)

    _print_section("iso-time", iso_present, iso_missing, len(iso_rows))
    _print_section("k-vote",   kv_present,  kv_missing,  len(kv_rows))

    # Reuse the orchestrator writers so schemas + formatting stay in lockstep
    # with the single-rig output.
    if iso_rows:
        _write_iso_csv(iso_rows, results_dir / "iso_time_results.csv")
        _emit_iso_plots(iso_rows, results_dir)
    else:
        print("\n[iso-time] no rows merged — skipping CSV + plots")

    if kv_rows:
        _write_kvote_csv(kv_rows, results_dir / "k_vote_results.csv")
        _emit_kvote_plots(kv_rows, results_dir)
    else:
        print("\n[k-vote] no rows merged — skipping CSV + plots")

    # Sanity-check expected shape only when --ignore-missing wasn't used, so
    # partial merges don't flood stderr with warnings.
    if not args.ignore_missing:
        if len(iso_rows) != len(RUNS):
            print(f"  [warn] expected {len(RUNS)} iso-time rows, got {len(iso_rows)}")

    # Report which CSV columns were written (useful when downstream scripts
    # grow new columns faster than the CSV schema).
    print(f"\n[aggregate] iso-time columns: {ISO_CSV_COLUMNS}")
    print(f"[aggregate] k-vote   columns: {KVOTE_CSV_COLUMNS}")


if __name__ == "__main__":
    main()
