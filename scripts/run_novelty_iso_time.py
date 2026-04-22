"""Orchestrate the 6-run iso-wall-clock novelty sweep for the final report.

Every model gets the same TRM_MAX_TRAIN_SECONDS budget (default 2.5 hr).
Heavy artifacts land in $TRM_WORK_DIR/novelty-<label>-seed<N>/ (local,
non-OneDrive); aggregated CSV + plots land in results/novelty/ so they
sync with the repo for the writeup.

Run order is load-bearing: distill-sudoku and distill-maze consume the
teacher latest.pt produced by runs 3 and 4. --skip-runs respects this.

Usage:
    python scripts/run_novelty_iso_time.py \
        [--seed 0] [--max-train-seconds 9000] \
        [--work-dir C:/ml-trm-work] [--skip-runs 1,2] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Orchestrator must be invokable on system Python without importing the
# project's training deps, so we only pull in the two modules it truly
# needs (workdir resolution + HF-path constants). matplotlib is imported
# lazily inside _emit_plots so --dry-run stays dep-free.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cli.paths import HF_REMAPPED_SUDOKU_MLP, HF_REMAPPED_MAZE  # noqa: E402
from src.cli.workdir import _resolve_work_dir  # noqa: E402


@dataclass
class RunSpec:
    idx: int
    label: str
    task: str          # "sudoku" | "maze" — for grouping plots
    model: str         # "trm-mlp" | "trm-att" | "qwen" | "distilled" — x-axis label
    config: str        # relative path under repo
    mode: str          # "train" | "distill"
    init_weights: Optional[str] = None
    # For distill runs: idx of the prior run whose latest.pt is the teacher.
    teacher_run_idx: Optional[int] = None


RUNS: list[RunSpec] = [
    RunSpec(1, "trm-mlp-sudoku",  "sudoku", "trm-mlp",   "configs/trm_official_sudoku_mlp.yaml", "train",
            init_weights=HF_REMAPPED_SUDOKU_MLP),
    RunSpec(2, "trm-att-maze",    "maze",   "trm-att",   "configs/trm_official_maze.yaml",      "train",
            init_weights=HF_REMAPPED_MAZE),
    RunSpec(3, "qwen-sudoku",     "sudoku", "qwen",      "configs/llm_qwen.yaml",               "train"),
    RunSpec(4, "qwen-maze",       "maze",   "qwen",      "configs/llm_qwen_maze.yaml",          "train"),
    RunSpec(5, "distill-sudoku",  "sudoku", "distilled", "configs/llm_qwen.yaml",               "distill",
            teacher_run_idx=3),
    RunSpec(6, "distill-maze",    "maze",   "distilled", "configs/llm_qwen_maze.yaml",          "distill",
            teacher_run_idx=4),
]


@dataclass
class RunResult:
    spec: RunSpec
    artifact_dir: Path
    status: str = "pending"  # "ok" | "error" | "skipped"
    error: str = ""
    wall_clock_sec: float = 0.0
    # Parsed from training_results.json + emissions.csv + train_log.csv.
    epochs_completed: int = 0
    energy_kwh: float = 0.0
    emissions_kg: float = 0.0
    final_cell_acc: float = 0.0
    final_puzzle_acc: float = 0.0
    best_puzzle_acc: float = 0.0
    checkpoint_path: str = ""


# ---------------------------------------------------------------------------
# Path + artifact helpers
# ---------------------------------------------------------------------------

def _artifact_dir(work_dir: Path, label: str, seed: int) -> Path:
    return work_dir / f"novelty-{label}-seed{seed}"


def _teacher_checkpoint(spec: RunSpec, results: dict[int, RunResult]) -> Path:
    """Resolve the teacher latest.pt path for a distill run.

    LLM runs save as `<model_tag>_latest.pt` (see trainer_llm.py), where
    model_tag = f"{llm_name.split('/')[-1].lower().replace('-', '_')}_{dataset}".
    For Qwen2.5-0.5B that's `qwen2.5_0.5b_<sudoku|maze>_latest.pt`.
    """
    assert spec.teacher_run_idx is not None
    teacher_result = results.get(spec.teacher_run_idx)
    if teacher_result is None:
        raise RuntimeError(
            f"distill run {spec.idx} needs run {spec.teacher_run_idx} "
            f"but that run was skipped/missing"
        )
    # Derived tag for Qwen2.5-0.5B. If you swap the LLM, update this to
    # match trainer_llm.py's _save_checkpoint filename.
    dataset = "maze" if spec.task == "maze" else "sudoku"
    filename = f"qwen2.5_0.5b_{dataset}_latest.pt"
    return teacher_result.artifact_dir / filename


# ---------------------------------------------------------------------------
# Subprocess argv + env construction
# ---------------------------------------------------------------------------

def _build_argv(
    spec: RunSpec, seed: int, teacher_path: Optional[Path]
) -> list[str]:
    argv = [
        sys.executable, str(REPO_ROOT / "main.py"),
        "--mode", spec.mode,
        "--config", spec.config,
        "--seed", str(seed),
    ]
    if spec.mode == "train" and spec.init_weights:
        # HF-init is optional: only pass if the remapped file exists, so a
        # fresh clone without the 2.47 GB reference tarball still runs.
        if Path(spec.init_weights).is_file():
            argv += ["--init-weights", spec.init_weights]
    if spec.mode == "distill":
        assert teacher_path is not None
        argv += ["--checkpoint", str(teacher_path)]
    return argv


def _build_env(artifact_dir: Path, max_seconds: int) -> dict[str, str]:
    env = os.environ.copy()
    env["TRM_MAX_TRAIN_SECONDS"] = str(max_seconds)
    env["TRM_CHECKPOINT_DIR"] = str(artifact_dir)
    env["TRM_EXPERIMENT_DIR"] = str(artifact_dir)
    return env


# ---------------------------------------------------------------------------
# Result extraction — three trainers, three JSON shapes
# ---------------------------------------------------------------------------

def _parse_emissions(data: dict) -> tuple[float, float]:
    """Pull (kWh, kg CO2eq) out of the `emissions` key, which is either a dict
    (carbon.stop() return) or None if the tracker bailed out."""
    e = data.get("emissions") or {}
    if isinstance(e, dict):
        return float(e.get("energy_kwh") or 0.0), float(e.get("emissions_kg") or 0.0)
    # Very old runs stored just a float (kg). Energy unknown → 0.
    try:
        return 0.0, float(e)
    except (TypeError, ValueError):
        return 0.0, 0.0


def _find_results_json(artifact_dir: Path) -> Optional[Path]:
    """Trainers write different filenames. Pick whichever exists:

    - trainer_official → training_results.json
    - trainer_llm      → <model_tag>_training_results.json
    - trainer_distill  → distill_<dataset>_results.json
    """
    for pat in (
        "training_results.json",
        "*_training_results.json",
        "*_results.json",
    ):
        hits = sorted(artifact_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _parse_train_log(artifact_dir: Path) -> tuple[int, float, float]:
    """Grab (epochs_completed, final_cell_acc, final_puzzle_acc) from CSV.

    trainer_official writes <model_type>_train_log.csv with val_cell_acc /
    val_puzzle_acc columns. trainer_llm + trainer_distill use val_cell_acc
    and val_puzzle_acc too (different filename prefixes). Last non-blank
    row wins — blank cells mean eval hadn't run yet that epoch.
    """
    hits = sorted(artifact_dir.glob("*train_log.csv"))
    if not hits:
        return 0, 0.0, 0.0
    with hits[0].open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, 0.0, 0.0

    def _flt(row: dict, *keys: str) -> float:
        for k in keys:
            if k in row and row[k] not in ("", None):
                try:
                    return float(row[k])
                except ValueError:
                    pass
        return 0.0

    # Scan backwards for the last row with a non-blank val_cell_acc.
    final_cell = final_puzzle = 0.0
    for row in reversed(rows):
        cell = _flt(row, "val_cell_acc")
        if cell > 0 or row.get("val_cell_acc") not in ("", None):
            final_cell = cell
            final_puzzle = _flt(row, "val_puzzle_acc")
            break
    epochs = 0
    try:
        epochs = int(rows[-1].get("epoch", 0))
    except (TypeError, ValueError):
        pass
    return epochs, final_cell, final_puzzle


def _collect_metrics(result: RunResult) -> None:
    """Populate the metric fields on `result` from on-disk artifacts."""
    results_json = _find_results_json(result.artifact_dir)
    best_puzzle = 0.0
    kwh = kg = 0.0
    if results_json is not None:
        try:
            with results_json.open() as f:
                data = json.load(f)
            best_puzzle = float(data.get("best_puzzle_acc") or 0.0)
            kwh, kg = _parse_emissions(data)
        except (json.JSONDecodeError, OSError):
            pass

    epochs, cell, puzzle = _parse_train_log(result.artifact_dir)
    # LLM + distill trainers don't emit best_puzzle_acc — fall back to the
    # last logged val_puzzle_acc so the column isn't empty for those rows.
    result.best_puzzle_acc = best_puzzle or puzzle
    result.energy_kwh = kwh
    result.emissions_kg = kg
    result.epochs_completed = epochs
    result.final_cell_acc = cell
    result.final_puzzle_acc = puzzle

    # Figure out which checkpoint this run would hand to a downstream step.
    # TRM official writes latest.pt; LLM/distill use a prefixed filename.
    candidates = (
        result.artifact_dir / "latest.pt",
        *sorted(result.artifact_dir.glob("*_latest.pt")),
    )
    for cand in candidates:
        if Path(cand).is_file():
            result.checkpoint_path = str(cand)
            break


# ---------------------------------------------------------------------------
# Run driver
# ---------------------------------------------------------------------------

def _run_one(
    spec: RunSpec, seed: int, work_dir: Path, max_seconds: int,
    results_so_far: dict[int, RunResult], dry_run: bool,
) -> RunResult:
    artifact_dir = _artifact_dir(work_dir, spec.label, seed)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result = RunResult(spec=spec, artifact_dir=artifact_dir)

    try:
        teacher = _teacher_checkpoint(spec, results_so_far) if spec.mode == "distill" else None
        if teacher is not None and not teacher.is_file() and not dry_run:
            # Fail this run loudly but don't abort the whole sweep — later
            # runs might still be independent of this one.
            raise FileNotFoundError(f"teacher checkpoint missing: {teacher}")
    except Exception as e:
        result.status = "error"
        result.error = str(e)
        print(f"  [run {spec.idx}] SKIPPED: {e}")
        return result

    argv = _build_argv(spec, seed, teacher)
    env = _build_env(artifact_dir, max_seconds)

    print(f"\n{'=' * 72}")
    print(f"[run {spec.idx}/{len(RUNS)}] {spec.label}")
    print(f"  config     : {spec.config}")
    print(f"  mode       : {spec.mode}")
    print(f"  artifact   : {artifact_dir}")
    print(f"  budget     : {max_seconds}s  (TRM_MAX_TRAIN_SECONDS)")
    if teacher is not None:
        print(f"  teacher    : {teacher}")
    print(f"  argv       : {' '.join(argv)}")
    print(f"{'=' * 72}\n")

    if dry_run:
        result.status = "skipped"
        return result

    t0 = time.time()
    log_path = artifact_dir / "stdout.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log_fh:
        # Tee: stream to the run log, not the orchestrator's terminal. The
        # orchestrator's own stdout stays readable (you want to watch run
        # transitions, not 10,000 tqdm bar refreshes). tail -F the log to
        # follow a specific run in another pane.
        proc = subprocess.run(
            argv,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
        )
    result.wall_clock_sec = time.time() - t0

    if proc.returncode != 0:
        result.status = "error"
        result.error = f"exit code {proc.returncode} — see {log_path}"
        print(f"  [run {spec.idx}] FAILED ({result.error})")
    else:
        result.status = "ok"
        print(f"  [run {spec.idx}] OK in {result.wall_clock_sec / 60:.1f} min")

    _collect_metrics(result)
    return result


# ---------------------------------------------------------------------------
# Aggregation — CSV + plots
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "run_id", "label", "task", "model", "config",
    "epochs_completed", "wall_clock_sec", "kwh", "emissions_kg",
    "final_puzzle_acc", "final_cell_acc", "best_puzzle_acc",
    "checkpoint_path", "status", "error",
]


def _write_csv(results: list[RunResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in results:
            w.writerow([
                r.spec.idx, r.spec.label, r.spec.task, r.spec.model, r.spec.config,
                r.epochs_completed, f"{r.wall_clock_sec:.1f}",
                f"{r.energy_kwh:.4f}", f"{r.emissions_kg:.6f}",
                f"{r.final_puzzle_acc:.4f}", f"{r.final_cell_acc:.4f}",
                f"{r.best_puzzle_acc:.4f}", r.checkpoint_path,
                r.status, r.error,
            ])
    print(f"\nCSV -> {out_path}")


def _emit_plots(results: list[RunResult], out_dir: Path) -> None:
    # Only emit for runs that produced metrics. Error/skipped rows stay
    # in the CSV for traceability but don't belong on the comparison plots.
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        print("  [plots] no successful runs to plot")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({r.spec.model for r in ok})
    tasks = sorted({r.spec.task for r in ok})

    def _grouped_bar(metric_fn, ylabel: str, fname: str) -> None:
        import numpy as np
        x = np.arange(len(models))
        width = 0.8 / max(1, len(tasks))
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for j, task in enumerate(tasks):
            vals = []
            for m in models:
                hits = [r for r in ok if r.spec.model == m and r.spec.task == task]
                vals.append(metric_fn(hits[0]) if hits else 0.0)
            ax.bar(x + j * width - 0.4 + width / 2, vals, width, label=task)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Iso-wall-clock: {ylabel} by model × task")
        ax.legend(title="task")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  plot -> {out_dir / fname}")

    _grouped_bar(lambda r: r.best_puzzle_acc, "puzzle accuracy",
                 "iso_time_accuracy_by_model.png")
    _grouped_bar(lambda r: r.energy_kwh, "energy (kWh)",
                 "iso_time_energy_by_model.png")

    # Scatter: kWh vs puzzle acc. Color=model, marker=task.
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    markers = {t: m for t, m in zip(tasks, ("o", "s", "^", "D"))}
    # Distinct colors pulled from the default matplotlib cycle by index.
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_for = {m: colors[i % len(colors)] for i, m in enumerate(models)}
    for r in ok:
        ax.scatter(
            r.energy_kwh, r.best_puzzle_acc,
            c=color_for[r.spec.model], marker=markers.get(r.spec.task, "o"),
            s=90, edgecolors="black", linewidths=0.6,
            label=f"{r.spec.model}/{r.spec.task}",
        )
    # Dedupe legend entries (one per model/task combo).
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    ax.set_xlabel("energy (kWh)")
    ax.set_ylabel("best puzzle accuracy")
    ax.set_title("Iso-wall-clock: accuracy vs energy cost")
    fig.tight_layout()
    fig.savefig(out_dir / "iso_time_acc_vs_kwh.png", dpi=150)
    plt.close(fig)
    print(f"  plot -> {out_dir / 'iso_time_acc_vs_kwh.png'}")


def _print_summary(results: list[RunResult]) -> None:
    print("\n" + "=" * 88)
    print(f"{'#':>2}  {'label':<18}  {'status':<8}  {'epochs':>6}  "
          f"{'wall_s':>8}  {'kWh':>6}  {'puzzle':>7}  {'best':>7}")
    print("-" * 88)
    for r in results:
        print(
            f"{r.spec.idx:>2}  {r.spec.label:<18}  {r.status:<8}  "
            f"{r.epochs_completed:>6}  {r.wall_clock_sec:>8.0f}  "
            f"{r.energy_kwh:>6.3f}  {r.final_puzzle_acc:>7.3f}  "
            f"{r.best_puzzle_acc:>7.3f}"
        )
    print("=" * 88)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def _parse_skip(raw: str) -> set[int]:
    if not raw:
        return set()
    out: set[int] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.add(int(tok))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-seconds", type=int, default=9000,
                   help="per-run wall-clock budget (default 9000 = 2.5 hr)")
    p.add_argument("--work-dir", type=str, default="",
                   help="override TRM_WORK_DIR (refuses OneDrive paths)")
    p.add_argument("--skip-runs", type=str, default="",
                   help="comma-separated run indices to skip (1..6)")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would run without launching subprocesses")
    args = p.parse_args()

    # CLI --work-dir wins over env, but we route it through the same
    # OneDrive-refusal logic by shoving it into the env before _resolve_work_dir.
    if args.work_dir:
        os.environ["TRM_WORK_DIR"] = args.work_dir
    work_dir = Path(_resolve_work_dir()).resolve()

    skip = _parse_skip(args.skip_runs)
    print(f"\n[orchestrator] work_dir        : {work_dir}")
    print(f"[orchestrator] seed            : {args.seed}")
    print(f"[orchestrator] budget / run    : {args.max_train_seconds}s")
    if skip:
        print(f"[orchestrator] skipping runs   : {sorted(skip)}")
    if args.dry_run:
        print("[orchestrator] DRY RUN — no subprocesses will be launched")

    results: dict[int, RunResult] = {}
    for spec in RUNS:
        if spec.idx in skip:
            # Still create a placeholder so distill can find teacher paths
            # from a prior session's artifact dir.
            artifact_dir = _artifact_dir(work_dir, spec.label, args.seed)
            placeholder = RunResult(spec=spec, artifact_dir=artifact_dir, status="skipped")
            if artifact_dir.is_dir():
                _collect_metrics(placeholder)
            results[spec.idx] = placeholder
            print(f"\n[run {spec.idx}] {spec.label}: SKIPPED via --skip-runs")
            continue
        results[spec.idx] = _run_one(
            spec, args.seed, work_dir, args.max_train_seconds,
            results, args.dry_run,
        )

    ordered = [results[s.idx] for s in RUNS]
    out_dir = REPO_ROOT / "results" / "novelty"
    _write_csv(ordered, out_dir / "iso_time_results.csv")
    if not args.dry_run:
        _emit_plots(ordered, out_dir)
    _print_summary(ordered)


if __name__ == "__main__":
    main()
