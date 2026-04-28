"""K-vote driver for the post-Fix-B Qwen Sudoku checkpoint (M2's task).

Wraps `scripts/run_kvote_llm_single.py` (per-label, non-clobbering writer)
rather than `run_novelty_k_vote.py` (which without --rig overwrites the
canonical merged k_vote_results.csv and would destroy existing TRM rows).

Steps:
  1. `start.py status` — confirm venv/wandb/data ✓ before launching.
  2. Locate M5's Fix-B checkpoint (passed via --m5-checkpoint).
  3. Invoke run_kvote_llm_single.py with --label qwen-sudoku-fixb so the
     output CSV is `results/novelty/k_vote_results-qwen-sudoku-fixb.csv`
     (untouched merged file).
  4. K=4 OOM fallback: detect CUDA OOM, retry with K=4 dropped.
  5. Generate k_vote_pareto_qwen.png + k_vote_accuracy_qwen.png from the
     resulting CSV (per the M2 task brief's named filenames).
  6. Contract A.5 fleet copy: copy outputs to
     `C:/ml-trm-work/checkpoints to use/machine2/k_vote_fixed/`.
  7. Apply Contract B (§B.2/§B.3) to the K=1 row; print verdict block
     ready for findings.md §5 transcription.

Usage:
    python scripts/run_qwen_fixb_kvote.py \
        --m5-checkpoint <path-to-M5-best.pt> \
        [--config configs/llm_qwen.yaml] \
        [--k-values 1,2,4] [--batch-size 2]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Contract B §B.2/§B.3 expected band for any LLM + LoRA on Sudoku.
LLM_SUDOKU_PUZZLE_MAX = 0.05
LLM_SUDOKU_CELL_LO = 0.091   # 1/11 chance floor
LLM_SUDOKU_CELL_HI = 0.30    # well above the §B.9 calibration anchor (0.125)

LABEL = "qwen-sudoku-fixb"
RESULTS_NOVELTY = REPO_ROOT / "results" / "novelty"
PER_LABEL_CSV = RESULTS_NOVELTY / f"k_vote_results-{LABEL}.csv"
KVOTE_RUN_DIR = RESULTS_NOVELTY / "k_vote_runs" / LABEL
FIG_PARETO = REPO_ROOT / "results" / "figures" / "k_vote_pareto_qwen.png"
FIG_ACC = REPO_ROOT / "results" / "figures" / "k_vote_accuracy_qwen.png"
REDUNDANCY_DEST = Path("C:/ml-trm-work/checkpoints to use/machine2/k_vote_fixed")


def _run(cmd: list[str], desc: str, *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n=== {desc} ===\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if check and proc.returncode != 0:
        raise SystemExit(f"command failed (rc={proc.returncode}): {' '.join(cmd)}")
    return proc


def _start_py_sanity(python_exe: str) -> None:
    """`start.py status` per user directive. Warns on stale stages but
    does not block — the K-vote sweep has minimal deps and stale `sync`
    typically won't bite.
    """
    proc = _run([python_exe, "start.py", "status"], "start.py sanity check",
                check=False)
    out = (proc.stdout or "") + (proc.stderr or "")
    blocking = []
    for stage in ("venv", "env", "wandb", "data", "transfer"):
        # the spinner output uses [ ] for incomplete, [✓] for complete
        if f"[ ] {stage}" in out or f"[ ]   {stage}" in out:
            blocking.append(stage)
    if blocking:
        print(f"\n[warn] start.py reports incomplete blocking stages: {blocking}",
              file=sys.stderr)
        print(f"[warn] run `{python_exe} start.py` once to advance, then retry",
              file=sys.stderr)
        # Don't auto-block here — let the K-vote subprocess fail loudly if it actually matters.


def _invoke_single_kvote(
    python_exe: str, m5_ckpt: Path, config: Path, k_values: list[int],
    batch_size: int,
) -> tuple[list[int], int]:
    """Invoke run_kvote_llm_single.py. Returns (k_values_actually_used, rc).
    Auto-retries with K=4 dropped on OOM.
    """
    base = [
        python_exe, "scripts/run_kvote_llm_single.py",
        "--config", str(config),
        "--checkpoint", str(m5_ckpt),
        "--label", LABEL,
        "--task", "sudoku",
        "--kind", "baseline",   # Qwen + LoRA, not distill
        "--k-values", ",".join(str(k) for k in k_values),
        "--batch-size", str(batch_size),
        "--output-csv", str(PER_LABEL_CSV),
    ]
    proc = _run(base, f"K-vote sweep K={k_values}", check=False)
    if proc.returncode == 0:
        return k_values, 0
    is_oom = "CUDA out of memory" in (proc.stdout or "") + (proc.stderr or "")
    if is_oom and 4 in k_values:
        retry_k = [k for k in k_values if k != 4]
        print(f"\n[oom] K=4 OOMed at batch={batch_size} - retrying with K={retry_k}",
              file=sys.stderr)
        base[base.index("--k-values") + 1] = ",".join(str(k) for k in retry_k)
        proc = _run(base, f"K-vote retry K={retry_k}", check=True)
        return retry_k, proc.returncode
    raise SystemExit(f"K-vote failed (rc={proc.returncode}, OOM={is_oom})")


def _load_kvote_csv(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _emit_plots(rows: list[dict]) -> None:
    """Write k_vote_pareto_qwen.png + k_vote_accuracy_qwen.png. Same
    visual conventions as run_novelty_k_vote.py:_emit_plots — log-x for
    accuracy-vs-K, log-x for Pareto kWh/puzzle vs accuracy.
    """
    if not rows:
        print("[plots] no rows; skipping", file=sys.stderr)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: int(_to_float(r.get("k", 0))))
    ks = [int(_to_float(r.get("k", 0))) for r in rows]
    pa = [_to_float(r.get("puzzle_acc", 0)) for r in rows]
    ca = [_to_float(r.get("cell_acc", 0)) for r in rows]
    kwh = [_to_float(r.get("kwh_per_puzzle", 0)) for r in rows]

    FIG_ACC.parent.mkdir(parents=True, exist_ok=True)

    # 1. accuracy vs K
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(ks, pa, marker="o", linewidth=1.8, color="#c0392b",
            label="puzzle_acc", markersize=7)
    ax.plot(ks, ca, marker="s", linewidth=1.6, color="#2980b9",
            label="cell_acc", markersize=6, alpha=0.85)
    if len(set(ks)) > 1:
        ax.set_xscale("log", base=2)
    ax.set_xlabel("K (samples per puzzle)")
    ax.set_ylabel("accuracy")
    ax.set_title("Qwen Sudoku post-Fix-B — K-vote accuracy vs K")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(FIG_ACC), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {FIG_ACC}")

    # 2. Pareto: kWh/puzzle (log) vs puzzle_acc
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(kwh, pa, linestyle="--", color="#c0392b", linewidth=1.0, alpha=0.6)
    ax.scatter(kwh, pa, c="#c0392b", s=80, edgecolors="black",
               linewidths=0.5, label="qwen-sudoku-fixb")
    for k, x, y in zip(ks, kwh, pa):
        ax.annotate(f"K={k}", (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    if all(v > 0 for v in kwh):
        ax.set_xscale("log")
    ax.set_xlabel("kWh per puzzle")
    ax.set_ylabel("puzzle_acc")
    ax.set_title("Qwen Sudoku post-Fix-B — K-vote Pareto: accuracy vs energy")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(FIG_PARETO), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {FIG_PARETO}")


def _check_contract_b(rows: list[dict]) -> tuple[str, list[str]]:
    if not rows:
        return "metric realism violation", ["no K-vote rows produced"]
    k1 = next(
        (r for r in rows if int(_to_float(r.get("k", 0))) == 1),
        None,
    )
    if k1 is None:
        return "metric realism violation", ["no K=1 row in K-vote results"]
    puzzle = _to_float(k1.get("puzzle_acc", 0))
    cell = _to_float(k1.get("cell_acc", 0))
    reasons: list[str] = []
    if puzzle > LLM_SUDOKU_PUZZLE_MAX:
        reasons.append(
            f"K=1 puzzle_acc={puzzle:.4f} > {LLM_SUDOKU_PUZZLE_MAX:.4f} "
            f"(LLM should pin at 0)"
        )
    if cell < LLM_SUDOKU_CELL_LO:
        reasons.append(
            f"K=1 cell_acc={cell:.4f} < {LLM_SUDOKU_CELL_LO:.4f} chance floor "
            f"(model not learning per-token statistics)"
        )
    if cell > LLM_SUDOKU_CELL_HI:
        reasons.append(
            f"K=1 cell_acc={cell:.4f} > {LLM_SUDOKU_CELL_HI:.4f} expected high "
            f"(possible mask-bug or contamination)"
        )
    if reasons:
        return "metric realism violation", reasons
    return "viability gate passed", [
        f"K=1 puzzle_acc={puzzle:.4f}, cell_acc={cell:.4f} inside LLM-Sudoku band"
    ]


def _fleet_copy() -> None:
    """Contract A.5 — one-shot copy to `machine2/k_vote_fixed/`."""
    REDUNDANCY_DEST.mkdir(parents=True, exist_ok=True)
    if PER_LABEL_CSV.is_file():
        shutil.copy2(str(PER_LABEL_CSV), str(REDUNDANCY_DEST / PER_LABEL_CSV.name))
    if KVOTE_RUN_DIR.is_dir():
        for src in KVOTE_RUN_DIR.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(KVOTE_RUN_DIR)
            dst = REDUNDANCY_DEST / "k_vote_runs" / LABEL / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
    for fig in (FIG_ACC, FIG_PARETO):
        if fig.is_file():
            shutil.copy2(str(fig), str(REDUNDANCY_DEST / fig.name))
    print(f"[fleet-copy] -> {REDUNDANCY_DEST}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--m5-checkpoint", required=True,
                   help="Path to M5's Qwen Sudoku Fix-B best.pt")
    p.add_argument("--config", default="configs/llm_qwen.yaml",
                   help="LLM config (defaults to Qwen Sudoku)")
    p.add_argument("--k-values", default="1,2,4")
    p.add_argument("--batch-size", type=int, default=2,
                   help="batch size for K-vote inference (default 2 for "
                        "0.5B + 12GB headroom at K=4)")
    p.add_argument("--python-exe", default=sys.executable)
    args = p.parse_args(argv)

    os.chdir(REPO_ROOT)
    m5_ckpt = Path(args.m5_checkpoint).resolve()
    if not m5_ckpt.is_file():
        raise SystemExit(f"M5 checkpoint not found: {m5_ckpt}")
    config = (REPO_ROOT / args.config).resolve()
    if not config.is_file():
        raise SystemExit(f"config not found: {config}")

    k_values = [int(t) for t in args.k_values.split(",") if t.strip()]

    # 1. start.py sanity (per user directive).
    _start_py_sanity(args.python_exe)

    # 2-4. Run K-vote with OOM fallback.
    used_k, _ = _invoke_single_kvote(
        args.python_exe, m5_ckpt, config, k_values, args.batch_size,
    )

    # 5. Plot the per-Qwen K-vote figures (matched filenames per M2 brief).
    rows = _load_kvote_csv(PER_LABEL_CSV)
    _emit_plots(rows)

    # 6. Fleet copy (Contract A.5).
    _fleet_copy()

    # 7. Contract B verdict.
    verdict, reasons = _check_contract_b(rows)

    print(f"\n=== Contract B verdict: {verdict} ===")
    for r in reasons:
        print(f"  - {r}")
    print(f"\nUsed K values: {used_k}")
    print(f"Per-label CSV: {PER_LABEL_CSV}")
    print(f"Plots: {FIG_ACC}\n       {FIG_PARETO}")

    print("\n--- suggested findings.md §5 entry ---")
    if verdict == "viability gate passed":
        print(f"K-vote on Qwen Sudoku Fix-B (M5 ckpt {m5_ckpt.name}): "
              f"viability gate passed.")
        print(f"K values used: {used_k}; per-label CSV at {PER_LABEL_CSV.name}.")
        print(f"Plots: {FIG_ACC.name}, {FIG_PARETO.name}.")
    else:
        print(f"K-vote on Qwen Sudoku Fix-B: metric realism violation.")
        for r in reasons:
            print(f"  - {r}")
        print(f"Suspect K-vote CSV: {PER_LABEL_CSV.name}; do NOT include in "
              f"headline figures until investigated.")

    return 0 if verdict == "viability gate passed" else 1


if __name__ == "__main__":
    sys.exit(main())
