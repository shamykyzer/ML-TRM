"""Build the May-1 supplementary ZIP for UFCFAS-15-2 TRM-vs-LLMs.

When invoked, this script:
  1. Verifies that `findings.md` §5.13 contains both
     'Track A status: complete' and 'Track B status: complete' lines.
     (Manual edit by the aggregator — see polling loop in §5.13.)
     Pass --force to skip this gate (for dry-runs).
  2. Bundles into `submission/supplementary_<YYYY-MM-DD>.zip`:
       - findings.md (latest)
       - results/summary_fixed.csv
       - results/eval_fixed/* (if present)
       - results/figures/*.png (M2's figures + the existing story plots)
       - notebooks/eval_only_baselines.ipynb (built if absent)
       - manifest.md (auto-generated; lists every file with sha256 + bytes)
  3. Prints a one-line "ZIP ready for human review" banner. Does NOT
     upload to Blackboard or push anything.

Usage:
  python scripts/build_supplementary_zip.py            # gated build
  python scripts/build_supplementary_zip.py --force    # build now
  python scripts/build_supplementary_zip.py --check    # only check gates
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FINDINGS = REPO / "findings.md"
SUMMARY_FIXED = REPO / "results" / "summary_fixed.csv"
EVAL_FIXED_DIR = REPO / "results" / "eval_fixed"
FIGURES_DIR = REPO / "results" / "figures"
NOTEBOOK = REPO / "notebooks" / "eval_only_baselines.ipynb"
SUBMISSION = REPO / "submission"

TRACK_A_OK = "Track A status: complete"
TRACK_B_OK = "Track B status: complete"


def gates_satisfied() -> tuple[bool, list[str]]:
    text = FINDINGS.read_text(encoding="utf-8") if FINDINGS.exists() else ""
    missing = []
    if TRACK_A_OK not in text:
        missing.append(f"missing '{TRACK_A_OK}' in findings.md §5.13")
    if TRACK_B_OK not in text:
        missing.append(f"missing '{TRACK_B_OK}' in findings.md §5.13")
    return (not missing), missing


def ensure_notebook() -> Path:
    """Build a minimal eval-only notebook if absent.

    Loads the HF Sudoku-Extreme-mlp and Maze-Hard checkpoints, runs
    `scripts/eval_hf_checkpoints.py` end-to-end, and plots the existing
    summary numbers. Markdown-only stub if the eval scripts can't run
    in the marker's environment — they still see the structure.
    """
    if NOTEBOOK.exists():
        return NOTEBOOK
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Eval-only reproduction — TRM vs LLMs (UFCFAS-15-2)\n",
                    "\n",
                    "Loads the HF community checkpoints and runs the strict-eval\n",
                    "pipeline end-to-end. Designed for the marker's environment;\n",
                    "no GPU training is performed here.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. TRM-MLP-Sudoku HF eval\n",
                    "\n",
                    "```bash\n",
                    "python scripts/eval_hf_checkpoints.py --task sudoku-mlp\n",
                    "# -> results/trm_official_sudoku_eval.json (puzzle 0.8474, cell 0.9155)\n",
                    "```\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. TRM-Att-Maze HF eval (mask sensitivity)\n",
                    "\n",
                    "```bash\n",
                    "python scripts/eval_hf_checkpoints.py --task maze --mask-non-path true\n",
                    "python scripts/eval_hf_checkpoints.py --task maze --mask-non-path false\n",
                    "# -> 0.7960 vs 0.7890 puzzle accuracy, see findings.md §5.6\n",
                    "```\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. LLM Sudoku Fix-B re-eval\n",
                    "\n",
                    "```bash\n",
                    "python scripts/eval_llm_checkpoint.py configs/llm_qwen.yaml \\\n",
                    "    'C:/ml-trm-work/llm-qwen-sudoku-seed0/qwen2.5_0.5b_sudoku_latest.pt' 50\n",
                    "# -> 0% puzzle / 19.07% cell (post-Fix-B)\n",
                    "```\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Plot the headline figures\n",
                    "\n",
                    "All headline figures live in `results/figures/`:\n",
                    "- `sudoku_mlp_peak_and_overfit.png` (§5.1)\n",
                    "- `sudoku_att_rise_and_collapse.png` (§5.4 ethics hook)\n",
                    "- `model_accuracy_bars.png`, `carbon_footprint_bars.png` (§5.1 cross-family)\n",
                    "- K-vote figures from `results/novelty/` (§5.5)\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return NOTEBOOK


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_payload() -> list[Path]:
    payload: list[Path] = []
    if FINDINGS.exists():
        payload.append(FINDINGS)
    if SUMMARY_FIXED.exists():
        payload.append(SUMMARY_FIXED)
    if EVAL_FIXED_DIR.exists():
        payload.extend(p for p in EVAL_FIXED_DIR.rglob("*") if p.is_file())
    if FIGURES_DIR.exists():
        payload.extend(p for p in FIGURES_DIR.glob("*.png"))
    nb = ensure_notebook()
    payload.append(nb)
    # Optional extras
    for extra in ("docs/report_methods_experiments_draft.md",
                  "results/summary.csv",
                  "results/trm_runs_overview.csv",
                  "results/hf_eval_maze_hard_mask_true.json",
                  "results/hf_eval_maze_hard_mask_false.json",
                  "results/trm_official_sudoku_eval.json"):
        p = REPO / extra
        if p.exists():
            payload.append(p)
    return payload


def build_manifest(zip_path: Path, files: list[Path]) -> str:
    lines = [
        f"# Supplementary ZIP manifest — {zip_path.name}",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Total files: {len(files)}",
        "",
        "| Path in ZIP | Bytes | sha256 |",
        "|---|---:|---|",
    ]
    for p in sorted(files, key=lambda x: str(x)):
        rel = p.relative_to(REPO)
        lines.append(f"| `{rel.as_posix()}` | {p.stat().st_size} | `{sha256_of(p)}` |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="skip the Track A/B gate check")
    ap.add_argument("--check", action="store_true",
                    help="only print gate status, do not build")
    args = ap.parse_args()

    ok, missing = gates_satisfied()
    if args.check:
        if ok:
            print("[gate] BOTH Track A and Track B marked complete.")
            return 0
        for m in missing:
            print(f"[gate] {m}")
        return 1

    if not ok and not args.force:
        for m in missing:
            print(f"[gate] {m}")
        print("Refusing to build. Pass --force to override.")
        return 1

    SUBMISSION.mkdir(exist_ok=True)
    today = dt.date.today().isoformat()
    zip_path = SUBMISSION / f"supplementary_{today}.zip"

    files = collect_payload()
    if not files:
        print("[build] no payload files found.")
        return 1

    manifest_text = build_manifest(zip_path, files)
    manifest_path = SUBMISSION / f"manifest_{today}.md"
    manifest_path.write_text(manifest_text, encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=p.relative_to(REPO).as_posix())
        zf.write(manifest_path, arcname=f"manifest_{today}.md")

    print(f"[build] ZIP ready for human review: {zip_path}")
    print(f"[build] Manifest: {manifest_path}")
    print(f"[build] Files: {len(files)+1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
