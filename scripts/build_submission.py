"""Build the supplementary code ZIP for submission.

Creates ML-TRM-Code.zip containing:
  - ML_TRM_Walkthrough.ipynb  (guided notebook the spec requires)
  - Source code (src/, main.py, start.py)
  - Configs (configs/)
  - Data builders + metadata (data/)
  - Key scripts (scripts/)
  - Results (results/)
  - Experiment logs (experiments/)
  - WandB docs (docs/wandb_metrics_glossary.md)
  - requirements.txt, README.md

Output: /mnt/c/Users/adsha/Downloads/ML final/ML-TRM-Code.zip
"""
from __future__ import annotations

import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path("/mnt/c/Users/adsha/Downloads/ML final")
ZIP_NAME = "ML-TRM-Code.zip"

INCLUDE_DIRS = [
    "src",
    "configs",
    "data",
    "results",
    "tests",
    "experiments",
]

INCLUDE_SCRIPTS = [
    "scripts/gen_training_curve.py",
    "scripts/gen_energy_scatter.py",
    "scripts/eval_llm_checkpoint.py",
    "scripts/eval_distill_checkpoint.py",
    "scripts/eval_hf_checkpoints.py",
    "scripts/aggregate_metrics.py",
    "scripts/aggregate_wandb_runs.py",
    "scripts/publish_wandb_metrics_report.py",
    "scripts/run_seed.sh",
    "scripts/run_finetune_queue.sh",
    "scripts/run_kvote_llm_single.py",
    "scripts/run_novelty_k_vote.py",
    "scripts/run_novelty_aggregate.py",
    "scripts/run_novelty_iso_time.py",
    "scripts/strict_eval.py",
    "scripts/plot_results.py",
    "scripts/plot_sudoku_mlp_overfit.py",
    "scripts/plot_sudoku_att_story.py",
    "scripts/finalize_results.py",
    "scripts/build_submission.py",
]

INCLUDE_DOCS = [
    "docs/wandb_metrics_glossary.md",
    "docs/setup-guide.txt",
]

INCLUDE_ROOT_FILES = [
    "main.py",
    "start.py",
    "requirements.txt",
    "run.sh",
    ".env.example",
    "LICENSE",
]

SKIP_PATTERNS = {
    "__pycache__",
    ".pyc",
    ".git",
    ".venv",
    ".env",
    "node_modules",
    ".remember",
    ".npy",
    ".pt",
    ".pth",
}


def should_skip(path: Path) -> bool:
    for part in path.parts:
        for pattern in SKIP_PATTERNS:
            if pattern in part:
                return True
    return False


def add_directory(zf: zipfile.ZipFile, src_dir: Path, arc_prefix: str) -> int:
    count = 0
    for f in sorted(src_dir.rglob("*")):
        if f.is_dir() or should_skip(f):
            continue
        arc_name = f"{arc_prefix}/{f.relative_to(REPO)}"
        zf.write(f, arc_name)
        count += 1
    return count


def main() -> None:
    prefix = "ML-TRM-Code"
    zip_path = OUT_DIR / ZIP_NAME

    nb_src = REPO / "ML_TRM_Walkthrough.ipynb"
    if not nb_src.exists():
        print(f"ERROR: {nb_src} not found. Generate the notebook first.")
        return

    readme_src = REPO / "SUBMISSION_README.md"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        total = 0

        zf.write(nb_src, f"{prefix}/ML_TRM_Walkthrough.ipynb")
        total += 1

        if readme_src.exists():
            zf.write(readme_src, f"{prefix}/README.md")
            total += 1

        for f_rel in INCLUDE_ROOT_FILES:
            f = REPO / f_rel
            if f.exists():
                zf.write(f, f"{prefix}/{f_rel}")
                total += 1

        for d_name in INCLUDE_DIRS:
            d = REPO / d_name
            if d.exists():
                total += add_directory(zf, d, prefix)

        for s_rel in INCLUDE_SCRIPTS:
            s = REPO / s_rel
            if s.exists():
                zf.write(s, f"{prefix}/{s_rel}")
                total += 1

        for d_rel in INCLUDE_DOCS:
            d = REPO / d_rel
            if d.exists():
                zf.write(d, f"{prefix}/{d_rel}")
                total += 1

        print(f"Wrote {zip_path}")
        print(f"  {total} files, {zip_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
