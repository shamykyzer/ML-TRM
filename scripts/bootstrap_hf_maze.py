"""Bootstrap hf_checkpoints/Maze-Hard/remapped_for_local.pt on a training machine.

Idempotent — safe to re-run. Skips work that's already done.

Steps:
  1. If hf_checkpoints/Maze-Hard/remapped_for_local.pt already exists, exit.
  2. If hf_checkpoints/Maze-Hard/step_9765 is missing, download it from
     Sanjin2024/TinyRecursiveModel-Maze-Hard (public, no HF token needed).
  3. Run scripts/remap_maze.py to write remapped_for_local.pt.
  4. Print final size + path so you can verify before launching training.

Usage:
    python scripts/bootstrap_hf_maze.py

After this, you can launch the maze fine-tune with:
    python main.py --mode train \\
      --config configs/trm_official_maze_finetune.yaml \\
      --seed 0 \\
      --init-weights hf_checkpoints/Maze-Hard/remapped_for_local.pt
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "hf_checkpoints" / "Maze-Hard" / "remapped_for_local.pt"
SOURCE = REPO_ROOT / "hf_checkpoints" / "Maze-Hard" / "step_9765"
REMAP = REPO_ROOT / "scripts" / "remap_maze.py"


def mb(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.1f} MB"


def main() -> int:
    if TARGET.exists():
        print(f"OK  {TARGET} already exists ({mb(TARGET)}) - nothing to do.")
        return 0

    SOURCE.parent.mkdir(parents=True, exist_ok=True)

    if not SOURCE.exists():
        print(f"Downloading Sanjin2024/TinyRecursiveModel-Maze-Hard/step_9765 ...")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("ERROR: huggingface_hub not installed. Run:", file=sys.stderr)
            print("    pip install huggingface_hub", file=sys.stderr)
            return 1

        hf_hub_download(
            repo_id="Sanjin2024/TinyRecursiveModel-Maze-Hard",
            filename="step_9765",
            local_dir=str(SOURCE.parent),
        )
        if not SOURCE.exists():
            print(f"ERROR: download did not produce {SOURCE}", file=sys.stderr)
            return 1
        print(f"OK  downloaded to {SOURCE} ({mb(SOURCE)})")
    else:
        print(f"OK  {SOURCE} already present ({mb(SOURCE)}) - skipping download.")

    if not REMAP.exists():
        print(f"ERROR: {REMAP} not found. Did you `git pull`?", file=sys.stderr)
        return 1

    print(f"Running {REMAP.name} ...")
    result = subprocess.run(
        [sys.executable, str(REMAP)],
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        print(f"ERROR: {REMAP.name} exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    if not TARGET.exists():
        print(f"ERROR: remap finished but {TARGET} not produced.", file=sys.stderr)
        return 1

    print()
    print(f"OK  bootstrap complete:")
    print(f"    {TARGET} ({mb(TARGET)})")
    print()
    print(f"Now launch training with:")
    print(f"    python main.py --mode train \\")
    print(f"      --config configs/trm_official_maze_finetune.yaml \\")
    print(f"      --seed 0 \\")
    print(f"      --init-weights hf_checkpoints/Maze-Hard/remapped_for_local.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
