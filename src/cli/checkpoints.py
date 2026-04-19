"""Checkpoint discovery + run-dir introspection.

Helpers for the interactive resume picker:
  - scan a root for subdirs containing epoch_<N>.pt files
  - derive the YAML config from a run dir's basename
  - parse the seed integer out of a `<task>-seed<N>` directory name
"""
import os
from typing import List, Tuple

# Map a checkpoint dir (or its task-prefix) to the YAML config that produced
# it. The fleet dirs use `<task>-seed<N>` naming where <task> is one of the
# TASK_DISPATCH keys; legacy / non-fleet runs (older `models/sudoku/`,
# `models/maze/`) use the simple-TRM configs. New entries can be added here
# without touching anything else.
RESUME_CONFIG_BY_PREFIX: dict = {
    "sudoku-mlp":        "configs/trm_official_sudoku_mlp.yaml",
    "sudoku-att":        "configs/trm_official_sudoku.yaml",
    "sudoku-official":   "configs/trm_official_sudoku.yaml",  # legacy alias
    "maze":              "configs/trm_official_maze.yaml",
    "llm-sudoku":        "configs/llm_qwen.yaml",             # legacy alias
    "llm-gpt2-sudoku":   "configs/llm_config.yaml",
    "llm-smollm-sudoku": "configs/llm_smollm.yaml",
    "llm-qwen-sudoku":   "configs/llm_qwen.yaml",
    "llm-llama-sudoku":  "configs/llm_llama.yaml",
    "llm-gpt2-maze":     "configs/llm_gpt2_maze.yaml",
    "llm-smollm-maze":   "configs/llm_smollm_maze.yaml",
    "llm-qwen-maze":     "configs/llm_qwen_maze.yaml",
    "llm-llama-maze":    "configs/llm_llama_maze.yaml",
    # Legacy simple-TRM dirs (trainer_trm.py, not trainer_official.py):
    "sudoku":            "configs/trm_sudoku.yaml",
}


def _scan_for_checkpoints(root: str) -> List[Tuple[str, int, str]]:
    """Walk one root for subdirs containing epoch_<N>.pt files.

    Returns a list of (dir_path, max_epoch_int, latest_pt_path) tuples,
    one per discovered run dir, sorted by mtime descending so the most
    recent run shows first in the picker. Skips any dir with no
    epoch_*.pt files. We deliberately accept BOTH a `latest.pt` (only
    written by trainer_trm at end-of-run) and the highest `epoch_N.pt`
    (the granular crash-recovery point) — the picker prefers epoch_N.pt
    because it's available mid-run after Ctrl+C, which `latest.pt` is not.
    """
    import re
    if not os.path.isdir(root):
        return []
    results: List[Tuple[str, int, str]] = []
    epoch_re = re.compile(r"epoch_(\d+)\.pt$")
    for entry in sorted(os.listdir(root)):
        run_dir = os.path.join(root, entry)
        if not os.path.isdir(run_dir):
            continue
        max_ep = -1
        max_path = ""
        try:
            files = os.listdir(run_dir)
        except OSError:
            continue
        for fname in files:
            m = epoch_re.match(fname)
            if not m:
                continue
            ep = int(m.group(1))
            if ep > max_ep:
                max_ep = ep
                max_path = os.path.join(run_dir, fname)
        if max_ep >= 0:
            results.append((run_dir, max_ep, max_path))
    # Sort newest-mtime first (the run the user just Ctrl+C'd should be
    # at the top so they don't have to scroll).
    results.sort(key=lambda r: os.path.getmtime(r[2]), reverse=True)
    return results


def _config_for_run_dir(run_dir: str) -> str:
    """Best-effort: derive the YAML config from the run dir's basename.

    Strips a trailing `-seed<N>` suffix and looks the prefix up in
    RESUME_CONFIG_BY_PREFIX. Returns "" when the prefix is unknown — the
    caller prompts for a manual config path in that case.
    """
    import re
    base = os.path.basename(run_dir.rstrip(os.sep).rstrip("/"))
    base = re.sub(r"-seed\d+$", "", base)
    return RESUME_CONFIG_BY_PREFIX.get(base, "")


def _seed_for_run_dir(run_dir: str) -> int:
    """Parse the seed N out of a `<task>-seed<N>` directory name.

    Returns 0 when the dir doesn't follow the convention (legacy runs
    written before the seed-variance launcher existed).
    """
    import re
    m = re.search(r"-seed(\d+)$", os.path.basename(run_dir.rstrip(os.sep).rstrip("/")))
    return int(m.group(1)) if m else 0
