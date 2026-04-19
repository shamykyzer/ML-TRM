"""Task metadata registry.

Loads configs/tasks.yaml into a dict of TaskSpec dataclasses, used by the
new dashboard/menu subcommands for display metadata (paper_target,
approx_minutes, family labels).

Runtime authority for every pre-existing invocation remains
src.cli.bootstrap.TASK_DISPATCH — this module never feeds the training
launcher. If the two drift, tests/test_tasks_yaml.py fails loudly.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Optional

from src.cli import paths as cli_paths
from src.cli.paths import ROOT

_TASKS_YAML = os.path.join(ROOT, "configs", "tasks.yaml")


@dataclass(frozen=True)
class TaskSpec:
    """One row of the task registry."""
    id: str
    family: str
    label: str
    config: str
    init_weights: str
    mlp_t: Optional[bool] = None
    paper_target: Optional[float] = None
    approx_minutes: Optional[int] = None


def _resolve_placeholder(value: str) -> str:
    """Resolve ${HF_REMAPPED_*} placeholders against src.cli.paths constants.

    Keeps the YAML free of absolute paths (portable across machines) while
    still surfacing the real checkpoint location at load time.
    """
    if not isinstance(value, str):
        return value
    pattern = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

    def sub(match: re.Match) -> str:
        name = match.group(1)
        return str(getattr(cli_paths, name, match.group(0)))

    return pattern.sub(sub, value)


def load_tasks(path: str = _TASKS_YAML) -> Dict[str, TaskSpec]:
    """Read configs/tasks.yaml and return {id: TaskSpec}.

    Fallback: if the YAML is missing or PyYAML isn't installed, falls back
    to src.cli.bootstrap.TASK_DISPATCH so the caller always gets SOMETHING
    usable. The fallback carries fewer fields (no paper_target,
    approx_minutes, etc.) — the dashboard degrades those panels gracefully.

    Raises ValueError if two tasks share an id (protects the
    sudoku-mlp vs sudoku-att distinction).
    """
    try:
        import yaml
    except ImportError:
        return _load_from_task_dispatch()

    if not os.path.exists(path):
        return _load_from_task_dispatch()

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_tasks = data.get("tasks", [])
    result: Dict[str, TaskSpec] = {}
    for entry in raw_tasks:
        spec = TaskSpec(
            id=entry["id"],
            family=entry.get("family", ""),
            label=entry.get("label", entry["id"]),
            config=entry.get("config", ""),
            init_weights=_resolve_placeholder(entry.get("init_weights", "")),
            mlp_t=entry.get("mlp_t"),
            paper_target=entry.get("paper_target"),
            approx_minutes=entry.get("approx_minutes"),
        )
        if spec.id in result:
            raise ValueError(f"Duplicate task id in {path}: {spec.id}")
        result[spec.id] = spec

    # Invariant: sudoku-mlp and sudoku-att must be distinct entries with
    # different configs. If a future edit merges them, the dashboard's
    # Experiments panel would show one row instead of two — wrong.
    if "sudoku-mlp" in result and "sudoku-att" in result:
        if result["sudoku-mlp"].config == result["sudoku-att"].config:
            raise ValueError(
                "sudoku-mlp and sudoku-att must have different configs "
                "(mlp_t=true vs mlp_t=false)"
            )

    return result


def _load_from_task_dispatch() -> Dict[str, TaskSpec]:
    """Fallback path when tasks.yaml isn't readable.

    Converts the tuple-of-three TASK_DISPATCH into a dict of TaskSpec with
    just the three fields that dispatch carries. Enough for the dashboard
    to render a degraded view.
    """
    # Late import to avoid circular dep — tasks.py is imported by
    # dashboard.py which is imported by app.py which is imported by
    # bootstrap.py.
    from src.cli.bootstrap import TASK_DISPATCH

    result: Dict[str, TaskSpec] = {}
    for task_id, (config, init, desc) in TASK_DISPATCH.items():
        family = "llm" if task_id.startswith("llm-") else "trm"
        mlp_t = True if task_id == "sudoku-mlp" else (False if task_id == "sudoku-att" else None)
        result[task_id] = TaskSpec(
            id=task_id,
            family=family,
            label=desc,
            config=config,
            init_weights=init,
            mlp_t=mlp_t,
        )
    return result
