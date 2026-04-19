"""Drift-detection for configs/tasks.yaml vs src.cli.bootstrap.TASK_DISPATCH.

The YAML is metadata-only (consumed by the new dashboard/menu subcommands);
the authoritative runtime registry is TASK_DISPATCH. If an edit adds or
renames a task in one but not the other, the dashboard would show stale
rows or the launcher would reject a task the menu advertised. This test
fails CI in that case.

Runs under pytest AND plain `python tests/test_tasks_yaml.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python tests/test_tasks_yaml.py` to find src/.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.bootstrap import TASK_DISPATCH  # noqa: E402
from src.cli.tasks import load_tasks  # noqa: E402


def test_yaml_and_task_dispatch_have_identical_ids():
    """Every task in TASK_DISPATCH must be in tasks.yaml and vice versa."""
    yaml_ids = set(load_tasks().keys())
    dispatch_ids = set(TASK_DISPATCH.keys())
    missing_from_yaml = dispatch_ids - yaml_ids
    extra_in_yaml = yaml_ids - dispatch_ids
    assert not missing_from_yaml, (
        f"tasks.yaml is missing these TASK_DISPATCH entries: {sorted(missing_from_yaml)}"
    )
    assert not extra_in_yaml, (
        f"tasks.yaml has entries not in TASK_DISPATCH: {sorted(extra_in_yaml)}"
    )


def test_yaml_configs_match_task_dispatch():
    """Per-task config path in the YAML must match TASK_DISPATCH."""
    yaml_tasks = load_tasks()
    for task_id, (dispatch_config, _init, _desc) in TASK_DISPATCH.items():
        spec = yaml_tasks[task_id]
        assert spec.config == dispatch_config, (
            f"config drift for {task_id!r}: "
            f"yaml={spec.config!r} vs TASK_DISPATCH={dispatch_config!r}"
        )


def test_sudoku_mlp_and_sudoku_att_are_distinct():
    """The two sudoku variants must never collapse into one entry.

    sudoku-mlp (mlp_t=true) hits paper 84.80%; sudoku-att (mlp_t=false)
    hits paper 77.70%. They share a ModelType but different YAMLs —
    merging them silently would break the experimental comparison.
    """
    yaml_tasks = load_tasks()
    assert "sudoku-mlp" in yaml_tasks
    assert "sudoku-att" in yaml_tasks
    mlp = yaml_tasks["sudoku-mlp"]
    att = yaml_tasks["sudoku-att"]
    assert mlp.config != att.config, (
        f"sudoku-mlp and sudoku-att must have different configs; both are {mlp.config}"
    )
    assert mlp.mlp_t is True, f"sudoku-mlp must have mlp_t=true, got {mlp.mlp_t}"
    assert att.mlp_t is False, f"sudoku-att must have mlp_t=false, got {att.mlp_t}"


def test_fallback_when_yaml_missing(tmp_path=None):
    """If tasks.yaml isn't readable, load_tasks falls back to TASK_DISPATCH."""
    nonexistent = str(REPO_ROOT / "configs" / "__does_not_exist__.yaml")
    result = load_tasks(nonexistent)
    assert set(result.keys()) == set(TASK_DISPATCH.keys())


def test_all_tasks_have_a_family():
    """Every task must be in one of {trm, llm}."""
    for spec in load_tasks().values():
        assert spec.family in {"trm", "llm"}, (
            f"task {spec.id!r} has unexpected family {spec.family!r}"
        )


def _run_all() -> int:
    """Discover + run every top-level test_* function and report results."""
    module = sys.modules[__name__]
    names = sorted(n for n in dir(module) if n.startswith("test_"))
    failures: list[tuple[str, BaseException]] = []
    for name in names:
        fn = getattr(module, name)
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL {name}: {exc}")
        else:
            print(f"PASS {name}")
    print()
    print(f"{len(names) - len(failures)}/{len(names)} tests passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(_run_all())
