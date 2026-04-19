"""CLI dispatch for ML-TRM.

Phase 2: thin delegate that re-exports `main` from `bootstrap`. The existing
router logic (bare invocation, `status`, `--skip-wandb`, direct-launch
`<task> <seed>`) lives verbatim in `src/cli/bootstrap.main()`.

Later phases will add the `dashboard` and `menu` subcommands here as pure
additions — the pre-existing routing logic stays untouched.
"""
from src.cli.bootstrap import main

__all__ = ["main"]
