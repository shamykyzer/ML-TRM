"""CLI dispatch for ML-TRM.

Routes the user-visible commands to the right handler. Pre-existing
invocations (bare `python start.py`, `status`, `--skip-wandb`, direct-launch
`<task> <seed>`) fall through to `src.cli.bootstrap.main()` — which is the
verbatim lift of the original start.py main(). The NEW commands `dashboard`
and `menu` are handled here BEFORE falling through, so they don't touch
the existing code path.

Rule of thumb for anything new: check for it here, return if handled,
otherwise delegate to bootstrap.main(). Never rewrite bootstrap's router.
"""
import sys

from src.cli.bootstrap import main as _bootstrap_main


def main() -> None:
    """Dispatch argv to the right handler.

    New commands are checked BEFORE anything else so they win over the
    pre-existing router. Unknown arguments — including anything that
    looks like a direct-launch `<task> <seed>` pair — fall through to
    bootstrap.main() unchanged.
    """
    args = sys.argv[1:]

    # Extract --skip-wandb so it can coexist with `dashboard` / `menu`
    # (pre-existing semantics). Doesn't affect these new commands but
    # we preserve the flag so downstream code never sees it.
    positional = [a for a in args if a != "--skip-wandb"]

    if positional and positional[0] == "dashboard":
        # Late import so the dashboard module (which reads CSVs) isn't
        # loaded until the user asks for it. Keeps the setup path fast.
        from src.cli.dashboard import render_dashboard
        render_dashboard()
        return

    if positional and positional[0] == "menu":
        # Force the copy-paste menu without running any setup stage.
        # Delegates to the existing _print_training_menu which uses
        # every detail from bootstrap (FLEET_PLAN, TASK_DISPATCH, etc.).
        from src.cli.menus import _print_training_menu
        _print_training_menu()
        return

    # Everything else — bare invocation, status, --skip-wandb,
    # direct-launch <task> <seed>, status, or any unknown first-positional
    # — hands off to the pre-existing bootstrap router. No behaviour change.
    _bootstrap_main()


__all__ = ["main"]
