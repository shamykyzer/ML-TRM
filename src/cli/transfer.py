"""HF reference checkpoint remap coordination.

Idempotent stage action that regenerates the remapped checkpoint and
verifies it loads cleanly. Delegates to scripts/remap_hf_checkpoint.py
and scripts/verify_remap_loads.py via subprocess.
"""
import os

from src.cli.console import CYAN, DIM, GREEN, RESET
from src.cli.paths import HF_SOURCE_CKPT, PYTHON, REMAP_SCRIPT, VERIFY_SCRIPT


def _setup_transfer() -> None:
    """Remap the HF reference checkpoint and verify it loads cleanly.

    Idempotent: always regenerates the remapped file when this stage fires,
    on the theory that if we got here something was either missing or broken,
    and rebuilding from source is cheaper than trying to diagnose which.
    Both sub-scripts are streamed so the user sees the full transfer report
    and the pass/fail verdict inline — that's the whole point of automating
    this (no more eyeballing trainer startup logs to catch a silent breakage).
    """
    # Late import of _run to avoid a circular dependency: bootstrap.py
    # imports this module at its top level, so importing bootstrap at
    # this module's top level would break the initial load. The runtime
    # cost of a per-call import is negligible (this runs once per setup).
    from src.cli.bootstrap import _run

    if not os.path.exists(HF_SOURCE_CKPT):
        # User isn't doing transfer learning — nothing to do.
        print(f"{DIM}No source checkpoint at {HF_SOURCE_CKPT} — skipping.{RESET}")
        return
    print(f"{CYAN}Remapping HF reference checkpoint \u2192 local TRMOfficial shape...{RESET}")
    _run([PYTHON, REMAP_SCRIPT])
    print(f"\n{CYAN}Verifying remapped checkpoint loads into a fresh TRMOfficial...{RESET}")
    _run([PYTHON, VERIFY_SCRIPT])
    print(f"\n{GREEN}\u2713 Transfer-learning init weights verified and ready.{RESET}")
