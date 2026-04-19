#!/usr/bin/env python
"""ML-TRM single entry point. Stdlib-only shim over src/cli.

Historically this file held the full 1,500+ lines of bootstrap + menu logic.
Phase 2 of the refactor moved that logic verbatim into src/cli/bootstrap.py.
Every pre-existing invocation keeps working identically:

    python start.py                       run the next missing setup stage
    python start.py status                show stage status
    python start.py --skip-wandb          skip wandb auth stage
    python start.py <task> <seed>         direct-launch with preflight

Equivalent package entry point: python -m src.cli
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.cli.app import main  # noqa: E402  (path bootstrap must run first)

if __name__ == "__main__":
    main()
