"""ANSI color constants + stdout reconfigure for Windows Unicode safety.

Stdlib-only, zero dependencies. Imported early so the unicode glyphs
printed by bootstrap.py, menus.py, dashboard.py, etc. don't crash on
the default Windows cp1252 console.
"""
import sys

# ANSI color codes (best-effort; Windows terminals >= Win10 support these)
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def reconfigure_stdout() -> None:
    """Force UTF-8 on stdout/stderr so unicode glyphs (checkmarks, box
    characters, etc.) don't crash with UnicodeEncodeError on the default
    Windows cp1252 console. Python 3.7+ supports reconfigure(); older
    streams (StringIO in tests, certain subprocess setups) are silently
    skipped via the try/except.
    """
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass


# Call at import time so every caller gets the benefit without having to
# remember to call this manually. Mirrors what start.py did at module
# level pre-refactor.
reconfigure_stdout()
