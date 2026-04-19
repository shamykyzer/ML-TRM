"""Wandb authentication helpers.

Stdlib-only so start.py can call these before the venv is built. Handles:
  - reading wandb_api.txt
  - auto-bootstrapping .env + ~/_netrc from the file
  - printing manual-login instructions when the token isn't available
  - the _wandb_ready() stage check
"""
import os
import shutil

from src.cli.console import BOLD, CYAN, DIM, GREEN, RESET, YELLOW
from src.cli.paths import ACTIVATE_HINT, PYTHON, ROOT, WANDB

_WANDB_API_FILE = os.path.join(ROOT, "wandb_api.txt")


def _read_wandb_api_file() -> str:
    """Return the stripped token in wandb_api.txt, or '' if missing/short.

    Mirrors the same helper in src/training/wandb_utils.py — kept duplicated
    rather than imported because start.py runs on system Python (no third-
    party deps), and `from src.training...` would pull in torch/wandb.
    """
    if not os.path.exists(_WANDB_API_FILE):
        return ""
    try:
        with open(_WANDB_API_FILE, encoding="utf-8") as f:
            tok = f.read().strip()
    except OSError:
        return ""
    return tok if len(tok) >= 40 else ""


def _bootstrap_wandb_from_file() -> bool:
    """If wandb_api.txt has a token, plumb it into .env and ~/_netrc.

    This is the auto-setup the user gets when they drop their key into
    wandb_api.txt at the repo root and re-run `python start.py`. Running
    after the file is created upgrades the wandb stage from "instructions"
    to "actually configured" without requiring manual `wandb login` calls.

    Idempotent and safe to call on every start.py invocation:
      - Reads wandb_api.txt; bails with False if missing/empty/too-short.
      - Updates the WANDB_API_KEY line in .env in-place if present, else
        appends; .env is created if absent (using .env.example as the
        template) so `load_dotenv()` in main.py picks it up.
      - Rewrites the api.wandb.ai stanza in ~/_netrc, preserving any
        other machine entries (github creds, etc).
      - Returns True on success, False if no token was available.
    """
    token = _read_wandb_api_file()
    if not token:
        return False

    # --- .env: create from .env.example if absent, then upsert the key ---
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        example = os.path.join(ROOT, ".env.example")
        if os.path.exists(example):
            shutil.copy(example, env_path)
            print(f"{DIM}[wandb-bootstrap] created .env from .env.example{RESET}")
        else:
            open(env_path, "w").close()

    with open(env_path, encoding="utf-8") as f:
        lines = f.readlines()
    found = False
    for i, line in enumerate(lines):
        if line.startswith("WANDB_API_KEY="):
            lines[i] = f"WANDB_API_KEY={token}\n"
            found = True
            break
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"WANDB_API_KEY={token}\n")
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"{DIM}[wandb-bootstrap] wrote WANDB_API_KEY to .env{RESET}")

    # --- ~/_netrc: rewrite only the api.wandb.ai stanza, preserve others ---
    netrc_path = os.path.join(os.path.expanduser("~"), "_netrc")
    stanza = f"machine api.wandb.ai\n  login user\n  password {token}\n"
    existing = ""
    if os.path.exists(netrc_path):
        try:
            with open(netrc_path, encoding="utf-8") as f:
                existing = f.read()
            # Strip any prior api.wandb.ai block (up to next `machine` or EOF).
            import re
            existing = re.sub(
                r"machine\s+api\.wandb\.ai.*?(?=\nmachine\s|\Z)",
                "",
                existing,
                flags=re.DOTALL,
            ).rstrip()
            if existing:
                existing += "\n\n"
        except OSError:
            existing = ""
    with open(netrc_path, "w", encoding="utf-8") as f:
        f.write(existing + stanza)
    try:
        os.chmod(netrc_path, 0o600)  # best-effort on git-bash/Windows
    except OSError:
        pass
    print(f"{DIM}[wandb-bootstrap] wrote api.wandb.ai stanza to {netrc_path}{RESET}")

    print(f"\n{GREEN}\u2713 wandb auto-bootstrapped from wandb_api.txt{RESET}")
    print(f"{DIM}  (you can delete wandb_api.txt now \u2014 the key is persisted in .env + _netrc){RESET}\n")
    return True


def _wandb_instructions() -> None:
    """Auto-bootstrap from wandb_api.txt when present, else print manual steps.

    Stage action for the `wandb` stage. Called only when _wandb_ready()
    returns False (no env var, no netrc, no wandb_api.txt) — but also
    re-invokable manually if the user wants to refresh the netrc entry
    after dropping a new token into wandb_api.txt.

    If the user's current shell has the venv activated, `wandb` will be on
    PATH and we show the short form. Otherwise we hand them the full path
    to the venv's wandb.exe — that works from any shell without activation,
    which is the common stumble point on Windows/PowerShell.
    """
    # Auto-bootstrap path: a token in wandb_api.txt means we can configure
    # wandb without any user interaction. This handles the common workflow
    # of "drop the key in a file, re-run start.py, training picks it up".
    if _bootstrap_wandb_from_file():
        print(f"{DIM}[wandb-bootstrap] consider deleting wandb_api.txt now \u2014{RESET}")
        print(f"{DIM}   it sits inside the OneDrive-synced repo and will upload to{RESET}")
        print(f"{DIM}   the cloud. The key now lives in .env and ~/_netrc.{RESET}")
        return

    print(f"\n{YELLOW}\u26a0  wandb not authed.{RESET} Your configs have use_wandb=true.")
    print(f"{DIM}   (Training will still work \u2014 wandb tracking just gets disabled.){RESET}")
    print(f"\n{BOLD}\u2192 Easiest:{RESET} paste your key into {CYAN}wandb_api.txt{RESET} at the repo root,")
    print(f"           then re-run {CYAN}python start.py{RESET} \u2014 auto-configures.")

    on_path = shutil.which("wandb") is not None
    venv_wandb_exists = os.path.exists(WANDB)

    print(f"\n{BOLD}\u2192 To enable wandb:{RESET}")
    if on_path:
        print(f"   {CYAN}wandb login{RESET}           # paste your API key when prompted")
    elif venv_wandb_exists:
        print(f"   {DIM}(venv not activated in this shell \u2014 `wandb` isn't on PATH){RESET}")
        print(f"   {CYAN}\"{WANDB}\" login{RESET}")
        print(f"   {DIM}\u2026or, equivalently (works from any shell):{RESET}")
        print(f"   {CYAN}\"{PYTHON}\" -m wandb login{RESET}")
        print(f"   {DIM}\u2026or activate the venv first, then use the short form:{RESET}")
        print(f"   {CYAN}{ACTIVATE_HINT}{RESET}")
        print(f"   {CYAN}wandb login{RESET}")
    else:
        print(f"   {CYAN}wandb login{RESET}           # paste your API key when prompted")
        print(f"   {DIM}(make sure your venv is activated first){RESET}")
    print(f"   {CYAN}python start.py{RESET}       # continue")

    print(f"\n{BOLD}\u2192 To skip wandb and continue anyway:{RESET}")
    print(f"   {CYAN}python start.py --skip-wandb{RESET}")


def _wandb_ready() -> bool:
    """WANDB_API_KEY env var set, OR netrc has wandb creds, OR wandb_api.txt
    holds a plausible token.

    Checks `~/.netrc` (POSIX) and `~/_netrc` (Windows — what the wandb CLI
    writes on Windows). The wandb_api.txt path is the auto-bootstrap source
    consumed by `_bootstrap_wandb_from_file()` (and by init_wandb at
    trainer-import time): if that file is present and non-empty, the wandb
    stage's setup action is a no-op because the trainer will load it
    automatically on the next run.
    """
    if os.getenv("WANDB_API_KEY"):
        return True
    home = os.path.expanduser("~")
    for name in (".netrc", "_netrc"):
        path = os.path.join(home, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                if "api.wandb.ai" in f.read():
                    return True
        except OSError:
            continue
    if _read_wandb_api_file():
        return True
    return False
