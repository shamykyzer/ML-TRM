#!/usr/bin/env python
"""Stage-aware onboarding for ML-TRM. Runs on system Python, no third-party deps.

One command, one stage at a time. Each invocation runs the next missing
setup stage, prints "done" + what to run next, and exits. When every
stage is satisfied, prints a menu of training commands to copy-paste.

Usage:
    python start.py               # run the next missing setup stage
    python start.py status        # show stage status without running anything
    python start.py --skip-wandb  # skip the wandb auth stage (and continue)
"""
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# Force UTF-8 on stdout/stderr so the unicode glyphs we print (✓, ▕, ▏, etc)
# don't crash with UnicodeEncodeError on the default Windows cp1252 console.
# Python 3.7+ supports `reconfigure()`; older streams (StringIO in tests,
# certain subprocess setups) are silently skipped via the try/except.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

# --- Venv path detection ---
# Windows uses a shorter path to avoid MAX_PATH issues during pip install.
if platform.system() == "Windows":
    VENV_DIR = os.environ.get(
        "TRM_VENV_DIR",
        os.path.join(os.path.expanduser("~"), ".venvs", "ml-trm"),
    )
    PYTHON = os.path.join(VENV_DIR, "Scripts", "python.exe")
    PIP = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    WANDB = os.path.join(VENV_DIR, "Scripts", "wandb.exe")
    ACTIVATE_HINT = f'"{os.path.join(VENV_DIR, "Scripts", "activate")}"'
else:
    VENV_DIR = os.environ.get("TRM_VENV_DIR", os.path.join(ROOT, ".venv"))
    PYTHON = os.path.join(VENV_DIR, "bin", "python")
    PIP = os.path.join(VENV_DIR, "bin", "pip")
    WANDB = os.path.join(VENV_DIR, "bin", "wandb")
    ACTIVATE_HINT = f"source {os.path.join(VENV_DIR, 'bin', 'activate')}"

# Marker file written into the venv after a successful sync, containing the
# SHA-1 of requirements.txt at sync time. If the file's current hash differs
# (or the marker is absent), the `sync` stage re-runs pip install.
REQUIREMENTS_HASH_FILE = os.path.join(VENV_DIR, ".trm_requirements_hash")
REQUIREMENTS_TXT = os.path.join(ROOT, "requirements.txt")

# Transfer-learning artifacts: the HF reference TRM checkpoint (trained on
# ARC-AGI-2) can be remapped into our local TRMOfficial shape and used as
# `--init-weights` to transfer the ~99.8% reasoning core. This whole pipeline
# is optional — only runs when the source file is present on disk. See
# scripts/remap_hf_checkpoint.py for the full rationale.
HF_SOURCE_CKPT = os.path.join(ROOT, "hf_checkpoints", "ARC", "step_723914")
HF_REMAPPED_CKPT = os.path.join(ROOT, "hf_checkpoints", "ARC", "remapped_for_local.pt")
REMAP_SCRIPT = os.path.join(ROOT, "scripts", "remap_hf_checkpoint.py")
VERIFY_SCRIPT = os.path.join(ROOT, "scripts", "verify_remap_loads.py")

# Per-task remapped HF checkpoints used by the seed-variance fine-tune plan.
# Each of these is a paper-faithful reference checkpoint that the trainer
# loads via --init-weights to start close to the published accuracy rather
# than training from scratch on RTX-5070 compute (which would take months).
HF_REMAPPED_SUDOKU_MLP = os.path.join(ROOT, "hf_checkpoints", "Sudoku-Extreme-mlp", "remapped_for_local.pt")
HF_REMAPPED_SUDOKU_ATT = os.path.join(ROOT, "hf_checkpoints", "Sudoku-Extreme-att", "remapped_for_local.pt")
HF_REMAPPED_MAZE = os.path.join(ROOT, "hf_checkpoints", "Maze-Hard", "remapped_for_local.pt")

# The six-seed fleet plan: machine index (1..6) -> (task, seed).
# Machines 1-3 run sudoku-att seeds 0-2; machines 4-6 run maze seeds 3-5.
# Three seeds per task gives mean ± std for the sudoku vs maze comparison.
# Each machine owns one (task, seed) pair so per-seed output dirs
# (<task>-seed<N>) are unambiguous.
FLEET_PLAN: List[Tuple[int, str, int]] = [
    (1, "sudoku-att", 0),
    (2, "sudoku-att", 1),
    (3, "sudoku-att", 2),
    (4, "maze", 3),
    (5, "maze", 4),
    (6, "maze", 5),
]

# Where per-seed outputs land. MUST be a local non-OneDrive path: the 6-box
# fleet shares one OneDrive tree for code + data + HF weights (good), but
# parallel training writes inside the sync folder would corrupt checkpoints
# during upload. We auto-pick the first writable non-OneDrive candidate and
# export it as TRM_WORK_DIR so subprocesses (and any `echo $TRM_WORK_DIR`)
# see the same value the launcher used. Override by setting TRM_WORK_DIR
# yourself before running start.py.
_DEFAULT_WORK_DIR_CACHE: str = ""


def _default_work_dir() -> str:
    """Pick (and cache) a safe per-machine default for TRM_WORK_DIR.

    Tries short/clean paths first. On Windows that's C:/ml-trm-work, which
    sidesteps two problems at once: (1) long-path issues when PyTorch nests
    checkpoint subdirs several levels deep under a long user profile path,
    and (2) the confusion of "is %USERPROFILE% actually inside OneDrive on
    this machine?" — it usually isn't, but enterprise Known Folder Move
    setups can make it so. If C:/ml-trm-work can't be created or written
    to (e.g. a locked-down enterprise box where non-admins can't touch
    C:\\ root), we fall back to %USERPROFILE%\\ml-trm-work, which is
    almost always writable.

    OneDrive paths are skipped even if they're the only writable candidate;
    the caller's OneDrive guard will fire a loud error in that impossible
    case — better than silently training into a corrupting location.
    """
    global _DEFAULT_WORK_DIR_CACHE
    if _DEFAULT_WORK_DIR_CACHE:
        return _DEFAULT_WORK_DIR_CACHE

    if platform.system() == "Windows":
        candidates = [
            "C:/ml-trm-work",
            os.path.join(
                os.environ.get("USERPROFILE") or os.path.expanduser("~"),
                "ml-trm-work",
            ),
        ]
    else:
        candidates = [os.path.join(os.path.expanduser("~"), "ml-trm-work")]

    for cand in candidates:
        if "onedrive" in cand.lower():
            continue
        try:
            os.makedirs(cand, exist_ok=True)
            # os.makedirs(exist_ok=True) succeeds even on a read-only dir,
            # so probe with an actual file write to confirm writability.
            probe = os.path.join(cand, ".trm_write_probe")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            _DEFAULT_WORK_DIR_CACHE = cand
            return cand
        except OSError:
            continue

    # Nothing was writable. Return the last candidate anyway so the caller
    # surfaces the real error when it tries to use it, rather than us
    # raising an opaque "no default found" here.
    _DEFAULT_WORK_DIR_CACHE = candidates[-1]
    return _DEFAULT_WORK_DIR_CACHE

# Task dispatch table — single source of truth for every fine-tuneable task.
# Used by both the interactive launcher and the printed copy-paste commands.
# An empty init string means "random init" (the LLM tasks load their own HF
# weights via transformers; only the TRM tasks accept a remapped HF init).
# sudoku-att and sudoku-mlp share one model type enum but different YAMLs —
# trm_official_sudoku.yaml is the attention variant (mlp_t=false),
# trm_official_sudoku_mlp.yaml is MLP-t.
#
# LLM coverage: every LLM × {sudoku, maze} so the coursework's three-way
# comparison (TRM vs fine-tuned LLM vs distilled student) has a full scaling
# sweep on both reasoning tasks. Sudoku and maze configs differ in vocab_size
# (11 vs 6), seq_len (81 vs 900), and per-LLM batch_size + grad_accum_steps.
TASK_DISPATCH = {
    "sudoku-mlp": (
        "configs/trm_official_sudoku_mlp.yaml",
        HF_REMAPPED_SUDOKU_MLP,
        "Sudoku-Extreme MLP-t (paper 84.80%)",
    ),
    "sudoku-att": (
        "configs/trm_official_sudoku.yaml",
        HF_REMAPPED_SUDOKU_ATT,
        "Sudoku-Extreme attention (paper 77.70%)",
    ),
    "maze": (
        "configs/trm_official_maze.yaml",
        HF_REMAPPED_MAZE,
        "Maze-Hard 30x30 (paper 85.30%)",
    ),
    "llm-gpt2-sudoku":   ("configs/llm_config.yaml",        "", "GPT-2 (124M) LoRA on Sudoku"),
    "llm-smollm-sudoku": ("configs/llm_smollm.yaml",        "", "SmolLM2-360M LoRA on Sudoku"),
    "llm-qwen-sudoku":   ("configs/llm_qwen.yaml",          "", "Qwen2.5-0.5B LoRA on Sudoku"),
    "llm-llama-sudoku":  ("configs/llm_llama.yaml",         "", "Llama-3.2-1B LoRA on Sudoku"),
    "llm-gpt2-maze":     ("configs/llm_gpt2_maze.yaml",     "", "GPT-2 (124M) LoRA on Maze"),
    "llm-smollm-maze":   ("configs/llm_smollm_maze.yaml",   "", "SmolLM2-360M LoRA on Maze"),
    "llm-qwen-maze":     ("configs/llm_qwen_maze.yaml",     "", "Qwen2.5-0.5B LoRA on Maze"),
    "llm-llama-maze":    ("configs/llm_llama_maze.yaml",    "", "Llama-3.2-1B LoRA on Maze"),
}


# ANSI color codes (best-effort; Windows terminals >= Win10 support these)
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _run(cmd: List[str], cwd: str = None) -> None:
    """Run a subprocess, stream output, abort on failure."""
    print(f"\n{DIM}>>> {' '.join(cmd)}{RESET}\n")
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    if result.returncode != 0:
        print(f"\n{YELLOW}!!! Command failed (exit {result.returncode}){RESET}")
        sys.exit(result.returncode)


# ============================================================
# Stage actions — what to do when a stage is not ready
# ============================================================

def _setup_venv() -> None:
    """Create venv and install CUDA torch + requirements."""
    os.makedirs(os.path.dirname(VENV_DIR), exist_ok=True)
    _run([sys.executable, "-m", "venv", VENV_DIR])
    _run([PYTHON, "-m", "pip", "install", "--upgrade", "pip"])
    _run([
        PIP, "install", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    ])
    _run([PIP, "install", "-r", "requirements.txt"])
    # Record the requirements.txt hash so the `sync` stage starts green —
    # otherwise we'd immediately re-run pip install after a fresh venv build.
    _write_requirements_hash()
    _run([
        PYTHON, "-c",
        "import torch; "
        "print(f'CUDA: {torch.cuda.is_available()}, "
        "GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')",
    ])


def _sync_venv() -> None:
    """Install/update packages so the venv matches requirements.txt."""
    print(f"{CYAN}requirements.txt changed (or venv was built externally) — syncing...{RESET}")
    _run([PIP, "install", "-r", "requirements.txt"])
    _write_requirements_hash()
    print(f"\n{GREEN}✓ Venv synced to requirements.txt{RESET}")


def _bootstrap_env() -> None:
    """Copy .env.example to .env and tell the user to edit it."""
    src = os.path.join(ROOT, ".env.example")
    dst = os.path.join(ROOT, ".env")
    if not os.path.exists(src):
        print(f"{YELLOW}!!! .env.example not found — can't bootstrap .env{RESET}")
        sys.exit(1)
    shutil.copy(src, dst)
    print(f"\n{GREEN}✓ Created .env from .env.example{RESET}")
    print(f"\n{BOLD}→ Edit .env to set at least:{RESET}")
    print(f"   {CYAN}WANDB_API_KEY{RESET}   (from https://wandb.ai/authorize)")
    print(f"\n{DIM}Then re-run:{RESET} python start.py")


def _bootstrap_wandb_from_file() -> bool:
    """If wandb_api.txt has a token, plumb it into .env and ~/_netrc.

    This is the auto-setup the user gets when they drop their key into
    wandb_api.txt at the repo root and re-run `python start.py`. Running
    after the file is created upgrades the wandb stage from "instructions"
    to "actually configured" without requiring manual `wandb login` calls.

    Idempotent and safe to call on every start.py invocation:
      • Reads wandb_api.txt; bails with False if missing/empty/too-short.
      • Updates the WANDB_API_KEY line in .env in-place if present, else
        appends; .env is created if absent (using .env.example as the
        template) so `load_dotenv()` in main.py picks it up.
      • Rewrites the api.wandb.ai stanza in ~/_netrc, preserving any
        other machine entries (github creds, etc).
      • Returns True on success, False if no token was available.
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

    print(f"\n{GREEN}✓ wandb auto-bootstrapped from wandb_api.txt{RESET}")
    print(f"{DIM}  (you can delete wandb_api.txt now — the key is persisted in .env + _netrc){RESET}\n")
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
        print(f"{DIM}[wandb-bootstrap] consider deleting wandb_api.txt now —{RESET}")
        print(f"{DIM}   it sits inside the OneDrive-synced repo and will upload to{RESET}")
        print(f"{DIM}   the cloud. The key now lives in .env and ~/_netrc.{RESET}")
        return

    print(f"\n{YELLOW}⚠  wandb not authed.{RESET} Your configs have use_wandb=true.")
    print(f"{DIM}   (Training will still work — wandb tracking just gets disabled.){RESET}")
    print(f"\n{BOLD}→ Easiest:{RESET} paste your key into {CYAN}wandb_api.txt{RESET} at the repo root,")
    print(f"           then re-run {CYAN}python start.py{RESET} — auto-configures.")

    on_path = shutil.which("wandb") is not None
    venv_wandb_exists = os.path.exists(WANDB)

    print(f"\n{BOLD}→ To enable wandb:{RESET}")
    if on_path:
        print(f"   {CYAN}wandb login{RESET}           # paste your API key when prompted")
    elif venv_wandb_exists:
        print(f"   {DIM}(venv not activated in this shell — `wandb` isn't on PATH){RESET}")
        print(f"   {CYAN}\"{WANDB}\" login{RESET}")
        print(f"   {DIM}…or, equivalently (works from any shell):{RESET}")
        print(f"   {CYAN}\"{PYTHON}\" -m wandb login{RESET}")
        print(f"   {DIM}…or activate the venv first, then use the short form:{RESET}")
        print(f"   {CYAN}{ACTIVATE_HINT}{RESET}")
        print(f"   {CYAN}wandb login{RESET}")
    else:
        print(f"   {CYAN}wandb login{RESET}           # paste your API key when prompted")
        print(f"   {DIM}(make sure your venv is activated first){RESET}")
    print(f"   {CYAN}python start.py{RESET}       # continue")

    print(f"\n{BOLD}→ To skip wandb and continue anyway:{RESET}")
    print(f"   {CYAN}python start.py --skip-wandb{RESET}")


def _setup_transfer() -> None:
    """Remap the HF reference checkpoint and verify it loads cleanly.

    Idempotent: always regenerates the remapped file when this stage fires,
    on the theory that if we got here something was either missing or broken,
    and rebuilding from source is cheaper than trying to diagnose which.
    Both sub-scripts are streamed so the user sees the full transfer report
    and the pass/fail verdict inline — that's the whole point of automating
    this (no more eyeballing trainer startup logs to catch a silent breakage).
    """
    if not os.path.exists(HF_SOURCE_CKPT):
        # User isn't doing transfer learning — nothing to do.
        print(f"{DIM}No source checkpoint at {HF_SOURCE_CKPT} — skipping.{RESET}")
        return
    print(f"{CYAN}Remapping HF reference checkpoint → local TRMOfficial shape...{RESET}")
    _run([PYTHON, REMAP_SCRIPT])
    print(f"\n{CYAN}Verifying remapped checkpoint loads into a fresh TRMOfficial...{RESET}")
    _run([PYTHON, VERIFY_SCRIPT])
    print(f"\n{GREEN}✓ Transfer-learning init weights verified and ready.{RESET}")


def _bootstrap_data() -> None:
    """Download whichever datasets are missing."""
    data_dir = os.path.join(ROOT, "data")
    sudoku_ok = os.path.exists(os.path.join(data_dir, "sudoku-extreme-full/train/all__inputs.npy"))
    maze_ok = os.path.exists(os.path.join(data_dir, "maze-30x30-hard-1k-aug/train/all__inputs.npy"))

    if not sudoku_ok:
        print(f"{CYAN}Downloading Sudoku dataset...{RESET}")
        _run([
            PYTHON, "build_sudoku_dataset.py",
            "--output-dir", "sudoku-extreme-full",
            "--subsample-size", "1000",
        ], cwd=data_dir)

    if not maze_ok:
        # `--aug` enables 8× dihedral augmentation on the train split (1000
        # → 8000 samples). Without it the maze train set saturates train
        # accuracy in a few hundred epochs and the learning curve is flat
        # for the rest of training. Test split is unaffected (aug only
        # applies to train in build_maze_dataset.py).
        print(f"{CYAN}Downloading Maze dataset (with 8x dihedral aug)...{RESET}")
        _run([
            PYTHON, "build_maze_dataset.py",
            "--output-dir", "maze-30x30-hard-1k-aug",
            "--aug",
        ], cwd=data_dir)


# ============================================================
# Stage checks — pure os-level probes, no third-party imports
# ============================================================

def _venv_ready() -> bool:
    """Venv exists AND its python can import torch."""
    if not os.path.exists(PYTHON):
        return False
    result = subprocess.run(
        [PYTHON, "-c", "import torch"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _requirements_hash() -> str:
    """SHA-1 of requirements.txt, or '' if the file is missing."""
    if not os.path.exists(REQUIREMENTS_TXT):
        return ""
    with open(REQUIREMENTS_TXT, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def _write_requirements_hash() -> None:
    """Write the current requirements.txt hash into the venv marker file."""
    os.makedirs(os.path.dirname(REQUIREMENTS_HASH_FILE), exist_ok=True)
    with open(REQUIREMENTS_HASH_FILE, "w") as f:
        f.write(_requirements_hash())


def _sync_ready() -> bool:
    """Venv's recorded requirements.txt hash matches the current file.

    The marker file is written whenever we successfully run pip install on
    requirements.txt. If the file gets edited (new package added, version
    bumped), the stored hash no longer matches and the sync stage fires.
    """
    if not os.path.exists(PYTHON):
        return False  # venv stage will handle this first
    if not os.path.exists(REQUIREMENTS_HASH_FILE):
        return False
    try:
        with open(REQUIREMENTS_HASH_FILE) as f:
            stored = f.read().strip()
    except OSError:
        return False
    return stored == _requirements_hash()


def _env_ready() -> bool:
    return os.path.exists(os.path.join(ROOT, ".env"))


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


def _transfer_ready() -> bool:
    """Remapped HF checkpoint exists AND verifies cleanly — or source isn't present.

    Three outcomes:
      1. No source checkpoint on disk → N/A, return ready (most users).
      2. Source present, remapped file missing → not ready (build it).
      3. Both present → delegate to scripts/verify_remap_loads.py via the venv
         python. Exit code 0 means load report matched exactly: zero unexpected
         keys, and the only "missing" keys are the intentionally-skipped
         embed/task_emb/lm_head/rotary buffers. Anything else → not ready,
         force a rebuild.

    Stays silent (captures output) because this runs during the status
    display. The loud re-run happens in _setup_transfer if this returns False.
    """
    if not os.path.exists(HF_SOURCE_CKPT):
        return True  # N/A — user isn't doing transfer learning
    if not os.path.exists(HF_REMAPPED_CKPT):
        return False
    if not os.path.exists(PYTHON):
        return False  # venv stage handles this first; we'll re-check next run
    result = subprocess.run(
        [PYTHON, VERIFY_SCRIPT],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _data_ready() -> bool:
    return (
        os.path.exists(os.path.join(ROOT, "data/sudoku-extreme-full/train/all__inputs.npy"))
        and os.path.exists(os.path.join(ROOT, "data/maze-30x30-hard-1k-aug/train/all__inputs.npy"))
    )


# ============================================================
# Stage definition
# ============================================================

@dataclass
class Stage:
    key: str
    label: str
    check: Callable[[], bool]
    action: Callable[[], None]
    blocking: bool = True


STAGES: List[Stage] = [
    Stage("venv",     "Python venv with CUDA torch",    _venv_ready,     _setup_venv),
    Stage("sync",     "Venv matches requirements.txt",  _sync_ready,     _sync_venv),
    Stage("env",      "Machine-local .env file",        _env_ready,      _bootstrap_env),
    Stage("wandb",    "wandb logged in",                _wandb_ready,    _wandb_instructions, blocking=False),
    # Non-blocking because most users won't have the 2.47 GB source file —
    # _transfer_ready() returns True when the source is absent, so this stage
    # silently passes for anyone not doing ARC→local transfer learning.
    Stage("transfer", "HF reference remapped + verified", _transfer_ready, _setup_transfer, blocking=False),
    Stage("data",     "Sudoku + Maze datasets",         _data_ready,     _bootstrap_data),
]


# ============================================================
# Output helpers
# ============================================================

def _print_stage_status(results: List[Tuple[Stage, bool]], skip_wandb: bool) -> None:
    print(f"\n{BOLD}=== ML-TRM Setup Status ==={RESET}\n")
    for stage, done in results:
        if stage.key == "wandb" and skip_wandb and not done:
            mark, color, suffix = "~", DIM, " (skipped)"
        elif done:
            mark, color, suffix = "✓", GREEN, ""
        else:
            mark, color, suffix = " ", "", ""
        print(f"  [{color}{mark}{RESET}] {stage.key:<7s}  {stage.label}{suffix}")
    print()


def _resolve_work_dir() -> str:
    """Resolve TRM_WORK_DIR, auto-setting a safe default when unset.

    The single most important hygiene check in the 6-box fleet. Every machine
    syncs code + data + HF weights via OneDrive, but writing training outputs
    into the sync folder would corrupt checkpoints mid-run as OneDrive uploads
    partial tensor files. The warning in src/utils/config.py is a belt; this
    is the suspenders — we refuse to launch if the path looks OneDrive-ish.

    When TRM_WORK_DIR is unset (the common case on a fresh machine), we pick
    a safe local default via _default_work_dir() AND export it into the
    current process env so every subprocess we spawn — and anything the
    trainer re-reads from os.environ — sees the same value the launcher
    used. That makes the effective work dir discoverable at a glance
    (`echo $env:TRM_WORK_DIR` in PowerShell) instead of hidden inside
    start.py's resolution logic.
    """
    env_work_dir = os.environ.get("TRM_WORK_DIR")
    if env_work_dir:
        work_dir = env_work_dir
        auto_set = False
    else:
        work_dir = _default_work_dir()
        os.environ["TRM_WORK_DIR"] = work_dir
        auto_set = True

    if "onedrive" in work_dir.lower():
        print(f"\n{YELLOW}!!! TRM_WORK_DIR='{work_dir}' looks like a OneDrive path.{RESET}")
        print(f"{YELLOW}    Parallel training on shared OneDrive will corrupt checkpoints.{RESET}")
        print(f"{DIM}    Pick a local path and re-run:{RESET}")
        if platform.system() == "Windows":
            print(f"      {CYAN}$env:TRM_WORK_DIR = 'C:/ml-trm-work'{RESET}")
        else:
            print(f"      {CYAN}export TRM_WORK_DIR=$HOME/ml-trm-work{RESET}")
        sys.exit(3)

    if auto_set:
        print(f"{DIM}[start.py] TRM_WORK_DIR not set — auto-picked: {work_dir}{RESET}")
        print(f"{DIM}           (override by setting TRM_WORK_DIR before launch){RESET}")

    return work_dir


def _prompt(msg: str, default: str = "") -> str:
    """Prompt user on stdin, return default on empty input or EOF."""
    suffix = f" [{default}]" if default else ""
    try:
        reply = input(f"{CYAN}{msg}{suffix}: {RESET}").strip()
    except EOFError:
        reply = ""
    return reply or default


def _prompt_task_and_seed() -> Tuple[str, int]:
    """Interactive picker for task label and seed int.

    Shows every task in TASK_DISPATCH as a numbered menu, grouped TRM-first
    then LLM, so the operator can scan by family. Tasks whose HF init file is
    missing get a yellow flag (the run still works — it just starts from
    random init, exploratory rather than paper-faithful). Defaults to seed 0
    (first row of FLEET_PLAN, a safe starter on any machine).
    """
    tasks = list(TASK_DISPATCH.keys())
    trm_tasks = [t for t in tasks if not t.startswith("llm-")]
    llm_tasks = [t for t in tasks if t.startswith("llm-")]

    print(f"\n{BOLD}Which task?{RESET}")
    print(f"  {DIM}-- TRM (paper architectures) --{RESET}")
    for t in trm_tasks:
        i = tasks.index(t) + 1
        _, init, desc = TASK_DISPATCH[t]
        suffix = (
            f"  {YELLOW}(HF init missing — will use random init){RESET}"
            if init and not os.path.exists(init) else ""
        )
        print(f"  {CYAN}{i:>2}{RESET}) {t:<20s}  {DIM}{desc}{RESET}{suffix}")
    print(f"  {DIM}-- LLM baselines (LoRA fine-tune) --{RESET}")
    for t in llm_tasks:
        i = tasks.index(t) + 1
        _, _, desc = TASK_DISPATCH[t]
        print(f"  {CYAN}{i:>2}{RESET}) {t:<20s}  {DIM}{desc}{RESET}")

    choice = _prompt(f"Pick 1-{len(tasks)}", default="1")
    try:
        task = tasks[int(choice) - 1]
        if int(choice) < 1:
            raise IndexError
    except (ValueError, IndexError):
        print(f"{YELLOW}!!! Invalid task choice '{choice}'.{RESET}")
        sys.exit(2)

    seed_str = _prompt("Seed (non-negative int)", default="0")
    try:
        seed = int(seed_str)
        if seed < 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}!!! Seed must be a non-negative integer, got '{seed_str}'.{RESET}")
        sys.exit(2)

    return task, seed


def _dispatch_training(
    task: str,
    seed: int,
    dry_run: bool = False,
    epochs: Optional[int] = None,
) -> None:
    """Build main.py argv and exec, routing checkpoints to a per-seed local dir.

    Python twin of `scripts/run_seed.sh <task> <seed>`. Sets TRM_CHECKPOINT_DIR
    and TRM_EXPERIMENT_DIR in the child env so the trainer writes to
    <TRM_WORK_DIR>/<task>-seed<N>/, never into the OneDrive-synced repo. Shell
    launchers are kept for automation (cron, CI); this path is for users who
    drive everything from start.py interactively.

    When `epochs` is passed, it forwards as --epochs N to main.py (overriding
    YAML's training.epochs for this run only). dry_run wins if both are set.
    """
    config, init, description = TASK_DISPATCH[task]

    work_dir = _resolve_work_dir()
    task_dir = os.path.join(work_dir, f"{task}-seed{seed}")
    os.makedirs(task_dir, exist_ok=True)

    env = os.environ.copy()
    env["TRM_CHECKPOINT_DIR"] = task_dir
    env["TRM_EXPERIMENT_DIR"] = task_dir

    args = [PYTHON, "main.py", "--mode", "train", "--config", config, "--seed", str(seed)]
    if init and os.path.exists(init):
        args.extend(["--init-weights", init])
    if dry_run:
        args.extend(["--epochs", "5"])
        epochs_label = "5 (dry run — pipeline smoke test)"
    elif epochs is not None:
        args.extend(["--epochs", str(epochs)])
        epochs_label = f"{epochs} (overridden via --epochs)"
    else:
        epochs_label = "from YAML config.training.epochs"

    bar = "=" * 64
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"  task               : {CYAN}{task}{RESET}  {DIM}({description}){RESET}")
    print(f"  seed               : {CYAN}{seed}{RESET}")
    print(f"  config             : {config}")
    init_label = init if (init and os.path.exists(init)) else "<none, random init>"
    print(f"  init_weights       : {init_label}")
    print(f"  TRM_CHECKPOINT_DIR : {task_dir}")
    print(f"  TRM_EXPERIMENT_DIR : {task_dir}")
    print(f"  epochs             : {epochs_label}")
    print(f"{BOLD}{bar}{RESET}\n")

    result = subprocess.run(args, env=env)
    sys.exit(result.returncode)


# ============================================================
# Resume / extend training
# ============================================================

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


def _resume_training_picker() -> None:
    """Interactive: pick a finished/interrupted run and extend it by N epochs.

    Discovery scans both:
      • $TRM_WORK_DIR (fleet runs from `start.py` dispatch / run_seed.sh)
      • <repo>/models/  (legacy simple-TRM runs that wrote inside the repo)
    Each candidate run is shown with its highest epoch_<N>.pt checkpoint —
    NOT `latest.pt`, because latest.pt is only written when a run completes
    cleanly, but Ctrl+C recovery has to start from the most recent
    epoch_N.pt that the save_interval cadence wrote to disk.

    The user picks a run, the picker resolves the right config via the dir
    name, asks how many ADDITIONAL epochs to run, then dispatches main.py
    with --resume <ckpt> --epochs <max_ep + extra>. main.py's _run_train
    treats --epochs as the new total target — the trainer loop runs from
    the resumed start_epoch through to the new total.
    """
    work_dir = os.environ.get("TRM_WORK_DIR") or _default_work_dir()
    candidates = (
        _scan_for_checkpoints(work_dir)
        + _scan_for_checkpoints(os.path.join(ROOT, "models"))
    )

    if not candidates:
        print(f"\n{YELLOW}!!! No resumable runs found.{RESET}")
        print(f"{DIM}    Looked in:{RESET}")
        print(f"{DIM}      • {work_dir}{RESET}")
        print(f"{DIM}      • {os.path.join(ROOT, 'models')}{RESET}")
        print(f"{DIM}    A run is resumable when its directory contains at least one{RESET}")
        print(f"{DIM}    epoch_<N>.pt file (written every save_interval epochs).{RESET}\n")
        return

    print(f"\n{BOLD}Resumable runs:{RESET}  {DIM}(newest first){RESET}")
    for i, (run_dir, max_ep, ckpt_path) in enumerate(candidates, 1):
        cfg = _config_for_run_dir(run_dir) or f"{YELLOW}<unknown — will prompt>{RESET}"
        seed = _seed_for_run_dir(run_dir)
        size_mb = os.path.getsize(ckpt_path) / 1e6
        print(
            f"  {CYAN}{i:>2}{RESET}) {os.path.basename(run_dir):<28s}"
            f"  epoch={CYAN}{max_ep:>4}{RESET}"
            f"  seed={seed}"
            f"  ({size_mb:.0f} MB)"
        )
        print(f"      {DIM}{ckpt_path}{RESET}")
        print(f"      {DIM}config: {cfg}{RESET}")

    raw = _prompt(f"\nPick run 1-{len(candidates)}", default="1")
    try:
        idx = int(raw) - 1
        run_dir, max_ep, ckpt_path = candidates[idx]
    except (ValueError, IndexError):
        print(f"{YELLOW}!!! Invalid choice '{raw}'.{RESET}")
        return

    config = _config_for_run_dir(run_dir)
    if not config:
        config = _prompt(
            f"Config YAML for this run (relative to repo root)",
            default="configs/trm_sudoku.yaml",
        )
        if not os.path.exists(os.path.join(ROOT, config)):
            print(f"{YELLOW}!!! Config not found: {config}{RESET}")
            return

    # The "how many MORE epochs?" framing is the natural one for the user
    # ("I just hit 500, give me 200 more"); we convert to the absolute
    # total that main.py / the trainer loop expect via --epochs.
    extra_str = _prompt(
        f"Run for how many MORE epochs? (current = {max_ep})",
        default="100",
    )
    try:
        extra = int(extra_str)
        if extra <= 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}!!! Need a positive integer, got '{extra_str}'.{RESET}")
        return
    new_total = max_ep + extra
    seed = _seed_for_run_dir(run_dir)

    # Pin checkpoint/experiment dirs back to the SAME run dir so the new
    # epoch_<N>.pt files, train_log.csv, and emissions.csv append to the
    # existing artifacts instead of starting a fresh dir somewhere else.
    env = os.environ.copy()
    env["TRM_CHECKPOINT_DIR"] = run_dir
    env["TRM_EXPERIMENT_DIR"] = run_dir

    args = [
        PYTHON, "main.py",
        "--mode", "train",
        "--config", config,
        "--resume", ckpt_path,
        "--epochs", str(new_total),
        "--seed", str(seed),
    ]

    bar = "=" * 64
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"  resume from        : {CYAN}epoch_{max_ep}.pt{RESET}  {DIM}({ckpt_path}){RESET}")
    print(f"  config             : {config}")
    print(f"  current epoch      : {max_ep}")
    print(f"  extra epochs       : {CYAN}{extra}{RESET}")
    print(f"  new total epochs   : {CYAN}{new_total}{RESET}")
    print(f"  seed               : {seed}")
    print(f"  TRM_CHECKPOINT_DIR : {run_dir}")
    print(f"  wandb              : "
          f"{GREEN}✓ ready{RESET}" if _wandb_ready() else f"{YELLOW}not configured (will train without){RESET}")
    print(f"{BOLD}{bar}{RESET}\n")

    confirm = _prompt(f"Launch? [y/N]", default="N").lower()
    if confirm not in ("y", "yes"):
        print(f"{DIM}Aborted — nothing launched.{RESET}\n")
        return

    result = subprocess.run(args, env=env, cwd=ROOT)
    sys.exit(result.returncode)


# ============================================================
# Fresh-start (overwrite) launcher
# ============================================================

def _fresh_start_launcher() -> None:
    """Launch a NEW run for a chosen task+seed, wiping any existing run dir.

    Counterpart to the resume picker: instead of continuing from the latest
    epoch_<N>.pt, this path guarantees epoch 0 by deleting everything under
    <TRM_WORK_DIR>/<task>-seed<N>/ first. The user picks task, seed, and
    total epochs (forwarded as --epochs to main.py so the YAML stays
    untouched). A confirmation prompt lists what's about to be deleted
    because `rm -rf <run_dir>` with stale checkpoints is not recoverable.
    """
    task, seed = _prompt_task_and_seed()

    epochs_str = _prompt("Epochs (total — overrides YAML)", default="2000")
    try:
        epochs = int(epochs_str)
        if epochs <= 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}!!! Need a positive integer, got '{epochs_str}'.{RESET}")
        return

    work_dir = _resolve_work_dir()
    task_dir = os.path.join(work_dir, f"{task}-seed{seed}")

    if os.path.isdir(task_dir):
        try:
            existing = sorted(os.listdir(task_dir))
        except OSError:
            existing = []
        if existing:
            print(f"\n{YELLOW}⚠  Existing run directory will be WIPED:{RESET}")
            print(f"   {task_dir}")
            print(f"{DIM}   Contains {len(existing)} item(s):{RESET}")
            for name in existing[:10]:
                full = os.path.join(task_dir, name)
                if os.path.isfile(full):
                    size_mb = os.path.getsize(full) / 1e6
                    size_label = f"  ({size_mb:.1f} MB)" if size_mb >= 0.1 else ""
                    print(f"{DIM}     • {name}{size_label}{RESET}")
                else:
                    print(f"{DIM}     • {name}/{RESET}")
            if len(existing) > 10:
                print(f"{DIM}     ... and {len(existing) - 10} more{RESET}")
            confirm = _prompt(
                f"\n{BOLD}Delete these and start fresh?{RESET} [y/N]",
                default="N",
            ).lower()
            if confirm not in ("y", "yes"):
                print(f"{DIM}Aborted — existing run preserved, nothing launched.{RESET}\n")
                return
            try:
                shutil.rmtree(task_dir)
            except OSError as exc:
                print(f"{YELLOW}!!! Could not remove {task_dir}: {exc}{RESET}")
                return
            print(f"{DIM}[fresh-start] removed {task_dir}{RESET}")
    else:
        print(f"\n{DIM}[fresh-start] no existing dir at {task_dir} — clean launch.{RESET}")

    _dispatch_training(task, seed, epochs=epochs)


# ============================================================
# Interactive launcher
# ============================================================

def _interactive_launcher() -> None:
    """Prompt the user to pick an action after all setup stages are ready.

    Main entry for the 6-box workflow: re-run `python start.py` per seed,
    pick a task + seed from the menu, train. Option 3 is the Regime A
    one-shot (no training); option 4 dumps the full copy-paste command
    list for scripted workflows. Option 5 is the resume/extend path —
    finds the latest epoch_<N>.pt for any prior run (including ones the
    user Ctrl+C'd) and extends training by a chosen number of epochs.
    """
    print(f"\n{BOLD}What do you want to run?{RESET}")
    print(f"  {CYAN}1{RESET}) Dry run         {DIM}(5-epoch pipeline smoke test — always do this first){RESET}")
    print(f"  {CYAN}2{RESET}) Seed-variance   {DIM}(full fine-tune from HF init — the 6-machine plan){RESET}")
    print(f"  {CYAN}3{RESET}) Evaluate HF     {DIM}(Regime A — all 3 paper checkpoints, no training){RESET}")
    print(f"  {CYAN}4{RESET}) Show commands   {DIM}(print copy-paste commands and exit){RESET}")
    print(f"  {CYAN}5{RESET}) Resume/extend   {DIM}(continue a finished or Ctrl+C'd run for N more epochs){RESET}")
    print(f"  {CYAN}6{RESET}) Fresh start     {DIM}(new run — overwrite existing seed dir, pick epochs){RESET}")
    print(f"  {CYAN}Q{RESET}) Quit")

    choice = _prompt("Pick", default="Q").upper()

    if choice in ("Q", ""):
        print(f"\n{DIM}Nothing launched. Re-run `python start.py` when ready.{RESET}\n")
        return

    if choice == "3":
        _run([PYTHON, os.path.join("scripts", "eval_hf_checkpoints.py")])
        return

    if choice == "4":
        _print_copy_paste_commands()
        return

    if choice == "5":
        _resume_training_picker()
        return

    if choice == "6":
        _fresh_start_launcher()
        return

    if choice in ("1", "2"):
        task, seed = _prompt_task_and_seed()
        _dispatch_training(task, seed, dry_run=(choice == "1"))
        return

    print(f"{YELLOW}!!! Unknown choice '{choice}'.{RESET}")


def _print_copy_paste_commands() -> None:
    """Print all manual training commands. Fallback for scripted workflows."""
    py_quoted = f'"{PYTHON}"'
    is_powershell = (
        platform.system() == "Windows"
        and os.environ.get("PSModulePath") is not None
    )
    py = f"& {py_quoted}" if is_powershell else py_quoted

    print(f"\n{BOLD}Regime A — Evaluate paper checkpoints (no training):{RESET}")
    print(f"  {CYAN}{py} scripts/eval_hf_checkpoints.py{RESET}\n")

    print(f"{BOLD}Regime B — Seed-variance fine-tune:{RESET} {DIM}(direct main.py){RESET}")
    for task, (config, init, desc) in TASK_DISPATCH.items():
        init_arg = f" --init-weights {init}" if init and os.path.exists(init) else ""
        print(f"  {CYAN}{py} main.py --mode train --config {config}{init_arg} --seed 0{RESET} {DIM}# {task}{RESET}")
    print()

    print(f"{BOLD}...or the shell launchers (they auto-set TRM_CHECKPOINT_DIR per seed):{RESET}")
    if platform.system() == "Windows":
        print(f"  {CYAN}scripts/run_seed.ps1 -Task sudoku-mlp -Seed 0{RESET}")
        print(f"  {CYAN}scripts/run_seed.ps1 -Task sudoku-mlp -Seed 0 -DryRun{RESET}  {DIM}# 5-epoch smoke{RESET}")
    else:
        print(f"  {CYAN}scripts/run_seed.sh sudoku-mlp 0{RESET}")
        print(f"  {CYAN}scripts/run_seed.sh sudoku-mlp 0 --dry-run{RESET}  {DIM}# 5-epoch smoke{RESET}")
    print()

    print(f"{BOLD}Other (non-fleet) TRM configs:{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/trm_sudoku.yaml --seed 0{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/trm_maze.yaml --seed 0{RESET}\n")

    print(f"{BOLD}LLM baselines on Sudoku (LoRA fine-tune):{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_config.yaml --seed 0{RESET}  {DIM}# GPT-2 (124M){RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_smollm.yaml --seed 0{RESET}  {DIM}# SmolLM2-360M{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_qwen.yaml --seed 0{RESET}    {DIM}# Qwen2.5-0.5B{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_llama.yaml --seed 0{RESET}   {DIM}# Llama-3.2-1B{RESET}\n")

    print(f"{BOLD}LLM baselines on Maze (LoRA fine-tune):{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_gpt2_maze.yaml --seed 0{RESET}    {DIM}# GPT-2 (124M){RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_smollm_maze.yaml --seed 0{RESET}  {DIM}# SmolLM2-360M{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_qwen_maze.yaml --seed 0{RESET}    {DIM}# Qwen2.5-0.5B{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_llama_maze.yaml --seed 0{RESET}   {DIM}# Llama-3.2-1B{RESET}\n")

    print(f"{BOLD}Distill a fine-tuned LLM into a small student:{RESET}")
    print(f"  {CYAN}{py} main.py --mode distill --config configs/llm_qwen.yaml --checkpoint models/llm/qwen2.5_0.5b_sudoku_latest.pt{RESET}\n")

    print(f"{BOLD}Evaluate (after training):{RESET}")
    print(f"  {CYAN}{py} main.py --mode eval --config configs/<name>.yaml --checkpoint models/<name>/best.pt{RESET}\n")

    print(f"{BOLD}Resume from last checkpoint:{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/<name>.yaml --resume models/<name>/latest.pt --seed 0{RESET}\n")

    print(f"{BOLD}Reproducibility — seed convention:{RESET}")
    print(f"  {DIM}Every command above passes {RESET}{CYAN}--seed N{RESET}{DIM} explicitly so the seed shows up in{RESET}")
    print(f"  {DIM}shell history AND the wandb run name. Omitting --seed inherits the{RESET}")
    print(f"  {DIM}{RESET}{CYAN}seed:{RESET}{DIM} field from the YAML (default 42 — reproducible). Set {RESET}{CYAN}seed: -1{RESET}{DIM}{RESET}")
    print(f"  {DIM}in the YAML for wall-clock seeding.{RESET}\n")

    print(f"{DIM}Tip: run `python start.py status` any time to re-check stages.{RESET}")
    print(f"{DIM}Tip: Weave traces appear at wandb.ai/<entity>/<project>/weave/monitors{RESET}\n")


def _print_training_menu() -> None:
    print(f"\n{GREEN}{BOLD}✓ All setup complete!{RESET}\n")

    # 6-machine fleet assignment table. Each operator picks the row matching
    # their box — seeds 0..5 across tasks give the full seed-variance set the
    # coursework report needs for mean ± std on the three-way model comparison.
    print(f"{BOLD}6-Machine Fleet Plan:{RESET} {DIM}(pick the row for this box){RESET}")
    print(f"  {DIM}machine   task          seed{RESET}")
    for idx, task, seed in FLEET_PLAN:
        print(f"  {CYAN}{idx:<9}{RESET} {task:<13s} {CYAN}{seed}{RESET}")
    print()

    # TRM_WORK_DIR status + reminder. The launcher will exit(3) later if this
    # resolves inside OneDrive; this printout is the friendly early warning.
    print(f"{BOLD}Work dir for training outputs:{RESET}")
    current_work = os.environ.get("TRM_WORK_DIR")
    if current_work:
        work_label = f"{CYAN}{current_work}{RESET}"
    else:
        work_label = f"{CYAN}{_default_work_dir()}{RESET}  {DIM}(auto-picked — TRM_WORK_DIR not set){RESET}"
    print(f"  TRM_WORK_DIR = {work_label}")
    if platform.system() == "Windows":
        print(f"  {DIM}To set for this shell: {RESET}{CYAN}$env:TRM_WORK_DIR = 'C:/ml-trm-work'{RESET}")
    else:
        print(f"  {DIM}To set for this shell: {RESET}{CYAN}export TRM_WORK_DIR=$HOME/ml-trm-work{RESET}")
    print(f"  {DIM}(MUST be a local non-OneDrive path — parallel runs on the shared{RESET}")
    print(f"  {DIM} OneDrive would corrupt each machine's checkpoints during upload.){RESET}\n")

    # wandb reminder — re-check here even though the setup stage covered it,
    # because it's common to lose the netrc entry when switching machines.
    if _wandb_ready():
        print(f"{BOLD}wandb:{RESET} {GREEN}✓ logged in{RESET}  {DIM}(runs will track to your wandb project){RESET}\n")
    else:
        print(f"{BOLD}wandb:{RESET} {YELLOW}not logged in{RESET}  {DIM}(runs still train, just without cloud tracking){RESET}")
        if os.path.exists(WANDB):
            print(f"  {CYAN}\"{WANDB}\" login{RESET}  {DIM}# paste your key from wandb.ai/authorize{RESET}\n")
        else:
            print(f"  {CYAN}wandb login{RESET}  {DIM}# paste your key from wandb.ai/authorize{RESET}\n")

    # Hand off to the interactive launcher. If the user picks a task, it
    # dispatches via subprocess.run + sys.exit — we never return here. If
    # they quit or pick "show commands", we fall through and exit naturally.
    _interactive_launcher()


# ============================================================
# Entry points
# ============================================================

def main() -> None:
    args = sys.argv[1:]
    skip_wandb = "--skip-wandb" in args
    args = [a for a in args if a != "--skip-wandb"]

    if args and args[0] == "status":
        results = [(s, s.check()) for s in STAGES]
        _print_stage_status(results, skip_wandb)
        missing = [s for s, done in results if not done and s.blocking]
        if not missing:
            print(f"{GREEN}All blocking stages ready.{RESET} Run: python start.py")
        else:
            print(f"{BOLD}Next stage:{RESET} {missing[0].key}. Run: python start.py")
        return

    results = [(s, s.check()) for s in STAGES]
    _print_stage_status(results, skip_wandb)

    for stage, done in results:
        if done:
            continue
        if stage.key == "wandb" and skip_wandb:
            continue
        print(f"{BOLD}[{stage.key}] {stage.label}{RESET} — not ready. Running...")
        stage.action()
        if stage.check():
            print(f"\n{GREEN}✓ [{stage.key}] ready.{RESET}")
            print(f"\n{BOLD}→ Next:{RESET} python start.py")
        return

    # All stages done
    _print_training_menu()


if __name__ == "__main__":
    main()
