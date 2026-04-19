"""OneDrive-safe work directory resolution for ML-TRM.

Where per-seed outputs land. MUST be a local non-OneDrive path: the 6-box
fleet shares one OneDrive tree for code + data + HF weights (good), but
parallel training writes inside the sync folder would corrupt checkpoints
during upload. We auto-pick the first writable non-OneDrive candidate and
export it as TRM_WORK_DIR so subprocesses (and any `echo $TRM_WORK_DIR`)
see the same value the launcher used. Override by setting TRM_WORK_DIR
yourself before running start.py.
"""
import os
import platform
import sys

from src.cli.console import CYAN, DIM, RESET, YELLOW

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
