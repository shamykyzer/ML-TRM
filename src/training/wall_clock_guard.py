"""Wall-clock training cap for the novelty iso-time experiment.

Armed by `TRM_MAX_TRAIN_SECONDS` env var. Unset = dormant, every helper
is a no-op, and regular fleet runs see exactly the behavior they had
before this module existed.

Usage (trainer hook at top of epoch loop):

    from src.training.wall_clock_guard import wall_clock_expired

    for epoch in range(total_epochs):
        if wall_clock_expired():
            # save the checkpoint and break
            break
        ...

The first call to any helper starts the clock. Placing that first call
inside the epoch loop (not at module import) excludes setup time —
model init, HF weight download, dataset materialization — from the
budget, so the 2.5 hr cap is "training time" in the intuitive sense.
"""
import os
import time
from typing import Optional


_start_time: Optional[float] = None
_max_seconds: Optional[float] = None
_initialized: bool = False


def _init() -> None:
    """Read env var on first call. Idempotent on subsequent calls."""
    global _start_time, _max_seconds, _initialized
    if _initialized:
        return
    _initialized = True

    raw = os.environ.get("TRM_MAX_TRAIN_SECONDS", "").strip()
    if not raw:
        return
    try:
        parsed = float(raw)
    except ValueError:
        return
    if parsed <= 0:
        return
    _max_seconds = parsed
    _start_time = time.time()


def is_active() -> bool:
    """Whether the guard is armed (env var set to a positive float)."""
    _init()
    return _max_seconds is not None


def wall_clock_expired() -> bool:
    """True once elapsed wall-clock exceeds the budget. Dormant => always False."""
    _init()
    if _max_seconds is None or _start_time is None:
        return False
    return (time.time() - _start_time) >= _max_seconds


def seconds_remaining() -> Optional[float]:
    """Seconds left in budget, or None if dormant."""
    _init()
    if _max_seconds is None or _start_time is None:
        return None
    return max(0.0, _max_seconds - (time.time() - _start_time))


def seconds_elapsed() -> Optional[float]:
    """Seconds since first call, or None if dormant."""
    _init()
    if _start_time is None:
        return None
    return time.time() - _start_time


def max_seconds() -> Optional[float]:
    """The configured budget in seconds, or None if dormant."""
    _init()
    return _max_seconds
