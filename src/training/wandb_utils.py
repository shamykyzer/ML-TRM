"""Shared W&B + Weave initialization helper for all trainers.

One function, `init_wandb(config)`, handles auth check, hostname-tagged run
name, graceful degradation, and Weave trace initialization — so the four
trainers (`trainer_official`, `trainer_trm`, `trainer_llm`, `trainer_distill`)
don't each reimplement the same init block.

Graceful degradation rules:
    use_wandb=False                    → return False, no wandb, no weave
    wandb not installed                → print warning, return False
    use_wandb=True but no API key      → print warning, return False
    use_wandb=True and authed          → wandb.init(...), weave.init(...) if enabled, return True
    weave not installed or fails       → warn and continue (wandb stays active)
"""
import os
import socket
import time

from src.utils.config import ExperimentConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import weave
    WEAVE_AVAILABLE = True
    weave_op = weave.op  # real decorator
except ImportError:
    WEAVE_AVAILABLE = False

    # No-op fallback so trainers can write `@weave_op()` unconditionally:
    # when weave isn't installed, the decorator just returns the function.
    def weave_op(fn=None, **_kwargs):  # type: ignore[no-redef]
        if fn is None:
            return lambda f: f
        return fn


def init_wandb(config: ExperimentConfig) -> bool:
    """Initialize W&B (and Weave) if enabled and credentials are present.

    Returns True if wandb is active for this run, False otherwise.
    When wandb is active AND config.training.use_weave is True AND weave is
    installed, also calls weave.init() so that @weave.op()-decorated functions
    generate traces for the wandb.ai/<entity>/<project>/weave/monitors page.
    """
    tc = config.training
    if not tc.use_wandb:
        return False
    if not WANDB_AVAILABLE:
        print("[W&B] use_wandb=true but wandb is not installed — disabling.")
        print("[W&B] Fix: pip install wandb")
        return False

    # Auth check: WANDB_API_KEY env var OR a netrc file with wandb creds.
    # On Windows, the wandb CLI writes ~/_netrc (underscore) rather than
    # ~/.netrc, so we check both filenames regardless of platform.
    has_netrc_auth = False
    home = os.path.expanduser("~")
    for name in (".netrc", "_netrc"):
        path = os.path.join(home, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                if "api.wandb.ai" in f.read():
                    has_netrc_auth = True
                    break
        except OSError:
            continue

    if not (os.getenv("WANDB_API_KEY") or has_netrc_auth):
        print("[W&B] use_wandb=true but no WANDB_API_KEY set and no netrc auth — disabling wandb.")
        print("[W&B] Fix: run `wandb login` (writes ~/.netrc or ~/_netrc) or set WANDB_API_KEY in .env")
        return False

    hostname = socket.gethostname()
    task = config.model.model_type.value
    run_name = f"{task}_{hostname}_{int(time.time())}"

    wandb.init(
        entity=tc.wandb_entity or None,
        project=tc.wandb_project,
        name=run_name,
        tags=[task, hostname],
        config=config.model_dump(),
    )
    print(f"[W&B] Run: {run_name}")

    # Weave — traces for the monitors page at wandb.ai/<entity>/<project>/weave/monitors
    if tc.use_weave and WEAVE_AVAILABLE:
        try:
            # weave.init takes "entity/project" — fall back to just project if
            # no entity (weave will use the default from wandb login).
            weave_target = f"{tc.wandb_entity}/{tc.wandb_project}" if tc.wandb_entity else tc.wandb_project
            weave.init(weave_target)
            print(f"[Weave] Traces initialized → {weave_target}")
        except Exception as e:
            print(f"[Weave] init failed, continuing without Weave traces: {e}")
    elif tc.use_weave and not WEAVE_AVAILABLE:
        print("[Weave] use_weave=true but weave is not installed — skipping. (pip install weave)")

    return True


def define_common_metrics(
    use_wandb: bool,
    namespaces: tuple[str, ...] = ("train", "val", "carbon", "system"),
    summaries: dict[str, str] | None = None,
) -> None:
    """Register the shared wandb panel structure used by all TRM trainers.

    Without this, metrics logged by ``trainer_trm`` / ``trainer_llm`` /
    ``trainer_distill`` end up at the default global root in the wandb UI
    and show up as an unsorted pile. Calling ``define_common_metrics`` right
    after ``init_wandb`` makes the UI group them into ``train/``, ``val/``,
    ``carbon/``, and ``system/`` panels, with sensible summary aggregations
    (``max`` for accuracies, ``min`` for losses, etc).

    Parameters
    ----------
    use_wandb : bool
        Return value of ``init_wandb`` — when False, the function is a no-op
        so callers can write ``define_common_metrics(self.use_wandb)`` without
        any conditional.
    namespaces : tuple[str, ...]
        Panel prefixes to register. Every ``<namespace>/*`` metric is x-axised
        against the (hidden) ``epoch`` step metric.
    summaries : dict[str, str] | None
        Optional override of the default summary aggregations. Keys are wandb
        glob patterns (``val/*_acc``, ``*/loss``, ``carbon/*`` ...), values
        are aggregation names (``max``, ``min``, ``mean``, ``last``). Merges
        on top of the defaults below; pass ``None`` to accept the defaults.

    Defaults
    --------
    ``val/*_acc`` → ``max``        (accuracies — we care about the peak)
    ``*/loss``    → ``min``        (ce_loss / q_loss / lm_loss / loss)
    ``carbon/*``  → ``last``       (cumulative counters)
    ``system/*``  → ``max``        (GPU mem / util peaks)
    ``train/lr``  → ``last``       (schedule snapshot)
    ``*/_sec``    → ``mean``       (per-step timings)

    Notes
    -----
    ``init_wandb`` must run first — this function assumes a wandb run is
    already active. If ``use_wandb`` is False the function returns silently
    without importing wandb, so it is safe on machines without the package
    installed.
    """
    if not use_wandb:
        return

    # Local import so test / no-wandb environments don't pay the import cost.
    import wandb

    wandb.define_metric("epoch", hidden=True)
    for ns in namespaces:
        wandb.define_metric(f"{ns}/*", step_metric="epoch")

    default_summaries: dict[str, str] = {
        "val/*_acc": "max",
        "*/loss": "min",
        "carbon/*": "last",
        "system/*": "max",
        "train/lr": "last",
        "*/_sec": "mean",
    }
    if summaries:
        default_summaries.update(summaries)

    for pattern, agg in default_summaries.items():
        wandb.define_metric(pattern, summary=agg)
