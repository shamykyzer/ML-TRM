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

# Repo root resolved from this file's path: src/training/wandb_utils.py → repo
# Used to find wandb_api.txt regardless of process cwd (matters for tests
# and for direct trainer imports outside the start.py / main.py launchers).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WANDB_API_FILE = os.path.join(_REPO_ROOT, "wandb_api.txt")


def _load_api_key_from_file() -> str:
    """Read wandb_api.txt at the repo root, return the stripped token or ''.

    Convenience for users who don't want to deal with `wandb login` or
    edit `.env`: drop the key in `wandb_api.txt` and any trainer process
    will pick it up via this helper. Idempotent and safe to call when the
    file is missing — returns empty string.

    Token validation is deliberately minimal (length sanity check). The
    real verification happens when wandb.init() actually contacts the API.
    """
    if not os.path.exists(_WANDB_API_FILE):
        return ""
    try:
        with open(_WANDB_API_FILE, encoding="utf-8") as f:
            token = f.read().strip()
    except OSError:
        return ""
    # Wandb API keys are >= 40 chars (legacy hex) or longer (new prefixed
    # format). Anything shorter is almost certainly a typo or a placeholder
    # — better to silently ignore than to send junk to api.wandb.ai.
    return token if len(token) >= 40 else ""

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

    # Auth-key bootstrap from wandb_api.txt — runs BEFORE the env+netrc
    # check so a freshly-dropped token activates wandb on the very next
    # run with no other config touched. We only set the env var when it
    # is unset; an existing WANDB_API_KEY (e.g. from .env via load_dotenv)
    # takes precedence so users can override per-run from the shell.
    if not os.getenv("WANDB_API_KEY"):
        token = _load_api_key_from_file()
        if token:
            os.environ["WANDB_API_KEY"] = token
            print(f"[W&B] Loaded API key from {_WANDB_API_FILE}")

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
        print("[W&B] Fix: paste your key into wandb_api.txt at the repo root,")
        print("[W&B]      OR run `wandb login`, OR set WANDB_API_KEY in .env")
        return False

    hostname = socket.gethostname()
    base_task = config.model.model_type.value

    # All 4 LLM-finetune YAMLs (gpt2/smollm/qwen/llama) and both distill
    # variants share the same model_type enum, so a naked `task` would collapse
    # the 8-run LLM × {sudoku,maze} sweep into one undistinguishable label in
    # the wandb sidebar. Synthesize a richer task_label from llm_name + dataset
    # so every row in the sweep gets a unique name (e.g. llm_qwen2_5_0_5b_maze).
    if base_task in ("llm_finetune", "llm_distill"):
        llm_short = (
            config.model.llm_name.split("/")[-1]
            .lower()
            .replace("-", "_")
            .replace(".", "_")
        )
        dataset = config.data.dataset
        kind = "distill" if base_task == "llm_distill" else "llm"
        task_label = f"{kind}_{llm_short}_{dataset}"
        extra_tags = [llm_short, dataset]
    else:
        task_label = base_task
        extra_tags = []

    seed = config.seed
    run_name = f"{task_label}_seed{seed}_{hostname}_{int(time.time())}"
    tags = [task_label, hostname, f"seed{seed}", *extra_tags]

    # `group` lets wandb's UI fold multi-seed runs of the same (model, dataset)
    # together; `job_type` distinguishes seeds inside that group. Together they
    # turn the runs table into a clean grid: rows = task_label, cols = seedN.
    wandb.init(
        entity=tc.wandb_entity or None,
        project=tc.wandb_project,
        name=run_name,
        group=task_label,
        job_type=f"seed{seed}",
        tags=tags,
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
        # Accuracies — peak value across the run is the headline number.
        "val/*_acc": "max",
        "val/accuracy": "max",         # alias of val/cell_acc (trainer_llm/distill)
        "val/exact_accuracy": "max",   # alias of val/puzzle_acc (trainer_llm/distill)
        "val/puzzle_acc": "max",       # explicit fallback in case glob rejected
        "val/cell_acc": "max",         # explicit fallback
        # Losses — lowest value reached is "best".
        "*/loss": "min",
        "train/loss": "min",           # explicit fallback
        "val/loss": "min",             # explicit fallback
        # Cumulative counters / snapshots.
        "carbon/*": "last",
        "system/*": "max",
        "train/lr": "last",
        "train/elapsed_min": "last",
        "*/_sec": "mean",
    }
    if summaries:
        default_summaries.update(summaries)

    # wandb's define_metric currently only accepts suffix globs (e.g. "val/*"),
    # not infix ones ("val/*_acc"). Any pattern it rejects raises wandb.Error
    # mid-trainer-init — a fatal stop just to set summary preferences. Swallow
    # rejections per pattern so unsupported entries degrade to "no preference"
    # without taking the training run with them.
    for pattern, agg in default_summaries.items():
        try:
            wandb.define_metric(pattern, summary=agg)
        except wandb.Error:
            continue
