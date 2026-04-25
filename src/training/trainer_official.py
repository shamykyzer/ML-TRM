"""Trainer for the official TRM architecture.

Key differences from trainer_trm.py:
- Carry-based ACT loop (model manages halt state, not fixed N_sup steps)
- ACTLossHead computes all losses (StableMax CE + Q-learning)
- AdamATan2 optimizer with separate param groups for task embeddings
- bfloat16 native forward (no GradScaler needed)
- W&B logging alongside CSV
"""

import csv
import json
import os
import re
import shutil
import sys
import time
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class _TqdmNewlineFile:
    """File wrapper that converts tqdm's in-place refresh into a new line.

    tqdm refreshes progress bars by writing "\\r<bar>" — the carriage return
    moves the cursor back so the next write overwrites the same terminal
    line ("one slot, updating"). This wrapper strips the carriage return
    and any ANSI cursor-movement escapes, then appends a newline, so every
    refresh emits a fresh line instead of overwriting.

    Result: append-only progress log that survives in terminal scrollback
    and captures cleanly when stdout is piped to a file or tailed by a
    log collector. Trade-off: bars no longer share a single row, so
    nested-bar `position=` layouts must be dropped (the cursor-up escapes
    they rely on are stripped by this wrapper).
    """

    # Matches CSI cursor-up / cursor-next-line sequences that nested tqdm
    # bars emit to jump between rows. We strip them because in newline mode
    # they'd either leak as garbage or do nothing meaningful.
    _ANSI_CURSOR_MOVE = re.compile(r"\x1b\[\d*[AFJK]")

    def __init__(self, fp):
        # Reconfigure the underlying stream to UTF-8 so the unicode block
        # characters used in the progress bar (▕ ▏ ▎ ▍ ▌ ▋ ▊ ▉ █) don't
        # raise UnicodeEncodeError on legacy Windows consoles that default
        # to cp1252. `reconfigure` is a no-op on streams that don't support
        # it; `backslashreplace` keeps any future non-encodable byte from
        # crashing training mid-epoch.
        if hasattr(fp, "reconfigure"):
            try:
                fp.reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass
        self.fp = fp

    def write(self, s: str) -> None:
        if not s:
            return
        s = s.lstrip("\r")
        s = self._ANSI_CURSOR_MOVE.sub("", s)
        s = s.rstrip()
        if not s:
            return
        self.fp.write(s + "\n")

    def flush(self) -> None:
        self.fp.flush()

    # tqdm probes terminal size (for `dynamic_ncols`) by calling `.fileno()`
    # on the file object it was constructed with, then handing that fd to an
    # ioctl / shutil.get_terminal_size() call. Forward to the wrapped fp so
    # tqdm sees the real stderr fd and can detect the actual terminal width
    # — without this, tqdm falls back to a fixed 80 columns regardless of
    # how wide the terminal is.
    def fileno(self) -> int:
        return self.fp.fileno()

    def isatty(self) -> bool:
        return getattr(self.fp, "isatty", lambda: False)()

from src.models.losses_official import ACTLossHead
from src.training.carbon_tracker import CarbonTracker
from src.training.ema import EMA
from src.training.wall_clock_guard import wall_clock_expired
from src.training.wandb_utils import init_wandb, weave_op
from src.utils.config import ExperimentConfig

try:
    import wandb  # still needed for wandb.log / wandb.finish / wandb.Artifact
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def _build_optimizer(model: nn.Module, config: ExperimentConfig):
    """Build optimizer with separate param groups for task embeddings."""
    tc = config.training

    # Separate task embedding params for different lr/wd
    task_emb_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "task_emb" in name:
            task_emb_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "lr": tc.lr, "weight_decay": tc.weight_decay},
    ]
    if task_emb_params:
        param_groups.append({
            "params": task_emb_params,
            "lr": tc.task_emb_lr,
            "weight_decay": tc.task_emb_weight_decay,
        })

    if tc.optimizer == "adam_atan2":
        # Prefer the official CUDA build (`adam_atan2.AdamATan2`) when it is
        # actually usable — on most systems (and always on Windows CPU-builds)
        # the PyPI `adam-atan2` package ships only a wrapper around a CUDA
        # kernel that is NOT bundled, so importing it raises ModuleNotFoundError
        # for `adam_atan2_backend`. Fall back to the pure-Python
        # `adam_atan2_pytorch` implementation (lucidrains), which is the same
        # algorithm and is what the "Less is More" recipe calls for. We do NOT
        # silently fall back to AdamW — AdamATan2 is mandatory for official runs.
        AdamAtan2Cls = None
        impl = None
        try:
            from adam_atan2 import AdamATan2 as _CudaAdamATan2  # type: ignore
            AdamAtan2Cls = _CudaAdamATan2
            impl = "adam_atan2 (CUDA backend)"
        except Exception:
            try:
                from adam_atan2_pytorch import AdamAtan2 as _PyAdamAtan2  # type: ignore
                AdamAtan2Cls = _PyAdamAtan2
                impl = "adam_atan2_pytorch (pure Python)"
            except ImportError as e:
                raise ImportError(
                    "AdamATan2 is required for the official TRM recipe "
                    "(Less-is-More paper). Neither `adam_atan2` (with its CUDA "
                    "backend) nor `adam_atan2_pytorch` (pure Python) is "
                    "importable. Install the pure-Python version with:\n"
                    "    pip install adam-atan2-pytorch\n"
                    "Do NOT switch to AdamW for official runs."
                ) from e
        optimizer = AdamAtan2Cls(param_groups, betas=tc.betas)
        print(f"[Optimizer] Using AdamATan2 via {impl}")
        return optimizer

    optimizer = torch.optim.AdamW(param_groups, betas=tc.betas)
    print("[Optimizer] Using AdamW")
    return optimizer


class OfficialTRMTrainer:
    """Trainer for the official TRM architecture with Q-learning ACT."""

    def __init__(
        self,
        model: nn.Module,
        loss_head: ACTLossHead,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ExperimentConfig,
        resume_checkpoint: str = "",
        init_weights: str = "",
    ):
        self.model = model
        self.loss_head = loss_head
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tc = config.training
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Native bfloat16 forward — no GradScaler needed (bfloat16 has float32's exponent range)
        self.forward_dtype = getattr(torch, config.model.forward_dtype, torch.bfloat16)
        self.model.to(device=self.device, dtype=self.forward_dtype)
        self.loss_head.to(device=self.device, dtype=self.forward_dtype)

        self.optimizer = _build_optimizer(model, config)

        # Linear warmup scheduler
        self.global_step = 0
        self.start_epoch = 0
        self.best_acc = 0.0

        def lr_lambda(step: int) -> float:
            if step < self.tc.warmup_steps:
                return step / max(1, self.tc.warmup_steps)
            return 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.ema = EMA(model, decay=self.tc.ema_decay)

        self.carbon = CarbonTracker(
            f"{config.model.model_type.value}_train",
            output_dir=config.experiment_dir,
        )

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.experiment_dir, exist_ok=True)

        # Pre-compute milestone epochs: {epoch_number (1-indexed) -> percent label}
        self.milestone_epochs: dict[int, int] = {}
        if self.tc.milestone_checkpoints:
            for frac in self.tc.milestone_fractions:
                ep = max(1, int(round(self.tc.epochs * frac)))
                self.milestone_epochs[ep] = int(round(frac * 100))

        # HuggingFace Hub
        self.hf_repo_id = getattr(self.tc, "hf_repo_id", "")
        if self.hf_repo_id and HF_AVAILABLE:
            self.hf_api = HfApi()
            self.hf_api.create_repo(self.hf_repo_id, exist_ok=True, private=True)
            print(f"[HF Hub] Syncing checkpoints to {self.hf_repo_id}")
        else:
            self.hf_api = None

        # Init weights (transfer learning) — load model weights only, start
        # optimizer/step/epoch fresh. Mutually exclusive with resume_checkpoint
        # (main.py enforces this). Use --init-weights to bootstrap training
        # from a pretrained checkpoint (e.g. the remapped HF reference weights)
        # without inheriting its training state.
        if init_weights and os.path.isfile(init_weights):
            self._load_init_weights(init_weights)
            # self.ema was constructed above (line ~160) from the model's
            # random init, BEFORE these HF weights were loaded. Reseed the
            # shadow now so it tracks the actual starting point of training
            # instead of the random init that no longer exists in the model.
            # Without this reseed, the shadow slowly decays away from random
            # init toward the HF weights over ~1/(1-decay) ≈ 1000 steps,
            # wasting the first ~0.5 epoch of eval signal.
            self._reseed_ema_shadow("post-init-weights")

        # Resume
        if resume_checkpoint and os.path.isfile(resume_checkpoint):
            self._load_checkpoint(resume_checkpoint)

        # W&B + Weave — hostname-tagged run name, graceful auth check,
        # and Weave trace init all handled in the shared helper.
        self.use_wandb = init_wandb(config)

        # Predefine metrics so the wandb dashboard auto-organizes on open:
        # x-axis = epoch for every namespace, summary aggregations for the
        # run-overview card and sortable runs table. Must come BEFORE the
        # first wandb.log call (which happens in train() after _init_log).
        self._define_wandb_metrics()

        # One-shot flag for live-syncing best.pt to the wandb Files tab.
        # wandb.save(policy="live") is a "register once, watcher handles
        # overwrites forever" primitive — calling it on every new-best event
        # would just churn the watcher. So we set this True after the first
        # best.pt is saved + registered, and skip the call on subsequent bests.
        self._best_wandb_registered = False

        # One-shot regression alert — fires the first time val drops more than
        # tc.regression_alert_threshold below self.best_acc. Catches dz3tkge9-
        # style fine-tune regressions in one eval cycle instead of 23 hours.
        self._regression_alert_fired = False

        # CSV log
        self.log_path = os.path.join(
            config.experiment_dir, f"{config.model.model_type.value}_train_log.csv"
        )

    def _init_log(self) -> None:
        mode = "a" if self.start_epoch > 0 and os.path.exists(self.log_path) else "w"
        if mode == "w":
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "lm_loss", "q_halt_loss", "q_continue_loss",
                    "accuracy", "exact_accuracy", "q_halt_accuracy", "avg_steps",
                    "val_cell_acc", "val_puzzle_acc", "best_puzzle_acc", "elapsed_min",
                ])
        # Register the CSV for live sync to the wandb run's Files tab. The
        # file exists on disk by this point (whether newly-created with the
        # header row above, or preserved across a resume). One call here
        # covers every subsequent _append_log write — wandb's watcher picks
        # up each modification and re-uploads it.
        self._wandb_save_live(self.log_path)

    def _append_log(self, row: list) -> None:
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _define_wandb_metrics(self) -> None:
        """Predefine metric axes and summaries so the wandb dashboard auto-organizes.

        Three things get set up, all via wandb.define_metric (idempotent,
        persists for the life of the run, no effect if wandb is disabled):

        1. **x-axis = epoch**. By default wandb plots everything against its
           internal "Step" counter. We declare an explicit `epoch` metric
           and point every namespace (train/, val/, carbon/, system/) at it
           via step_metric="epoch", so all charts show "Epoch" on the x-axis
           and resumed runs line up cleanly instead of creating a Step-axis
           discontinuity. The `hidden=True` on the epoch declaration stops
           wandb from also plotting epoch as its own 1:1 diagonal line chart.

        2. **Summary aggregations** — the single number that appears in the
           run-overview card and the sortable runs table. wandb defaults to
           "last value" which is wrong for most of our metrics:
             - val/train accuracy → max (peak is the achievement)
             - losses            → min (trough is the achievement)
             - throughput / time → mean (per-run average is the useful number)
             - carbon counters   → last (they're cumulative — last = total)
             - GPU peak mem      → max (true peak across epochs = sizing info)
             - learning rate     → last (final value after warmup; flat thereafter)
           Setting these explicitly means you can sort your runs table by
           `val/puzzle_acc` to find the best run without clicking in.

        3. **Glob patterns** (`train/*` etc.) cover present-and-future
           metrics under each prefix, so adding a new metric in _train_epoch
           tomorrow doesn't require editing this function.

        Must run AFTER init_wandb succeeds and BEFORE the first wandb.log
        call — i.e. exactly where we call it from __init__.
        """
        if not self.use_wandb:
            return

        # The x-axis metric itself. hidden=True keeps it out of the workspace
        # as a standalone chart (it'd just be a 1:1 line, noise).
        wandb.define_metric("epoch", hidden=True)

        # =====================================================================
        # ORDERING CONTRACT — summary/ FIRST so it renders at the very top of
        # the workspace, then val/, train/, carbon/, system/. wandb's
        # workspace renders panel sections in the order metrics are first
        # registered (define_metric counts as registration). Do NOT reshuffle
        # these blocks unless you want the workspace to flip — train/* gets
        # logged on every epoch while val/* only fires at eval_interval, so
        # without this ordering the train section dominates by sheer log
        # frequency.
        # =====================================================================

        # 0. SUMMARY — thesis-critical metrics pinned to the top of the
        #    workspace. These are aliases of live metrics already logged under
        #    val/, train/, carbon/ — duplicated here so the user gets a single
        #    "at a glance" section combining val accuracy, train accuracy,
        #    loss, and carbon in one place (wandb groups panels strictly by
        #    namespace prefix, so without this mirror the same numbers are
        #    scattered across three sections). Aggregations follow the same
        #    conventions as the underlying metric: max for accuracies, min
        #    for losses, last for cumulative counters.
        for m, agg in (
            ("summary/val_accuracy",            "max"),
            ("summary/val_exact_accuracy",      "max"),
            ("summary/val_q_halt_accuracy",     "max"),
            ("summary/train_accuracy",          "max"),
            ("summary/train_exact_accuracy",    "max"),
            ("summary/train_q_halt_accuracy",   "max"),
            ("summary/train_loss",              "min"),
            ("summary/train_q_halt_loss",       "min"),
            ("summary/train_q_continue_loss",   "min"),
            ("summary/train_frac_at_max_steps", "max"),
            ("summary/carbon_emissions_kg",     "last"),
        ):
            wandb.define_metric(m, step_metric="epoch", summary=agg)
        wandb.define_metric("summary/*", step_metric="epoch")

        # 1. PRIORITY — headline thesis numbers, pinned to the very top.
        #    Defined BEFORE the namespace globs so they are the first
        #    explicit metrics wandb sees (globs come after, register the
        #    rest implicitly when first logged). We register the
        #    symmetric-with-train aliases (val/accuracy, val/exact_accuracy)
        #    FIRST inside this block so they out-rank the legacy short
        #    names in the dashboard column order.
        for m in (
            "val/accuracy",        # alias of val/cell_acc — symmetric w/ train/accuracy
            "val/exact_accuracy",  # = puzzle_acc — symmetric w/ train/exact_accuracy
            "val/puzzle_acc",      # legacy name kept for backward-compat
            "val/cell_acc",        # legacy name kept for backward-compat
        ):
            wandb.define_metric(m, step_metric="epoch", summary="max")

        # 2. Validation losses (min — small is good).
        for m in ("val/lm_loss", "val/q_halt_loss", "val/q_continue_loss"):
            wandb.define_metric(m, step_metric="epoch", summary="min")

        # 3. Validation halting dynamics — diagnostic for ACT collapse.
        #    Both new (val/q_halt_accuracy, val/avg_steps) and legacy
        #    (val/q_halt_acc, val/avg_act_steps) names registered.
        for m in (
            "val/q_halt_accuracy", "val/q_halt_acc",
            "val/avg_steps",       "val/avg_act_steps",
            "val/frac_at_max_steps",
        ):
            wandb.define_metric(m, step_metric="epoch", summary="max")

        # 4. Namespace glob for any future val/* metric not enumerated above.
        #    Has to be in this position (after explicit val/* metrics, before
        #    train/*) so newly-added val metrics still beat train/ panels to
        #    registration.
        wandb.define_metric("val/*", step_metric="epoch")

        # 5. Training accuracy (max — sanity check that train > val).
        for m in ("train/accuracy", "train/exact_accuracy", "train/q_halt_accuracy"):
            wandb.define_metric(m, step_metric="epoch", summary="max")

        # 6. Training losses (min).
        for m in ("train/lm_loss", "train/q_halt_loss", "train/q_continue_loss"):
            wandb.define_metric(m, step_metric="epoch", summary="min")

        # 7. Throughput / wall-clock — mean across run is steady-state rate.
        for m in ("train/samples_per_sec", "train/epoch_time_sec"):
            wandb.define_metric(m, step_metric="epoch", summary="mean")

        # 8. Train halting dynamics — high values flag ACT-ceiling failures.
        wandb.define_metric("train/frac_at_max_steps", step_metric="epoch", summary="max")

        # 9. Learning rate snapshot — post-warmup steady-state value.
        wandb.define_metric("train/lr", step_metric="epoch", summary="last")

        # 10. Namespace glob for any other train/* metric.
        wandb.define_metric("train/*", step_metric="epoch")

        # 11. Carbon counters — monotonic, last value IS cumulative total.
        for m in ("carbon/emissions_kg", "carbon/energy_kwh"):
            wandb.define_metric(m, step_metric="epoch", summary="last")
        wandb.define_metric("carbon/*", step_metric="epoch")

        # 12. System resource peaks — high-water mark for sizing.
        wandb.define_metric("system/gpu_mem_gb", step_metric="epoch", summary="max")
        wandb.define_metric("system/*", step_metric="epoch")

        # Top-level thesis-favourite: best_puzzle_acc is already tracked
        # internally via self.best_acc and printed on every log interval,
        # but we don't currently log it to wandb as its own time-series.
        # (The max summary on val/puzzle_acc covers this for the overview
        # card, so no extra log call is needed.)

    def _wandb_save_live(self, path: str) -> None:
        """Register a file for live sync to the wandb run's Files tab.

        wandb.save(policy="live") watches `path` and re-uploads on every
        modification, so a single call covers all subsequent overwrites /
        appends. We pass base_path=dirname(path) so the file lands at the
        root of the Files tab under its basename, not nested inside the
        absolute local path (wandb otherwise mirrors the full dir tree).

        Silent on failure: an upload glitch or unusual filesystem layout
        shouldn't crash training — the file still exists on disk as the
        authoritative copy, and the next checkpoint will re-try.
        """
        if not self.use_wandb:
            return
        try:
            wandb.save(path, base_path=os.path.dirname(path), policy="live")
        except Exception as e:
            tqdm.write(f"[wandb] live-save registration failed for {path}: {e}")

    def _load_init_weights(self, path: str) -> None:
        """Load model weights only (partial state_dict OK) for transfer learning.

        Unlike _load_checkpoint, this does NOT restore the optimizer, EMA,
        global_step, or start_epoch. Training starts from step 0 using the
        loaded weights as initialization, with any keys not present in the
        file left at their random init (strict=False).

        Prints a summary of which keys were loaded vs which stayed random,
        so we can verify the remap covered the reasoning core and only the
        intentionally-excluded keys (embed_tokens, lm_head, task_emb, rotary
        buffers) are flagged as missing.
        """
        print(f"Loading init weights from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        if "note" in ckpt:
            print(f"  source note: {ckpt['note']}")

        result = self.model.load_state_dict(state, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)

        loaded_count = len(state) - len(unexpected)
        print(f"  loaded:     {loaded_count}/{len(state)} keys from checkpoint")
        if missing:
            print(f"  missing   ({len(missing)}): keys in model but not in checkpoint (fresh random init):")
            for k in missing:
                print(f"    {k}")
        if unexpected:
            print(f"  unexpected ({len(unexpected)}): keys in checkpoint but not in model (silently dropped):")
            for k in unexpected:
                print(f"    {k}")
        print("  optimizer/EMA/global_step unchanged — training starts from step 0")

    def _load_checkpoint(self, path: str) -> None:
        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Skip ema_state_dict even when it's in the checkpoint.
        #
        # Checkpoints written prior to the ema.py fp32-shadow fix contain a
        # bf16 shadow that was frozen at its init values (the native-bf16
        # forward path + EMA's per-step ~0.1% delta falls below bf16's
        # ~0.8% mantissa precision — every update rounds to a no-op). Those
        # shadows report the accuracy of the random-init model forever, so
        # loading them back in would re-poison eval on the resumed run.
        #
        # The fp32 shadow built from the fresh model_state_dict above is
        # the right starting point: it's the actual trained weights at the
        # resume epoch, and training continues the EMA from there instead
        # of inheriting the broken history.
        #
        # Trade-off: we lose the smoothed EMA history across the resume
        # boundary. But since the pre-fix history is garbage, "lose the
        # smoothing" is a feature — and after 1/(1-decay) ≈ 1000 steps
        # (well under one epoch for this config) the new shadow has
        # effectively the same smoothing characteristics it would have
        # had with a continuous fp32 run.
        if "ema_state_dict" in ckpt:
            print("  skipping checkpoint's ema_state_dict (pre-fix broken shadow)")
            print("  reseeding EMA shadow from loaded model weights in fp32")
        self._reseed_ema_shadow("post-resume")

        self.global_step = ckpt.get("global_step", 0)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_acc = ckpt.get("best_puzzle_acc", 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(self.global_step):
                self.scheduler.step()
        print(f"Resumed at epoch {self.start_epoch}, step {self.global_step}, best_acc {self.best_acc:.4f}")

    def _reseed_ema_shadow(self, reason: str) -> None:
        """Rebuild ``self.ema.shadow`` from the model's current parameters in fp32.

        Called from two places:
        1. After ``_load_init_weights`` in ``__init__`` — the shadow built
           at construction time tracks the pre-HF-load random init, not
           the actual starting point of training.
        2. From ``_load_checkpoint`` — pre-fix checkpoints contain a
           frozen bf16 shadow that must not be trusted.

        Cheap enough to do unconditionally (it's O(num_params) tensor
        clones, all on self.device) so no need to gate it on anything.
        """
        self.ema.shadow = {
            name: param.detach().clone().to(torch.float32)
            for name, param in self.model.named_parameters()
        }
        print(f"[EMA] shadow reseeded from model params ({reason}, fp32)")

    def _maybe_alert_on_regression(self, current_puzzle_acc: float, epoch_1based: int) -> None:
        """Fire wandb.alert when val_puzzle_acc drops below best by threshold.

        One-shot per run (further drops in the same run don't re-fire — the
        first signal is enough to triage). Skipped silently when wandb is off,
        when threshold is 0, or before the first new-best has been seen
        (self.best_acc==0 means we have no baseline to compare against yet).
        """
        threshold = self.tc.regression_alert_threshold
        if (
            threshold <= 0
            or self._regression_alert_fired
            or not self.use_wandb
            or self.best_acc <= 0
        ):
            return

        drop = self.best_acc - current_puzzle_acc
        if drop < threshold:
            return

        self._regression_alert_fired = True
        title = f"val_puzzle_acc regressed {drop * 100:.2f} pp"
        text = (
            f"Epoch {epoch_1based}: val_puzzle_acc={current_puzzle_acc:.4f} "
            f"(best={self.best_acc:.4f}, drop {drop:.4f}). "
            f"Threshold {threshold}. Consider stopping the run — see "
            f"analysis_run_dz3tkge9.md for the seed-4 precedent."
        )
        try:
            wandb.alert(title=title, text=text, level=wandb.AlertLevel.WARN)
            tqdm.write(f"[ALERT] {title} — {text}")
        except Exception as e:
            tqdm.write(f"[ALERT] wandb.alert() failed: {e}; printing instead — {title}: {text}")

    @weave_op()
    def _trace_eval_puzzle(
        self,
        *,
        epoch: int,
        index: int,
        puzzle: list[int],
        label: list[int],
        prediction: list[int],
        halt_step: int,
        cell_correct: int,
        cell_total: int,
        puzzle_correct: bool,
    ) -> dict:
        """Per-puzzle Weave trace for sampled eval puzzles.

        The decorator is what emits the trace — the function body just packages
        the data into a dict that shows up in the Weave UI under the parent
        evaluate() trace. Callers are responsible for sampling so we don't emit
        one trace per puzzle on a 6.6k-puzzle eval.
        """
        return {
            "epoch": epoch,
            "index": index,
            "puzzle": puzzle,
            "label": label,
            "prediction": prediction,
            "halt_step": halt_step,
            "cell_correct": cell_correct,
            "cell_total": cell_total,
            "cell_acc": cell_correct / max(1, cell_total),
            "puzzle_correct": puzzle_correct,
        }

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds >= 86400:
            return f"{seconds / 86400:.1f}d"
        if seconds >= 3600:
            return f"{seconds / 3600:.1f}h"
        return f"{seconds / 60:.0f}m"

    # Visual constants for the tqdm progress bars. Routed through
    # _TqdmNewlineFile, which means stderr isn't a tty from tqdm's POV —
    # without `ascii=_BAR_FILL` and `dynamic_ncols=False` on the constructor,
    # tqdm falls back to its plain digit-subdivision ASCII bar at width 10.
    _BAR_FILL = " ▏▎▍▌▋▊▉█"  # 8 sub-block subdivisions per cell

    # NOTE: `{bar}` (no width spec) lets tqdm compute the fill width as
    # `ncols - len(everything-else-in-the-format-string)`. Combined with
    # `dynamic_ncols=True` on the constructor, the bar re-stretches to fill
    # the current terminal width on each refresh. For that to work, tqdm
    # must be able to probe terminal size through the `file=` object — see
    # `_TqdmNewlineFile.fileno` / `isatty` below, which forward to the
    # underlying tty so tqdm's ioctl-based size probe succeeds.
    # Note the lack of whitespace between `{rate_fmt}` and `{postfix}` (and
    # between `{remaining}` and `{postfix}` on the epoch bar): tqdm's
    # `format_meter` unconditionally prepends `", "` to any non-empty
    # postfix string, so letting that auto-comma supply the separator keeps
    # the rendered line free of a dangling `  , loss=...` wart.
    BAR_FORMAT = (
        "  {desc}{n_fmt:>3}/{total_fmt:<3} ▕{bar}▏ "
        "{elapsed}<{remaining} · {rate_fmt}{postfix}"
    )
    EPOCH_BAR_FORMAT = (
        "{desc}{n_fmt:>4}/{total_fmt:<4} ▕{bar}▏ "
        "{elapsed}<{remaining}{postfix}"
    )

    @staticmethod
    def _fmt_train_postfix(loss: float, act: float, step: int) -> str:
        # Fixed-width columns so values stay vertically aligned across iters
        # in the append-only newline log — much easier to scan than tqdm's
        # default `, key=val` rendering, which jitters with value width.
        return f"loss={loss:6.3f}  act={act:4.1f}  step={step:>7d}"

    @staticmethod
    def _fmt_epoch_postfix(loss: float, best: float, step: int) -> str:
        return f"best={best:6.3f}  loss={loss:6.3f}  step={step:>7d}"

    @staticmethod
    def _fmt_eval_postfix(cell: float, puzzle: float) -> str:
        return f"cell={cell:6.3f}  puzzle={puzzle:6.3f}"

    @staticmethod
    def _term_width() -> int:
        # shutil.get_terminal_size order of resolution:
        #   1. `COLUMNS` env var (if set)
        #   2. os.get_terminal_size() on stdout — works in a real tty
        #   3. fallback tuple below (160 cols, 24 rows)
        # Resolved ONCE per tqdm construction so the bar keeps a stable
        # width for the whole epoch. We prefer this to `dynamic_ncols=True`
        # because tqdm's dynamic path re-ioctls on every refresh — which
        # fails (→ 80-col fallback) whenever stderr is teed / piped to a
        # log file, even though the attached terminal may be wider.
        return shutil.get_terminal_size((160, 24)).columns

    def train(self) -> None:
        self._init_log()
        self.carbon.start()
        t_start = time.time()

        epoch_times = []
        last_val = {}
        # Tracks the most recent epoch we actually completed. Equals
        # self.tc.epochs - 1 when the loop runs to completion. Updated
        # at the end of each iteration so a wall-clock break saves
        # latest.pt with the correct epoch number.
        last_completed_epoch = self.start_epoch - 1

        epoch_iter = tqdm(
            range(self.start_epoch, self.tc.epochs),
            desc="epochs ",
            initial=self.start_epoch,
            total=self.tc.epochs,
            leave=True,
            bar_format=self.EPOCH_BAR_FORMAT,
            ascii=self._BAR_FILL,
            ncols=self._term_width(),
            file=_TqdmNewlineFile(sys.stderr),
            mininterval=0,
        )
        for epoch in epoch_iter:
            # Novelty iso-time cap. No-op unless TRM_MAX_TRAIN_SECONDS is set
            # (regular fleet runs unaffected). Checked at top of epoch so the
            # in-flight epoch is never torn down mid-step.
            if wall_clock_expired():
                tqdm.write(
                    f"[wall-clock] budget exhausted before epoch {epoch + 1}/{self.tc.epochs} "
                    f"— halting; latest.pt will snapshot epoch {last_completed_epoch + 1}."
                )
                break

            # MUST be the first line: resets the lazy checkpoint payload per
            # epoch. If we skipped this and a later epoch's save_interval fired
            # without the log_interval branch having rebuilt the payload, we'd
            # silently write a stale state_dict from an earlier epoch.
            payload: dict | None = None
            slim: dict | None = None  # model+EMA only — for best.pt + milestones

            epoch_start = time.time()
            metrics = self._train_epoch(epoch)
            epoch_elapsed = time.time() - epoch_start
            epoch_times.append(epoch_elapsed)

            recent = epoch_times[-10:]
            avg_sec = sum(recent) / len(recent)
            eta_sec = (self.tc.epochs - (epoch + 1)) * avg_sec
            elapsed = time.time() - t_start

            # Pull side-channel values out before we iterate `metrics` for
            # wandb logging — these need histogram/scalar special handling.
            train_halt_tensor = metrics.pop("_halt_steps_tensor", None)
            n_samples_seen = metrics.pop("_n_samples_seen", 0)

            epoch_iter.set_postfix_str(
                self._fmt_epoch_postfix(
                    loss=metrics["lm_loss"],
                    best=self.best_acc,
                    step=self.global_step,
                )
            )

            if self.use_wandb:
                # NOTE: do NOT reuse the name `payload` here — that identifier
                # is already the lazy checkpoint dict (declared None at the top
                # of each epoch). Shadowing it would leak the wandb dict into
                # _save_checkpoint a few lines down.
                wandb_payload = {f"train/{k}": v for k, v in metrics.items()}

                # Tier 1 system/optimizer scalars
                wandb_payload["train/lr"] = self.optimizer.param_groups[0]["lr"]
                wandb_payload["train/epoch_time_sec"] = epoch_elapsed
                wandb_payload["train/samples_per_sec"] = (
                    n_samples_seen / epoch_elapsed if epoch_elapsed > 0 else 0.0
                )
                if torch.cuda.is_available():
                    wandb_payload["system/gpu_mem_gb"] = (
                        torch.cuda.max_memory_allocated() / 1e9
                    )
                    # Reset the peak so each epoch measures its own high-water
                    # mark, not the all-time max since process start.
                    torch.cuda.reset_peak_memory_stats()

                # frac_at_max_steps is a scalar and useful at log cadence —
                # the expensive-to-render histogram moves to eval cadence
                # below (fires once per eval_interval epochs, e.g. every 50).
                if train_halt_tensor is not None and train_halt_tensor.numel() > 0:
                    wandb_payload["train/frac_at_max_steps"] = float(
                        (train_halt_tensor >= self.config.model.halt_max_steps)
                        .float()
                        .mean()
                    )

                # Live carbon snapshot — one curve instead of one scalar at
                # the end. flush() is cheap and safe to call every log step.
                carbon_snapshot = self.carbon.flush()
                wandb_payload["carbon/emissions_kg"] = carbon_snapshot["emissions_kg"]
                wandb_payload["carbon/energy_kwh"] = carbon_snapshot["energy_kwh"]

                # Summary-section mirrors — see _define_wandb_metrics step 0.
                # Key-existence guards so a trainer refactor that renames a key
                # doesn't crash the run; the summary chart just flatlines until
                # fixed. frac_at_max_steps is mirrored inside the halt-tensor
                # branch below (only defined when train_halt_tensor is non-empty).
                if "accuracy" in metrics:
                    wandb_payload["summary/train_accuracy"] = metrics["accuracy"]
                if "exact_accuracy" in metrics:
                    wandb_payload["summary/train_exact_accuracy"] = metrics["exact_accuracy"]
                if "q_halt_accuracy" in metrics:
                    wandb_payload["summary/train_q_halt_accuracy"] = metrics["q_halt_accuracy"]
                if "lm_loss" in metrics:
                    wandb_payload["summary/train_loss"] = metrics["lm_loss"]
                if "q_halt_loss" in metrics:
                    wandb_payload["summary/train_q_halt_loss"] = metrics["q_halt_loss"]
                if "q_continue_loss" in metrics:
                    wandb_payload["summary/train_q_continue_loss"] = metrics["q_continue_loss"]
                if "train/frac_at_max_steps" in wandb_payload:
                    wandb_payload["summary/train_frac_at_max_steps"] = wandb_payload["train/frac_at_max_steps"]
                wandb_payload["summary/carbon_emissions_kg"] = carbon_snapshot["emissions_kg"]

                # Explicit epoch for the step_metric declared in
                # _define_wandb_metrics — this is the value the dashboard
                # charts plot against on the x-axis.
                wandb_payload["epoch"] = epoch + 1
                wandb.log(wandb_payload, step=epoch + 1)

            # Eval cadence is independent of log cadence when eval_interval > 0.
            # eval_interval == 0 preserves legacy behavior (eval fused to log).
            eff_eval_interval = self.tc.eval_interval or self.tc.log_interval
            new_best = False

            # Force one extra eval on the FIRST epoch processed by this
            # process — fires whenever the trainer starts, including after
            # a --resume. Without this, a resumed run from epoch 500 with
            # eval_interval=50 has zero val/* data in wandb until epoch 550
            # (~24 minutes at 28s/epoch), making the dashboard look empty
            # for the user. The cost is a single extra eval pass per
            # process invocation, which is one full validation set forward
            # pass — small relative to the per-epoch training cost.
            is_first_epoch_of_process = (epoch == self.start_epoch)

            if (epoch + 1) % eff_eval_interval == 0 or is_first_epoch_of_process:
                # Stash the 1-based epoch so per-puzzle Weave traces can tag
                # themselves without changing evaluate()'s signature (kept
                # untouched so diagnose_real_weights.py's 1:1 mirror still
                # matches).
                self._current_eval_epoch_label = epoch + 1
                last_val = self.evaluate()

                if self.use_wandb:
                    val_halt_tensor = last_val.pop("_halt_steps_tensor", None)
                    val_payload = {f"val/{k}": v for k, v in last_val.items()}

                    # Histograms (both train and val) are throttled to eval
                    # cadence — once per eval_interval epochs — so the wandb
                    # dashboard isn't flooded with a new histogram every 5
                    # epochs. The train tensor logged here is the *most recent
                    # epoch's* halt distribution (not a 50-epoch rolling mix),
                    # which matches how the scalar train/ metrics logged in
                    # the same call behave — a snapshot, not an accumulation.
                    if val_halt_tensor is not None and val_halt_tensor.numel() > 0:
                        val_payload["val/halt_steps_hist"] = wandb.Histogram(
                            val_halt_tensor.numpy()
                        )
                    if train_halt_tensor is not None and train_halt_tensor.numel() > 0:
                        val_payload["train/halt_steps_hist"] = wandb.Histogram(
                            train_halt_tensor.numpy()
                        )

                    # Summary-section mirrors — see _define_wandb_metrics step 0.
                    # evaluate() doesn't compute a loss (no loss_head call in the
                    # eval loop), so there's no val_loss to mirror here.
                    if "accuracy" in last_val:
                        val_payload["summary/val_accuracy"] = last_val["accuracy"]
                    if "exact_accuracy" in last_val:
                        val_payload["summary/val_exact_accuracy"] = last_val["exact_accuracy"]
                    if "q_halt_accuracy" in last_val:
                        val_payload["summary/val_q_halt_accuracy"] = last_val["q_halt_accuracy"]

                    # Same step_metric contract as the train log above.
                    val_payload["epoch"] = epoch + 1
                    wandb.log(val_payload, step=epoch + 1)
                else:
                    # Keep last_val clean for downstream best-tracking code
                    # regardless of whether wandb is enabled.
                    last_val.pop("_halt_steps_tensor", None)

                # Regression alert (one-shot per run). The condition must run
                # BEFORE self.best_acc is potentially updated by the new-best
                # branch below, otherwise the threshold drifts upward in the
                # same step that just set a new best — the alert would never
                # see the *previous* best.
                self._maybe_alert_on_regression(last_val["puzzle_acc"], epoch + 1)

                if last_val["puzzle_acc"] > self.best_acc:
                    self.best_acc = last_val["puzzle_acc"]
                    payload = payload or self._checkpoint_payload(epoch)
                    slim = slim or {k: v for k, v in payload.items() if k != "optimizer_state_dict"}
                    self._save_checkpoint("best.pt", slim)
                    # Live-sync best.pt to the wandb Files tab. First new-best
                    # event registers the watcher; every subsequent best.pt
                    # overwrite is picked up automatically. Must come AFTER
                    # _save_checkpoint so the file exists before wandb.save
                    # stats it. Separate from the wandb_best_artifact path
                    # below (which creates versioned Artifacts, a different
                    # wandb feature for a different use case).
                    if self.use_wandb and not self._best_wandb_registered:
                        best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
                        self._wandb_save_live(best_path)
                        self._best_wandb_registered = True
                    if self.use_wandb and self.tc.wandb_best_artifact:
                        self._log_best_to_wandb(epoch)
                    new_best = True

            if (epoch + 1) % self.tc.log_interval == 0:
                val_cell = last_val.get("cell_acc") if last_val else None
                val_puzzle = last_val.get("puzzle_acc") if last_val else None
                new_best_suffix = " NEW BEST!" if new_best else ""

                if val_cell is None:
                    # log_interval fired before the first eval_interval — no val yet
                    tqdm.write(
                        f"Epoch {epoch + 1}/{self.tc.epochs} - "
                        f"lm_loss: {metrics['lm_loss']:.4f}  "
                        f"(no val yet)  "
                        f"elapsed: {self._fmt_time(elapsed)}  "
                        f"ETA: {self._fmt_time(eta_sec)}"
                    )
                else:
                    tqdm.write(
                        f"Epoch {epoch + 1}/{self.tc.epochs} - "
                        f"lm_loss: {metrics['lm_loss']:.4f}  "
                        f"cell_acc: {val_cell:.4f}  "
                        f"puzzle_acc: {val_puzzle:.4f}  "
                        f"best: {self.best_acc:.4f}  "
                        f"elapsed: {self._fmt_time(elapsed)}  "
                        f"ETA: {self._fmt_time(eta_sec)}"
                        f"{new_best_suffix}"
                    )

                self._append_log([
                    epoch + 1,
                    f"{metrics['lm_loss']:.4f}",
                    f"{metrics['q_halt_loss']:.4f}",
                    f"{metrics.get('q_continue_loss', 0):.4f}",
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['exact_accuracy']:.4f}",
                    f"{metrics['q_halt_accuracy']:.4f}",
                    f"{metrics['avg_steps']:.1f}",
                    f"{val_cell:.4f}" if val_cell is not None else "",
                    f"{val_puzzle:.4f}" if val_puzzle is not None else "",
                    f"{self.best_acc:.4f}",
                    f"{elapsed / 60:.1f}",
                ])

            if (epoch + 1) % self.tc.save_interval == 0:
                payload = payload or self._checkpoint_payload(epoch)
                self._save_checkpoint(f"epoch_{epoch + 1}.pt", payload)

            # Rolling crash-recovery backup (independent cadence)
            if (
                self.tc.rolling_checkpoint_dir
                and (epoch + 1) % self.tc.rolling_checkpoint_interval == 0
            ):
                payload = payload or self._checkpoint_payload(epoch)
                self._save_rolling_checkpoint(epoch, payload)

            # Milestone snapshots at fixed fractions (resume-safe).
            # Single .get() replaces the old two-step (in + .get()) pattern.
            pct = self.milestone_epochs.get(epoch + 1) if self.milestone_epochs else None
            if pct is not None:
                payload = payload or self._checkpoint_payload(epoch)
                slim = slim or {k: v for k, v in payload.items() if k != "optimizer_state_dict"}
                self._save_milestone_checkpoint(pct, slim)

            last_completed_epoch = epoch

        # Pin the final save to whichever epoch we actually finished. For a
        # full run this equals self.tc.epochs - 1 (unchanged behavior). For
        # a wall-clock-halted run, using last_completed_epoch instead keeps
        # latest.pt's stored epoch number honest.
        final_payload = self._checkpoint_payload(last_completed_epoch)
        self._save_checkpoint("latest.pt", final_payload)
        emissions = self.carbon.stop()

        if self.use_wandb:
            wandb.finish()

        results_path = os.path.join(self.config.experiment_dir, "training_results.json")
        with open(results_path, "w") as f:
            json.dump({"best_puzzle_acc": self.best_acc, "emissions": emissions}, f, indent=2)

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.loss_head.train()

        totals = {}
        n_batches = 0
        grad_norm_sum = 0.0
        grad_norm_count = 0
        n_samples_seen = 0

        # Per-epoch buffer of halt-step counts for every sample that halted this
        # epoch. Built from carry.steps[carry.halted] after each ACT inner step,
        # so every halt event (including re-halts after carry reset) contributes
        # — same population that avg_steps averages over, making the two metrics
        # directly comparable on the wandb dashboard.
        halt_steps_chunks: list[torch.Tensor] = []

        # Throttle train bar to one log line per 50 iters. `mininterval=0` +
        # explicit `miniters=50` makes iteration count the sole gate. Must be
        # paired with `set_postfix(refresh=False)` inside the loop; otherwise
        # set_postfix forces a refresh every iter and defeats the throttle.
        #
        # `maxinterval=float('inf')` disables tqdm's background TMonitor
        # thread, which otherwise force-sets `miniters = 1` whenever a bar
        # hasn't refreshed within `maxinterval` (default 10s). At maze's
        # ~3s/step, 50 iters ≈ 150s between refreshes — the 10s monitor
        # trips after ~4 iters, drops miniters to 1, and prints every iter
        # for the rest of the epoch.
        pbar = tqdm(
            self.train_loader,
            desc=f"train epoch {epoch + 1} ",
            bar_format=self.BAR_FORMAT,
            ascii=self._BAR_FILL,
            ncols=self._term_width(),
            leave=False,
            file=_TqdmNewlineFile(sys.stderr),
            miniters=50,
            mininterval=0,
            maxinterval=float("inf"),
        )
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            n_samples_seen += batch["inputs"].shape[0]

            carry = self.loss_head.initial_carry(batch)

            batch_metrics = {}
            steps_this_batch = 0

            for _act_step in range(self.config.model.halt_max_steps):
                carry, loss, metrics, _outputs, all_halted = self.loss_head(
                    return_keys=(), carry=carry, batch=batch,
                )

                loss.backward()
                # Capture pre-clip L2 norm — clip_grad_norm_ returns it as a
                # scalar tensor. Logging the pre-clip value is the right signal
                # for stability diagnostics: "was the gradient actually large?"
                # not "was it clipped?".
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tc.max_grad_norm
                )
                grad_norm_sum += float(grad_norm)
                grad_norm_count += 1

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.ema.update()
                self.scheduler.step()
                self.global_step += 1
                steps_this_batch += 1

                # Record halt events for the histogram (detached, moved to CPU
                # so we don't pin grow-only GPU memory across the epoch).
                if carry.halted.any():
                    halt_steps_chunks.append(
                        carry.steps[carry.halted].detach().to(torch.int32).cpu()
                    )

                # Accumulate metrics from halted samples
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    batch_metrics[k] = batch_metrics.get(k, 0) + v

                if all_halted:
                    break

            # Normalize by count
            count = max(1, batch_metrics.get("count", 1))
            normalized = {
                "lm_loss": batch_metrics.get("lm_loss", 0) / count,
                "q_halt_loss": batch_metrics.get("q_halt_loss", 0) / count,
                "q_continue_loss": batch_metrics.get("q_continue_loss", 0) / count,
                "accuracy": batch_metrics.get("accuracy", 0) / count,
                "exact_accuracy": batch_metrics.get("exact_accuracy", 0) / count,
                "q_halt_accuracy": batch_metrics.get("q_halt_accuracy", 0) / count,
                "avg_steps": batch_metrics.get("steps", 0) / count,
            }

            for k, v in normalized.items():
                totals[k] = totals.get(k, 0) + v
            n_batches += 1

            # refresh=False: update postfix state only. The next iteration's
            # auto refresh (gated by mininterval=1.0) will pick up the new
            # values in a single write — without this, set_postfix forces
            # an extra refresh per iter, doubling every line in the newline
            # log (one from update(1), one from set_postfix).
            pbar.set_postfix_str(
                self._fmt_train_postfix(
                    loss=normalized["lm_loss"],
                    act=normalized["avg_steps"],
                    step=self.global_step,
                ),
                refresh=False,
            )

        epoch_metrics: dict = {k: v / max(1, n_batches) for k, v in totals.items()}
        epoch_metrics["grad_norm"] = grad_norm_sum / max(1, grad_norm_count)
        # Non-scalar side-channels returned alongside the averaged metrics.
        # Prefixed with "_" so the `wandb.log({f"train/{k}": v ...})` loop in
        # train() can filter them out and handle histograms/samples explicitly.
        if halt_steps_chunks:
            epoch_metrics["_halt_steps_tensor"] = torch.cat(halt_steps_chunks)
        else:
            epoch_metrics["_halt_steps_tensor"] = torch.empty(0, dtype=torch.int32)
        epoch_metrics["_n_samples_seen"] = n_samples_seen
        return epoch_metrics

    @weave_op()
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.ema.apply_shadow()
        self.model.eval()
        self.loss_head.eval()

        total_cell_correct = 0
        total_cells = 0
        total_puzzle_correct = 0
        total_puzzles = 0
        total_steps = 0
        total_q_halt_correct = 0
        n_samples = 0

        # Per-puzzle Weave trace plumbing. Sample uniformly across the dataset
        # so the traces span easy + hard puzzles. Stride is computed once at
        # eval-start; budget is decremented per emitted trace. When sample
        # size is 0 the inner block becomes a no-op (no @weave_op calls fire).
        trace_target = self.tc.eval_trace_sample_size
        try:
            dataset_len = len(self.val_loader.dataset)  # type: ignore[arg-type]
        except (AttributeError, TypeError):
            dataset_len = 0
        if trace_target > 0 and dataset_len > 0:
            trace_stride = max(1, dataset_len // trace_target)
        else:
            trace_stride = 0
        trace_budget = trace_target
        # Set by train() right before each evaluate() call so traces are
        # tagged with the right epoch. Falls back to start_epoch+1 for the
        # rare standalone evaluate() (e.g. resume's first-epoch warm-up eval).
        trace_epoch = getattr(self, "_current_eval_epoch_label", self.start_epoch + 1)

        # Per-sample "first step at which the Q-head would halt in deployment".
        # The model itself disables early halting outside self.training (see
        # trm_official.py:332), so carry.halted is useless in eval — we have
        # to reconstruct the would-halt event manually from q_halt vs
        # q_continue logits at each forward.
        max_steps = self.config.model.halt_max_steps
        halt_steps_chunks: list[torch.Tensor] = []
        no_act_continue = self.config.model.no_ACT_continue

        # Throttle eval bar to one log line per 50 iters — see the matching
        # train-bar block for why `maxinterval=float('inf')` is required.
        pbar = tqdm(
            self.val_loader,
            desc="eval ",
            bar_format=self.BAR_FORMAT,
            ascii=self._BAR_FILL,
            ncols=self._term_width(),
            leave=False,
            file=_TqdmNewlineFile(sys.stderr),
            miniters=50,
            mininterval=0,
            maxinterval=float("inf"),
        )
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            B = batch["inputs"].shape[0]

            carry = self.loss_head.initial_carry(batch)

            # Samples that never cross the halt threshold stay pinned at
            # max_steps — that matches how they'd actually be deployed.
            first_halt_step = torch.full(
                (B,), max_steps, dtype=torch.int32, device=self.device
            )
            ever_halted = torch.zeros(B, dtype=torch.bool, device=self.device)

            # Run for full halt_max_steps (no early stopping during eval)
            for step_idx in range(max_steps):
                carry, _outputs = self.model(carry=carry, batch=batch)

                q_halt = _outputs["q_halt_logits"]
                q_cont = _outputs["q_continue_logits"]
                would_halt = (q_halt > 0) if no_act_continue else (q_halt > q_cont)
                newly = would_halt & ~ever_halted
                if newly.any():
                    first_halt_step = torch.where(
                        newly,
                        torch.full_like(first_halt_step, step_idx + 1),
                        first_halt_step,
                    )
                    ever_halted = ever_halted | would_halt

            halt_steps_chunks.append(first_halt_step.detach().cpu())

            # Get final predictions from last forward
            logits = _outputs["logits"]
            preds = logits.argmax(-1)
            labels = carry.current_data["labels"]
            mask = labels != -100

            total_cell_correct += ((preds == labels) & mask).sum().item()
            total_cells += mask.sum().item()

            puzzle_correct = ((preds == labels) | ~mask).all(dim=-1)
            total_puzzle_correct += puzzle_correct.sum().item()
            total_puzzles += B

            total_steps += first_halt_step.sum().item()
            n_samples += B

            # Q-halt accuracy
            q_halt_correct = (_outputs["q_halt_logits"] >= 0) == puzzle_correct
            total_q_halt_correct += q_halt_correct.sum().item()

            # Sampled per-puzzle Weave traces. Stride==0 short-circuits the
            # whole block when traces are disabled. Conversion to Python lists
            # is intentional — Weave needs JSON-serialisable values, not torch
            # tensors that hold device handles.
            if trace_stride > 0 and trace_budget > 0:
                global_start = n_samples - B  # global index of batch[0]
                for i in range(B):
                    if trace_budget <= 0:
                        break
                    global_idx = global_start + i
                    if global_idx % trace_stride != 0:
                        continue
                    cell_correct_i = int(((preds[i] == labels[i]) & mask[i]).sum().item())
                    cell_total_i = int(mask[i].sum().item())
                    self._trace_eval_puzzle(
                        epoch=trace_epoch,
                        index=global_idx,
                        puzzle=batch["inputs"][i].cpu().tolist(),
                        label=labels[i].cpu().tolist(),
                        prediction=preds[i].cpu().tolist(),
                        halt_step=int(first_halt_step[i].item()),
                        cell_correct=cell_correct_i,
                        cell_total=cell_total_i,
                        puzzle_correct=bool(puzzle_correct[i].item()),
                    )
                    trace_budget -= 1

            # refresh=False: update postfix state only. tqdm's 50-iter auto
            # refresh (configured above) will pick up the latest values.
            pbar.set_postfix_str(
                self._fmt_eval_postfix(
                    cell=total_cell_correct / max(1, total_cells),
                    puzzle=total_puzzle_correct / max(1, total_puzzles),
                ),
                refresh=False,
            )

        self.ema.restore()

        halt_tensor = (
            torch.cat(halt_steps_chunks)
            if halt_steps_chunks
            else torch.empty(0, dtype=torch.int32)
        )
        frac_at_max = (
            (halt_tensor >= max_steps).float().mean().item()
            if halt_tensor.numel() > 0
            else 0.0
        )

        # Naming aliases — the train side uses `accuracy`, `exact_accuracy`,
        # `q_halt_accuracy`, `avg_steps`, and the val side historically used
        # the shorter `cell_acc`, `puzzle_acc`, `q_halt_acc`, `avg_act_steps`.
        # The asymmetric naming made val panels look "missing" relative to
        # their train counterparts in the wandb workspace (no val/accuracy
        # to pair with train/accuracy). We now emit BOTH names so:
        #   • old dashboards / CSV consumers reading val/cell_acc still work
        #   • new lookups for val/accuracy (the symmetric name) succeed
        # Pure aliasing — both names carry the same value, no extra compute.
        cell_accuracy = total_cell_correct / max(1, total_cells)
        puzzle_accuracy = total_puzzle_correct / max(1, total_puzzles)
        q_halt_accuracy = total_q_halt_correct / max(1, n_samples)
        avg_steps = total_steps / max(1, n_samples)

        return {
            # Symmetric-with-train names (preferred for new code):
            "accuracy":         cell_accuracy,    # ↔ train/accuracy
            "exact_accuracy":   puzzle_accuracy,  # ↔ train/exact_accuracy
            "q_halt_accuracy":  q_halt_accuracy,  # ↔ train/q_halt_accuracy
            "avg_steps":        avg_steps,        # ↔ train/avg_steps
            # Legacy short names (kept for backward compat with CSV / old
            # wandb workspaces that already reference these keys):
            "cell_acc":         cell_accuracy,
            "puzzle_acc":       puzzle_accuracy,
            "q_halt_acc":       q_halt_accuracy,
            "avg_act_steps":    avg_steps,
            "frac_at_max_steps": frac_at_max,
            "_halt_steps_tensor": halt_tensor,
        }

    def _safe_torch_save(self, payload: dict, path: str, tag: str) -> bool:
        """Wrap torch.save with uniform error logging. Returns True on success.

        Broad Exception catch (not OSError) — torch.save can raise pickle errors,
        __reduce__ failures, and other non-filesystem errors we still want to
        log-and-continue on rather than crash a long training run.
        """
        try:
            torch.save(payload, path)
            return True
        except Exception as e:
            tqdm.write(f"[{tag}] Save failed: {e}")
            return False

    def _checkpoint_payload(self, epoch: int) -> dict:
        """Shared dict builder so all save paths serialize the same fields."""
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "config": self.config.model_dump(),
            "seed": self.config.seed,
            "global_step": self.global_step,
            "best_puzzle_acc": self.best_acc,
        }

    def _save_checkpoint(self, filename: str, payload: dict) -> None:
        path = os.path.join(self.config.checkpoint_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not self._safe_torch_save(payload, path, "checkpoint"):
            return
        if self.hf_api:
            try:
                self.hf_api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=f"{self.config.checkpoint_dir}/{filename}",
                    repo_id=self.hf_repo_id,
                    commit_message=f"checkpoint epoch {payload['epoch'] + 1} (acc={payload['best_puzzle_acc']:.4f})",
                )
            except Exception as e:
                tqdm.write(f"[HF Hub] Upload failed: {e}")

    def _save_rolling_checkpoint(self, epoch: int, payload: dict) -> None:
        """Save to rolling-window dir (e.g. D: drive), prune oldest beyond max."""
        rolling_dir = self.tc.rolling_checkpoint_dir
        if not rolling_dir:
            return
        try:
            os.makedirs(rolling_dir, exist_ok=True)
        except OSError as e:
            tqdm.write(f"[rolling] Cannot create {rolling_dir}: {e}")
            return

        filename = f"epoch_{epoch + 1:06d}.pt"
        path = os.path.join(rolling_dir, filename)
        if not self._safe_torch_save(payload, path, "rolling"):
            return

        # Prune oldest: zero-padded names sort chronologically
        existing = sorted(
            f for f in os.listdir(rolling_dir)
            if f.startswith("epoch_") and f.endswith(".pt")
        )
        excess = len(existing) - self.tc.rolling_checkpoint_max
        if excess > 0:
            for old in existing[:excess]:
                try:
                    os.remove(os.path.join(rolling_dir, old))
                except OSError as e:
                    tqdm.write(f"[rolling] Delete failed for {old}: {e}")

    def _save_milestone_checkpoint(self, pct: int, payload_slim: dict) -> None:
        """Save a fixed-fraction thesis milestone with HF Hub + wandb mirror.

        payload_slim omits optimizer state (decision 6D) — milestones are for
        analysis/inference, not resume. Resume from epoch_N.pt or latest.pt.

        Idempotent: if the local file already exists (resume past milestone
        epoch, or a re-run that reached the same epoch), this is a no-op —
        no duplicate HF commits or wandb artifact versions.
        """
        dataset = self.config.data.dataset
        epoch_num = payload_slim["epoch"] + 1
        filename = f"snapshots_for_thesis/{dataset}_milestone_{pct:02d}pct_epoch{epoch_num}.pt"
        path = os.path.join(self.config.checkpoint_dir, filename)

        if os.path.isfile(path):
            return

        # Local save + HF Hub upload (shared helper; creates subdir via _save_checkpoint)
        self._save_checkpoint(filename, payload_slim)
        tqdm.write(f"  [milestone] saved {filename}")

        # wandb Artifact mirror — decision 3D
        if self.use_wandb and os.path.isfile(path):
            try:
                artifact = wandb.Artifact(
                    name=f"{dataset}-milestone-{pct:02d}pct",
                    type="model",
                    metadata={
                        "puzzle_acc": float(self.best_acc),
                        "epoch": epoch_num,
                        "global_step": self.global_step,
                        "pct": pct,
                    },
                )
                artifact.add_file(path)
                wandb.log_artifact(artifact, aliases=[f"milestone_{pct:02d}pct"])
            except Exception as e:
                tqdm.write(f"[wandb] Milestone artifact upload failed: {e}")

    def _log_best_to_wandb(self, epoch: int) -> None:
        """Upload best.pt as a versioned wandb Artifact (non-blocking background sync)."""
        best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
        if not os.path.isfile(best_path):
            return
        try:
            artifact = wandb.Artifact(
                name=f"{self.config.model.model_type.value}-best",
                type="model",
                metadata={
                    "puzzle_acc": float(self.best_acc),
                    "epoch": epoch + 1,
                    "global_step": self.global_step,
                },
            )
            artifact.add_file(best_path)
            wandb.log_artifact(artifact, aliases=["best"])
        except Exception as e:
            tqdm.write(f"[wandb] Artifact upload failed: {e}")
