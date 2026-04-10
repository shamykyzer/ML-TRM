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
import time
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.losses_official import ACTLossHead
from src.training.carbon_tracker import CarbonTracker
from src.training.ema import EMA
from src.utils.config import ExperimentConfig

try:
    import wandb
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
        try:
            from adam_atan2 import AdamAtan2
            optimizer = AdamAtan2(param_groups, betas=tc.betas)
            print("[Optimizer] Using AdamAtan2")
            return optimizer
        except ImportError:
            print("[Optimizer] adam-atan2 not installed, falling back to AdamW")

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

        # Resume
        if resume_checkpoint and os.path.isfile(resume_checkpoint):
            self._load_checkpoint(resume_checkpoint)

        # W&B
        self.use_wandb = self.tc.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                entity=self.tc.wandb_entity or None,
                project=self.tc.wandb_project,
                config=config.model_dump(),
            )

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

    def _append_log(self, row: list) -> None:
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _load_checkpoint(self, path: str) -> None:
        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_acc = ckpt.get("best_puzzle_acc", 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(self.global_step):
                self.scheduler.step()
        print(f"Resumed at epoch {self.start_epoch}, step {self.global_step}, best_acc {self.best_acc:.4f}")

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds >= 86400:
            return f"{seconds / 86400:.1f}d"
        if seconds >= 3600:
            return f"{seconds / 3600:.1f}h"
        return f"{seconds / 60:.0f}m"

    BAR_FORMAT = "  {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}, {rate_fmt}]"

    def train(self) -> None:
        self._init_log()
        self.carbon.start()
        t_start = time.time()

        epoch_times = []
        last_val = {}

        for epoch in range(self.start_epoch, self.tc.epochs):
            # MUST be the first line: resets the lazy checkpoint payload per
            # epoch. If we skipped this and a later epoch's save_interval fired
            # without the log_interval branch having rebuilt the payload, we'd
            # silently write a stale state_dict from an earlier epoch.
            payload: dict | None = None

            epoch_start = time.time()
            metrics = self._train_epoch(epoch)
            epoch_times.append(time.time() - epoch_start)

            recent = epoch_times[-10:]
            avg_sec = sum(recent) / len(recent)
            eta_sec = (self.tc.epochs - (epoch + 1)) * avg_sec
            elapsed = time.time() - t_start

            if self.use_wandb:
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=epoch + 1)

            if (epoch + 1) % self.tc.log_interval == 0:
                last_val = self.evaluate()

                if self.use_wandb:
                    wandb.log({f"val/{k}": v for k, v in last_val.items()}, step=epoch + 1)

                new_best = ""
                if last_val["puzzle_acc"] > self.best_acc:
                    self.best_acc = last_val["puzzle_acc"]
                    payload = payload or self._checkpoint_payload(epoch)
                    self._save_checkpoint("best.pt", payload)
                    if self.use_wandb and self.tc.wandb_best_artifact:
                        self._log_best_to_wandb(epoch)
                    new_best = " NEW BEST!"

                print(
                    f"Epoch {epoch + 1}/{self.tc.epochs} - "
                    f"lm_loss: {metrics['lm_loss']:.4f}  "
                    f"cell_acc: {last_val['cell_acc']:.4f}  "
                    f"puzzle_acc: {last_val['puzzle_acc']:.4f}  "
                    f"best: {self.best_acc:.4f}  "
                    f"elapsed: {self._fmt_time(elapsed)}  "
                    f"ETA: {self._fmt_time(eta_sec)}"
                    f"{new_best}"
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
                    f"{last_val['cell_acc']:.4f}",
                    f"{last_val['puzzle_acc']:.4f}",
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
                self._save_milestone_checkpoint(pct, payload)

        final_payload = self._checkpoint_payload(self.tc.epochs - 1)
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

        for batch in tqdm(self.train_loader, bar_format=self.BAR_FORMAT):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            carry = self.loss_head.initial_carry(batch)

            batch_metrics = {}
            steps_this_batch = 0

            for _act_step in range(self.config.model.halt_max_steps):
                carry, loss, metrics, _outputs, all_halted = self.loss_head(
                    return_keys=(), carry=carry, batch=batch,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.ema.update()
                self.scheduler.step()
                self.global_step += 1
                steps_this_batch += 1

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

        return {k: v / max(1, n_batches) for k, v in totals.items()}

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

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            B = batch["inputs"].shape[0]

            carry = self.loss_head.initial_carry(batch)

            # Run for full halt_max_steps (no early stopping during eval)
            for _step in range(self.config.model.halt_max_steps):
                carry, _outputs = self.model(carry=carry, batch=batch)

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

            total_steps += carry.steps.sum().item()
            n_samples += B

            # Q-halt accuracy
            q_halt_correct = (_outputs["q_halt_logits"] >= 0) == puzzle_correct
            total_q_halt_correct += q_halt_correct.sum().item()

        self.ema.restore()

        return {
            "cell_acc": total_cell_correct / max(1, total_cells),
            "puzzle_acc": total_puzzle_correct / max(1, total_puzzles),
            "avg_act_steps": total_steps / max(1, n_samples),
            "q_halt_acc": total_q_halt_correct / max(1, n_samples),
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

    def _save_milestone_checkpoint(self, pct: int, payload: dict) -> None:
        """Save a fixed-fraction training milestone (idempotent, never rotated)."""
        filename = f"milestone_{pct:02d}pct.pt"
        path = os.path.join(self.config.checkpoint_dir, filename)
        if os.path.isfile(path):
            return  # already saved on a prior run — resume-safe
        if self._safe_torch_save(payload, path, "milestone"):
            tqdm.write(f"  [milestone] saved {filename} at epoch {payload['epoch'] + 1}")

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
