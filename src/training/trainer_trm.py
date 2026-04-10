import csv
import json
import os
import time
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.layers import StableMaxCrossEntropy
from src.models.recursion import deep_supervision_step
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


class TRMTrainer:
    """Trainer for TRM models with deep supervision, ACT, EMA, and CodeCarbon."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ExperimentConfig,
        resume_checkpoint: str = "",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tc = config.training
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.tc.lr,
            betas=self.tc.betas,
            weight_decay=self.tc.weight_decay,
        )

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

        self.loss_fn = StableMaxCrossEntropy(ignore_index=0)
        self.ema = EMA(model, decay=self.tc.ema_decay)

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.carbon = CarbonTracker(
            f"{config.model.model_type.value}_train",
            output_dir=config.experiment_dir,
        )

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.experiment_dir, exist_ok=True)

        # HuggingFace Hub checkpoint sync
        self.hf_repo_id = config.training.hf_repo_id if hasattr(config.training, "hf_repo_id") else ""
        if self.hf_repo_id and HF_AVAILABLE:
            self.hf_api = HfApi()
            self.hf_api.create_repo(self.hf_repo_id, exist_ok=True, private=True)
            print(f"[HF Hub] Syncing checkpoints to {self.hf_repo_id}")
        else:
            self.hf_api = None

        # Resume from checkpoint if provided
        if resume_checkpoint and os.path.isfile(resume_checkpoint):
            self._load_checkpoint(resume_checkpoint)

        if self.tc.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.tc.wandb_project,
                config=config.model_dump(),
            )

        # CSV log for remote progress tracking
        self.log_path = os.path.join(config.experiment_dir, f"{config.model.model_type.value}_train_log.csv")

    def _init_log(self) -> None:
        mode = "a" if self.start_epoch > 0 and os.path.exists(self.log_path) else "w"
        if mode == "w":
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "ce_loss", "q_mean", "steps_taken", "val_cell_acc", "val_puzzle_acc", "best_puzzle_acc", "elapsed_min"])

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
        # Advance scheduler to match global_step
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

        best_acc = self.best_acc
        epoch_times = []
        last_val = {}

        for epoch in range(self.start_epoch, self.tc.epochs):
            epoch_start = time.time()
            metrics = self._train_epoch(epoch)
            epoch_times.append(time.time() - epoch_start)

            # Rolling ETA from last 10 epochs
            recent = epoch_times[-10:]
            avg_sec = sum(recent) / len(recent)
            eta_sec = (self.tc.epochs - (epoch + 1)) * avg_sec
            elapsed = time.time() - t_start

            if (epoch + 1) % self.tc.log_interval == 0:
                last_val = self.evaluate()

                if self.tc.use_wandb and WANDB_AVAILABLE:
                    wandb.log({**metrics, **{f"val_{k}": v for k, v in last_val.items()}})

                new_best = ""
                if last_val["puzzle_acc"] > best_acc:
                    best_acc = last_val["puzzle_acc"]
                    self._save_checkpoint(epoch, "best.pt", best_acc)
                    new_best = " NEW BEST!"

                print(
                    f"Epoch {epoch + 1}/{self.tc.epochs} - "
                    f"ce_loss: {metrics['ce_loss']:.4f}  "
                    f"cell_acc: {last_val['cell_acc']:.4f}  "
                    f"puzzle_acc: {last_val['puzzle_acc']:.4f}  "
                    f"best: {best_acc:.4f}  "
                    f"elapsed: {self._fmt_time(elapsed)}  "
                    f"ETA: {self._fmt_time(eta_sec)}"
                    f"{new_best}"
                )

                self._append_log([
                    epoch + 1, f"{metrics['ce_loss']:.4f}", f"{metrics['q_mean']:.3f}",
                    f"{metrics['steps_taken']:.1f}", f"{last_val['cell_acc']:.4f}",
                    f"{last_val['puzzle_acc']:.4f}", f"{best_acc:.4f}", f"{elapsed / 60:.1f}",
                ])

            if (epoch + 1) % self.tc.save_interval == 0:
                self._save_checkpoint(epoch, f"epoch_{epoch + 1}.pt", best_acc)

        self._save_checkpoint(self.tc.epochs - 1, "latest.pt", best_acc)
        emissions = self.carbon.stop()

        results_path = os.path.join(self.config.experiment_dir, "training_results.json")
        with open(results_path, "w") as f:
            json.dump({"best_puzzle_acc": best_acc, "emissions": emissions}, f, indent=2)

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        epoch_metrics = {
            "ce_loss": 0.0, "q_loss": 0.0, "q_mean": 0.0,
            "steps_taken": 0.0, "puzzle_acc": 0.0,
        }
        n_batches = 0

        for inputs, labels in tqdm(self.train_loader, bar_format=self.BAR_FORMAT):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            metrics = deep_supervision_step(
                model=self.model,
                inputs=inputs,
                labels=labels,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                ema=self.ema,
                n=self.tc.n_latent,
                T=self.tc.T_deep,
                N_sup=self.tc.N_sup,
                act_threshold=self.tc.act_threshold,
                max_grad_norm=self.tc.max_grad_norm,
                scaler=self.scaler if self.use_amp else None,
            )

            self.scheduler.step()
            self.global_step += 1

            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
            n_batches += 1

        return {k: v / max(1, n_batches) for k, v in epoch_metrics.items()}

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.ema.apply_shadow()
        self.model.eval()

        total_cell_correct = 0
        total_cells = 0
        total_puzzle_correct = 0
        total_puzzles = 0

        for inputs, labels in self.val_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Run inference: deep recursion with ACT
            B = inputs.shape[0]
            x = self.model.embedding(inputs)
            y = self.model.y_init.expand(B, -1, -1).clone()
            z = self.model.z_init.expand(B, -1, -1).clone()

            from src.models.recursion import deep_recursion

            for _ in range(self.tc.N_sup):
                (y, z), logits, q, _q_logits = deep_recursion(
                    self.model.block, self.model.output_head, self.model.q_head,
                    x, y, z, n=self.tc.n_latent, T=self.tc.T_deep,
                )
                if q.mean().item() > self.tc.act_threshold:
                    break

            preds = logits.argmax(-1)
            mask = labels != 0

            total_cell_correct += ((preds == labels) & mask).sum().item()
            total_cells += mask.sum().item()

            puzzle_correct = ((preds == labels) | ~mask).all(dim=-1)
            total_puzzle_correct += puzzle_correct.sum().item()
            total_puzzles += B

        self.ema.restore()

        return {
            "cell_acc": total_cell_correct / max(1, total_cells),
            "puzzle_acc": total_puzzle_correct / max(1, total_puzzles),
        }

    def _save_checkpoint(self, epoch: int, filename: str, best_acc: float = 0.0) -> None:
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "ema_state_dict": self.ema.state_dict(),
                "config": self.config.model_dump(),
                "seed": self.config.seed,
                "global_step": self.global_step,
                "best_puzzle_acc": best_acc,
            },
            path,
        )
        if self.hf_api:
            try:
                self.hf_api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=f"{self.config.checkpoint_dir}/{filename}",
                    repo_id=self.hf_repo_id,
                    commit_message=f"checkpoint epoch {epoch + 1} (acc={best_acc:.4f})",
                )
            except Exception as e:
                tqdm.write(f"[HF Hub] Upload failed: {e}")
