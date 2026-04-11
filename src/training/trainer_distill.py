import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.carbon_tracker import CarbonTracker
from src.training.wandb_utils import define_common_metrics, init_wandb, weave_op
from src.utils.config import ExperimentConfig

try:
    import wandb  # needed for wandb.log / wandb.finish when use_wandb=True
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DistillationLoss(nn.Module):
    """Knowledge distillation loss: soft KL-div + hard CE."""

    def __init__(self, alpha: float = 0.7, temperature: float = 4.0, ignore_index: int = 0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        T = self.temperature

        # Hard label loss
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=self.ignore_index,
        )

        # Soft label loss (KL divergence with temperature scaling)
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        kl_loss = F.kl_div(
            soft_student.view(-1, soft_student.size(-1)),
            soft_teacher.view(-1, soft_teacher.size(-1)),
            reduction="batchmean",
        ) * (T * T)

        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss


class DistillationTrainer:
    """Trainer for knowledge distillation from a fine-tuned LLM to a small student."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ExperimentConfig,
    ):
        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tc = config.training
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Freeze teacher
        self.teacher.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student.to(self.device)

        self.optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=self.tc.lr,
            weight_decay=0.01,
        )

        self.loss_fn = DistillationLoss(
            alpha=self.tc.distill_alpha,
            temperature=self.tc.distill_temperature,
        )

        self.carbon = CarbonTracker("distill_train", output_dir=config.experiment_dir)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.experiment_dir, exist_ok=True)

        # W&B + Weave — graceful auth check, hostname-tagged run name
        self.use_wandb = init_wandb(config)
        # Shared panel structure: train/ val/ carbon/ system/
        define_common_metrics(self.use_wandb)

        self.log_path = os.path.join(config.experiment_dir, "distill_train_log.csv")

    def _init_log(self) -> None:
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "val_puzzle_acc", "elapsed_min"])

    def _append_log(self, row: list) -> None:
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def train(self) -> None:
        self._init_log()
        self.carbon.start()
        t_start = time.time()

        for epoch in range(self.tc.epochs):
            metrics = self._train_epoch(epoch)

            if (epoch + 1) % self.tc.log_interval == 0:
                val_metrics = self.evaluate()
                elapsed = (time.time() - t_start) / 60.0
                tqdm.write(
                    f"Distill Epoch {epoch + 1}/{self.tc.epochs} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['puzzle_acc']:.4f} | "
                    f"Time: {elapsed:.0f}min"
                )
                self._append_log([
                    epoch + 1, f"{metrics['loss']:.4f}",
                    f"{val_metrics['puzzle_acc']:.4f}", f"{elapsed:.1f}",
                ])

                if self.use_wandb:
                    wandb.log(
                        {
                            "train/loss": metrics["loss"],
                            "val/puzzle_acc": val_metrics["puzzle_acc"],
                            "train/elapsed_min": elapsed,
                        },
                        step=epoch + 1,
                    )

        self._save_checkpoint("distill_latest.pt")
        emissions = self.carbon.stop()

        if self.use_wandb:
            wandb.finish()

        results_path = os.path.join(self.config.experiment_dir, "distill_results.json")
        with open(results_path, "w") as f:
            json.dump({"emissions": emissions}, f, indent=2)

    def _train_epoch(self, epoch: int) -> dict:
        self.student.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Distill Epoch {epoch + 1}", leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = self.teacher(inputs)

            student_logits = self.student(inputs)

            loss = self.loss_fn(student_logits, teacher_logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / max(1, n_batches)}

    @weave_op()
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.student.eval()
        total_correct = 0
        total_puzzles = 0

        for inputs, labels in self.val_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            logits = self.student(inputs)
            preds = logits.argmax(-1)

            mask = labels != 0
            puzzle_correct = ((preds == labels) | ~mask).all(dim=-1)
            total_correct += puzzle_correct.sum().item()
            total_puzzles += inputs.shape[0]

        return {"puzzle_acc": total_correct / max(1, total_puzzles)}

    def _save_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(
            {
                "model_state_dict": self.student.state_dict(),
                "config": self.config.model_dump(),
            },
            path,
        )
