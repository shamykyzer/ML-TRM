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
from src.utils.seed import set_seed

try:
    import wandb  # needed for wandb.log / wandb.finish when use_wandb=True
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DistillationLoss(nn.Module):
    """Knowledge distillation loss: soft KL-div + hard CE.

    Datasets emit label=0 to mean "ignore this position" (sudoku clues, maze
    walls/start/goal). Both branches of this loss must skip those positions:
    CE via ignore_index, KL via masking the per-token KL contributions.
    """

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

        # Soft label loss — per-token KL, masked to skip ignore positions so
        # the student isn't pulled toward the teacher's distribution on cells
        # the teacher had no reason to predict (e.g. pre-filled clues).
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        kl_per_tok = F.kl_div(
            soft_student, soft_teacher, reduction="none",
        ).sum(dim=-1)  # [B, L]
        mask = (labels != self.ignore_index).float()
        kl_loss = (kl_per_tok * mask).sum() / mask.sum().clamp(min=1.0) * (T * T)

        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss


def _teacher_logits(teacher: nn.Module, inputs: torch.Tensor, kind: str) -> torch.Tensor:
    """Extract logits from either a HF-wrapped BaselineLLM or a raw DistilledLLM.

    BaselineLLM.forward returns a HuggingFace CausalLMOutputWithPast; we want
    its .logits tensor. DistilledLLM.forward returns the logits tensor directly.
    """
    if kind == "baseline_llm":
        out = teacher(input_ids=inputs)
        return out.logits
    return teacher(inputs)


class DistillationTrainer:
    """Trainer for knowledge distillation from a fine-tuned LLM to a small student."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ExperimentConfig,
        teacher_kind: str = "distilled_llm",
    ):
        # Reproducibility — match LLMTrainer / TRMTrainer behavior.
        set_seed(config.seed)

        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tc = config.training
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.teacher_kind = teacher_kind

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

        # Tag distillation outputs by dataset so sudoku/maze runs don't collide.
        self.tag = f"distill_{config.data.dataset}"

        self.carbon = CarbonTracker(self.tag, output_dir=config.experiment_dir)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.experiment_dir, exist_ok=True)

        # W&B + Weave — graceful auth check, hostname-tagged run name
        self.use_wandb = init_wandb(config)
        # Shared panel structure: train/ val/ carbon/ system/
        define_common_metrics(self.use_wandb)

        self.log_path = os.path.join(config.experiment_dir, f"{self.tag}_train_log.csv")

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

            if (epoch + 1) % self.tc.save_interval == 0:
                self._save_checkpoint(f"{self.tag}_epoch_{epoch + 1}.pt")

        self._save_checkpoint(f"{self.tag}_latest.pt")
        emissions = self.carbon.stop()

        if self.use_wandb:
            wandb.finish()

        results_path = os.path.join(self.config.experiment_dir, f"{self.tag}_results.json")
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
                teacher_logits = _teacher_logits(self.teacher, inputs, self.teacher_kind)

            student_logits = self.student(inputs)

            # Teacher and student vocab sizes can differ (BaselineLLM has the
            # HF model's full vocab, ~50K; DistilledLLM has the puzzle vocab,
            # 6 or 11). Slice teacher logits to the student vocab so KL is
            # computed over the same token set on both sides.
            v = student_logits.size(-1)
            if teacher_logits.size(-1) != v:
                teacher_logits = teacher_logits[..., :v]

            loss = self.loss_fn(student_logits, teacher_logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

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
