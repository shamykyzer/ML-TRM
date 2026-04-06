import csv
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.carbon_tracker import CarbonTracker
from src.utils.config import ExperimentConfig


class LLMTrainer:
    """Trainer for fine-tuning LLM baselines (GPT-2 / TinyLlama with LoRA)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ExperimentConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tc = config.training
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Only train LoRA parameters; use lower weight_decay for LLMs
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.tc.lr,
            weight_decay=0.01,  # NOT 1.0 -- that's TRM-specific
        )

        # Derive a short tag from model name for unique filenames
        self.model_tag = config.model.llm_name.split("/")[-1].lower().replace("-", "_")

        self.carbon = CarbonTracker(
            f"{self.model_tag}_train",
            output_dir=config.experiment_dir,
        )

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.experiment_dir, exist_ok=True)

        self.log_path = os.path.join(config.experiment_dir, f"{self.model_tag}_train_log.csv")

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
                    f"Epoch {epoch + 1}/{self.tc.epochs} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['puzzle_acc']:.4f} | "
                    f"Time: {elapsed:.0f}min"
                )
                self._append_log([
                    epoch + 1, f"{metrics['loss']:.4f}",
                    f"{val_metrics['puzzle_acc']:.4f}", f"{elapsed:.1f}",
                ])

        self._save_checkpoint(self.tc.epochs - 1, f"{self.model_tag}_latest.pt")
        emissions = self.carbon.stop()

        results_path = os.path.join(self.config.experiment_dir, f"{self.model_tag}_training_results.json")
        with open(results_path, "w") as f:
            json.dump({"emissions": emissions}, f, indent=2)

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"LLM Epoch {epoch + 1}", leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return {"loss": total_loss / max(1, n_batches)}

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        total_correct = 0
        total_puzzles = 0

        for inputs, labels in self.val_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(input_ids=inputs)
            preds = outputs.logits.argmax(-1)

            mask = labels != 0
            puzzle_correct = ((preds == labels) | ~mask).all(dim=-1)
            total_correct += puzzle_correct.sum().item()
            total_puzzles += inputs.shape[0]

        return {"puzzle_acc": total_correct / max(1, total_puzzles)}

    def _save_checkpoint(self, epoch: int, filename: str) -> None:
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "config": self.config.model_dump(),
            },
            path,
        )
