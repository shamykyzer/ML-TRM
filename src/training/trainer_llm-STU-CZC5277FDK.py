import csv
import json
import os
import time

import torch
import torch.nn as nn
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


# Datasets emit label 0 to mean "ignore this position" (pre-filled clue cells in
# sudoku, walls/start/goal in maze). HuggingFace causal LM uses ignore_index=-100
# in its internal CrossEntropyLoss, so feeding raw 0s would train the model to
# predict token 0 at every clue position — silently broken. Remap before forward.
HF_IGNORE_INDEX = -100


class LLMTrainer:
    """Trainer for fine-tuning LLM baselines (GPT-2 / SmolLM2 / Qwen / Llama with LoRA).

    Works for both Sudoku-Extreme (seq_len=81, vocab=11) and Maze-Hard
    (seq_len=900, vocab=6). The dataset/loader is chosen by main.py based on
    config.data.dataset; this trainer is task-agnostic.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ExperimentConfig,
    ):
        # Reproducibility: LLMTrainer used to skip set_seed, so reruns of the
        # same YAML produced different LoRA inits and shuffle orders.
        set_seed(config.seed)

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

        # Derive a short tag from model name + dataset for unique filenames
        # so a sudoku run and a maze run on the same LLM don't collide.
        model_short = config.model.llm_name.split("/")[-1].lower().replace("-", "_")
        self.model_tag = f"{model_short}_{config.data.dataset}"

        self.carbon = CarbonTracker(
            f"{self.model_tag}_train",
            output_dir=config.experiment_dir,
        )

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.experiment_dir, exist_ok=True)

        # W&B + Weave — graceful auth check, hostname-tagged run name
        self.use_wandb = init_wandb(config)
        # Shared panel structure: train/ val/ carbon/ system/
        define_common_metrics(self.use_wandb)

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
                self._save_checkpoint(epoch, f"{self.model_tag}_epoch_{epoch + 1}.pt")

        self._save_checkpoint(self.tc.epochs - 1, f"{self.model_tag}_latest.pt")
        emissions = self.carbon.stop()

        if self.use_wandb:
            wandb.finish()

        results_path = os.path.join(self.config.experiment_dir, f"{self.model_tag}_training_results.json")
        with open(results_path, "w") as f:
            json.dump({"emissions": emissions}, f, indent=2)

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        accum = max(1, self.tc.grad_accum_steps)

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"LLM Epoch {epoch + 1}", leave=False)
        for step, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Remap dataset's ignore sentinel (0) to HF's ignore_index (-100)
            # so the LoRA only learns to predict target positions, not pad/clues.
            labels_for_loss = labels.masked_fill(labels == 0, HF_IGNORE_INDEX)

            outputs = self.model(input_ids=inputs, labels=labels_for_loss)
            loss = outputs.loss / accum
            loss.backward()

            if (step + 1) % accum == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += outputs.loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

        # Flush trailing micro-batches when len(loader) % accum != 0
        if (len(self.train_loader) % accum) != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {"loss": total_loss / max(1, n_batches)}

    @weave_op()
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

            # mask: True at positions the model must predict (label != 0 ignore).
            # A puzzle is solved when every must-predict position is correct.
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
