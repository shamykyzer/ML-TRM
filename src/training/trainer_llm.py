import csv
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.carbon_tracker import CarbonTracker
from src.training.wall_clock_guard import wall_clock_expired
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
            csv.writer(f).writerow(
                ["epoch", "loss", "val_loss", "val_puzzle_acc", "val_cell_acc", "elapsed_min"]
            )

    def _append_log(self, row: list) -> None:
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def train(self) -> None:
        self._init_log()
        self.carbon.start()
        t_start = time.time()

        # Early stopping state: a non-zero patience arms it. `best` starts at the
        # worst possible value for the chosen mode so the first eval always wins.
        es_patience = int(self.tc.early_stop_patience or 0)
        es_mode = self.tc.early_stop_mode
        es_metric = self.tc.early_stop_metric
        es_min_delta = float(self.tc.early_stop_min_delta)
        best_value = float("-inf") if es_mode == "max" else float("inf")
        best_epoch = 0
        stopped_early = False
        last_epoch = self.tc.epochs - 1

        # Pre-training eval at step 0, before any gradient update. Anchors the
        # plateau plot: if val metrics at step 0 ≈ step N, the model never moved.
        # Without this baseline, the first data point is at step=log_interval
        # (e.g. 10) and reviewers can ask "how do we know it didn't briefly
        # learn and then stall?" — this answers that definitively.
        val_metrics_initial = self.evaluate()
        tqdm.write(
            f"[baseline] Epoch 0/{self.tc.epochs} | "
            f"ValLoss: {val_metrics_initial['loss']:.4f} | "
            f"Puzzle: {val_metrics_initial['puzzle_acc']:.4f} | "
            f"Cell: {val_metrics_initial['cell_acc']:.4f}"
        )
        self._append_log([
            0, "",  # no train_loss at epoch 0 — no gradient steps taken yet
            f"{val_metrics_initial['loss']:.4f}",
            f"{val_metrics_initial['puzzle_acc']:.4f}",
            f"{val_metrics_initial['cell_acc']:.4f}",
            "0.0",
        ])
        if self.use_wandb:
            wandb.log(
                {
                    "val/loss": val_metrics_initial["loss"],
                    "val/lm_loss": val_metrics_initial["loss"],
                    "val/puzzle_acc": val_metrics_initial["puzzle_acc"],
                    "val/cell_acc": val_metrics_initial["cell_acc"],
                    "val/accuracy": val_metrics_initial["cell_acc"],
                    "val/exact_accuracy": val_metrics_initial["puzzle_acc"],
                },
                step=0,
            )

        for epoch in range(self.tc.epochs):
            # Novelty iso-time cap. Dormant unless TRM_MAX_TRAIN_SECONDS is set.
            # Reuses the existing early-stop bookkeeping (last_epoch,
            # stopped_early) so latest.pt records the right epoch.
            if wall_clock_expired():
                tqdm.write(
                    f"[wall-clock] budget exhausted before epoch {epoch + 1}/{self.tc.epochs} "
                    f"— halting."
                )
                last_epoch = epoch - 1 if epoch > 0 else 0
                stopped_early = True
                break
            metrics = self._train_epoch(epoch)

            if (epoch + 1) % self.tc.log_interval == 0:
                val_metrics = self.evaluate()
                elapsed = (time.time() - t_start) / 60.0
                tqdm.write(
                    f"Epoch {epoch + 1}/{self.tc.epochs} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"ValLoss: {val_metrics['loss']:.4f} | "
                    f"Puzzle: {val_metrics['puzzle_acc']:.4f} | "
                    f"Cell: {val_metrics['cell_acc']:.4f} | "
                    f"Time: {elapsed:.0f}min"
                )
                self._append_log([
                    epoch + 1, f"{metrics['loss']:.4f}",
                    f"{val_metrics['loss']:.4f}",
                    f"{val_metrics['puzzle_acc']:.4f}",
                    f"{val_metrics['cell_acc']:.4f}",
                    f"{elapsed:.1f}",
                ])

                if self.use_wandb:
                    # Primary names + symmetric aliases matching trainer_official:
                    # val/accuracy mirrors train/accuracy (cell-level),
                    # val/exact_accuracy mirrors train/exact_accuracy (puzzle-level).
                    # lm/loss and val/lm_loss are explicit semantic names for the
                    # LM cross-entropy — same scalars as train/loss, val/loss;
                    # duplicate keys so existing dashboards keep working.
                    wandb.log(
                        {
                            "train/loss": metrics["loss"],
                            "lm/loss": metrics["loss"],
                            "val/loss": val_metrics["loss"],
                            "val/lm_loss": val_metrics["loss"],
                            "val/puzzle_acc": val_metrics["puzzle_acc"],
                            "val/cell_acc": val_metrics["cell_acc"],
                            "val/accuracy": val_metrics["cell_acc"],
                            "val/exact_accuracy": val_metrics["puzzle_acc"],
                            "train/elapsed_min": elapsed,
                        },
                        step=epoch + 1,
                    )

                if es_patience > 0:
                    current = {
                        "val_cell_acc": val_metrics["cell_acc"],
                        "val_puzzle_acc": val_metrics["puzzle_acc"],
                        "train_loss": metrics["loss"],
                    }[es_metric]
                    improved = (
                        current > best_value + es_min_delta if es_mode == "max"
                        else current < best_value - es_min_delta
                    )
                    if improved:
                        best_value = current
                        best_epoch = epoch + 1
                    elif (epoch + 1) - best_epoch >= es_patience:
                        tqdm.write(
                            f"[early-stop] {es_metric} has not improved for "
                            f"{(epoch + 1) - best_epoch} epochs "
                            f"(best={best_value:.4f} at epoch {best_epoch}). "
                            f"Halting at epoch {epoch + 1}/{self.tc.epochs}."
                        )
                        last_epoch = epoch
                        stopped_early = True
                        break

            if (epoch + 1) % self.tc.save_interval == 0:
                self._save_checkpoint(epoch, f"{self.model_tag}_epoch_{epoch + 1}.pt")

        self._save_checkpoint(last_epoch, f"{self.model_tag}_latest.pt")
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

            # Per-step LM loss logging — only enabled for runs that set
            # training.log_per_step: true in their YAML (e.g. DeepSeek
            # plateau runs). Off by default to avoid dashboard clutter.
            if self.use_wandb and self.tc.log_per_step:
                wandb.log({"lm/step_loss": outputs.loss.item()})

        # Flush trailing micro-batches when len(loader) % accum != 0
        if (len(self.train_loader) % accum) != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {"loss": total_loss / max(1, n_batches)}

    @weave_op()
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        total_puzzles_correct = 0
        total_cells_correct = 0
        total_cells_graded = 0
        total_puzzles = 0
        total_loss_sum = 0.0
        total_loss_batches = 0

        for inputs, labels in self.val_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Pass labels so HF computes val cross-entropy internally with the
            # correct shift + ignore_index=-100 mask. Mirrors the training loss
            # computation so val_loss diverging from train_loss is a direct
            # overfitting signal.
            labels_for_loss = labels.masked_fill(labels == 0, HF_IGNORE_INDEX)
            outputs = self.model(input_ids=inputs, labels=labels_for_loss)
            total_loss_sum += outputs.loss.item()
            total_loss_batches += 1

            # Accuracy uses the same shift HF applied internally for the loss:
            # logits[i] predicts labels[i+1], so compare preds[:-1] with labels[1:].
            preds = outputs.logits[:, :-1, :].argmax(-1)
            labels_shifted = labels[:, 1:]

            mask = labels_shifted != 0
            puzzle_correct = ((preds == labels_shifted) | ~mask).all(dim=-1)
            cells_correct = ((preds == labels_shifted) & mask).sum().item()

            total_puzzles_correct += puzzle_correct.sum().item()
            total_cells_correct += cells_correct
            total_cells_graded += mask.sum().item()
            total_puzzles += inputs.shape[0]

        return {
            "loss": total_loss_sum / max(1, total_loss_batches),
            "puzzle_acc": total_puzzles_correct / max(1, total_puzzles),
            "cell_acc": total_cells_correct / max(1, total_cells_graded),
        }

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
