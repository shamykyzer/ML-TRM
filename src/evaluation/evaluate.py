import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import cell_accuracy, puzzle_accuracy
from src.models.recursion import deep_recursion
from src.training.carbon_tracker import CarbonTracker
from src.training.ema import EMA
from src.utils.config import ExperimentConfig, ModelType


def evaluate_trm(
    model: nn.Module,
    test_loader: DataLoader,
    config: ExperimentConfig,
    ema: EMA = None,
) -> dict:
    """Evaluate a TRM model with ACT inference and CodeCarbon tracking."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    tc = config.training

    carbon = CarbonTracker(
        f"{config.model.model_type.value}_inference",
        output_dir=config.experiment_dir,
    )

    if ema is not None:
        ema.apply_shadow()

    model.eval()
    carbon.start()

    total_cell_correct = 0
    total_cells = 0
    total_puzzle_correct = 0
    total_puzzles = 0
    total_steps = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            B = inputs.shape[0]

            x = model.embedding(inputs)
            y = model.y_init.expand(B, -1, -1).clone()
            z = model.z_init.expand(B, -1, -1).clone()

            steps = 0
            for step in range(tc.N_sup):
                (y, z), logits, q = deep_recursion(
                    model.block, model.output_head, model.q_head,
                    x, y, z, n=tc.n_latent, T=tc.T_deep,
                )
                steps = step + 1
                if q.mean().item() > tc.act_threshold:
                    break

            preds = logits.argmax(-1)
            mask = labels != 0

            total_cell_correct += ((preds == labels) & mask).sum().item()
            total_cells += mask.sum().item()

            puzzle_correct = ((preds == labels) | ~mask).all(dim=-1)
            total_puzzle_correct += puzzle_correct.sum().item()
            total_puzzles += B
            total_steps += steps

    emissions = carbon.stop()

    if ema is not None:
        ema.restore()

    return {
        "cell_accuracy": total_cell_correct / max(1, total_cells),
        "puzzle_accuracy": total_puzzle_correct / max(1, total_puzzles),
        "avg_act_steps": total_steps / max(1, len(test_loader)),
        "inference_emissions": emissions,
    }


def evaluate_standard(
    model: nn.Module,
    test_loader: DataLoader,
    config: ExperimentConfig,
    model_label: str = "model",
) -> dict:
    """Evaluate a standard (non-TRM) model with CodeCarbon tracking."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    carbon = CarbonTracker(f"{model_label}_inference", output_dir=config.experiment_dir)

    model.eval()
    carbon.start()

    total_cell_correct = 0
    total_cells = 0
    total_puzzle_correct = 0
    total_puzzles = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating {model_label}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            if hasattr(logits, "logits"):
                logits = logits.logits

            preds = logits.argmax(-1)
            mask = labels != 0

            total_cell_correct += ((preds == labels) & mask).sum().item()
            total_cells += mask.sum().item()

            puzzle_correct = ((preds == labels) | ~mask).all(dim=-1)
            total_puzzle_correct += puzzle_correct.sum().item()
            total_puzzles += inputs.shape[0]

    emissions = carbon.stop()

    return {
        "cell_accuracy": total_cell_correct / max(1, total_cells),
        "puzzle_accuracy": total_puzzle_correct / max(1, total_puzzles),
        "inference_emissions": emissions,
    }


def load_and_evaluate(
    checkpoint_path: str,
    test_loader: DataLoader,
    config: ExperimentConfig,
) -> dict:
    """Load a checkpoint and evaluate it."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_type = config.model.model_type
    if model_type in (ModelType.TRM_SUDOKU, ModelType.TRM_MAZE):
        from src.models.trm_sudoku import TRMMaze, TRMSudoku

        if model_type == ModelType.TRM_SUDOKU:
            model = TRMSudoku(
                vocab_size=config.model.vocab_size,
                seq_len=config.model.seq_len,
                d_model=config.model.d_model,
                ff_hidden=config.model.ff_hidden,
                num_classes=config.model.num_classes,
            )
        else:
            model = TRMMaze(
                vocab_size=config.model.vocab_size,
                seq_len=config.model.seq_len,
                d_model=config.model.d_model,
                ff_hidden=config.model.ff_hidden,
                num_classes=config.model.num_classes,
            )

        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore EMA if available
        ema = None
        if "ema_state_dict" in checkpoint:
            ema = EMA(model, decay=config.training.ema_decay)
            ema.load_state_dict(checkpoint["ema_state_dict"])

        results = evaluate_trm(model, test_loader, config, ema=ema)

    elif model_type == ModelType.LLM_DISTILL:
        from src.models.distilled_llm import DistilledLLM

        model = DistilledLLM(
            vocab_size=config.model.vocab_size,
            seq_len=config.model.seq_len,
            d_model=config.model.distill_d_model,
            n_layers=config.model.distill_n_layers,
            ff_hidden=config.model.distill_ff_hidden,
            n_heads=config.model.distill_n_heads,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        results = evaluate_standard(model, test_loader, config, "distilled_llm")

    else:
        raise ValueError(f"Use trainer_llm evaluation for {model_type}")

    return results


def save_results(results: dict, output_dir: str, model_name: str) -> None:
    """Save evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}_eval.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def print_sudoku_grid(tokens: list[int], title: str = "") -> None:
    """Print a 9x9 sudoku grid from flat token list (values 1-10, 0=pad)."""
    if title:
        print(f"\n{title}")
    print("+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+")
    for i in range(9):
        row = tokens[i * 9 : (i + 1) * 9]
        cells = []
        for j, v in enumerate(row):
            digit = v - 1 if v > 0 else 0  # undo +1 offset
            cells.append(str(digit) if digit > 0 else ".")
            if j in (2, 5):
                cells.append("|")
        print("| " + " ".join(cells) + " |")
        if i in (2, 5):
            print("+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+")
    print("+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+")
