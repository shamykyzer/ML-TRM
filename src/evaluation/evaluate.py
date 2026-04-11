import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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
                (y, z), logits, q, _q_logits = deep_recursion(
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


def evaluate_official(
    model: nn.Module,
    test_loader: DataLoader,
    config: ExperimentConfig,
    ema: EMA = None,
) -> dict:
    """Evaluate an official TRM model with full ACT steps."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    # Cast to forward_dtype (typically bf16) BEFORE the first forward — the
    # CastedEmbedding inside the model casts its output to forward_dtype, so
    # if the linear layers are still fp32 (PyTorch default), the first matmul
    # crashes with a dtype mismatch. The trainer already does this in
    # OfficialTRMTrainer.__init__ at trainer_official.py:163-165; eval was
    # missing the equivalent cast.
    forward_dtype = getattr(torch, config.model.forward_dtype, torch.bfloat16)
    model.to(device=device, dtype=forward_dtype)

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
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            B = batch["inputs"].shape[0]

            carry = model.initial_carry(batch)

            for _step in range(config.model.halt_max_steps):
                carry, outputs = model(carry=carry, batch=batch)

            preds = outputs["logits"].argmax(-1)
            labels = carry.current_data["labels"]
            mask = labels != -100

            total_cell_correct += ((preds == labels) & mask).sum().item()
            total_cells += mask.sum().item()

            puzzle_correct = ((preds == labels) | ~mask).all(dim=-1)
            total_puzzle_correct += puzzle_correct.sum().item()
            total_puzzles += B
            total_steps += carry.steps.sum().item()
            n_samples += B

    emissions = carbon.stop()

    if ema is not None:
        ema.restore()

    return {
        "cell_accuracy": total_cell_correct / max(1, total_cells),
        "puzzle_accuracy": total_puzzle_correct / max(1, total_puzzles),
        "avg_act_steps": total_steps / max(1, n_samples),
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

    elif model_type in (ModelType.TRM_OFFICIAL_SUDOKU, ModelType.TRM_OFFICIAL_MAZE):
        from src.models.trm_official import TRMOfficial

        model_config = {
            "batch_size": config.training.batch_size,
            "seq_len": config.model.seq_len,
            "vocab_size": config.model.vocab_size,
            "num_task_types": config.model.num_task_types,
            "task_emb_ndim": config.model.task_emb_ndim,
            "task_emb_len": config.model.task_emb_len,
            "hidden_size": config.model.d_model,
            "expansion": config.model.ff_hidden / config.model.d_model,
            "num_heads": config.model.n_heads,
            "L_layers": config.model.L_layers,
            "H_cycles": config.model.H_cycles,
            "L_cycles": config.model.L_cycles,
            "halt_max_steps": config.model.halt_max_steps,
            "halt_exploration_prob": config.model.halt_exploration_prob,
            "no_ACT_continue": config.model.no_ACT_continue,
            "forward_dtype": config.model.forward_dtype,
            "mlp_t": config.model.mlp_t,
        }
        model = TRMOfficial(model_config)

        # strict=False so we can load remapped HF reference checkpoints
        # (e.g. hf_checkpoints/Sudoku-Extreme-mlp/remapped_for_local.pt) that
        # only contain model weights — no optimizer/EMA/global_step. The only
        # legitimately-missing keys are the rotary_emb buffers (deterministic,
        # reconstructed by RotaryEmbedding.__init__). A trainer-saved
        # checkpoint will have everything and report 0/0.
        result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        print(f"[Eval] Loaded {len(checkpoint['model_state_dict'])} keys "
              f"(missing: {len(missing)}, unexpected: {len(unexpected)})")
        if unexpected:
            print(f"[Eval] WARNING: unexpected keys (should be zero): {unexpected[:5]}"
                  f"{' ...' if len(unexpected) > 5 else ''}")
        non_rotary_missing = [k for k in missing if "rotary_emb" not in k]
        if non_rotary_missing:
            print(f"[Eval] WARNING: missing keys beyond rotary_emb buffers: "
                  f"{non_rotary_missing[:5]}{' ...' if len(non_rotary_missing) > 5 else ''}")
        if "source" in checkpoint:
            print(f"[Eval] Checkpoint source: {checkpoint['source']}")

        ema = None
        if "ema_state_dict" in checkpoint:
            ema = EMA(model, decay=config.training.ema_decay)
            ema.load_state_dict(checkpoint["ema_state_dict"])
        else:
            print("[Eval] No EMA shadow in checkpoint — evaluating raw model weights")

        results = evaluate_official(model, test_loader, config, ema=ema)

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
