"""Pre-flight sanity check: evaluate the *raw* trained model weights,
bypassing the EMA swap, to prove that training has been working even
though val_cell_acc has been stuck at 0.1233 in the CSV log.

Why this script exists
----------------------
As of the conversation where this was written, the EMA shadow stored
in ``src/training/ema.py`` had been inheriting the model's bf16 dtype
from ``OfficialTRMTrainer``'s native-bf16 optimization (trainer_official.py:141).
bf16's 7-bit mantissa can't resolve the EMA's per-step ~0.1% delta, so
the shadow was frozen at its initial value and eval (which runs on
``self.ema.apply_shadow()``-swapped weights) was reporting the accuracy
of a frozen random-init model every single epoch. Meanwhile ``self.model``
had been training normally — the train CSV columns ``accuracy`` and
``exact_accuracy`` show real progress (18% -> 74% cell acc, 0% -> 29%
exact solve across epochs 5-95).

This script loads a checkpoint, skips the ``ema_state_dict`` entirely,
and runs the exact same eval loop as ``OfficialTRMTrainer.evaluate()``
(trainer_official.py:793) minus the EMA swap. The output is the val
cell/puzzle accuracy of the *actually trained* weights — the number
the CSV *should* have been reporting all along.

Usage
-----
    python scripts/diagnose_real_weights.py
    python scripts/diagnose_real_weights.py --checkpoint "C:/TRM checkpoints/sudoku-official/snapshots_for_thesis/sudoku_milestone_10pct_epoch50.pt"
    python scripts/diagnose_real_weights.py --device cpu

The default checkpoint is the most recent milestone in the checkpoint
dir (falling back through: epoch_75.pt, epoch_50.pt, epoch_25.pt). This
is because the currently-running training process holds the actual
``latest.pt`` writes via ``_safe_torch_save``, but the periodic
``epoch_*.pt`` files are stable on disk.

Note: this script builds its own model + val_loader (no
``OfficialTRMTrainer`` construction, so no EMA gets re-poisoned). If you
run it while the main training is still going on the same GPU, you'll
fight for VRAM — pass ``--device cpu`` or kill the training first.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Make the project root importable so `src.*` works when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from torch.utils.data import DataLoader

from src.data.collate import official_collate_fn
from src.data.sudoku_dataset import SudokuDataset
from src.models.losses_official import ACTLossHead
from src.models.trm_official import TRMOfficial
from src.utils.config import load_config


def _pick_default_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the most recent stable checkpoint in the dir.

    Prefers milestone snapshots (``epoch_*.pt``) over latest.pt because
    the running training process may be holding latest.pt mid-write.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    # Collect epoch_*.pt files and sort by epoch number
    candidates: list[tuple[int, str]] = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("epoch_") and name.endswith(".pt"):
            try:
                n = int(name[len("epoch_"):-len(".pt")])
                candidates.append((n, os.path.join(checkpoint_dir, name)))
            except ValueError:
                continue

    if candidates:
        candidates.sort(reverse=True)  # highest epoch first
        return candidates[0][1]

    # Fall back to latest.pt if no epoch_*.pt files
    latest = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.isfile(latest):
        return latest
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/trm_official_sudoku.yaml",
        help="YAML config (must match the checkpoint's architecture)",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path (default: most recent epoch_*.pt in checkpoint_dir)",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Override device (default: config.device, usually cuda)",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="If >0, eval only the first N batches (fast smoke test)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config:      {args.config}")
    print(f"  d_model={cfg.model.d_model}  ff_hidden={cfg.model.ff_hidden}  "
          f"halt_max_steps={cfg.model.halt_max_steps}  forward_dtype={cfg.model.forward_dtype}")

    # Resolve checkpoint
    ckpt_path = args.checkpoint or _pick_default_checkpoint(cfg.checkpoint_dir)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"ERROR: no checkpoint found (tried {ckpt_path!r})", file=sys.stderr)
        print(f"  checkpoint_dir: {cfg.checkpoint_dir}", file=sys.stderr)
        return 2
    print(f"Checkpoint:  {ckpt_path}")

    # Resolve device
    device_str = args.device or cfg.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("  cuda requested but unavailable, falling back to cpu")
        device_str = "cpu"
    device = torch.device(device_str)
    forward_dtype = getattr(torch, cfg.model.forward_dtype, torch.bfloat16)
    print(f"Device:      {device}  forward_dtype={forward_dtype}")

    # Build model + loss_head exactly like main.py:147-167
    model_config = {
        "batch_size": cfg.training.batch_size,
        "seq_len": cfg.model.seq_len,
        "vocab_size": cfg.model.vocab_size,
        "num_task_types": cfg.model.num_task_types,
        "task_emb_ndim": cfg.model.task_emb_ndim,
        "task_emb_len": cfg.model.task_emb_len,
        "hidden_size": cfg.model.d_model,
        "expansion": cfg.model.ff_hidden / cfg.model.d_model,
        "num_heads": cfg.model.n_heads,
        "L_layers": cfg.model.L_layers,
        "H_cycles": cfg.model.H_cycles,
        "L_cycles": cfg.model.L_cycles,
        "halt_max_steps": cfg.model.halt_max_steps,
        "halt_exploration_prob": cfg.model.halt_exploration_prob,
        "no_ACT_continue": cfg.model.no_ACT_continue,
        "forward_dtype": cfg.model.forward_dtype,
        "mlp_t": cfg.model.mlp_t,
    }
    print("\nBuilding model ...")
    model = TRMOfficial(model_config)
    loss_head = ACTLossHead(model)
    print(f"  params: {model.param_count():,}")

    # Cast to bf16 (matches trainer_official.py:141 — important: if we
    # loaded fp32 weights into a bf16 model via a state_dict whose tensors
    # are bf16, load_state_dict wouldn't need to cast; but the checkpoint
    # IS bf16 because the running trainer is native bf16. Cast first,
    # then load — the shapes stay the same either way).
    model.to(device=device, dtype=forward_dtype)
    loss_head.to(device=device, dtype=forward_dtype)

    # Load ONLY model_state_dict — explicitly skip ema_state_dict so we
    # measure the actual trained weights, not the frozen EMA shadow.
    print(f"\nLoading {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        print(f"ERROR: checkpoint has no 'model_state_dict' key. Keys: {list(ckpt.keys())}",
              file=sys.stderr)
        return 3

    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    print(f"  loaded epoch={ckpt.get('epoch', '?')}  "
          f"global_step={ckpt.get('global_step', '?')}  "
          f"best_puzzle_acc={ckpt.get('best_puzzle_acc', 0.0):.4f}")
    if missing:
        print(f"  missing ({len(missing)}): {missing[:4]}{' ...' if len(missing) > 4 else ''}")
    if unexpected:
        print(f"  unexpected ({len(unexpected)}): {unexpected[:4]}{' ...' if len(unexpected) > 4 else ''}")
    if "ema_state_dict" in ckpt:
        print(f"  DELIBERATELY IGNORED: ema_state_dict "
              f"({len(ckpt['ema_state_dict'])} tensors) — that's what we're bypassing")

    # Build val_loader — same as main.py:199-203 for the sudoku dataset
    print(f"\nLoading val dataset from {cfg.data.data_dir} ...")
    test_ds = SudokuDataset(cfg.data.data_dir, "test")
    print(f"  {len(test_ds)} test samples")
    collate_fn = official_collate_fn(cfg.training.task_id)
    val_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,  # force 0 so we don't compete with training for workers
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # Eval loop — this is a 1:1 copy of OfficialTRMTrainer.evaluate()
    # (trainer_official.py:793-911) MINUS the self.ema.apply_shadow() swap
    # at line 794 and the self.ema.restore() at line 888. No tqdm because
    # this is a one-shot script and a plain per-batch print is more
    # diagnostic-friendly.
    model.eval()
    loss_head.eval()

    total_cell_correct = 0
    total_cells = 0
    total_puzzle_correct = 0
    total_puzzles = 0
    total_q_halt_correct = 0
    n_samples = 0

    max_steps = cfg.model.halt_max_steps
    no_act_continue = cfg.model.no_ACT_continue

    print(f"\nRunning eval (max_steps={max_steps}, batch_size={cfg.training.batch_size}) ...")
    t0 = time.time()
    n_batches_total = len(val_loader)
    n_batches_done = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if args.limit_batches and batch_idx >= args.limit_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            B = batch["inputs"].shape[0]

            carry = loss_head.initial_carry(batch)

            first_halt_step = torch.full(
                (B,), max_steps, dtype=torch.int32, device=device
            )
            ever_halted = torch.zeros(B, dtype=torch.bool, device=device)

            _outputs = None
            for step_idx in range(max_steps):
                carry, _outputs = model(carry=carry, batch=batch)
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

            logits = _outputs["logits"]
            preds = logits.argmax(-1)
            labels = carry.current_data["labels"]
            mask = labels != -100

            total_cell_correct += ((preds == labels) & mask).sum().item()
            total_cells += mask.sum().item()

            puzzle_correct = ((preds == labels) | ~mask).all(dim=-1)
            total_puzzle_correct += puzzle_correct.sum().item()
            total_puzzles += B

            q_halt_correct = (_outputs["q_halt_logits"] >= 0) == puzzle_correct
            total_q_halt_correct += q_halt_correct.sum().item()

            n_samples += B
            n_batches_done += 1

            # Periodic progress line so a long CPU-mode eval doesn't look hung
            if n_batches_done % 50 == 0 or n_batches_done == n_batches_total:
                cell_acc = total_cell_correct / max(1, total_cells)
                puz_acc = total_puzzle_correct / max(1, total_puzzles)
                elapsed = time.time() - t0
                print(f"  batch {n_batches_done}/{n_batches_total}  "
                      f"cell={cell_acc:.4f}  puzzle={puz_acc:.4f}  "
                      f"[{elapsed:.1f}s]")

    total_elapsed = time.time() - t0

    cell_acc = total_cell_correct / max(1, total_cells)
    puzzle_acc = total_puzzle_correct / max(1, total_puzzles)
    q_halt_acc = total_q_halt_correct / max(1, n_samples)

    print(f"\n=== RAW MODEL WEIGHTS EVAL (EMA swap bypassed) ===")
    print(f"  checkpoint:       {ckpt_path}")
    print(f"  epoch:            {ckpt.get('epoch', '?')}")
    print(f"  samples evaluated: {n_samples}")
    print(f"  wall time:        {total_elapsed:.1f}s")
    print(f"")
    print(f"  cell_acc:         {cell_acc:.4f}")
    print(f"  puzzle_acc:       {puzzle_acc:.4f}")
    print(f"  q_halt_acc:       {q_halt_acc:.4f}")
    print(f"")
    print(f"  CSV reported (EMA-swapped, frozen): cell=0.1233  puzzle=0.0000")
    print(f"")
    if cell_acc > 0.20:
        print(f"  DIAGNOSIS CONFIRMED: trained model has real accuracy, "
              f"the EMA shadow bug was hiding it all along.")
    else:
        print(f"  WARNING: raw model cell_acc is not substantially above the "
              f"random baseline either. Either the diagnosis is wrong, or the "
              f"trained model legitimately hasn't learned enough yet at this "
              f"checkpoint epoch.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
