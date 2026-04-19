"""Diagnose which state_dict keys are missing when HF Maze checkpoint is loaded."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.utils.config import load_config
from src.models.trm_official import TRMOfficial


def build_model(config):
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
    return TRMOfficial(model_config)


def main():
    ckpt_path = "hf_checkpoints/Maze-Hard/remapped_for_local.pt"
    config_path = "configs/trm_official_maze.yaml"

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Top-level keys in checkpoint file: {list(ckpt.keys())}")
    if "note" in ckpt:
        print(f"  note: {ckpt['note']}")
    if "source" in ckpt:
        print(f"  source: {ckpt['source']}")

    ckpt_sd = ckpt["model_state_dict"]
    print(f"# checkpoint model_state_dict keys: {len(ckpt_sd)}")

    config = load_config(config_path)
    model = build_model(config)
    model_sd = model.state_dict()

    ckpt_keys = set(ckpt_sd.keys())
    model_keys = set(model_sd.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    loaded = ckpt_keys & model_keys

    print("\n================ DIFF ================")
    print(f"missing (model has, ckpt lacks)   : {len(missing)}")
    print(f"unexpected (ckpt has, model lacks): {len(unexpected)}")
    print(f"loaded (in both)                  : {len(loaded)}")

    print("\n--- MISSING keys ---")
    named_params = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    for k in sorted(missing):
        shp = tuple(model_sd[k].shape)
        dtype = model_sd[k].dtype
        is_buffer = k in named_buffers
        is_param = k in named_params
        requires_grad = named_params[k].requires_grad if is_param else None
        print(f"  {k}")
        print(f"    shape={shp}, dtype={dtype}, is_buffer={is_buffer}, "
              f"is_param={is_param}, requires_grad={requires_grad}")

    print("\n--- SHAPE MISMATCHES among loaded keys ---")
    mismatches = [
        (k, tuple(ckpt_sd[k].shape), tuple(model_sd[k].shape))
        for k in sorted(loaded)
        if tuple(ckpt_sd[k].shape) != tuple(model_sd[k].shape)
    ]
    if mismatches:
        for k, c, m in mismatches:
            print(f"  {k}  ckpt={c}  model={m}")
    else:
        print("  (none)")

    print("\n--- torch's strict=False load ---")
    result = model.load_state_dict(ckpt_sd, strict=False)
    print(f"  missing_keys   ({len(result.missing_keys)}): {result.missing_keys}")
    print(f"  unexpected_keys({len(result.unexpected_keys)}): {result.unexpected_keys}")

    # Determinism sanity check: fresh model rebuilt from same config should
    # produce IDENTICAL rotary buffers. If so, partial load leaving these
    # keys untouched is exactly equivalent to restoring them from a full ckpt.
    print("\n--- Rotary buffer determinism check ---")
    model2 = build_model(config)
    model2_sd = model2.state_dict()
    all_equal = True
    for k in sorted(missing):
        a = model.state_dict()[k]  # after ckpt load (untouched)
        b = model2_sd[k]           # fresh construction
        eq = torch.allclose(a, b, atol=0, rtol=0)
        print(f"  {k}: identical across two fresh inits? {eq}")
        all_equal = all_equal and eq
    print(f"  => all rotary buffers deterministic: {all_equal}")

    # Compare model hidden-dim head_dim to the buffer shape to be explicit
    head_dim = config.model.d_model // config.model.n_heads
    total_len = config.model.seq_len + config.model.task_emb_len
    print(f"\n  expected head_dim = d_model/n_heads = {head_dim}")
    print(f"  expected max_positions = seq_len + task_emb_len = {total_len}")
    print(f"  so cos/sin_cached.shape should be ({total_len}, {head_dim // 2 * 2 // 2}={head_dim//2}) "
          f"per formula dim/2 pairs stored as [total_len, dim/2]")
    print(f"  inv_freq.shape should be ({head_dim // 2},)  (dim/2)")


if __name__ == "__main__":
    main()
