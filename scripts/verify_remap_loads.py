"""Integration test: construct a fresh TRMOfficial with the new ff_hidden=1536
config and load the remapped reference weights via strict=False.

Pass criteria:
  - All 18 transferred keys load without shape errors.
  - Missing keys are ONLY the intentionally-dropped ones (embed_tokens,
    task_emb, lm_head, rotary cos/sin buffers).
  - No unexpected keys.

Run:
    python scripts/verify_remap_loads.py
"""
from __future__ import annotations

import os
import sys

# Make the project root importable so `src.*` works when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from src.models.trm_official import TRMOfficial
from src.utils.config import load_config

CONFIG_PATH = "configs/trm_official_sudoku.yaml"
REMAP_PATH = "hf_checkpoints/ARC/remapped_for_local.pt"


def main() -> int:
    cfg = load_config(CONFIG_PATH)
    print(f"Config: {CONFIG_PATH}")
    print(f"  d_model={cfg.model.d_model}  ff_hidden={cfg.model.ff_hidden}  "
          f"n_heads={cfg.model.n_heads}  expansion={cfg.model.ff_hidden / cfg.model.d_model}")

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

    print("\nConstructing TRMOfficial ...")
    model = TRMOfficial(model_config)
    print(f"  param count: {model.param_count():,}")

    print(f"\nLoading remapped state_dict from {REMAP_PATH} (strict=False) ...")
    payload = torch.load(REMAP_PATH, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]

    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    print(f"\nLoad result:")
    print(f"  Tried to load: {len(state)} keys")
    print(f"  Missing in checkpoint ({len(missing)} keys, expected to be fresh-init):")
    for k in missing:
        print(f"    {k}")
    print(f"  Unexpected in checkpoint ({len(unexpected)} keys — should be ZERO):")
    for k in unexpected:
        print(f"    {k}")

    # Pass criteria
    problems = []
    if unexpected:
        problems.append(f"{len(unexpected)} unexpected keys (remap produced keys the model doesn't have)")

    # Expected-missing keys: embed_tokens, task_emb, lm_head, rotary buffers.
    # These should all show up in `missing` (they're model params not in the checkpoint).
    # lm_head has a double-register due to CastedLinear aliasing (see its docstring
    # in scripts/remap_hf_checkpoint.py) — both lm_head.weight and lm_head.linear.weight
    # are expected to be missing because we intentionally skip the whole layer.
    # If anything ELSE is missing, something's off.
    expected_missing_substrings = (
        "embed_tokens",
        "task_emb",
        "lm_head",
        "rotary_emb",
    )
    unexpected_missing = [
        k for k in missing
        if not any(s in k for s in expected_missing_substrings)
    ]
    if unexpected_missing:
        problems.append(
            f"{len(unexpected_missing)} unexpectedly-missing keys "
            f"(should have been in the remap but weren't): {unexpected_missing}"
        )

    print()
    if problems:
        print("FAIL:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PASS: All transferred keys loaded cleanly. Missing keys are all expected.")
    print(f"  Fresh-init params: embed/task_emb/lm_head/rotary = model minus the "
          f"{sum(v.numel() for v in state.values()):,} transferred params")
    return 0


if __name__ == "__main__":
    sys.exit(main())
