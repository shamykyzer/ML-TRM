"""Integration test: load the remapped Sanjin2024 maze checkpoint into a fresh
TRMOfficial(mlp_t=False) for the maze task and verify the load is clean.

Pass criteria:
  - Zero unexpected keys.
  - The only missing keys are the rotary_emb buffers (the reference codebase
    doesn't save these because they're deterministic and reconstructed by
    RotaryEmbedding.__init__ at model construction; our local model has them
    correctly populated already, so the missing report is cosmetic).
  - The loaded weights actually changed parameter values.
  - task_emb row 1 (maze, since maze config uses task_id=1) transferred from
    the reference's puzzle_emb; row 0 (sudoku) stayed at zero.

Run:
    python scripts/verify_maze_loads.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from src.models.trm_official import TRMOfficial

REMAP_PATH = "hf_checkpoints/Maze-Hard/remapped_for_local.pt"

# Hardcoded maze model config matching the Sanjin2024 maze all_config.yaml.
# Independent of any YAML so this verifier can run before the maze config
# rewrite (if any) lands.
MODEL_CONFIG = {
    "batch_size": 8,
    "seq_len": 900,
    "vocab_size": 6,
    "num_task_types": 2,
    "task_emb_ndim": 512,
    "task_emb_len": 16,
    "hidden_size": 512,
    "expansion": 1536 / 512,   # = 3.0; gives mlp ff_hidden=1536 via the simple formula
    "num_heads": 8,
    "L_layers": 2,
    "H_cycles": 3,
    "L_cycles": 4,             # paper-faithful for maze (Sanjin all_config.yaml)
    "halt_max_steps": 16,
    "halt_exploration_prob": 0.1,
    "no_ACT_continue": True,   # paper-faithful (was false in local maze.yaml)
    "forward_dtype": "bfloat16",
    "mlp_t": False,            # attention variant -- maze only ships attention
}

MAZE_TASK_ROW = 1  # configs/trm_official_maze.yaml sets task_id=1


def main() -> int:
    if not os.path.isfile(REMAP_PATH):
        print(f"ERROR: remapped checkpoint not found at {REMAP_PATH}", file=sys.stderr)
        print(f"  Run scripts/remap_maze.py first.", file=sys.stderr)
        return 2

    print("Constructing TRMOfficial(mlp_t=False) for maze ...")
    model = TRMOfficial(MODEL_CONFIG)
    print(f"  param count: {model.param_count():,}")

    # Snapshot a few parameter values BEFORE the load.
    pre_load = {
        "self_attn.q_proj": model.inner.L_level.layers[0].self_attn.q_proj.weight[0, 0].item(),
        "embed_tokens": model.inner.embed_tokens.embedding.weight[0, 0].item(),
        "q_head_bias": model.inner.q_head.linear.bias[0].item(),
    }

    print(f"\nLoading remapped state_dict from {REMAP_PATH} (strict=False) ...")
    payload = torch.load(REMAP_PATH, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    print(f"  Source: {payload.get('source', '?')}")

    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    print(f"\nLoad result:")
    print(f"  Tried to load: {len(state)} keys")
    print(f"  Missing in checkpoint ({len(missing)} -- expected: only rotary_emb buffers):")
    for k in missing:
        print(f"    {k}")
    print(f"  Unexpected in checkpoint ({len(unexpected)} -- expected: ZERO):")
    for k in unexpected:
        print(f"    {k}")

    # --- Pass criteria ---
    problems: list[str] = []

    if unexpected:
        problems.append(
            f"{len(unexpected)} unexpected keys (remap produced keys the model doesn't have)"
        )

    expected_missing_substrings = ("rotary_emb",)
    unexpected_missing = [
        k for k in missing
        if not any(s in k for s in expected_missing_substrings)
    ]
    if unexpected_missing:
        problems.append(
            f"{len(unexpected_missing)} unexpectedly-missing keys: {unexpected_missing}"
        )

    post_load = {
        "self_attn.q_proj": model.inner.L_level.layers[0].self_attn.q_proj.weight[0, 0].item(),
        "embed_tokens": model.inner.embed_tokens.embedding.weight[0, 0].item(),
        "q_head_bias": model.inner.q_head.linear.bias[0].item(),
    }
    print("\nWeight change check (pre -> post load):")
    for k in pre_load:
        changed = abs(pre_load[k] - post_load[k]) > 1e-9
        marker = "[ok]" if changed else "[FAIL]"
        print(f"  {marker} {k:<20} {pre_load[k]:+.6f}  ->  {post_load[k]:+.6f}")
        if not changed:
            problems.append(f"{k} did not change after load")

    # task_emb sanity: row 1 (maze) transferred, row 0 (sudoku) stays zero
    task_emb = model.inner.task_emb.embedding.weight
    row0_norm = task_emb[0].abs().sum().item()
    row1_norm = task_emb[1].abs().sum().item()
    print(f"\ntask_emb row sanity (row 0=sudoku, row 1=maze, maze checkpoint -> row 1):")
    print(f"  row 0 |abs|.sum: {row0_norm:.4f}  (expected: 0, zero-padded for sudoku)")
    print(f"  row 1 |abs|.sum: {row1_norm:.4f}  (expected: > 0, transferred from maze puzzle_emb)")
    if row0_norm != 0:
        problems.append(f"task_emb row 0 (sudoku) is non-zero ({row0_norm:.4f}) -- should be zero-padded")
    if row1_norm <= 0:
        problems.append("task_emb row 1 (maze) is all-zero -- maze puzzle_emb did not transfer to row 1")

    print()
    if problems:
        print("FAIL:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PASS: All transferred keys loaded cleanly. Only rotary_emb buffers missing (cosmetic).")
    print("      task_emb correctly transferred maze row, zero-padded sudoku row.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
