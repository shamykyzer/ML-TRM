"""Integration test: load the remapped Sanjin2024 sudoku-mlp checkpoint into
a fresh TRMOfficial(mlp_t=True) and verify the load is clean.

Pass criteria:
  - Zero unexpected keys (the remap produced nothing the model doesn't have).
  - The only missing keys are the rotary_emb buffers (the MLP variant was
    trained with pos_encodings: none — the buffers exist locally for code-path
    compatibility but the mlp_t branch never reads them).
  - The loaded weights actually changed parameter values (catches "everything
    loads cosmetically but no values transferred" bugs).

Run:
    python scripts/verify_sudoku_mlp_loads.py
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

REMAP_PATH = "hf_checkpoints/Sudoku-Extreme-mlp/remapped_for_local.pt"

# Hardcoded model config matching the Sanjin2024 sudoku-mlp all_config.yaml.
# Independent of any YAML so this verifier doesn't depend on the sudoku config
# rewrite landing first.
MODEL_CONFIG = {
    "batch_size": 64,
    "seq_len": 81,
    "vocab_size": 11,
    "num_task_types": 2,
    "task_emb_ndim": 512,
    "task_emb_len": 16,
    "hidden_size": 512,
    "expansion": 1536 / 512,   # = 3.0; gives mlp ff_hidden=1536 via the simple formula
    "num_heads": 8,
    "L_layers": 2,
    "H_cycles": 3,
    "L_cycles": 6,             # paper-faithful (Sanjin all_config.yaml)
    "halt_max_steps": 16,
    "halt_exploration_prob": 0.1,
    "no_ACT_continue": True,   # paper-faithful
    "forward_dtype": "bfloat16",
    "mlp_t": True,             # the headline 84.80% variant
}


def main() -> int:
    if not os.path.isfile(REMAP_PATH):
        print(f"ERROR: remapped checkpoint not found at {REMAP_PATH}", file=sys.stderr)
        print(f"  Run scripts/remap_sudoku_mlp.py first.", file=sys.stderr)
        return 2

    print("Constructing TRMOfficial(mlp_t=True) ...")
    model = TRMOfficial(MODEL_CONFIG)
    print(f"  param count: {model.param_count():,}")

    # Snapshot a few parameter values BEFORE the load so we can detect that the
    # load actually changed them. We pick params from across the model: an
    # mlp_t weight (the new code path), an embed (now transfers), and the
    # q_head bias (special init to -5, definitely not in any clean transfer).
    pre_load = {
        "mlp_t.w1": model.inner.L_level.layers[0].mlp_t.w1.weight[0, 0].item(),
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

    # The MLP variant was trained without RoPE — only rotary_emb buffers are
    # legitimately allowed to be missing. Anything else is a remap bug.
    expected_missing_substrings = ("rotary_emb",)
    unexpected_missing = [
        k for k in missing
        if not any(s in k for s in expected_missing_substrings)
    ]
    if unexpected_missing:
        problems.append(
            f"{len(unexpected_missing)} unexpectedly-missing keys "
            f"(should have been in the remap but weren't): {unexpected_missing}"
        )

    # Verify the load actually changed parameter values — catches a bug where
    # the load reports "0 unexpected, 0 unexpected_missing" but silently never
    # assigned anything (e.g., dtype conversion issue, deepcopy issue).
    post_load = {
        "mlp_t.w1": model.inner.L_level.layers[0].mlp_t.w1.weight[0, 0].item(),
        "embed_tokens": model.inner.embed_tokens.embedding.weight[0, 0].item(),
        "q_head_bias": model.inner.q_head.linear.bias[0].item(),
    }
    print("\nWeight change check (pre -> post load):")
    for k in pre_load:
        changed = abs(pre_load[k] - post_load[k]) > 1e-9
        marker = "[ok]" if changed else "[FAIL]"
        print(f"  {marker} {k:<20} {pre_load[k]:+.6f}  ->  {post_load[k]:+.6f}")
        if not changed:
            problems.append(f"{k} did not change after load — values may not have transferred")

    # Verify task_emb row 0 transferred (sudoku) and row 1 stayed at zero (maze).
    task_emb = model.inner.task_emb.embedding.weight
    row0_norm = task_emb[0].abs().sum().item()
    row1_norm = task_emb[1].abs().sum().item()
    print(f"\ntask_emb row sanity (row 0=sudoku, row 1=maze):")
    print(f"  row 0 |abs|.sum: {row0_norm:.4f}  (expected: > 0, transferred from puzzle_emb)")
    print(f"  row 1 |abs|.sum: {row1_norm:.4f}  (expected: 0, zero-padded for maze)")
    if row0_norm <= 0:
        problems.append("task_emb row 0 (sudoku) is all-zero — puzzle_emb did not transfer")
    if row1_norm != 0:
        problems.append(f"task_emb row 1 (maze) is non-zero ({row1_norm:.4f}) — should be zero-padded")

    print()
    if problems:
        print("FAIL:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PASS: All transferred keys loaded cleanly. Only rotary_emb buffers missing (by design).")
    print("      task_emb correctly transferred sudoku row, zero-padded maze row.")
    print(f"      Weight change check confirmed values were actually assigned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
