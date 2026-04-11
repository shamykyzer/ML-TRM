"""Remap the HuggingFace reference TRM checkpoint into our local TRMOfficial.

Source: hf_checkpoints/ARC/step_723914
   Published by the "Less is More: Recursive Reasoning with Tiny Networks"
   (Jolicoeur-Martineau, 2025) reproduction repo. Trained on ARC-AGI-2 via
   torchrun on 8xH100. Raw state_dict, ~617M params (dominated by a 1.19M-entry
   puzzle_emb table that is irrelevant for sudoku/maze).

What transfers (the 90% that matters):
  - L_level reasoning layers (self_attn + SwiGLU) for both layers 0 and 1
  - H_init / L_init initial latent buffers
  - q_head (halting Q-learning head)

What does NOT transfer (intentionally reinitialized):
  - puzzle_emb: ARC has 1.19M augmented puzzle IDs; local uses 2 task types.
    Completely different concept, no meaningful mapping.
  - embed_tokens: ARC's 12-token vocab is colors/symbols; sudoku's 11-token
    vocab is digits+blank. Semantically disjoint — loading ARC weights would
    actively poison training.
  - lm_head: same argument as embed_tokens.

Key renames applied:
  1. Prefix strip:     _orig_mod.model.inner.X  ->  inner.X
     (reference was saved through torch.compile + a loss-head wrapper)
  2. QKV split:        self_attn.qkv_proj.weight (1536,512)
                         -> self_attn.{q,k,v}_proj.weight (512,512) each
     Local Attention uses separate q/k/v projections; reference used fused.
  3. SwiGLU gate/up split:   mlp.gate_up_proj.weight (3072,512)
                               -> mlp.w1.weight (gate, 1536,512)
                                  mlp.w3.weight (up,   1536,512)
     Reference fuses gate+up; local keeps them separate as w1/w3.
  4. SwiGLU down rename:     mlp.down_proj.weight (512,1536)
                               -> mlp.w2.weight (512,1536)
  5. CastedLinear wrapping:  q_head.weight -> q_head.linear.weight
                             q_head.bias   -> q_head.linear.bias
     Local CastedLinear wraps nn.Linear as self.linear (src/models/layers_official.py:87-98).

The output file is a partial state_dict saved as a dict {"model_state_dict": ...}.
Load it via `python main.py --config ... --init-weights hf_checkpoints/ARC/remapped_for_local.pt`.
The trainer loads with strict=False so the intentionally-missing embed/task_emb/lm_head
keys stay at their fresh random init.

Run:
    python scripts/remap_hf_checkpoint.py
"""
from __future__ import annotations

import os
import sys

import torch

SRC = "hf_checkpoints/ARC/step_723914"
DST = "hf_checkpoints/ARC/remapped_for_local.pt"

# Paper-faithful dims (must match the YAML we just updated)
HIDDEN_SIZE = 512
FF_HIDDEN = 1536     # from all_config.yaml's expansion=4 via Llama formula
NUM_HEADS = 8
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS   # 64
L_LAYERS = 2

# The reference state_dict has this prefix on every key (torch.compile + wrapper)
REF_PREFIX = "_orig_mod.model."


def _strip_prefix(k: str) -> str:
    assert k.startswith(REF_PREFIX), f"Unexpected key (no {REF_PREFIX!r} prefix): {k}"
    return k[len(REF_PREFIX):]


def remap(ref: dict) -> tuple[dict, list[str]]:
    """Produce a local-compatible partial state_dict from the reference dict.

    Returns (local_state_dict, skipped_keys).
    """
    out: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for raw_key, tensor in ref.items():
        k = _strip_prefix(raw_key)

        # --- Keys we deliberately drop (semantic mismatch with sudoku/maze) ---
        if k == "inner.puzzle_emb.weights":
            skipped.append(f"{k}  [1.19M ARC puzzle IDs, irrelevant for 2-task-type local]")
            continue
        if k == "inner.embed_tokens.embedding_weight":
            skipped.append(f"{k}  [ARC vocab (12) != sudoku vocab (11), semantics differ]")
            continue
        if k == "inner.lm_head.weight":
            skipped.append(f"{k}  [same as embed_tokens — semantic mismatch]")
            continue

        # --- Latent initial buffers: direct copy ---
        if k in ("inner.H_init", "inner.L_init"):
            out[k] = tensor.clone()
            continue

        # --- q_head: CastedLinear wrapper drift (populate BOTH aliases) ---
        # CastedLinear (src/models/layers_official.py:92-95) does
        #   self.weight = self.linear.weight
        # which registers the same Parameter under two state_dict keys:
        # q_head.weight AND q_head.linear.weight. Both must be in the
        # checkpoint or load_state_dict(strict=False) will cosmetically
        # flag the unpopulated one as "missing" even though they share
        # storage. Populating both keeps the load report clean.
        if k == "inner.q_head.weight":
            out["inner.q_head.weight"] = tensor.clone()
            out["inner.q_head.linear.weight"] = tensor.clone()
            continue
        if k == "inner.q_head.bias":
            out["inner.q_head.bias"] = tensor.clone()
            out["inner.q_head.linear.bias"] = tensor.clone()
            continue

        # --- L_level reasoning layers: attention + SwiGLU ---
        # These match the pattern: inner.L_level.layers.{i}.{self_attn|mlp}.*
        for layer_i in range(L_LAYERS):
            attn_prefix = f"inner.L_level.layers.{layer_i}.self_attn."
            mlp_prefix = f"inner.L_level.layers.{layer_i}.mlp."

            # Fused QKV -> split q/k/v
            if k == attn_prefix + "qkv_proj.weight":
                assert tensor.shape == (3 * HIDDEN_SIZE, HIDDEN_SIZE), (
                    f"Expected qkv_proj shape ({3*HIDDEN_SIZE}, {HIDDEN_SIZE}), got {tuple(tensor.shape)}"
                )
                out[attn_prefix + "q_proj.weight"] = tensor[0:HIDDEN_SIZE].clone()
                out[attn_prefix + "k_proj.weight"] = tensor[HIDDEN_SIZE:2*HIDDEN_SIZE].clone()
                out[attn_prefix + "v_proj.weight"] = tensor[2*HIDDEN_SIZE:3*HIDDEN_SIZE].clone()
                break

            # o_proj: same shape, same name, direct copy
            if k == attn_prefix + "o_proj.weight":
                assert tensor.shape == (HIDDEN_SIZE, HIDDEN_SIZE)
                out[k] = tensor.clone()
                break

            # Fused gate+up -> split into w1 (gate) and w3 (up)
            # Convention: SwiGLU(x) = down(silu(gate(x)) * up(x))
            # Local forward (layers_official.py:143): w2(silu(w1(x)) * w3(x))
            # So: w1 = gate, w3 = up, w2 = down
            # Fused layout: gate_up_proj = cat([gate_weight, up_weight], dim=0)
            if k == mlp_prefix + "gate_up_proj.weight":
                assert tensor.shape == (2 * FF_HIDDEN, HIDDEN_SIZE), (
                    f"Expected gate_up_proj shape ({2*FF_HIDDEN}, {HIDDEN_SIZE}), got {tuple(tensor.shape)}"
                )
                out[mlp_prefix + "w1.weight"] = tensor[0:FF_HIDDEN].clone()         # gate
                out[mlp_prefix + "w3.weight"] = tensor[FF_HIDDEN:2*FF_HIDDEN].clone()  # up
                break

            # down_proj -> w2
            if k == mlp_prefix + "down_proj.weight":
                assert tensor.shape == (HIDDEN_SIZE, FF_HIDDEN)
                out[mlp_prefix + "w2.weight"] = tensor.clone()
                break
        else:
            # Key didn't match any L_level pattern and wasn't handled above.
            skipped.append(f"{k}  [unhandled — not recognized]")

    return out, skipped


def main() -> int:
    if not os.path.isfile(SRC):
        print(f"ERROR: source checkpoint not found at {SRC}", file=sys.stderr)
        return 2

    print(f"Loading {SRC} (CPU) ...", flush=True)
    ref = torch.load(SRC, map_location="cpu", weights_only=True)
    print(f"  Loaded {len(ref)} tensors, {sum(v.numel() for v in ref.values()):,} params")

    remapped, skipped = remap(ref)

    print("\n=== Transfer report ===")
    print(f"\nTransferred ({len(remapped)} keys, {sum(v.numel() for v in remapped.values()):,} params):")
    for k, v in remapped.items():
        print(f"  {k:<70} {tuple(v.shape)}")

    print(f"\nSkipped ({len(skipped)} keys) — will be reinitialized fresh in the new model:")
    for s in skipped:
        print(f"  {s}")

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    torch.save(
        {
            "model_state_dict": remapped,
            "source": SRC,
            "note": "Partial state_dict remapped from ARC-AGI-2 reference checkpoint. "
                    "Load with --init-weights (not --resume). Optimizer/EMA/global_step "
                    "intentionally absent — training starts fresh with transferred weights "
                    "as initialization.",
        },
        DST,
    )
    print(f"\nWrote remapped state_dict to {DST}")
    print(f"Size on disk: roughly {sum(v.numel() * v.element_size() for v in remapped.values()) / 1e6:.1f} MB")

    del ref
    del remapped
    return 0


if __name__ == "__main__":
    sys.exit(main())
