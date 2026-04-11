"""Remap the Sanjin2024 Sudoku-Extreme-att checkpoint into our local TRMOfficial.

Source: hf_checkpoints/Sudoku-Extreme-att/step_21700
   Published by Sanjin2024 as a faithful reproduction of the "Less is More:
   Recursive Reasoning with Tiny Networks" (Jolicoeur-Martineau, 2025) paper's
   attention-variant ablation for Sudoku-Extreme. 77.70% test accuracy. Trained
   on 8xH200 with global_batch_size=4608 in 40 minutes. The attention variant
   (mlp_t=false, pos_encodings=rope) is the secondary Sudoku result -- the
   headline 84.80% number comes from the MLP-token-mixer variant
   (see scripts/remap_sudoku_mlp.py).

Why this script exists separately from remap_maze.py
----------------------------------------------------
Sudoku-att and maze are structurally identical at the state_dict level (both
use mlp_t=false, both have the same self_attn + SwiGLU layer structure, same
fused QKV / fused gate_up convention). The only differences:
  - vocab_size: sudoku=11 vs maze=6 (affects embed_tokens / lm_head shapes)
  - task_id row destination: sudoku writes to task_emb row 0 (configs/trm_official_sudoku.yaml
    sets task_id=0); maze writes to row 1.
A single parameterized remap script could handle both, but for consistency
with the per-source pattern (each variant gets its own remap + verifier), we
keep them separate. Refactoring is a future cleanup.

Key renames applied (identical recipe to remap_maze.py)
-------------------------------------------------------
  1. Prefix strip:           _orig_mod.model.inner.X  ->  inner.X
  2. embed_tokens key:       embed_tokens.embedding_weight  ->  embed_tokens.embedding.weight
  3. puzzle_emb -> task_emb: inner.puzzle_emb.weights (1, 512)
                               -> inner.task_emb.embedding.weight (2, 512)
                             Row 0 (sudoku) gets the reference data; row 1
                             (maze) is zero-padded.
  4. q_head dual-register:   q_head.weight  -> q_head.weight + q_head.linear.weight
                             q_head.bias    -> q_head.bias + q_head.linear.bias
  5. lm_head dual-register:  lm_head.weight -> lm_head.weight + lm_head.linear.weight
  6. self_attn fused QKV:    self_attn.qkv_proj.weight (1536, 512)
                               -> self_attn.q_proj.weight (512, 512)
                                  self_attn.k_proj.weight (512, 512)
                                  self_attn.v_proj.weight (512, 512)
  7. self_attn o_proj:       self_attn.o_proj.weight (512, 512)  -- direct copy
  8. mlp fused gate+up:      mlp.gate_up_proj.weight (3072, 512)
                               -> mlp.w1.weight (1536, 512)
                                  mlp.w3.weight (1536, 512)
  9. mlp down rename:        mlp.down_proj.weight (512, 1536)
                               -> mlp.w2.weight (512, 1536)

Run:
    python scripts/remap_sudoku_att.py
"""
from __future__ import annotations

import os
import sys

import torch

SRC = "hf_checkpoints/Sudoku-Extreme-att/step_21700"
DST = "hf_checkpoints/Sudoku-Extreme-att/remapped_for_local.pt"

# Reference architecture dims (from Sanjin2024 sudoku-att all_config.yaml).
HIDDEN_SIZE = 512
SEQ_LEN = 81
TASK_EMB_LEN = 16
SUDOKU_VOCAB = 11
MLP_FF = 1536            # regular FFN ff_hidden (Llama-rounded for hidden=512, expansion=4)
L_LAYERS = 2
NUM_TASK_TYPES_LOCAL = 2     # local packs sudoku+maze; reference is sudoku-only
SUDOKU_TASK_ROW = 0          # local task_id=0 for sudoku (see configs/trm_official_sudoku.yaml)

REF_PREFIX = "_orig_mod.model."


def _strip_prefix(k: str) -> str:
    assert k.startswith(REF_PREFIX), f"Unexpected key (no {REF_PREFIX!r} prefix): {k}"
    return k[len(REF_PREFIX):]


def remap(ref: dict) -> tuple[dict, list[str]]:
    """Produce a local-compatible state_dict from the Sanjin2024 sudoku-att dict."""
    out: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for raw_key, tensor in ref.items():
        k = _strip_prefix(raw_key)

        # --- Latent initial buffers ---
        if k in ("inner.H_init", "inner.L_init"):
            out[k] = tensor.clone()
            continue

        # --- embed_tokens: rename @property alias ---
        if k == "inner.embed_tokens.embedding_weight":
            assert tensor.shape == (SUDOKU_VOCAB, HIDDEN_SIZE), (
                f"Expected embed_tokens shape ({SUDOKU_VOCAB}, {HIDDEN_SIZE}), got {tuple(tensor.shape)}"
            )
            out["inner.embed_tokens.embedding.weight"] = tensor.clone()
            continue

        # --- lm_head: dual-register for CastedLinear ---
        if k == "inner.lm_head.weight":
            assert tensor.shape == (SUDOKU_VOCAB, HIDDEN_SIZE)
            out["inner.lm_head.weight"] = tensor.clone()
            out["inner.lm_head.linear.weight"] = tensor.clone()
            continue

        # --- q_head dual-register ---
        if k == "inner.q_head.weight":
            assert tensor.shape == (2, HIDDEN_SIZE)
            out["inner.q_head.weight"] = tensor.clone()
            out["inner.q_head.linear.weight"] = tensor.clone()
            continue
        if k == "inner.q_head.bias":
            assert tensor.shape == (2,)
            out["inner.q_head.bias"] = tensor.clone()
            out["inner.q_head.linear.bias"] = tensor.clone()
            continue

        # --- puzzle_emb -> task_emb (with row-0 placement for sudoku) ---
        if k == "inner.puzzle_emb.weights":
            assert tensor.shape == (1, HIDDEN_SIZE), (
                f"Expected puzzle_emb shape (1, {HIDDEN_SIZE}), got {tuple(tensor.shape)}"
            )
            expanded = torch.zeros(NUM_TASK_TYPES_LOCAL, HIDDEN_SIZE, dtype=tensor.dtype)
            expanded[SUDOKU_TASK_ROW] = tensor[0]
            out["inner.task_emb.embedding.weight"] = expanded
            continue

        # --- L_level reasoning layers: attention + SwiGLU ---
        handled = False
        for layer_i in range(L_LAYERS):
            attn_prefix = f"inner.L_level.layers.{layer_i}.self_attn."
            mlp_prefix = f"inner.L_level.layers.{layer_i}.mlp."

            if k == attn_prefix + "qkv_proj.weight":
                assert tensor.shape == (3 * HIDDEN_SIZE, HIDDEN_SIZE), (
                    f"Expected qkv_proj shape ({3*HIDDEN_SIZE}, {HIDDEN_SIZE}), "
                    f"got {tuple(tensor.shape)}"
                )
                out[attn_prefix + "q_proj.weight"] = tensor[0:HIDDEN_SIZE].clone()
                out[attn_prefix + "k_proj.weight"] = tensor[HIDDEN_SIZE:2*HIDDEN_SIZE].clone()
                out[attn_prefix + "v_proj.weight"] = tensor[2*HIDDEN_SIZE:3*HIDDEN_SIZE].clone()
                handled = True
                break

            if k == attn_prefix + "o_proj.weight":
                assert tensor.shape == (HIDDEN_SIZE, HIDDEN_SIZE)
                out[k] = tensor.clone()
                handled = True
                break

            if k == mlp_prefix + "gate_up_proj.weight":
                assert tensor.shape == (2 * MLP_FF, HIDDEN_SIZE), (
                    f"Expected mlp gate_up_proj shape ({2*MLP_FF}, {HIDDEN_SIZE}), "
                    f"got {tuple(tensor.shape)}"
                )
                out[mlp_prefix + "w1.weight"] = tensor[0:MLP_FF].clone()
                out[mlp_prefix + "w3.weight"] = tensor[MLP_FF:2*MLP_FF].clone()
                handled = True
                break

            if k == mlp_prefix + "down_proj.weight":
                assert tensor.shape == (HIDDEN_SIZE, MLP_FF), (
                    f"Expected mlp down_proj shape ({HIDDEN_SIZE}, {MLP_FF}), "
                    f"got {tuple(tensor.shape)}"
                )
                out[mlp_prefix + "w2.weight"] = tensor.clone()
                handled = True
                break

        if not handled:
            skipped.append(f"{k}  [unhandled -- not recognized]")

    return out, skipped


def main() -> int:
    if not os.path.isfile(SRC):
        print(f"ERROR: source checkpoint not found at {SRC}", file=sys.stderr)
        print(
            f"  Run the fetch step first:\n"
            f"    python -c \"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download(repo_id='Sanjin2024/TinyRecursiveModels-Sudoku-Extreme-att', "
            f"filename='step_21700', local_dir='hf_checkpoints/Sudoku-Extreme-att')\"",
            file=sys.stderr,
        )
        return 2

    print(f"Loading {SRC} (CPU) ...", flush=True)
    ref = torch.load(SRC, map_location="cpu", weights_only=True)
    print(f"  Loaded {len(ref)} tensors, {sum(v.numel() for v in ref.values()):,} params")

    remapped, skipped = remap(ref)

    print("\n=== Transfer report ===")
    print(f"\nTransferred ({len(remapped)} keys, "
          f"{sum(v.numel() for v in remapped.values()):,} elements):")
    for k, v in remapped.items():
        print(f"  {k:<70} {tuple(v.shape)}  {v.dtype}")

    if skipped:
        print(f"\nSkipped ({len(skipped)} keys) -- should be ZERO:")
        for s in skipped:
            print(f"  {s}")

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    torch.save(
        {
            "model_state_dict": remapped,
            "source": SRC,
            "note": "Partial state_dict remapped from Sanjin2024 Sudoku-Extreme-att "
                    "(77.70% test acc, attention variant of the Less-is-More Sudoku result; "
                    "the headline 84.80% number comes from the MLP variant). "
                    "Load with --init-weights (not --resume). Optimizer/EMA/global_step "
                    "intentionally absent. The maze row of task_emb is zero-padded "
                    "since the reference was sudoku-only.",
        },
        DST,
    )
    print(f"\nWrote remapped state_dict to {DST}")
    print(f"Size on disk: ~{sum(v.numel() * v.element_size() for v in remapped.values()) / 1e6:.1f} MB")

    del ref
    del remapped
    return 0


if __name__ == "__main__":
    sys.exit(main())
