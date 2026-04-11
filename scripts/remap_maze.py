"""Remap the Sanjin2024 Maze-Hard checkpoint into our local TRMOfficial.

Source: hf_checkpoints/Maze-Hard/step_9765
   Published by Sanjin2024 as a faithful reproduction of the "Less is More:
   Recursive Reasoning with Tiny Networks" (Jolicoeur-Martineau, 2025) paper's
   Maze-Hard 30x30 result. 78.70% test accuracy. Trained on 8xH200 with
   global_batch_size=4608 in 2h. This is the attention variant (mlp_t=false,
   pos_encodings=rope) -- the only published variant for maze, since
   self-attention beats MLP-token-mixing on the larger 900-token maze grid.

Why this script exists separately from remap_hf_checkpoint.py
-------------------------------------------------------------
The ARC remapper drops embed_tokens / lm_head / puzzle_emb because the ARC vocab
(12 colors) and 1.19M-puzzle table are semantically unrelated. This checkpoint
was trained directly on Maze-Hard with vocab=6, num_puzzle_identifiers=1, so we
TRANSFER embed_tokens / lm_head / q_head / puzzle_emb / both attention and
SwiGLU layers.

Compared to the sudoku-mlp remap (scripts/remap_sudoku_mlp.py), the only
structural difference is the L_level layer contents:
  - sudoku-mlp:  mlp_t (token-mixer SwiGLU) + mlp (regular SwiGLU)
  - maze:        self_attn (q/k/v/o projections) + mlp (regular SwiGLU)
Everything else (prefix strip, embed/lm_head/q_head/puzzle_emb handling) is
identical.

Key renames applied
-------------------
  1. Prefix strip:           _orig_mod.model.inner.X  ->  inner.X
  2. embed_tokens key:       embed_tokens.embedding_weight  ->  embed_tokens.embedding.weight
  3. puzzle_emb -> task_emb: inner.puzzle_emb.weights (1, 512)
                               -> inner.task_emb.embedding.weight (2, 512)
                             ZERO-PADS row 0 (sudoku) for a maze-only training
                             that uses task_id=1; ZERO-PADS row 1... wait, this
                             checkpoint trained the maze task. The local maze
                             config has task_id=1 (collate flag). So this
                             checkpoint's puzzle_emb data should land in row 1
                             (the maze row), not row 0.
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

The local model's rotary_emb buffers (inv_freq, cos_cached, sin_cached) are
NOT in the reference checkpoint -- they're deterministic buffers reconstructed
from RotaryEmbedding.__init__ on every model load, so they don't need to be
saved. They will appear as "missing" in the strict=False load report, which
is expected and correct.

Run:
    python scripts/remap_maze.py
"""
from __future__ import annotations

import os
import sys

import torch

SRC = "hf_checkpoints/Maze-Hard/step_9765"
DST = "hf_checkpoints/Maze-Hard/remapped_for_local.pt"

# Reference architecture dims (from Sanjin2024 maze all_config.yaml).
HIDDEN_SIZE = 512
SEQ_LEN = 900
TASK_EMB_LEN = 16
MAZE_VOCAB = 6
MLP_FF = 1536            # regular FFN ff_hidden (Llama-rounded for hidden=512, expansion=4)
L_LAYERS = 2
NUM_TASK_TYPES_LOCAL = 2     # local packs sudoku+maze; reference is maze-only
MAZE_TASK_ROW = 1            # local task_id=1 for maze (see configs/trm_official_maze.yaml)

REF_PREFIX = "_orig_mod.model."


def _strip_prefix(k: str) -> str:
    assert k.startswith(REF_PREFIX), f"Unexpected key (no {REF_PREFIX!r} prefix): {k}"
    return k[len(REF_PREFIX):]


def remap(ref: dict) -> tuple[dict, list[str]]:
    """Produce a local-compatible state_dict from the Sanjin2024 maze dict."""
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
            assert tensor.shape == (MAZE_VOCAB, HIDDEN_SIZE), (
                f"Expected embed_tokens shape ({MAZE_VOCAB}, {HIDDEN_SIZE}), got {tuple(tensor.shape)}"
            )
            out["inner.embed_tokens.embedding.weight"] = tensor.clone()
            continue

        # --- lm_head: dual-register for CastedLinear ---
        if k == "inner.lm_head.weight":
            assert tensor.shape == (MAZE_VOCAB, HIDDEN_SIZE)
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

        # --- puzzle_emb -> task_emb (with row-1 placement for maze) ---
        # The reference was maze-only with one puzzle ID. Local task_emb has
        # 2 rows: row 0 = sudoku (zero-init here, since this is a maze
        # checkpoint), row 1 = maze (transfers from reference). The maze
        # config sets task_id=1 (see configs/trm_official_maze.yaml), so the
        # collate function feeds task_id=1 into task_emb, picking row 1.
        if k == "inner.puzzle_emb.weights":
            assert tensor.shape == (1, HIDDEN_SIZE), (
                f"Expected puzzle_emb shape (1, {HIDDEN_SIZE}), got {tuple(tensor.shape)}"
            )
            expanded = torch.zeros(NUM_TASK_TYPES_LOCAL, HIDDEN_SIZE, dtype=tensor.dtype)
            expanded[MAZE_TASK_ROW] = tensor[0]
            out["inner.task_emb.embedding.weight"] = expanded
            continue

        # --- L_level reasoning layers: attention + SwiGLU ---
        handled = False
        for layer_i in range(L_LAYERS):
            attn_prefix = f"inner.L_level.layers.{layer_i}.self_attn."
            mlp_prefix = f"inner.L_level.layers.{layer_i}.mlp."

            # Fused QKV -> split q/k/v (each 512 x 512)
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

            # o_proj direct copy
            if k == attn_prefix + "o_proj.weight":
                assert tensor.shape == (HIDDEN_SIZE, HIDDEN_SIZE)
                out[k] = tensor.clone()
                handled = True
                break

            # Regular mlp gate+up split
            if k == mlp_prefix + "gate_up_proj.weight":
                assert tensor.shape == (2 * MLP_FF, HIDDEN_SIZE), (
                    f"Expected mlp gate_up_proj shape ({2*MLP_FF}, {HIDDEN_SIZE}), "
                    f"got {tuple(tensor.shape)}"
                )
                out[mlp_prefix + "w1.weight"] = tensor[0:MLP_FF].clone()
                out[mlp_prefix + "w3.weight"] = tensor[MLP_FF:2*MLP_FF].clone()
                handled = True
                break

            # mlp down rename
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
            f"hf_hub_download(repo_id='Sanjin2024/TinyRecursiveModel-Maze-Hard', "
            f"filename='step_9765', local_dir='hf_checkpoints/Maze-Hard')\"",
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
            "note": "Partial state_dict remapped from Sanjin2024 Maze-Hard "
                    "(78.70% test acc reproduction of the Less-is-More paper). "
                    "Load with --init-weights (not --resume). Optimizer/EMA/global_step "
                    "intentionally absent. The sudoku row of task_emb is zero-padded "
                    "since the reference was maze-only.",
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
