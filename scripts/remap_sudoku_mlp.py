"""Remap the Sanjin2024 Sudoku-Extreme-mlp checkpoint into our local TRMOfficial.

Source: hf_checkpoints/Sudoku-Extreme-mlp/step_16275
   Published by Sanjin2024 as a faithful reproduction of the "Less is More:
   Recursive Reasoning with Tiny Networks" (Jolicoeur-Martineau, 2025) paper's
   primary Sudoku-Extreme result. 84.80% test accuracy. Trained on 8×H200 with
   global_batch_size=4608 in 40 minutes. This is the MLP-token-mixer variant
   (mlp_t=true, pos_encodings=none) — the headline 84.80% number, NOT the 77.70%
   attention ablation.

Why this script exists separately from remap_hf_checkpoint.py
-------------------------------------------------------------
The ARC remapper drops embed_tokens / lm_head / puzzle_emb because the ARC vocab
(12 colors) and 1.19M-puzzle table are semantically unrelated to sudoku. This
checkpoint is the OPPOSITE situation: it was trained directly on Sudoku-Extreme
with vocab=11, num_puzzle_identifiers=1, mlp_t=true. So we want to TRANSFER
embed_tokens / lm_head / q_head / puzzle_emb / both SwiGLUs (regular FFN AND
token-mixer). Only the rotary_emb buffer is intentionally absent (the MLP
variant was trained with pos_encodings: none — the local model still constructs
rotary_emb buffers for code-path compatibility, but the mlp_t branch in
TRMBlock.forward never reads them, so leaving them at fresh-init is harmless).

Key renames applied
-------------------
  1. Prefix strip:           _orig_mod.model.inner.X  ->  inner.X
  2. embed_tokens key:       embed_tokens.embedding_weight  ->  embed_tokens.embedding.weight
                             (reference uses CastedEmbedding's @property attribute name;
                              local stores the underlying nn.Embedding submodule)
  3. puzzle_emb -> task_emb: inner.puzzle_emb.weights (1, 512)
                               -> inner.task_emb.embedding.weight (2, 512)
                             ZERO-PADS row 1 (the maze task slot). The reference
                             was sudoku-only (1 puzzle ID); our local model packs
                             sudoku+maze (num_task_types=2). Row 0 transfers; row 1
                             stays at zero (which is also the local fresh-init —
                             see trm_official.py:177).
  4. q_head dual-register:   q_head.weight  -> q_head.weight + q_head.linear.weight
                             q_head.bias    -> q_head.bias + q_head.linear.bias
  5. lm_head dual-register:  lm_head.weight -> lm_head.weight + lm_head.linear.weight
                             (CastedLinear assigns self.weight = self.linear.weight,
                              which Module.__setattr__ auto-registers as a Parameter
                              under the alias name — both keys must be present in
                              the checkpoint or strict=False will report the alias
                              as "missing" cosmetically.)
  6. mlp_t fused gate+up:    mlp_t.gate_up_proj.weight (1024, 97)
                               -> mlp_t.w1.weight (gate, 512, 97)
                                  mlp_t.w3.weight (up,   512, 97)
  7. mlp_t down rename:      mlp_t.down_proj.weight (97, 512)
                               -> mlp_t.w2.weight (97, 512)
  8. mlp fused gate+up:      mlp.gate_up_proj.weight (3072, 512)
                               -> mlp.w1.weight (1536, 512)
                                  mlp.w3.weight (1536, 512)
  9. mlp down rename:        mlp.down_proj.weight (512, 1536)
                               -> mlp.w2.weight (512, 1536)

Shape compatibility for the mlp_t (token-mixer) branch depends on the
src/models/layers_official.py SwiGLU patch + src/models/trm_official.py
TRMBlock patch (llama_rounded_ff helper) — without those, the local model
would construct mlp_t with shape (388, 97) instead of (512, 97). See the
TRMBlock.__init__ comment for details.

Run:
    python scripts/remap_sudoku_mlp.py
"""
from __future__ import annotations

import os
import sys

import torch

SRC = "hf_checkpoints/Sudoku-Extreme-mlp/step_16275"
DST = "hf_checkpoints/Sudoku-Extreme-mlp/remapped_for_local.pt"

# Reference architecture dims (from Sanjin2024 all_config.yaml: hidden_size=512,
# num_heads=8, mlp expansion via Llama formula → 1536; mlp_t expansion via Llama
# formula on hidden=97 → 512). These are the shapes IN the reference checkpoint
# — we use them as assertion checks while remapping, so a mismatched source
# fails loudly instead of producing silently-broken weights.
HIDDEN_SIZE = 512
SEQ_LEN = 81
TASK_EMB_LEN = 16
MLP_T_HIDDEN = SEQ_LEN + TASK_EMB_LEN     # 97
MLP_FF = 1536                              # regular FFN ff_hidden
MLP_T_FF = 512                             # token-mixer ff_hidden (Llama-rounded)
L_LAYERS = 2
NUM_TASK_TYPES_LOCAL = 2                   # local packs sudoku+maze; reference is sudoku-only

REF_PREFIX = "_orig_mod.model."


def _strip_prefix(k: str) -> str:
    assert k.startswith(REF_PREFIX), f"Unexpected key (no {REF_PREFIX!r} prefix): {k}"
    return k[len(REF_PREFIX):]


def remap(ref: dict) -> tuple[dict, list[str]]:
    """Produce a local-compatible state_dict from the Sanjin2024 sudoku-mlp dict.

    Returns (local_state_dict, skipped_keys). The returned dict is intended to
    be loaded with strict=False — only the rotary_emb buffers will be missing,
    and that's by design (the MLP variant was trained without RoPE).
    """
    out: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for raw_key, tensor in ref.items():
        k = _strip_prefix(raw_key)

        # --- Latent initial buffers: direct copy (registered as buffers locally) ---
        if k in ("inner.H_init", "inner.L_init"):
            out[k] = tensor.clone()
            continue

        # --- embed_tokens: rename @property alias to submodule.weight ---
        if k == "inner.embed_tokens.embedding_weight":
            assert tensor.shape == (11, HIDDEN_SIZE), (
                f"Expected embed_tokens shape (11, {HIDDEN_SIZE}), got {tuple(tensor.shape)}"
            )
            out["inner.embed_tokens.embedding.weight"] = tensor.clone()
            continue

        # --- lm_head: dual-register for CastedLinear's weight aliasing ---
        if k == "inner.lm_head.weight":
            assert tensor.shape == (11, HIDDEN_SIZE)
            out["inner.lm_head.weight"] = tensor.clone()
            out["inner.lm_head.linear.weight"] = tensor.clone()
            continue

        # --- q_head: dual-register weight + bias ---
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

        # --- puzzle_emb -> task_emb (with shape expansion 1 -> num_task_types) ---
        # Reference was sudoku-only with one puzzle ID. Local packs sudoku+maze,
        # so task_emb has 2 rows. Row 0 = sudoku (transfers from reference);
        # row 1 = maze (zero-init, matching trm_official.py:177's fresh-init).
        if k == "inner.puzzle_emb.weights":
            assert tensor.shape == (1, HIDDEN_SIZE), (
                f"Expected puzzle_emb shape (1, {HIDDEN_SIZE}), got {tuple(tensor.shape)}"
            )
            expanded = torch.zeros(NUM_TASK_TYPES_LOCAL, HIDDEN_SIZE, dtype=tensor.dtype)
            expanded[0] = tensor[0]
            out["inner.task_emb.embedding.weight"] = expanded
            continue

        # --- L_level reasoning layers ---
        handled = False
        for layer_i in range(L_LAYERS):
            mlp_t_prefix = f"inner.L_level.layers.{layer_i}.mlp_t."
            mlp_prefix = f"inner.L_level.layers.{layer_i}.mlp."

            # mlp_t fused gate+up -> w1 (gate) + w3 (up)
            if k == mlp_t_prefix + "gate_up_proj.weight":
                assert tensor.shape == (2 * MLP_T_FF, MLP_T_HIDDEN), (
                    f"Expected mlp_t gate_up_proj shape ({2*MLP_T_FF}, {MLP_T_HIDDEN}), "
                    f"got {tuple(tensor.shape)}"
                )
                out[mlp_t_prefix + "w1.weight"] = tensor[0:MLP_T_FF].clone()
                out[mlp_t_prefix + "w3.weight"] = tensor[MLP_T_FF:2*MLP_T_FF].clone()
                handled = True
                break

            # mlp_t down -> w2 (rename only)
            if k == mlp_t_prefix + "down_proj.weight":
                assert tensor.shape == (MLP_T_HIDDEN, MLP_T_FF), (
                    f"Expected mlp_t down_proj shape ({MLP_T_HIDDEN}, {MLP_T_FF}), "
                    f"got {tuple(tensor.shape)}"
                )
                out[mlp_t_prefix + "w2.weight"] = tensor.clone()
                handled = True
                break

            # Regular mlp fused gate+up -> w1 + w3
            if k == mlp_prefix + "gate_up_proj.weight":
                assert tensor.shape == (2 * MLP_FF, HIDDEN_SIZE), (
                    f"Expected mlp gate_up_proj shape ({2*MLP_FF}, {HIDDEN_SIZE}), "
                    f"got {tuple(tensor.shape)}"
                )
                out[mlp_prefix + "w1.weight"] = tensor[0:MLP_FF].clone()
                out[mlp_prefix + "w3.weight"] = tensor[MLP_FF:2*MLP_FF].clone()
                handled = True
                break

            # Regular mlp down -> w2
            if k == mlp_prefix + "down_proj.weight":
                assert tensor.shape == (HIDDEN_SIZE, MLP_FF), (
                    f"Expected mlp down_proj shape ({HIDDEN_SIZE}, {MLP_FF}), "
                    f"got {tuple(tensor.shape)}"
                )
                out[mlp_prefix + "w2.weight"] = tensor.clone()
                handled = True
                break

        if not handled:
            skipped.append(f"{k}  [unhandled — not recognized]")

    return out, skipped


def main() -> int:
    if not os.path.isfile(SRC):
        print(f"ERROR: source checkpoint not found at {SRC}", file=sys.stderr)
        print(
            f"  Run the fetch step first (downloads via huggingface_hub):\n"
            f"    python -c \"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download(repo_id='Sanjin2024/TinyRecursiveModels-Sudoku-Extreme-mlp', "
            f"filename='step_16275', local_dir='hf_checkpoints/Sudoku-Extreme-mlp')\"",
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
        print(f"\nSkipped ({len(skipped)} keys) — should be ZERO for this variant:")
        for s in skipped:
            print(f"  {s}")

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    torch.save(
        {
            "model_state_dict": remapped,
            "source": SRC,
            "note": "Partial state_dict remapped from Sanjin2024 Sudoku-Extreme-mlp "
                    "(84.80% test acc reproduction of the Less-is-More paper's "
                    "primary Sudoku result). Load with --init-weights (not --resume). "
                    "Optimizer/EMA/global_step intentionally absent — training starts "
                    "fresh with these weights as initialization. The maze row of "
                    "task_emb is zero-padded since the reference was sudoku-only.",
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
