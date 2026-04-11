"""One-shot diagnostic: load the HF reference checkpoint to CPU, print the
state_dict key structure + shapes, then release memory. Used to verify
whether the reference weights can be loaded into the local TRMOfficial.

Run: python scripts/inspect_hf_checkpoint.py
"""
import sys
import torch

CKPT = "hf_checkpoints/ARC/step_723914"

print(f"Loading {CKPT} (CPU, weights_only=True) ...", flush=True)
try:
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
except Exception as e:
    print(f"weights_only=True failed ({e}), retrying weights_only=False", flush=True)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)

print(f"Top-level type: {type(state).__name__}")
if isinstance(state, dict):
    top_keys = list(state.keys())
    print(f"Top-level keys ({len(top_keys)}): {top_keys[:10]}{' ...' if len(top_keys) > 10 else ''}")

    # If this is a wrapped checkpoint (e.g. {'model': ..., 'optimizer': ...}),
    # try to drill into the state_dict-looking child.
    candidate = state
    if all(isinstance(v, torch.Tensor) for v in state.values()):
        candidate = state
        print("Looks like a raw state_dict (all values are tensors).")
    else:
        for k in ("model", "model_state_dict", "state_dict", "module"):
            if k in state and isinstance(state[k], dict):
                print(f"Drilling into state['{k}']")
                candidate = state[k]
                break

    if isinstance(candidate, dict):
        tensor_items = [(k, v) for k, v in candidate.items() if isinstance(v, torch.Tensor)]
        print(f"\nTotal tensor entries: {len(tensor_items)}")
        print(f"Total params: {sum(v.numel() for _, v in tensor_items):,}")
        print("\n=== All keys with shapes ===")
        for k, v in tensor_items:
            print(f"  {k:<70} {tuple(v.shape)}  {v.dtype}")
else:
    print(f"Unexpected top-level type: {type(state)}")
    sys.exit(1)

# Free memory immediately
del state
print("\nDone. Memory released.")
