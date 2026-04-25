"""weave.Model wrappers around TRM Sudoku checkpoints.

Used by:
  - scripts/weave_compare_checkpoints.py — runs weave.Evaluation across
    HF-init / from-scratch / fine-tuned checkpoints on a shared sample of
    the test split, producing a leaderboard at
    https://wandb.ai/<entity>/<project>/weave/evaluations
  - the wandb Playground UI — once a TRMSudokuModel is published the first
    time (it auto-publishes on first weave.op call), users can pick it from
    the Playground model picker and test individual puzzles interactively.

Usage:
    import weave
    from src.evaluation.weave_models import TRMSudokuModel

    weave.init("TRM")
    model = TRMSudokuModel(
        name="hf-init",
        checkpoint_path="hf_checkpoints/Sudoku-Extreme-mlp/remapped_for_local.pt",
        config_path="configs/trm_official_sudoku_mlp.yaml",
    )
    out = model.predict(puzzle=[1, 1, 5, 1, ...])  # 81 tokens, stored space (1..10)
    print(out["prediction"], out["halt_step"])
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

try:
    import weave  # noqa: E402
    WEAVE_AVAILABLE = True
    _WeaveModelBase = weave.Model
    _weave_op = weave.op
except ImportError:
    WEAVE_AVAILABLE = False

    # Minimal pydantic-shaped fallback so importers don't crash when weave
    # isn't installed. Tests that don't actually need traces can still build
    # the wrapper, just without the @weave.op behaviour.
    from pydantic import BaseModel  # type: ignore[import]
    _WeaveModelBase = BaseModel  # type: ignore[misc,assignment]

    def _weave_op(fn=None, **_kwargs):  # type: ignore[no-redef]
        if fn is None:
            return lambda f: f
        return fn


from src.models.losses_official import ACTLossHead  # noqa: E402
from src.models.trm_official import TRMOfficial  # noqa: E402
from src.utils.config import load_config  # noqa: E402

IGNORE_LABEL_ID = -100


@lru_cache(maxsize=8)
def _build_and_load(
    config_path: str, checkpoint_path: str, device_str: str
) -> tuple[torch.nn.Module, ACTLossHead, dict]:
    """Build the TRM model + loss_head from config_path and load weights.

    Cached so multiple weave.Evaluation runs over the same checkpoint don't
    pay the disk-load + cuda-copy cost repeatedly. Cache key is the triple
    (config, checkpoint, device) so swapping any of them invalidates.
    """
    cfg = load_config(config_path)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    forward_dtype = getattr(torch, cfg.model.forward_dtype, torch.bfloat16)

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
        "halt_exploration_prob": 0.0,  # always off for inference
        "no_ACT_continue": cfg.model.no_ACT_continue,
        "forward_dtype": cfg.model.forward_dtype,
        "mlp_t": cfg.model.mlp_t,
    }

    model = TRMOfficial(model_config)
    loss_head = ACTLossHead(model)
    model.to(device=device, dtype=forward_dtype)
    loss_head.to(device=device, dtype=forward_dtype)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'model_state_dict'. "
            f"Keys: {list(ckpt.keys())}"
        )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    loss_head.eval()
    return model, loss_head, {
        "device": device,
        "halt_max_steps": cfg.model.halt_max_steps,
        "no_ACT_continue": cfg.model.no_ACT_continue,
        "task_id": cfg.training.task_id,
    }


class TRMSudokuModel(_WeaveModelBase):  # type: ignore[misc,valid-type]
    """A registered TRM Sudoku checkpoint, callable as a weave.Model.

    The canonical use is feeding several of these to weave.Evaluation. Each
    instance maps to one checkpoint file; multiple instances can coexist
    (e.g. one per seed / fine-tune attempt) so a single Evaluation run
    produces a side-by-side leaderboard.
    """

    name: str
    checkpoint_path: str
    config_path: str = "configs/trm_official_sudoku_mlp.yaml"
    device_str: str = "cuda"

    @_weave_op()
    def predict(self, puzzle: list[int]) -> dict:
        """Run the model on one puzzle.

        Args:
            puzzle: 81-element list of stored tokens (1=blank, 2..10=digits 1..9).

        Returns dict:
            prediction: 81-element list of stored tokens
            halt_step: 1..halt_max_steps; the first step the Q-head would have
                halted at in deployment (or halt_max_steps if it never crossed).
        """
        if len(puzzle) != 81:
            raise ValueError(f"Sudoku puzzle must be 81 tokens, got {len(puzzle)}")
        model, _loss_head, info = _build_and_load(
            self.config_path, self.checkpoint_path, self.device_str
        )

        device = info["device"]
        max_steps = info["halt_max_steps"]
        no_act_continue = info["no_ACT_continue"]
        task_id = info["task_id"]

        inputs = torch.tensor([puzzle], dtype=torch.long, device=device)
        # Eval doesn't have ground-truth labels at inference time — pass a
        # dummy filled-with-1 vector. The model uses labels only for the
        # carry's `current_data` bookkeeping; logits are independent of them.
        labels = torch.full((1, 81), IGNORE_LABEL_ID, dtype=torch.long, device=device)
        task_ids = torch.tensor([task_id], dtype=torch.long, device=device)

        batch = {"inputs": inputs, "labels": labels, "task_id": task_ids}
        carry = _loss_head_initial_carry(model, batch)

        first_halt_step = max_steps
        ever_halted = False
        with torch.no_grad():
            for step_idx in range(max_steps):
                carry, outputs = model(carry=carry, batch=batch)
                q_halt = outputs["q_halt_logits"]
                q_cont = outputs["q_continue_logits"]
                would_halt = (q_halt > 0) if no_act_continue else (q_halt > q_cont)
                if (not ever_halted) and bool(would_halt[0].item()):
                    first_halt_step = step_idx + 1
                    ever_halted = True

        logits = outputs["logits"]
        preds = logits.argmax(-1)
        return {
            "prediction": preds[0].cpu().tolist(),
            "halt_step": int(first_halt_step),
        }


def _loss_head_initial_carry(model: torch.nn.Module, batch: dict) -> object:
    """Build the initial ACT carry without going through ACTLossHead.

    ACTLossHead.initial_carry only forwards to model.initial_carry; calling it
    here directly avoids wrapping a bare batch dict in a loss_head needlessly.
    """
    return model.initial_carry(batch)
