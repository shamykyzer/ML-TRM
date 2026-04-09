"""Loss functions for the official TRM architecture.

Ported from the official TinyRecursiveModels codebase.
Key difference from layers.py StableMaxCrossEntropy:
- Uses s(x) = 1/(1-x) for negatives, x+1 for positives (no exp())
- Uses ignore_index=-100 (PyTorch convention)
- Returns per-position losses (not reduced), for per-sample normalization
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_LABEL_ID = -100


def _s(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    """StableMax activation: replaces exp() with a bounded function."""
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log-StableMax normalization (replaces log-softmax)."""
    s_x = _s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """StableMax cross-entropy loss (per-position, unreduced).

    Args:
        logits: [B, L, C] prediction logits
        labels: [B, L] target labels
        ignore_index: label value to ignore
        valid_mask: optional precomputed mask (labels != ignore_index)

    Returns:
        [B, L] per-position losses (0 at ignored positions)
    """
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


class ACTLossHead(nn.Module):
    """ACT loss wrapper for the official TRM model.

    Wraps the model, computes:
    - StableMax CE on logits vs labels
    - BCE on q_halt_logits vs sequence correctness
    - BCE on q_continue_logits vs bootstrapped target Q (if enabled)

    Returns (new_carry, total_loss, metrics_dict, detached_outputs, all_halted).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str] = (),
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    0,
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # LM loss (StableMax CE)
        lm_loss = (
            stablemax_cross_entropy(
                outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask
            )
            / loss_divisor
        ).sum()

        # Q-halt loss (BCE: does the model know when it's right?)
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )

        metrics["lm_loss"] = lm_loss.detach()
        metrics["q_halt_loss"] = q_halt_loss.detach()

        # Q-continue loss (bootstrapped target, TD-learning style)
        q_continue_loss = torch.tensor(0.0, device=lm_loss.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
