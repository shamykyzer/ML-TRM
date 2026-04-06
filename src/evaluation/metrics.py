import torch


def cell_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = 0
) -> float:
    """Per-cell accuracy, ignoring pad/pre-filled positions."""
    preds = logits.argmax(-1)
    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0
    correct = (preds == labels) & mask
    return (correct.sum().float() / mask.sum().float()).item()


def puzzle_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = 0
) -> float:
    """Per-puzzle accuracy: 1 if ALL non-ignored cells correct, else 0."""
    preds = logits.argmax(-1)
    mask = labels != ignore_index
    per_sample = ((preds == labels) | ~mask).all(dim=-1).float()
    return per_sample.mean().item()
