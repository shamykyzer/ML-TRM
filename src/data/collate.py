"""Collate function adapting existing datasets to the official TRM model's dict format.

Existing datasets return (inputs, masked_labels) tuples with ignore_index=0.
The official model expects dicts with ignore_index=-100.

The remap 0 -> -100 is safe because token 0 is never a valid answer in either
dataset (Sudoku uses tokens 1-10, Maze uses tokens 1-5). Label 0 only ever
means 'ignore this position'.
"""

import torch

IGNORE_LABEL_ID = -100


def official_collate_fn(task_id: int):
    """Return a collate function that wraps dataset output into official model format.

    Args:
        task_id: integer task identifier (0=sudoku, 1=maze)
    """
    def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        inputs = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])

        # Remap ignore_index: 0 -> -100
        labels = torch.where(labels == 0, IGNORE_LABEL_ID, labels)

        task_ids = torch.full((inputs.shape[0],), task_id, dtype=torch.long)

        return {
            "inputs": inputs,
            "labels": labels,
            "task_id": task_ids,
        }

    return collate
