import numpy as np
from pydantic import BaseModel


class PuzzleDatasetMetadata(BaseModel):
    seq_len: int
    vocab_size: int
    pad_id: int
    ignore_label_id: int
    blank_identifier_id: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int
    sets: list[str]


def dihedral_transform(grid: np.ndarray, aug_idx: int) -> np.ndarray:
    """Apply one of 8 dihedral group (D4) transformations to a square grid.

    aug_idx 0-3: rotations by 0, 90, 180, 270 degrees
    aug_idx 4-7: horizontal flip + rotation by 0, 90, 180, 270 degrees
    """
    if aug_idx < 4:
        return np.rot90(grid, k=aug_idx).copy()
    else:
        return np.rot90(np.fliplr(grid), k=aug_idx - 4).copy()
