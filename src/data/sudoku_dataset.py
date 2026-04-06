import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _shuffle_sudoku(inp: np.ndarray, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """On-the-fly augmentation: randomly shuffle digits, rows, columns, and transpose.

    Operates on stored token values (1-10 where 1=blank, 2-10=digits 1-9).
    Converts to raw 0-9 for shuffling, then back to 1-10.
    """
    # Convert from stored (1-10) to raw (0-9)
    raw_inp = (inp - 1).reshape(9, 9)
    raw_lab = (lab - 1).reshape(9, 9)

    # Random digit permutation: map digits 1-9 randomly, keep 0 (blank) unchanged
    digit_map = np.zeros(10, dtype=np.int64)
    digit_map[1:] = np.random.permutation(np.arange(1, 10))

    # Random transpose
    if np.random.rand() < 0.5:
        raw_inp = raw_inp.T
        raw_lab = raw_lab.T

    # Random row permutation (shuffle bands, then rows within bands)
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Random column permutation (shuffle stacks, then columns within stacks)
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Apply row and column permutation
    raw_inp = raw_inp[row_perm][:, col_perm]
    raw_lab = raw_lab[row_perm][:, col_perm]

    # Apply digit mapping
    raw_inp = digit_map[raw_inp].flatten()
    raw_lab = digit_map[raw_lab].flatten()

    # Convert back to stored (1-10)
    return raw_inp + 1, raw_lab + 1


class SudokuDataset(Dataset):
    """PyTorch Dataset for preprocessed Sudoku-Extreme data.

    Loads .npy files produced by data/build_sudoku_dataset.py.
    Token values: 1-10 (build script adds +1; blanks=1, digits 1-9 become 2-10).
    Labels are masked: pre-filled positions set to 0 (ignore_index) so loss
    only applies to blank cells the model must predict.
    """

    def __init__(self, data_dir: str, split: str = "train", augment: bool = False):
        split_dir = os.path.join(data_dir, split)

        with open(os.path.join(split_dir, "dataset.json")) as f:
            self.metadata = json.load(f)

        self.inputs = np.load(os.path.join(split_dir, "all__inputs.npy"))
        self.labels = np.load(os.path.join(split_dir, "all__labels.npy"))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp_np = self.inputs[idx].copy()
        lab_np = self.labels[idx].copy()

        if self.augment:
            inp_np, lab_np = _shuffle_sudoku(inp_np, lab_np)

        inp = torch.tensor(inp_np, dtype=torch.long)
        lab = torch.tensor(lab_np, dtype=torch.long)

        # Mask pre-filled positions: where input already equals the answer,
        # set label to 0 (ignore_label_id) so loss ignores them.
        mask = inp != lab  # True for blank positions the model must solve
        masked_lab = lab.clone()
        masked_lab[~mask] = 0
        return inp, masked_lab


def get_sudoku_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_ds = SudokuDataset(data_dir, "train", augment=True)
    test_ds = SudokuDataset(data_dir, "test")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
