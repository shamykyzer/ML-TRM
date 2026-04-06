import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SudokuDataset(Dataset):
    """PyTorch Dataset for preprocessed Sudoku-Extreme data.

    Loads .npy files produced by data/build_sudoku_dataset.py.
    Token values: 1-10 (build script adds +1; blanks=1, digits 1-9 become 2-10).
    Labels are masked: pre-filled positions set to 0 (ignore_index) so loss
    only applies to blank cells the model must predict.
    """

    def __init__(self, data_dir: str, split: str = "train"):
        split_dir = os.path.join(data_dir, split)

        with open(os.path.join(split_dir, "dataset.json")) as f:
            self.metadata = json.load(f)

        self.inputs = np.load(os.path.join(split_dir, "all__inputs.npy"))
        self.labels = np.load(os.path.join(split_dir, "all__labels.npy"))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.tensor(self.inputs[idx], dtype=torch.long)
        lab = torch.tensor(self.labels[idx], dtype=torch.long)

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
    train_ds = SudokuDataset(data_dir, "train")
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
