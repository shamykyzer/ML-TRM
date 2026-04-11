"""PyTorch Dataset for preprocessed Maze-Hard data.

Token schema (authoritative — must match ``data/build_maze_dataset.py``
``CHARSET = '# SGo'`` and ``src/data/encoding.py``):

    stored token | char | meaning
    -------------|------|------------
         0       |  —   | pad / ignore
         1       |  #   | wall
         2       | ' '  | open cell (corridor)
         3       |  S   | start
         4       |  G   | goal
         5       |  o   | path marker — the solution the model must output

Label masking convention:
    Positions where ``inputs[i] == labels[i]`` (walls, open cells, S, G)
    are replaced with 0. The CE loss ignores index 0 so the model is only
    graded on the ``o`` cells it has to fill in — i.e. marking the
    actual solution path from S to G.

A valid maze solution has exactly one S, exactly one G, and the o-marked
cells must form a 4-connected chain linking them
(``src/data/encoding.is_valid_maze_path`` enforces this at eval time).
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MazeDataset(Dataset):
    """PyTorch Dataset for preprocessed Maze-Hard data.

    Loads .npy files produced by data/build_maze_dataset.py.
    Token values: 1-5 (CHARSET '#'=1,' '=2,'S'=3,'G'=4,'o'=5), 0=pad.
    Labels masked: positions where input==label set to 0 (ignore).
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

        mask = inp != lab
        masked_lab = lab.clone()
        masked_lab[~mask] = 0
        return inp, masked_lab


def get_maze_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_ds = MazeDataset(data_dir, "train")
    test_ds = MazeDataset(data_dir, "test")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
