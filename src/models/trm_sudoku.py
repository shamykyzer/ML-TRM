import torch
import torch.nn as nn

from src.models.trm_block import TRMBlock


class TRMSudoku(nn.Module):
    """TRM-MLP for Sudoku: MLP-Mixer instead of attention.

    From "Less is More" (Jolicoeur-Martineau, 2025):
    - Single 2-layer network (Mixer+FFN, Mixer+FFN) applied recursively
    - MLP-Mixer replaces self-attention (L=81 <= D=512, so cheap)
    - y_init and z_init are [1, 1, D] broadcast across positions
    - ff_hidden=2048 (expansion=4) to match official implementation
    """

    def __init__(
        self,
        vocab_size: int = 11,
        seq_len: int = 81,
        d_model: int = 512,
        ff_hidden: int = 2048,
        num_classes: int = 11,
        n_heads: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Learned initial embeddings of shape [1, 1, D], broadcast across positions
        # Paper: "add an embedding of shape [0, 1, D]"
        self.y_init = nn.Parameter(torch.zeros(1, 1, d_model))
        self.z_init = nn.Parameter(torch.zeros(1, 1, d_model))

        # Shared 2-layer recursive block (MLP-Mixer, no attention)
        self.block = TRMBlock(
            d_model=d_model,
            ff_hidden=ff_hidden,
            seq_len=seq_len,
            use_attention=False,
        )

        # Output heads (no bias)
        self.output_head = nn.Linear(d_model, num_classes, bias=False)
        self.q_head = nn.Linear(d_model, 1, bias=False)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TRMMaze(nn.Module):
    """TRM-Att for Maze: Self-attention with RoPE.

    From "Less is More" (Jolicoeur-Martineau, 2025):
    - Single 2-layer network (Attn+FFN, Attn+FFN) applied recursively
    - Self-attention with RoPE for large grids (L=900 >> D)
    - ff_hidden=2048 (expansion=4) to match official implementation
    """

    def __init__(
        self,
        vocab_size: int = 6,
        seq_len: int = 900,
        d_model: int = 512,
        ff_hidden: int = 2048,
        num_classes: int = 6,
        n_heads: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Learned initial embeddings [1, 1, D], broadcast across positions
        self.y_init = nn.Parameter(torch.zeros(1, 1, d_model))
        self.z_init = nn.Parameter(torch.zeros(1, 1, d_model))

        # Shared 2-layer recursive block (self-attention with RoPE)
        self.block = TRMBlock(
            d_model=d_model,
            ff_hidden=ff_hidden,
            seq_len=seq_len,
            use_attention=True,
            n_heads=n_heads,
            max_seq_len=seq_len,
        )

        # Output heads (no bias)
        self.output_head = nn.Linear(d_model, num_classes, bias=False)
        self.q_head = nn.Linear(d_model, 1, bias=False)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
