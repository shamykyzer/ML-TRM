import torch
import torch.nn as nn


class DistilledLLM(nn.Module):
    """Small transformer student model for knowledge distillation.

    A lightweight encoder-only transformer that operates on the same
    puzzle token sequences as the TRM. Trained via distillation from
    a fine-tuned LLM teacher.

    Target: ~10-20M parameters (matching TRM scale for fair comparison).
    """

    def __init__(
        self,
        vocab_size: int = 11,
        seq_len: int = 81,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        ff_hidden: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device)
        h = self.embedding(x) + self.pos_embedding(pos)
        h = self.encoder(h)
        return self.head(h)  # [B, L, vocab_size]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
