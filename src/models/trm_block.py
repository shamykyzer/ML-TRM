import torch
import torch.nn as nn

from src.models.layers import (
    MLPMixerBlock,
    MultiHeadSelfAttention,
    RMSNorm,
    SwiGLUFFN,
)


class TRMBlock(nn.Module):
    """Shared 2-layer TRM block used recursively.

    From the paper (Figure 1b of Mamba-2 hybrid paper), the TRM network
    is 2 full transformer layers, each consisting of:
      - Sequence processing (Self-Attention or MLP-Mixer) + post-norm
      - Channel processing (SwiGLU FFN) + post-norm

    Total: 4 sub-layers with post-norm residual connections.

    Effective depth per supervision step = T * (n + 1) * n_layers
    With T=3, n=6, n_layers=2: 3 * 7 * 2 = 42
    """

    def __init__(
        self,
        d_model: int = 512,
        ff_hidden: int = 2048,
        seq_len: int = 81,
        use_attention: bool = False,
        n_heads: int = 8,
        max_seq_len: int = 1024,
    ):
        super().__init__()

        # Transformer layer 1: Attention/Mixer + FFN
        if use_attention:
            self.seq1 = MultiHeadSelfAttention(d_model, n_heads, max_seq_len)
        else:
            self.seq1 = MLPMixerBlock(seq_len, d_model)
        self.norm1 = RMSNorm(d_model)
        self.ffn1 = SwiGLUFFN(d_model, ff_hidden)
        self.norm2 = RMSNorm(d_model)

        # Transformer layer 2: Attention/Mixer + FFN
        if use_attention:
            self.seq2 = MultiHeadSelfAttention(d_model, n_heads, max_seq_len)
        else:
            self.seq2 = MLPMixerBlock(seq_len, d_model)
        self.norm3 = RMSNorm(d_model)
        self.ffn2 = SwiGLUFFN(d_model, ff_hidden)
        self.norm4 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Post-norm: norm AFTER residual addition
        # Layer 1
        x = self.norm1(self.seq1(x) + x)
        x = self.norm2(self.ffn1(x) + x)
        # Layer 2
        x = self.norm3(self.seq2(x) + x)
        x = self.norm4(self.ffn2(x) + x)
        return x
