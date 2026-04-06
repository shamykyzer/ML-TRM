import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (post-norm variant)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.gamma


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network. No bias anywhere."""

    def __init__(self, d_model: int, ff_hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_hidden, bias=False)  # gate projection
        self.w2 = nn.Linear(ff_hidden, d_model, bias=False)  # down projection
        self.w3 = nn.Linear(d_model, ff_hidden, bias=False)  # value projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # cos, sin: [seq_len, head_dim//2] -> broadcast to [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    sin = sin.unsqueeze(0).unsqueeze(0)
    # Repeat for full head_dim (pairs)
    cos = cos.repeat(1, 1, 1, 2)  # [1, 1, seq_len, head_dim]
    sin = sin.repeat(1, 1, 1, 2)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE. No bias, bidirectional."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v)  # [B, H, L, D_h]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class MLPMixerBlock(nn.Module):
    """Token-mixing MLP (replaces attention for short sequences like sudoku)."""

    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        mixer_hidden = seq_len * 4
        self.fc1 = nn.Linear(seq_len, mixer_hidden, bias=False)
        self.fc2 = nn.Linear(mixer_hidden, seq_len, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> transpose to [B, D, L] for token-mixing
        return self.fc2(F.silu(self.fc1(x.transpose(-1, -2)))).transpose(-1, -2)


class StableMaxCrossEntropy(nn.Module):
    """Numerically stable cross-entropy with ignore_index support."""

    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, L, C], targets: [B, L]
        B, L, C = logits.shape
        logits = logits.view(-1, C)
        targets = targets.view(-1)

        mask = targets != self.ignore_index
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Stable log-softmax: shift by max
        shifted = logits - logits.max(dim=-1, keepdim=True).values
        log_sum_exp = torch.log(torch.clamp(shifted.exp().sum(dim=-1), min=1e-10))
        nll = log_sum_exp - shifted.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)

        return (nll * mask.float()).sum() / mask.sum().clamp(min=1)
