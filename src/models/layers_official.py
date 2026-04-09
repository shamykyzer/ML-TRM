"""Layer primitives for the official TRM architecture.

Ported from the official TinyRecursiveModels codebase. These layers use
bfloat16 casting and functional RMSNorm, which differ from the existing
layers in layers.py (which use float32 and nn.Module-based RMSNorm).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, variance_epsilon: float = 1e-5) -> torch.Tensor:
    """Functional RMSNorm (no learnable parameters)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + variance_epsilon)


@dataclass
class CosSin:
    """Precomputed RoPE cos/sin values."""
    cos: torch.Tensor
    sin: torch.Tensor


class RotaryEmbedding(nn.Module):
    """RoPE for the official TRM architecture."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_position_embeddings).float()
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

    def forward(self) -> CosSin:
        return CosSin(cos=self.cos_cached, sin=self.sin_cached)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos_sin: CosSin) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos_sin.cos
    sin = cos_sin.sin

    seq_len = q.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    cos = cos.repeat(1, 1, 1, 2)
    sin = sin.repeat(1, 1, 1, 2)

    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class CastedEmbedding(nn.Module):
    """Embedding with output cast to a target dtype and custom init."""

    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float = 1.0, cast_to: torch.dtype = torch.bfloat16):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.cast_to = cast_to
        nn.init.trunc_normal_(self.embedding.weight, std=init_std)

    @property
    def embedding_weight(self):
        return self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x).to(self.cast_to)


class CastedLinear(nn.Module):
    """Linear layer with output cast to a target dtype."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight = self.linear.weight
        if bias:
            self.bias = self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Attention(nn.Module):
    """Multi-head attention matching the official TRM implementation.

    Non-causal (bidirectional), supports RoPE via CosSin argument.
    """

    def __init__(self, hidden_size: int, head_dim: int, num_heads: int,
                 num_key_value_heads: int, causal: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal = causal

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if cos_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos_sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward matching the official TRM implementation."""

    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        ff_hidden = int(hidden_size * expansion)
        self.w1 = nn.Linear(hidden_size, ff_hidden, bias=False)
        self.w2 = nn.Linear(ff_hidden, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, ff_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
