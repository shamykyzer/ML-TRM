"""Official TRM architecture ported from TinyRecursiveModels.

Adapted for Sudoku-Extreme and Maze-Hard:
- Task-type embeddings (0=sudoku, 1=maze) replace puzzle-specific embeddings
- Single-GPU (no DDP)
- Configurable vocab_size/seq_len per task

Architecture:
  TRMOfficial (ACT wrapper)
    -> TRMInner (core model)
         -> embed_tokens (CastedEmbedding)
         -> task_emb (CastedEmbedding for task-type)
         -> rotary_emb (RotaryEmbedding)
         -> L_level (TRMReasoningModule of TRMBlocks)
         -> lm_head, q_head (CastedLinear)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from src.models.layers_official import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
)

IGNORE_LABEL_ID = -100


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Truncated normal initialization."""
    nn.init.trunc_normal_(tensor, std=std)
    return tensor


class TRMConfig(BaseModel):
    """Configuration for the official TRM model."""
    batch_size: int = 32
    seq_len: int = 81
    vocab_size: int = 11

    # Task-type embeddings (replaces puzzle embeddings)
    num_task_types: int = 2
    task_emb_ndim: int = 512
    task_emb_len: int = 16

    # Architecture
    hidden_size: int = 512
    expansion: float = 4.0
    num_heads: int = 8
    L_layers: int = 2
    H_cycles: int = 3
    L_cycles: int = 4

    # ACT / halting
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    no_ACT_continue: bool = False

    # Precision
    forward_dtype: str = "bfloat16"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # MLP mixer variant (False = attention, True = MLP token mixing)
    mlp_t: bool = False


@dataclass
class InnerCarry:
    """Internal carry state for the TRM inner model."""
    z_H: torch.Tensor  # [B, L+task_emb_len, D]
    z_L: torch.Tensor  # [B, L+task_emb_len, D]


@dataclass
class Carry:
    """Full carry state for ACT wrapper."""
    inner_carry: InnerCarry
    steps: torch.Tensor       # [B] int32, current step count per sample
    halted: torch.Tensor      # [B] bool, whether each sample has halted
    current_data: Dict[str, torch.Tensor]  # cached batch data for non-halted samples


class TRMBlock(nn.Module):
    """Single reasoning block: attention/mixer + SwiGLU FFN, both with post-norm residual."""

    def __init__(self, config: TRMConfig) -> None:
        super().__init__()
        self.config = config

        if config.mlp_t:
            self.mlp_t = SwiGLU(
                hidden_size=config.seq_len + config.task_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class TRMReasoningModule(nn.Module):
    """Reasoning module: input injection + N blocks."""

    def __init__(self, layers: List[TRMBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TRMInner(nn.Module):
    """Core TRM model: embeddings, reasoning module, output heads."""

    def __init__(self, config: TRMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Token embedding
        self.embed_tokens = CastedEmbedding(
            config.vocab_size, config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype,
        )

        # Task-type embedding (replaces puzzle embedding)
        self.task_emb_len = config.task_emb_len
        if config.task_emb_ndim > 0:
            self.task_emb = CastedEmbedding(
                config.num_task_types, config.task_emb_ndim,
                init_std=embed_init_std, cast_to=self.forward_dtype,
            )
            # Zero-initialize task embeddings (original code used init_std=0)
            with torch.no_grad():
                self.task_emb.embedding.weight.zero_()

        # Output heads
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # RoPE
        total_len = config.seq_len + config.task_emb_len
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=total_len,
            base=config.rope_theta,
        )

        # Reasoning layers (shared for both z_L and z_H updates)
        self.L_level = TRMReasoningModule(
            layers=[TRMBlock(config) for _ in range(config.L_layers)]
        )

        # Initial latent states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init: start near "don't halt"
        with torch.no_grad():
            self.q_head.linear.weight.zero_()
            self.q_head.linear.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        """Compute input embeddings with task-type prefix."""
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.task_emb_ndim > 0:
            task_embedding = self.task_emb(task_id)  # [B, task_emb_ndim]

            pad_count = self.task_emb_len * self.config.hidden_size - task_embedding.shape[-1]
            if pad_count > 0:
                task_embedding = F.pad(task_embedding, (0, pad_count))

            task_embedding = task_embedding.view(-1, self.task_emb_len, self.config.hidden_size)
            embedding = torch.cat((task_embedding, embedding), dim=-2)

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int) -> InnerCarry:
        total_len = self.config.seq_len + self.task_emb_len
        return InnerCarry(
            z_H=torch.empty(batch_size, total_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, total_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: InnerCarry) -> InnerCarry:
        return InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self, carry: InnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        cos_sin = self.rotary_emb()

        input_embeddings = self._input_embeddings(batch["inputs"], batch["task_id"])

        z_H, z_L = carry.z_H, carry.z_L

        # H_cycles - 1 without grad (memory efficient)
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
                z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        # Last H_cycle with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
        z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.task_emb_len:]  # strip task embedding positions
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)  # Q-head from first position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TRMOfficial(nn.Module):
    """ACT wrapper: manages carry state, halting decisions, and Q-learning exploration."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRMConfig(**config_dict)
        self.inner = TRMInner(self.config)

    @property
    def task_emb(self):
        return self.inner.task_emb

    def param_count(self) -> int:
        return self.inner.param_count()

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> Carry:
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self, carry: Carry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Carry, Dict[str, torch.Tensor]]:

        # Reset carry for halted sequences, load new data
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v,
            )
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration: random halt timing
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q for bootstrapping
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(
                        new_inner_carry, new_current_data
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        new_carry = Carry(new_inner_carry, new_steps, halted, new_current_data)
        return new_carry, outputs
