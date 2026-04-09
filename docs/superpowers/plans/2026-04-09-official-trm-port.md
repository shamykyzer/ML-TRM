# Official TRM Architecture Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the official TinyRecursiveModels architecture into the ML-TRM codebase, adapted for Sudoku-Extreme and Maze-Hard tasks, with full Q-learning ACT, StableMax CE, AdamATan2, task-type embeddings, and W&B logging.

**Architecture:** New model (`trm_official.py`), loss (`losses_official.py`), and trainer (`trainer_official.py`) files alongside existing code. Data adaption via a thin collate function. Config system extended with new fields. Existing code untouched — both architectures selectable via config.

**Tech Stack:** PyTorch 2.2+, adam-atan2, wandb, huggingface_hub, codecarbon, pydantic, argdantic

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/models/layers_official.py` | CastedEmbedding, CastedLinear, functional rms_norm, official Attention, official SwiGLU |
| Create | `src/models/trm_official.py` | TRMBlock, TRMReasoningModule, TRMInner, TRMOfficial (ACT wrapper) |
| Create | `src/models/losses_official.py` | StableMax CE, ACTLossHead |
| Create | `src/data/collate.py` | Dict adapter + ignore_index remapping |
| Create | `src/training/trainer_official.py` | Carry-based training loop with AdamATan2, W&B, EMA, HF sync |
| Create | `configs/trm_official_sudoku.yaml` | Official arch config for Sudoku |
| Create | `configs/trm_official_maze.yaml` | Official arch config for Maze |
| Modify | `src/utils/config.py` | Add new config fields for official arch |
| Modify | `src/evaluation/evaluate.py` | Add carry-based eval path |
| Modify | `main.py` | Route to official trainer |
| Modify | `requirements.txt` | Add adam-atan2 |

**Decision: `layers_official.py` instead of modifying `layers.py`.** The official layers (CastedEmbedding, functional rms_norm, official Attention/SwiGLU) use different interfaces than the existing layers. Putting them in a separate file avoids namespace collisions and keeps the old models working without import changes.

---

### Task 1: Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add adam-atan2 to requirements**

In `requirements.txt`, add after the last line:

```
adam-atan2>=0.1.0         # arctan-bounded optimizer (official TRM recipe)
```

Note: `wandb>=0.16.0` is already in requirements.txt.

- [ ] **Step 2: Install new dependency**

Run: `pip install adam-atan2`

If compilation fails on Windows (missing ninja/MSVC), that's expected — the trainer has a fallback to AdamW. Note the result for later.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add adam-atan2 optimizer for official TRM port"
```

---

### Task 2: Official Layers

**Files:**
- Create: `src/models/layers_official.py`

- [ ] **Step 1: Create official layers module**

Create `src/models/layers_official.py` with the layer primitives the official TRM model uses. These are intentionally separate from `layers.py` to avoid breaking existing models.

```python
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
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "from src.models.layers_official import rms_norm, CastedEmbedding, CastedLinear, Attention, SwiGLU, RotaryEmbedding, CosSin; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/models/layers_official.py
git commit -m "feat: add official TRM layer primitives (CastedEmbedding, functional rms_norm, Attention, SwiGLU)"
```

---

### Task 3: Loss Module

**Files:**
- Create: `src/models/losses_official.py`

- [ ] **Step 1: Create the official loss module**

Create `src/models/losses_official.py` with StableMax CE and ACTLossHead, ported from the official `losses.py`:

```python
"""Loss functions for the official TRM architecture.

Ported from the official TinyRecursiveModels codebase.
Key difference from layers.py StableMaxCrossEntropy:
- Uses s(x) = 1/(1-x) for negatives, x+1 for positives (no exp())
- Uses ignore_index=-100 (PyTorch convention)
- Returns per-position losses (not reduced), for per-sample normalization
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_LABEL_ID = -100


def _s(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    """StableMax activation: replaces exp() with a bounded function."""
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log-StableMax normalization (replaces log-softmax)."""
    s_x = _s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """StableMax cross-entropy loss (per-position, unreduced).

    Args:
        logits: [B, L, C] prediction logits
        labels: [B, L] target labels
        ignore_index: label value to ignore
        valid_mask: optional precomputed mask (labels != ignore_index)

    Returns:
        [B, L] per-position losses (0 at ignored positions)
    """
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


class ACTLossHead(nn.Module):
    """ACT loss wrapper for the official TRM model.

    Wraps the model, computes:
    - StableMax CE on logits vs labels
    - BCE on q_halt_logits vs sequence correctness
    - BCE on q_continue_logits vs bootstrapped target Q (if enabled)

    Returns (new_carry, total_loss, metrics_dict, detached_outputs, all_halted).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str] = (),
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    0,
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # LM loss (StableMax CE)
        lm_loss = (
            stablemax_cross_entropy(
                outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask
            )
            / loss_divisor
        ).sum()

        # Q-halt loss (BCE: does the model know when it's right?)
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )

        metrics["lm_loss"] = lm_loss.detach()
        metrics["q_halt_loss"] = q_halt_loss.detach()

        # Q-continue loss (bootstrapped target, TD-learning style)
        q_continue_loss = torch.tensor(0.0, device=lm_loss.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "from src.models.losses_official import stablemax_cross_entropy, ACTLossHead, IGNORE_LABEL_ID; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/models/losses_official.py
git commit -m "feat: add official StableMax CE loss and ACTLossHead"
```

---

### Task 4: Official TRM Model

**Files:**
- Create: `src/models/trm_official.py`

- [ ] **Step 1: Create the official TRM model**

Create `src/models/trm_official.py`:

```python
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
                init_std=0, cast_to=self.forward_dtype,
            )

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
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "from src.models.trm_official import TRMOfficial, TRMConfig; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Quick smoke test — model instantiation and forward pass**

Run:

```bash
python -c "
import torch
from src.models.trm_official import TRMOfficial

model = TRMOfficial({
    'batch_size': 2, 'seq_len': 81, 'vocab_size': 11,
    'num_task_types': 2, 'task_emb_ndim': 512, 'task_emb_len': 16,
    'hidden_size': 512, 'expansion': 4.0, 'num_heads': 8,
    'L_layers': 2, 'H_cycles': 3, 'L_cycles': 4,
    'halt_max_steps': 4, 'halt_exploration_prob': 0.1,
    'no_ACT_continue': False, 'forward_dtype': 'float32',
})
print(f'Params: {model.param_count():,}')

batch = {
    'inputs': torch.randint(0, 11, (2, 81)),
    'labels': torch.randint(-100, 11, (2, 81)),
    'task_id': torch.zeros(2, dtype=torch.long),
}
carry = model.initial_carry(batch)
model.train()
carry, outputs = model(carry=carry, batch=batch)
print(f'logits: {outputs[\"logits\"].shape}')
print(f'q_halt: {outputs[\"q_halt_logits\"].shape}')
print(f'q_continue: {outputs[\"q_continue_logits\"].shape}')
print(f'halted: {carry.halted}')
print(f'has target_q_continue: {\"target_q_continue\" in outputs}')
print('PASS')
"
```

Expected: Model instantiates, forward pass runs, shapes are correct, `target_q_continue` is present (since `no_ACT_continue=False`).

- [ ] **Step 4: Commit**

```bash
git add src/models/trm_official.py
git commit -m "feat: add official TRM model with Q-learning ACT and task-type embeddings"
```

---

### Task 5: Collate Function

**Files:**
- Create: `src/data/collate.py`

- [ ] **Step 1: Create the collate adapter**

Create `src/data/collate.py`:

```python
"""Collate function adapting existing datasets to the official TRM model's dict format.

Existing datasets return (inputs, masked_labels) tuples with ignore_index=0.
The official model expects dicts with ignore_index=-100.

The remap 0 -> -100 is safe because token 0 is never a valid answer in either
dataset (Sudoku uses tokens 1-10, Maze uses tokens 1-5). Label 0 only ever
means 'ignore this position'.
"""

import torch

IGNORE_LABEL_ID = -100


def official_collate_fn(task_id: int):
    """Return a collate function that wraps dataset output into official model format.

    Args:
        task_id: integer task identifier (0=sudoku, 1=maze)
    """
    def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        inputs = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])

        # Remap ignore_index: 0 -> -100
        labels = torch.where(labels == 0, IGNORE_LABEL_ID, labels)

        task_ids = torch.full((inputs.shape[0],), task_id, dtype=torch.long)

        return {
            "inputs": inputs,
            "labels": labels,
            "task_id": task_ids,
        }

    return collate
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "from src.data.collate import official_collate_fn; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/data/collate.py
git commit -m "feat: add collate adapter for official TRM model (ignore_index remap)"
```

---

### Task 6: Config System Updates

**Files:**
- Modify: `src/utils/config.py`

- [ ] **Step 1: Add TRM_OFFICIAL to ModelType enum**

In `src/utils/config.py`, add a new enum value:

```python
class ModelType(str, Enum):
    TRM_SUDOKU = "trm_sudoku"
    TRM_MAZE = "trm_maze"
    TRM_OFFICIAL_SUDOKU = "trm_official_sudoku"
    TRM_OFFICIAL_MAZE = "trm_official_maze"
    LLM_FINETUNE = "llm_finetune"
    LLM_DISTILL = "llm_distill"
```

- [ ] **Step 2: Add official arch fields to ModelConfig**

Add after the existing `dropout` field:

```python
class ModelConfig(BaseModel):
    model_type: ModelType = ModelType.TRM_SUDOKU
    d_model: int = 512
    ff_hidden: int = 2048
    n_heads: int = 8
    vocab_size: int = 11
    seq_len: int = 81
    num_classes: int = 11
    dropout: float = 0.0

    # Official TRM architecture
    H_cycles: int = 3
    L_cycles: int = 4
    L_layers: int = 2
    num_task_types: int = 2
    task_emb_len: int = 16
    task_emb_ndim: int = 512
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    no_ACT_continue: bool = False
    forward_dtype: str = "bfloat16"
    mlp_t: bool = False

    # LLM-specific
    llm_name: str = "gpt2"
    lora_r: int = 8
    lora_alpha: int = 16
    use_qlora: bool = False

    # Distillation-specific
    distill_n_layers: int = 3
    distill_d_model: int = 256
    distill_ff_hidden: int = 1024
    distill_n_heads: int = 4
```

- [ ] **Step 3: Add official training fields to TrainingConfig**

Add after the existing `hf_repo_id` field:

```python
    # Official TRM optimizer
    optimizer: str = "adamw"  # "adamw" or "adam_atan2"
    task_emb_lr: float = 0.01
    task_emb_weight_decay: float = 0.1

    # Task ID for collate
    task_id: int = 0  # 0=sudoku, 1=maze
```

- [ ] **Step 4: Verify config loads**

Run: `python -c "from src.utils.config import load_config; c = load_config('configs/trm_sudoku.yaml'); print(c.model.H_cycles, c.training.optimizer)"`

Expected: `3 adamw` (defaults applied since the old config doesn't set these)

- [ ] **Step 5: Commit**

```bash
git add src/utils/config.py
git commit -m "feat: extend config with official TRM arch fields (H/L_cycles, task_emb, halt, optimizer)"
```

---

### Task 7: Config Files

**Files:**
- Create: `configs/trm_official_sudoku.yaml`
- Create: `configs/trm_official_maze.yaml`

- [ ] **Step 1: Create Sudoku config**

Create `configs/trm_official_sudoku.yaml`:

```yaml
model:
  model_type: trm_official_sudoku
  d_model: 512
  ff_hidden: 2048
  n_heads: 8
  vocab_size: 11
  seq_len: 81
  num_classes: 11

  # Official TRM architecture
  H_cycles: 3
  L_cycles: 4
  L_layers: 2
  num_task_types: 2
  task_emb_len: 16
  task_emb_ndim: 512
  halt_max_steps: 16
  halt_exploration_prob: 0.1
  no_ACT_continue: false
  forward_dtype: bfloat16
  mlp_t: false

training:
  lr: 0.0001
  betas: [0.9, 0.95]
  weight_decay: 0.1
  warmup_steps: 2000
  batch_size: 64
  epochs: 500
  ema_decay: 0.999
  max_grad_norm: 1.0

  optimizer: adam_atan2
  task_emb_lr: 0.01
  task_emb_weight_decay: 0.1
  task_id: 0

  use_wandb: true
  wandb_project: trm-official
  log_interval: 5
  save_interval: 25
  hf_repo_id: shamykyzer/ml-trm-checkpoints

data:
  dataset: sudoku
  data_dir: data/sudoku-extreme-full
  num_workers: 0

seed: 42
device: cuda
checkpoint_dir: models/sudoku-official
experiment_dir: experiments/sudoku-official
```

- [ ] **Step 2: Create Maze config**

Create `configs/trm_official_maze.yaml`:

```yaml
model:
  model_type: trm_official_maze
  d_model: 512
  ff_hidden: 2048
  n_heads: 8
  vocab_size: 6
  seq_len: 900
  num_classes: 6

  # Official TRM architecture
  H_cycles: 3
  L_cycles: 4
  L_layers: 2
  num_task_types: 2
  task_emb_len: 16
  task_emb_ndim: 512
  halt_max_steps: 16
  halt_exploration_prob: 0.1
  no_ACT_continue: false
  forward_dtype: bfloat16
  mlp_t: false

training:
  lr: 0.0001
  betas: [0.9, 0.95]
  weight_decay: 0.1
  warmup_steps: 2000
  batch_size: 8
  epochs: 5000
  ema_decay: 0.999
  max_grad_norm: 1.0

  optimizer: adam_atan2
  task_emb_lr: 0.01
  task_emb_weight_decay: 0.1
  task_id: 1

  use_wandb: true
  wandb_project: trm-official
  log_interval: 50
  save_interval: 500

data:
  dataset: maze
  data_dir: data/maze-30x30-hard-1k
  num_workers: 4

seed: 42
device: cuda
checkpoint_dir: models/maze-official
experiment_dir: experiments/maze-official
```

- [ ] **Step 3: Verify configs load**

Run:

```bash
python -c "
from src.utils.config import load_config
s = load_config('configs/trm_official_sudoku.yaml')
m = load_config('configs/trm_official_maze.yaml')
print(f'Sudoku: {s.model.model_type}, H={s.model.H_cycles}, L={s.model.L_cycles}, opt={s.training.optimizer}')
print(f'Maze: {m.model.model_type}, H={m.model.H_cycles}, L={m.model.L_cycles}, opt={m.training.optimizer}')
"
```

Expected:
```
Sudoku: trm_official_sudoku, H=3, L=4, opt=adam_atan2
Maze: trm_official_maze, H=3, L=4, opt=adam_atan2
```

- [ ] **Step 4: Commit**

```bash
git add configs/trm_official_sudoku.yaml configs/trm_official_maze.yaml
git commit -m "feat: add official TRM config files for Sudoku and Maze"
```

---

### Task 8: Official Trainer

**Files:**
- Create: `src/training/trainer_official.py`

- [ ] **Step 1: Create the official trainer**

Create `src/training/trainer_official.py`:

```python
"""Trainer for the official TRM architecture.

Key differences from trainer_trm.py:
- Carry-based ACT loop (model manages halt state, not fixed N_sup steps)
- ACTLossHead computes all losses (StableMax CE + Q-learning)
- AdamATan2 optimizer with separate param groups for task embeddings
- bfloat16 native forward (no GradScaler needed)
- W&B logging alongside CSV
"""

import csv
import json
import os
import time
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.losses_official import ACTLossHead
from src.training.carbon_tracker import CarbonTracker
from src.training.ema import EMA
from src.utils.config import ExperimentConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def _build_optimizer(model: nn.Module, config: ExperimentConfig):
    """Build optimizer with separate param groups for task embeddings."""
    tc = config.training

    # Separate task embedding params for different lr/wd
    task_emb_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "task_emb" in name:
            task_emb_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "lr": tc.lr, "weight_decay": tc.weight_decay},
    ]
    if task_emb_params:
        param_groups.append({
            "params": task_emb_params,
            "lr": tc.task_emb_lr,
            "weight_decay": tc.task_emb_weight_decay,
        })

    if tc.optimizer == "adam_atan2":
        try:
            from adam_atan2 import AdamAtan2
            optimizer = AdamAtan2(param_groups, betas=tc.betas)
            print("[Optimizer] Using AdamAtan2")
            return optimizer
        except ImportError:
            print("[Optimizer] adam-atan2 not installed, falling back to AdamW")

    optimizer = torch.optim.AdamW(param_groups, betas=tc.betas)
    print("[Optimizer] Using AdamW")
    return optimizer


class OfficialTRMTrainer:
    """Trainer for the official TRM architecture with Q-learning ACT."""

    def __init__(
        self,
        model: nn.Module,
        loss_head: ACTLossHead,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ExperimentConfig,
        resume_checkpoint: str = "",
    ):
        self.model = model
        self.loss_head = loss_head
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tc = config.training
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.loss_head.to(self.device)

        self.optimizer = _build_optimizer(model, config)

        # Linear warmup scheduler
        self.global_step = 0
        self.start_epoch = 0
        self.best_acc = 0.0

        def lr_lambda(step):
            if step < self.tc.warmup_steps:
                return step / max(1, self.tc.warmup_steps)
            return 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.ema = EMA(model, decay=self.tc.ema_decay)

        self.carbon = CarbonTracker(
            f"{config.model.model_type.value}_train",
            output_dir=config.experiment_dir,
        )

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.experiment_dir, exist_ok=True)

        # HuggingFace Hub
        self.hf_repo_id = getattr(self.tc, "hf_repo_id", "")
        if self.hf_repo_id and HF_AVAILABLE:
            self.hf_api = HfApi()
            self.hf_api.create_repo(self.hf_repo_id, exist_ok=True, private=True)
            print(f"[HF Hub] Syncing checkpoints to {self.hf_repo_id}")
        else:
            self.hf_api = None

        # Resume
        if resume_checkpoint and os.path.isfile(resume_checkpoint):
            self._load_checkpoint(resume_checkpoint)

        # W&B
        self.use_wandb = self.tc.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=self.tc.wandb_project,
                name=getattr(self.tc, "wandb_run_name", None) or None,
                config=config.model_dump(),
            )

        # CSV log
        self.log_path = os.path.join(
            config.experiment_dir, f"{config.model.model_type.value}_train_log.csv"
        )

    def _init_log(self) -> None:
        mode = "a" if self.start_epoch > 0 and os.path.exists(self.log_path) else "w"
        if mode == "w":
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "lm_loss", "q_halt_loss", "q_continue_loss",
                    "accuracy", "exact_accuracy", "q_halt_accuracy", "avg_steps",
                    "val_cell_acc", "val_puzzle_acc", "best_puzzle_acc", "elapsed_min",
                ])

    def _append_log(self, row: list) -> None:
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _load_checkpoint(self, path: str) -> None:
        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_acc = ckpt.get("best_puzzle_acc", 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(self.global_step):
                self.scheduler.step()
        print(f"Resumed at epoch {self.start_epoch}, step {self.global_step}, best_acc {self.best_acc:.4f}")

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds >= 86400:
            return f"{seconds / 86400:.1f}d"
        if seconds >= 3600:
            return f"{seconds / 3600:.1f}h"
        return f"{seconds / 60:.0f}m"

    def train(self) -> None:
        self._init_log()
        self.carbon.start()
        t_start = time.time()

        best_acc = self.best_acc
        epoch_times = []
        last_val = {}

        total_epochs = self.tc.epochs - self.start_epoch
        step_bar = tqdm(total=total_epochs, desc="Training", unit="ep", dynamic_ncols=True)

        for epoch in range(self.start_epoch, self.tc.epochs):
            epoch_start = time.time()
            metrics = self._train_epoch(epoch, step_bar)
            step_bar.update(1)
            epoch_times.append(time.time() - epoch_start)

            recent = epoch_times[-10:]
            avg_sec = sum(recent) / len(recent)
            eta_sec = (self.tc.epochs - (epoch + 1)) * avg_sec
            elapsed = time.time() - t_start

            step_bar.set_description_str(f"Training Ep {epoch + 1}/{self.tc.epochs}")
            step_bar.set_postfix_str(
                f"LM={metrics['lm_loss']:.3f}  "
                f"Qh={metrics['q_halt_loss']:.3f}  "
                f"Steps={metrics['avg_steps']:.1f}  "
                f"Acc={metrics['exact_accuracy']:.1%}  "
                f"Val={'%.1f%%' % (last_val.get('puzzle_acc', 0) * 100)}  "
                f"Best={'%.1f%%' % (best_acc * 100)}  "
                f"ETA={self._fmt_time(eta_sec)}"
            )

            if self.use_wandb:
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=epoch + 1)

            if (epoch + 1) % self.tc.log_interval == 0:
                last_val = self.evaluate()

                if self.use_wandb:
                    wandb.log({f"val/{k}": v for k, v in last_val.items()}, step=epoch + 1)

                new_best = ""
                if last_val["puzzle_acc"] > best_acc:
                    best_acc = last_val["puzzle_acc"]
                    self._save_checkpoint(epoch, "best.pt", best_acc)
                    new_best = " NEW BEST!"

                tqdm.write(
                    f"  [{epoch + 1}/{self.tc.epochs}] "
                    f"cell={last_val['cell_acc']:.1%}  "
                    f"puzzle={last_val['puzzle_acc']:.1%}  "
                    f"best={best_acc:.1%}  "
                    f"LM={metrics['lm_loss']:.4f}  "
                    f"elapsed={self._fmt_time(elapsed)}"
                    f"{new_best}"
                )

                self._append_log([
                    epoch + 1,
                    f"{metrics['lm_loss']:.4f}",
                    f"{metrics['q_halt_loss']:.4f}",
                    f"{metrics.get('q_continue_loss', 0):.4f}",
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['exact_accuracy']:.4f}",
                    f"{metrics['q_halt_accuracy']:.4f}",
                    f"{metrics['avg_steps']:.1f}",
                    f"{last_val['cell_acc']:.4f}",
                    f"{last_val['puzzle_acc']:.4f}",
                    f"{best_acc:.4f}",
                    f"{elapsed / 60:.1f}",
                ])

            if (epoch + 1) % self.tc.save_interval == 0:
                self._save_checkpoint(epoch, f"epoch_{epoch + 1}.pt", best_acc)

        step_bar.close()
        self._save_checkpoint(self.tc.epochs - 1, "latest.pt", best_acc)
        emissions = self.carbon.stop()

        if self.use_wandb:
            wandb.finish()

        results_path = os.path.join(self.config.experiment_dir, "training_results.json")
        with open(results_path, "w") as f:
            json.dump({"best_puzzle_acc": best_acc, "emissions": emissions}, f, indent=2)

    def _train_epoch(self, epoch: int, step_bar: tqdm) -> dict:
        self.model.train()
        self.loss_head.train()

        totals = {}
        n_batches = 0

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            carry = self.loss_head.initial_carry(batch)

            batch_metrics = {}
            steps_this_batch = 0

            for _act_step in range(self.config.model.halt_max_steps):
                carry, loss, metrics, _outputs, all_halted = self.loss_head(
                    return_keys=(), carry=carry, batch=batch,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.ema.update()
                self.scheduler.step()
                self.global_step += 1
                steps_this_batch += 1

                # Accumulate metrics from halted samples
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    batch_metrics[k] = batch_metrics.get(k, 0) + v

                if all_halted:
                    break

            # Normalize by count
            count = max(1, batch_metrics.get("count", 1))
            normalized = {
                "lm_loss": batch_metrics.get("lm_loss", 0) / count,
                "q_halt_loss": batch_metrics.get("q_halt_loss", 0) / count,
                "q_continue_loss": batch_metrics.get("q_continue_loss", 0) / count,
                "accuracy": batch_metrics.get("accuracy", 0) / count,
                "exact_accuracy": batch_metrics.get("exact_accuracy", 0) / count,
                "q_halt_accuracy": batch_metrics.get("q_halt_accuracy", 0) / count,
                "avg_steps": batch_metrics.get("steps", 0) / count,
            }

            for k, v in normalized.items():
                totals[k] = totals.get(k, 0) + v
            n_batches += 1

            # Per-batch progress
            total_batches = len(self.train_loader)
            step_bar.set_description_str(f"Training Ep {epoch + 1}/{self.tc.epochs}")
            step_bar.set_postfix_str(
                f"Batch={n_batches}/{total_batches}  "
                f"LM={normalized['lm_loss']:.3f}  "
                f"Steps={normalized['avg_steps']:.1f}  "
                f"Acc={normalized['exact_accuracy']:.1%}"
            )

        return {k: v / max(1, n_batches) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.ema.apply_shadow()
        self.model.eval()
        self.loss_head.eval()

        total_cell_correct = 0
        total_cells = 0
        total_puzzle_correct = 0
        total_puzzles = 0
        total_steps = 0
        total_q_halt_correct = 0
        n_samples = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            B = batch["inputs"].shape[0]

            carry = self.loss_head.initial_carry(batch)

            # Run for full halt_max_steps (no early stopping during eval)
            for _step in range(self.config.model.halt_max_steps):
                carry, _outputs = self.model(carry=carry, batch=batch)

            # Get final predictions from last forward
            logits = _outputs["logits"]
            preds = logits.argmax(-1)
            labels = carry.current_data["labels"]
            mask = labels != -100

            total_cell_correct += ((preds == labels) & mask).sum().item()
            total_cells += mask.sum().item()

            puzzle_correct = ((preds == labels) | ~mask).all(dim=-1)
            total_puzzle_correct += puzzle_correct.sum().item()
            total_puzzles += B

            total_steps += carry.steps.sum().item()
            n_samples += B

            # Q-halt accuracy
            q_halt_correct = (_outputs["q_halt_logits"] >= 0) == puzzle_correct
            total_q_halt_correct += q_halt_correct.sum().item()

        self.ema.restore()

        return {
            "cell_acc": total_cell_correct / max(1, total_cells),
            "puzzle_acc": total_puzzle_correct / max(1, total_puzzles),
            "avg_act_steps": total_steps / max(1, n_samples),
            "q_halt_acc": total_q_halt_correct / max(1, n_samples),
        }

    def _save_checkpoint(self, epoch: int, filename: str, best_acc: float = 0.0) -> None:
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "ema_state_dict": self.ema.state_dict(),
                "config": self.config.model_dump(),
                "seed": self.config.seed,
                "global_step": self.global_step,
                "best_puzzle_acc": best_acc,
            },
            path,
        )
        if self.hf_api:
            try:
                self.hf_api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=f"{self.config.checkpoint_dir}/{filename}",
                    repo_id=self.hf_repo_id,
                    commit_message=f"checkpoint epoch {epoch + 1} (acc={best_acc:.4f})",
                )
            except Exception as e:
                tqdm.write(f"[HF Hub] Upload failed: {e}")
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "from src.training.trainer_official import OfficialTRMTrainer; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/training/trainer_official.py
git commit -m "feat: add official TRM trainer with carry-based ACT loop, AdamATan2, W&B"
```

---

### Task 9: Evaluation Updates

**Files:**
- Modify: `src/evaluation/evaluate.py`

- [ ] **Step 1: Add official TRM evaluation function**

Add the following function and update `load_and_evaluate` in `src/evaluation/evaluate.py`.

After the existing `evaluate_standard` function (after line 132), add:

```python
def evaluate_official(
    model: nn.Module,
    test_loader: DataLoader,
    config: ExperimentConfig,
    ema: EMA = None,
) -> dict:
    """Evaluate an official TRM model with full ACT steps."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    carbon = CarbonTracker(
        f"{config.model.model_type.value}_inference",
        output_dir=config.experiment_dir,
    )

    if ema is not None:
        ema.apply_shadow()

    model.eval()
    carbon.start()

    total_cell_correct = 0
    total_cells = 0
    total_puzzle_correct = 0
    total_puzzles = 0
    total_steps = 0
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            B = batch["inputs"].shape[0]

            carry = model.initial_carry(batch)

            for _step in range(config.model.halt_max_steps):
                carry, outputs = model(carry=carry, batch=batch)

            preds = outputs["logits"].argmax(-1)
            labels = carry.current_data["labels"]
            mask = labels != -100

            total_cell_correct += ((preds == labels) & mask).sum().item()
            total_cells += mask.sum().item()

            puzzle_correct = ((preds == labels) | ~mask).all(dim=-1)
            total_puzzle_correct += puzzle_correct.sum().item()
            total_puzzles += B
            total_steps += carry.steps.sum().item()
            n_samples += B

    emissions = carbon.stop()

    if ema is not None:
        ema.restore()

    return {
        "cell_accuracy": total_cell_correct / max(1, total_cells),
        "puzzle_accuracy": total_puzzle_correct / max(1, total_puzzles),
        "avg_act_steps": total_steps / max(1, n_samples),
        "inference_emissions": emissions,
    }
```

- [ ] **Step 2: Update load_and_evaluate to handle official models**

In the `load_and_evaluate` function, add a branch for the official model types. Add this before the `elif model_type == ModelType.LLM_DISTILL:` line:

```python
    elif model_type in (ModelType.TRM_OFFICIAL_SUDOKU, ModelType.TRM_OFFICIAL_MAZE):
        from src.models.trm_official import TRMOfficial

        model_config = {
            "batch_size": config.training.batch_size,
            "seq_len": config.model.seq_len,
            "vocab_size": config.model.vocab_size,
            "num_task_types": config.model.num_task_types,
            "task_emb_ndim": config.model.task_emb_ndim,
            "task_emb_len": config.model.task_emb_len,
            "hidden_size": config.model.d_model,
            "expansion": config.model.ff_hidden / config.model.d_model,
            "num_heads": config.model.n_heads,
            "L_layers": config.model.L_layers,
            "H_cycles": config.model.H_cycles,
            "L_cycles": config.model.L_cycles,
            "halt_max_steps": config.model.halt_max_steps,
            "halt_exploration_prob": config.model.halt_exploration_prob,
            "no_ACT_continue": config.model.no_ACT_continue,
            "forward_dtype": config.model.forward_dtype,
            "mlp_t": config.model.mlp_t,
        }
        model = TRMOfficial(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])

        ema = None
        if "ema_state_dict" in checkpoint:
            ema = EMA(model, decay=config.training.ema_decay)
            ema.load_state_dict(checkpoint["ema_state_dict"])

        results = evaluate_official(model, test_loader, config, ema=ema)
```

- [ ] **Step 3: Add the ModelType import update**

Update the import at the top of the file. The existing import at line 12 is:

```python
from src.utils.config import ExperimentConfig, ModelType
```

This already imports `ModelType`, so the new enum values (`TRM_OFFICIAL_SUDOKU`, `TRM_OFFICIAL_MAZE`) will be available automatically.

- [ ] **Step 4: Verify import works**

Run: `python -c "from src.evaluation.evaluate import evaluate_official; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/evaluate.py
git commit -m "feat: add carry-based evaluation for official TRM model"
```

---

### Task 10: Main Entry Point

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add official TRM training path**

In `main.py`, add a new branch in `_run_train_once` for the official model types. Add this after the `elif model_type == ModelType.TRM_MAZE:` block (after line 108) and before the `elif model_type == ModelType.LLM_FINETUNE:` block:

```python
    elif model_type in (ModelType.TRM_OFFICIAL_SUDOKU, ModelType.TRM_OFFICIAL_MAZE):
        from src.data.collate import official_collate_fn
        from src.models.losses_official import ACTLossHead
        from src.models.trm_official import TRMOfficial
        from src.training.trainer_official import OfficialTRMTrainer

        model_config = {
            "batch_size": config.training.batch_size,
            "seq_len": config.model.seq_len,
            "vocab_size": config.model.vocab_size,
            "num_task_types": config.model.num_task_types,
            "task_emb_ndim": config.model.task_emb_ndim,
            "task_emb_len": config.model.task_emb_len,
            "hidden_size": config.model.d_model,
            "expansion": config.model.ff_hidden / config.model.d_model,
            "num_heads": config.model.n_heads,
            "L_layers": config.model.L_layers,
            "H_cycles": config.model.H_cycles,
            "L_cycles": config.model.L_cycles,
            "halt_max_steps": config.model.halt_max_steps,
            "halt_exploration_prob": config.model.halt_exploration_prob,
            "no_ACT_continue": config.model.no_ACT_continue,
            "forward_dtype": config.model.forward_dtype,
            "mlp_t": config.model.mlp_t,
        }
        model = TRMOfficial(model_config)
        loss_head = ACTLossHead(model)
        print(f"TRM-Official params: {model.param_count():,}")

        collate_fn = official_collate_fn(config.training.task_id)

        if config.data.dataset == "maze":
            from src.data.maze_dataset import MazeDataset
            from torch.utils.data import DataLoader

            train_ds = MazeDataset(config.data.data_dir, "train")
            test_ds = MazeDataset(config.data.data_dir, "test")
            train_loader = DataLoader(
                train_ds, batch_size=config.training.batch_size, shuffle=True,
                num_workers=config.data.num_workers, pin_memory=True,
                drop_last=True, collate_fn=collate_fn,
            )
            val_loader = DataLoader(
                test_ds, batch_size=config.training.batch_size, shuffle=False,
                num_workers=config.data.num_workers, pin_memory=True,
                collate_fn=collate_fn,
            )
        else:
            from src.data.sudoku_dataset import SudokuDataset
            from torch.utils.data import DataLoader

            train_ds = SudokuDataset(config.data.data_dir, "train", augment=True)
            test_ds = SudokuDataset(config.data.data_dir, "test")
            train_loader = DataLoader(
                train_ds, batch_size=config.training.batch_size, shuffle=True,
                num_workers=config.data.num_workers, pin_memory=True,
                drop_last=True, collate_fn=collate_fn,
            )
            val_loader = DataLoader(
                test_ds, batch_size=config.training.batch_size, shuffle=False,
                num_workers=config.data.num_workers, pin_memory=True,
                collate_fn=collate_fn,
            )

        trainer = OfficialTRMTrainer(
            model, loss_head, train_loader, val_loader, config,
            resume_checkpoint=resume,
        )
        trainer.train()
```

- [ ] **Step 2: Update the eval path**

In `_run_eval`, add a branch for official models. After the `elif model_type == ModelType.TRM_MAZE:` block in `_run_eval` (around line 155), add:

```python
    elif model_type in (ModelType.TRM_OFFICIAL_SUDOKU, ModelType.TRM_OFFICIAL_MAZE):
        from src.data.collate import official_collate_fn

        collate_fn = official_collate_fn(config.training.task_id)

        if config.data.dataset == "maze":
            from src.data.maze_dataset import MazeDataset
            from torch.utils.data import DataLoader

            test_ds = MazeDataset(config.data.data_dir, "test")
            test_loader = DataLoader(
                test_ds, batch_size=config.training.batch_size, shuffle=False,
                num_workers=config.data.num_workers, pin_memory=True,
                collate_fn=collate_fn,
            )
        else:
            from src.data.sudoku_dataset import SudokuDataset
            from torch.utils.data import DataLoader

            test_ds = SudokuDataset(config.data.data_dir, "test")
            test_loader = DataLoader(
                test_ds, batch_size=config.training.batch_size, shuffle=False,
                num_workers=config.data.num_workers, pin_memory=True,
                collate_fn=collate_fn,
            )
```

- [ ] **Step 3: Update the ModelType import**

The existing import at line 7 already imports `ModelType`:

```python
from src.utils.config import ExperimentConfig, ModelType, load_config
```

The new enum values are automatically available.

- [ ] **Step 4: Verify the full pipeline parses**

Run: `python main.py --help`

Expected: Help text prints without import errors.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: wire official TRM model into main entry point (train + eval)"
```

---

### Task 11: End-to-End Smoke Test

**Files:** None (testing only)

- [ ] **Step 1: Run a 1-epoch smoke test on Sudoku**

Create a minimal test that runs 1 epoch of training. This validates the entire pipeline: config loading, model creation, data loading, collation, training loop, evaluation.

Run:

```bash
python -c "
import torch
from src.utils.config import load_config
from src.utils.seed import set_seed

config = load_config('configs/trm_official_sudoku.yaml')
config.training.epochs = 1
config.training.batch_size = 4
config.training.log_interval = 1
config.training.save_interval = 1
config.training.use_wandb = False
config.training.hf_repo_id = ''
config.model.forward_dtype = 'float32'  # CPU-safe
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(42)

from src.data.sudoku_dataset import SudokuDataset
from src.data.collate import official_collate_fn
from torch.utils.data import DataLoader, Subset

train_ds = SudokuDataset(config.data.data_dir, 'train', augment=True)
test_ds = SudokuDataset(config.data.data_dir, 'test')

# Use tiny subsets
train_ds = Subset(train_ds, range(16))
test_ds = Subset(test_ds, range(8))

collate_fn = official_collate_fn(config.training.task_id)
train_loader = DataLoader(train_ds, batch_size=4, collate_fn=collate_fn, drop_last=True)
val_loader = DataLoader(test_ds, batch_size=4, collate_fn=collate_fn)

from src.models.trm_official import TRMOfficial
from src.models.losses_official import ACTLossHead
from src.training.trainer_official import OfficialTRMTrainer

model_config = {
    'batch_size': 4, 'seq_len': config.model.seq_len, 'vocab_size': config.model.vocab_size,
    'num_task_types': 2, 'task_emb_ndim': 512, 'task_emb_len': 16,
    'hidden_size': 512, 'expansion': 4.0, 'num_heads': 8,
    'L_layers': 2, 'H_cycles': 2, 'L_cycles': 2,  # smaller for speed
    'halt_max_steps': 4,  # fewer steps for speed
    'halt_exploration_prob': 0.1, 'no_ACT_continue': False,
    'forward_dtype': 'float32', 'mlp_t': False,
}
config.model.halt_max_steps = 4
config.model.H_cycles = 2
config.model.L_cycles = 2

model = TRMOfficial(model_config)
loss_head = ACTLossHead(model)
config.training.optimizer = 'adamw'  # skip adam_atan2 for smoke test

trainer = OfficialTRMTrainer(model, loss_head, train_loader, val_loader, config)
trainer.train()
print('SMOKE TEST PASSED')
"
```

Expected: Training runs for 1 epoch, evaluation runs, checkpoint saved, prints `SMOKE TEST PASSED`.

- [ ] **Step 2: Verify checkpoint was saved**

Run: `ls models/sudoku-official/`

Expected: `best.pt`, `latest.pt`, `epoch_1.pt` present.

- [ ] **Step 3: Commit (no code changes, just confirming pipeline works)**

No commit needed — this task is validation only.

---

### Task 12: GPU Config Update

**Files:**
- Modify: `src/utils/gpu_config.py`

- [ ] **Step 1: Ensure GPU config handles the official model's dataset names**

The current `apply_gpu_overrides` uses `config.data.dataset` to look up task-specific batch sizes. The official configs use `dataset: sudoku` and `dataset: maze`, which already match the existing GPU_PROFILES keys. No code change needed — just verify.

Run:

```bash
python -c "
from src.utils.config import load_config
from src.utils.gpu_config import apply_gpu_overrides
config = load_config('configs/trm_official_sudoku.yaml')
apply_gpu_overrides(config)
print(f'batch_size={config.training.batch_size}')
"
```

Expected: GPU detected, batch_size overridden based on your GPU profile.

- [ ] **Step 2: No commit needed — verification only**

---
