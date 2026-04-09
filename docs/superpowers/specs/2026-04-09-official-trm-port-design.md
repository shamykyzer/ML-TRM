# Official TRM Architecture Port — Design Spec

**Date:** 2026-04-09
**Goal:** Port the official TinyRecursiveModels architecture (from `arcprize/trm_arc_prize_verification`) into the ML-TRM codebase, adapted for Sudoku-Extreme and Maze-Hard tasks.

**Approach:** Full replace — new model, loss, and trainer files alongside existing code. Nothing breaks; old architecture remains usable via original configs.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Puzzle embeddings | Repurpose as **task-type embeddings** (0=sudoku, 1=maze) | Enables future unified model; reuses official embedding infrastructure |
| ACT halting | **Full Q-learning** (dual q_halt + q_continue heads, 10% exploration) | Most powerful option; full official recipe |
| Optimizer | **AdamATan2** | Official recipe; arctan-bounded updates stabilize recursive dynamics |
| Loss function | **Official StableMax CE** | Replaces softmax exp() with numerically stable s() function |
| Distributed training | **Single-GPU only** | Lab machines are single-GPU (RTX 3070/4060/4070) |
| W&B | **Enabled** | Live training monitoring alongside CSV logging |
| Ignore index | **-100** (PyTorch convention) | Unambiguous; no collision with valid tokens |

---

## Architecture Overview

### Model (`src/models/trm_official.py`)

Ports `TinyRecursiveReasoningModel_ACTV1` from the official checkpoint's `trm.py`.

**`TRMBlock`**
- 2-layer post-norm residual block
- Layer 1: Attention (or MLP-mixer if `mlp_t=True`) + RMSNorm
- Layer 2: SwiGLU FFN + RMSNorm
- Formula: `x = rms_norm(x + layer(x))`

**`TRMReasoningModule`**
- Wraps `L_layers` (default 2) TRMBlocks
- Forward: `hidden = hidden + input_injection`, then pass through all blocks
- Single shared module used for both z_L and z_H updates

**`TRMInner`** — Core model
- Token embedding: `CastedEmbedding(vocab_size, 512)` scaled by `sqrt(512)`
- Task-type embedding: `CastedSparseEmbedding(num_task_types=2, task_emb_ndim=512)` → reshaped to `(task_emb_len=16, 512)` and prepended to sequence
- RoPE positional encoding over `seq_len + task_emb_len`
- L_level: `TRMReasoningModule` with `L_layers=2` blocks
- Initial states: `H_init`, `L_init` — learned [512] vectors broadcast across positions
- Output heads:
  - `lm_head`: Linear(512 → vocab_size), logits for token prediction
  - `q_head`: Linear(512 → 2, bias=True), Q-values for halt/continue decisions. Initialized: weights=0, bias=-5 (starts near "don't halt")

**Forward pass (deep recursion):**
```
For H_step in 0..H_cycles-1:      # H_cycles=3
    (first H_cycles-1 without grad, last 1 with grad)
    For L_step in 0..L_cycles-1:   # L_cycles=4
        z_L = L_level(z_L, z_H + input_embeddings)   # refine latent
    z_H = L_level(z_H, z_L)                           # update answer

logits = lm_head(z_H)[:, task_emb_len:]    # strip task embedding positions
q_logits = q_head(z_H[:, 0])               # Q-values from first position
```

**`TRMOfficial`** — ACT wrapper
- Maintains carry state: `inner_carry` (z_H, z_L), `steps` counter, `halted` flags, `current_data`
- On each forward call:
  1. Reset carry for halted sequences (replace with learned init states)
  2. Load new data for halted sequences from the batch
  3. Run `TRMInner.forward()`
  4. Increment steps, apply halt decision:
     - Hard halt if `steps >= halt_max_steps` (16)
     - Q-learning halt if `q_halt_logits > 0`
     - Exploration: 10% chance of random early/late halt during training
  5. Return new carry + outputs (logits, q_halt_logits, q_continue_logits)

### Loss (`src/models/losses_official.py`)

Ports `ACTLossHead` from official `losses.py`.

**StableMax cross-entropy:**
```python
def s(x):
    return where(x < 0, 1/(1-x), x + 1)

def stablemax_cross_entropy(logits, labels, ignore_index=-100):
    s_x = s(logits)
    log_probs = log(s_x / sum(s_x))
    return -log_probs[labels]  # masked by ignore_index
```

Replaces `softmax(exp(x))` with `s(x)` — avoids exp() overflow/underflow in long recursive gradient paths.

**ACTLossHead.forward():**
- Calls model forward, gets carry + outputs
- Computes:
  - `lm_loss`: StableMax CE on logits vs labels (normalized per-sample by valid token count)
  - `q_halt_loss`: BCE — q_halt_logits vs whether the sequence is fully correct
  - `q_continue_loss`: BCE — q_continue_logits vs bootstrapped target Q-value (TD-learning style)
- Total loss: `lm_loss + 0.5 * (q_halt_loss + q_continue_loss)`
- Metrics: count, accuracy, exact_accuracy, q_halt_accuracy, steps, per-loss breakdowns
- Returns: `(new_carry, total_loss, metrics, detached_outputs, all_halted)`

### Layers (`src/models/layers.py` — additions)

New layer implementations added alongside existing ones:

- **`CastedEmbedding`** — `nn.Embedding` that casts output to bfloat16
- **`CastedLinear`** — `nn.Linear` that casts output to bfloat16
- **`CastedSparseEmbedding`** — sparse embedding for task-type tokens, cast to bfloat16
- **`rms_norm(x, eps)`** — functional RMSNorm (vs your existing `RMSNorm` module)
- **`Attention`** — multi-head attention with RoPE, non-causal, matching official implementation
- **`LinearSwish`** — linear + SiLU activation (used internally by SwiGLU)

Existing layers (`RMSNorm`, `SwiGLUFFN`, `RotaryEmbedding`, `MultiHeadSelfAttention`, `MLPMixerBlock`, `StableMaxCrossEntropy`) remain unchanged.

---

## Trainer (`src/training/trainer_official.py`)

### Training Loop

```
For each epoch:
    For each batch:
        carry = model.initial_carry(batch)
        For act_step in 0..halt_max_steps:
            carry, loss, metrics, outputs, all_halted = loss_head(carry, batch)
            loss.backward()
            clip_grad_norm_(max_norm=1.0)
            optimizer.step()  # AdamATan2
            optimizer.zero_grad()
            ema.update()
            scheduler.step()
            if all_halted:
                break
        Log metrics (CSV + W&B)
    
    Every eval_interval:
        Evaluate with EMA shadow weights
        Save checkpoint if best puzzle accuracy
        Sync to HF Hub
```

### Optimizer

- **AdamATan2**: lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1
- Separate lr/weight_decay for task-type embeddings: lr=0.01, weight_decay=0.1 (matching official `puzzle_emb_lr`)
- Linear warmup: 2000 steps, no decay after (lr_min_ratio=1.0)
- Gradient clipping: max_norm=1.0
- Fallback: if adam-atan2 fails to import, warn and use AdamW

### W&B Integration

- `wandb.init(project=config.training.wandb_project, name=config.training.wandb_run_name, config=config_dict)`
- Per-step logging: lm_loss, q_halt_loss, q_continue_loss, steps_taken, accuracy, exact_accuracy, q_halt_accuracy, learning_rate
- Per-eval logging: val_cell_acc, val_puzzle_acc, avg_act_steps
- `wandb.finish()` on completion

### Checkpointing

Saves:
- `model_state_dict`, `optimizer_state_dict`, `ema_state_dict`
- `epoch`, `global_step`, `best_puzzle_acc`, `config`, `seed`

Resume: restores all of the above + repositions scheduler

### Kept Infrastructure

- EMA (decay=0.999)
- bfloat16 native forward (no GradScaler needed — bfloat16 has float32's exponent range)
- GPU auto-tuning via `apply_gpu_overrides()`
- HF Hub checkpoint sync
- CSV logging
- CodeCarbon emissions tracking

---

## Data Integration (`src/data/collate.py`)

Thin adapter between existing datasets and the official model's expected input format.

**`official_collate_fn(task_id: int)`** — returns a collate function that:
1. Takes `(inputs, masked_labels)` tuples from existing datasets
2. Remaps `ignore_index`: labels `0` → `-100`
3. Returns dict: `{"inputs": inputs, "labels": labels, "task_id": task_id_tensor}`

No changes to `sudoku_dataset.py` or `maze_dataset.py`.

**Note:** The remap `0 → -100` is safe because token `0` is never a valid answer in either dataset (Sudoku uses tokens 1-10, Maze uses tokens 1-5). Label `0` only ever means "ignore this position."

---

## Config System (`src/utils/config.py` — additions)

New fields in `ModelConfig`:
```yaml
arch: "trm_official"
H_cycles: 3
L_cycles: 4
L_layers: 2
num_task_types: 2
task_emb_len: 16
task_emb_ndim: 512
halt_max_steps: 16
halt_exploration_prob: 0.1
no_ACT_continue: false
forward_dtype: "bfloat16"
```

New fields in `TrainingConfig`:
```yaml
optimizer: "adam_atan2"
task_emb_lr: 0.01
task_emb_weight_decay: 0.1
wandb_project: ""
wandb_run_name: ""
```

### New Config Files

**`configs/trm_official_sudoku.yaml`:**
- vocab_size=11, seq_len=81
- H_cycles=3, L_cycles=4, L_layers=2
- halt_max_steps=16, batch_size=32 (GPU-overridden)
- task_id=0

**`configs/trm_official_maze.yaml`:**
- vocab_size=6, seq_len=900
- H_cycles=3, L_cycles=4, L_layers=2
- halt_max_steps=16, batch_size=8 (GPU-overridden)
- task_id=1

---

## Evaluation (`src/evaluation/evaluate.py` — updated)

New `evaluate_official()` function:
- Runs model for full `halt_max_steps` (no early stopping during eval — official behavior)
- Uses EMA shadow weights
- Computes: cell_accuracy, puzzle_accuracy, avg_act_steps, q_halt_accuracy
- Logs to W&B and CSV

---

## Entry Point (`main.py` — updated)

```python
if config.model.arch == "trm_official":
    from src.training.trainer_official import OfficialTRMTrainer
    trainer = OfficialTRMTrainer(config, model, loss_head, train_loader, test_loader)
else:
    # existing path
    from src.training.trainer_trm import TRMTrainer
    ...
```

---

## Dependencies (`requirements.txt` — updated)

```
adam-atan2>=0.1.0
wandb>=0.16.0
```

Plus build dependencies for adam-atan2: `ninja`, `packaging`, `setuptools`.

---

## File Change Summary

| Action | File |
|---|---|
| **Create** | `src/models/trm_official.py` |
| **Create** | `src/models/losses_official.py` |
| **Create** | `src/training/trainer_official.py` |
| **Create** | `src/data/collate.py` |
| **Create** | `configs/trm_official_sudoku.yaml` |
| **Create** | `configs/trm_official_maze.yaml` |
| **Modify** | `src/models/layers.py` — add CastedEmbedding, CastedLinear, CastedSparseEmbedding, functional rms_norm, official Attention |
| **Modify** | `src/utils/config.py` — add new config fields |
| **Modify** | `src/evaluation/evaluate.py` — add carry-based eval path + W&B logging |
| **Modify** | `main.py` — route to official trainer |
| **Modify** | `requirements.txt` — add adam-atan2, wandb |
| **Unchanged** | All existing model, trainer, data, and config files |
