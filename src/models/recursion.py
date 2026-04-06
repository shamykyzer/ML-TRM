import torch
import torch.nn as nn
import torch.nn.functional as F


def latent_recursion(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    n: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inner recursion loop: refine latent state z for n steps, then update answer y.

    Args:
        net: The shared TRM block.
        x: Input embedding [B, L, D].
        y: Answer embedding [B, L, D].
        z: Latent reasoning embedding [B, L, D].
        n: Number of inner refinement steps.

    Returns:
        Updated (y, z).
    """
    for _ in range(n):
        z = net(x + y + z)
    y = net(y + z)
    return y, z


def deep_recursion(
    net: nn.Module,
    output_head: nn.Linear,
    q_head: nn.Linear,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    n: int = 6,
    T: int = 3,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Outer recursion loop: T passes of latent_recursion, only last has gradients.

    Args:
        net: The shared TRM block.
        output_head: Linear layer projecting to vocab logits.
        q_head: Linear layer producing halting probability.
        x: Input embedding [B, L, D].
        y: Answer embedding [B, L, D].
        z: Latent reasoning embedding [B, L, D].
        n: Inner recursion steps per pass.
        T: Number of outer recursion passes.

    Returns:
        (y_detached, z_detached): Detached states for carrying forward.
        logits: [B, L, num_classes] output predictions.
        q: [B] halting probability (sigmoid).
        q_logits: [B] raw halting logits (pre-sigmoid, for BCE loss).
    """
    # T-1 passes WITHOUT gradients (saves memory)
    with torch.no_grad():
        for _ in range(T - 1):
            y, z = latent_recursion(net, x, y, z, n)

    # Final pass WITH gradients
    y, z = latent_recursion(net, x, y, z, n)

    logits = output_head(y)
    q_logits = q_head(y).mean(dim=1).squeeze(-1)  # [B] raw logits
    q = torch.sigmoid(q_logits)  # [B] probability (for inference/ACT)

    return (y.detach(), z.detach()), logits, q, q_logits


def deep_supervision_step(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema=None,
    n: int = 6,
    T: int = 3,
    N_sup: int = 16,
    act_threshold: float = 0.5,
    max_grad_norm: float = 1.0,
    scaler=None,
) -> dict:
    """Full deep supervision training step.

    Each supervision step is a separate optimizer step. States y, z are
    detached between steps to prevent memory explosion.

    Args:
        model: TRM model (must have .embedding, .block, .output_head, .q_head, .y_init, .z_init).
        inputs: Input token IDs [B, L].
        labels: Target labels [B, L] (0 = ignore).
        loss_fn: StableMaxCrossEntropy loss.
        optimizer: AdamW optimizer.
        ema: Optional EMA wrapper.
        n, T, N_sup, act_threshold: Recursion hyperparameters.
        max_grad_norm: Gradient clipping norm.
        scaler: Optional torch.amp.GradScaler for mixed precision.

    Returns:
        Dict with training metrics.
    """
    model.train()
    device = inputs.device
    B = inputs.shape[0]
    use_amp = scaler is not None

    y = model.y_init.expand(B, -1, -1).clone()
    z = model.z_init.expand(B, -1, -1).clone()

    total_ce_loss = 0.0
    total_q_loss = 0.0
    steps_taken = 0

    for step in range(N_sup):
        optimizer.zero_grad()

        # Recompute x each step: optimizer updates embedding weights,
        # and we need a fresh graph for each backward() call.
        with torch.amp.autocast("cuda", enabled=use_amp):
            x = model.embedding(inputs)

            (y, z), logits, q, q_logits = deep_recursion(
                model.block, model.output_head, model.q_head, x, y, z, n, T
            )

            ce_loss = loss_fn(logits, labels)

            # Correctness: all non-ignored positions must match
            mask = labels != 0
            correct_per_sample = ((logits.argmax(-1) == labels) | ~mask).all(dim=-1).float()
            q_loss = F.binary_cross_entropy_with_logits(q_logits, correct_per_sample)

            loss = ce_loss + q_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if ema is not None:
            ema.update()

        total_ce_loss += ce_loss.item()
        total_q_loss += q_loss.item()
        steps_taken = step + 1

        # ACT early stopping
        if q.mean().item() > act_threshold:
            break

    return {
        "ce_loss": total_ce_loss / steps_taken,
        "q_loss": total_q_loss / steps_taken,
        "total_loss": (total_ce_loss + total_q_loss) / steps_taken,
        "q_mean": q.mean().item(),
        "steps_taken": steps_taken,
        "puzzle_acc": correct_per_sample.mean().item(),
    }
