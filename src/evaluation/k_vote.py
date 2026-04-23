"""K-sample inference-time ensembling with majority voting ("self-consistency").

Novel inference-time extension of Dillon (2026)'s training-time WTA technique:
run the model K times per puzzle with injected diversity, majority-vote
per-cell. A puzzle counts as solved only if every non-ignore cell matches
ground truth after the vote.

Diversity sources:
  - TRM  : seeded Gaussian noise added to the initial latent state (z_H, z_L)
           before the ACT recurrence. Reproducible per pass_idx.
  - LLM  : temperature-scaled categorical sampling from the teacher-forced
           logits (matches trainer_llm.evaluate's single-forward pattern,
           unlike autoregressive generate which would compound drift).

We re-instantiate CarbonTracker per K so the kWh reading isolates the K-pass
inference cost cleanly.
"""

import time
from dataclasses import replace

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.trm_official import InnerCarry
from src.training.carbon_tracker import CarbonTracker

IGNORE_LABEL_ID = -100


def _majority_vote(preds_stack: torch.Tensor) -> torch.Tensor:
    """Per-cell mode across K. preds_stack: [K, B, L] -> [B, L].

    torch.mode breaks ties arbitrarily; acceptable per spec.
    """
    return torch.mode(preds_stack, dim=0).values


def _score_batch(
    voted: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Count (cell_correct, cell_total, puzzle_correct, puzzle_total).

    Ignore positions (labels == IGNORE_LABEL_ID) are excluded from both cell
    accuracy and the puzzle exact-match check.
    """
    mask = labels != IGNORE_LABEL_ID
    cell_correct = ((voted == labels) & mask).sum().item()
    cell_total = mask.sum().item()
    puzzle_correct = ((voted == labels) | ~mask).all(dim=-1).sum().item()
    puzzle_total = labels.shape[0]
    return cell_correct, cell_total, puzzle_correct, puzzle_total


# ---------------------------------------------------------------------------
# TRM
# ---------------------------------------------------------------------------

def _trm_perturbed_carry(
    model,
    batch: dict,
    pass_idx: int,
    sigma: float,
) -> object:
    """Build an initial Carry whose z_H/z_L are H_init/L_init + seeded noise.

    We bypass the model.forward reset-on-halted path by pre-populating the
    latent buffers ourselves and flipping halted=False. This avoids any
    edit to TRMOfficial.forward (which would otherwise overwrite them via
    reset_carry during the first ACT step).
    """
    carry = model.initial_carry(batch)
    B = batch["inputs"].shape[0]
    inner = model.inner
    device = inner.H_init.device
    dtype = inner.H_init.dtype

    shape_H = carry.inner_carry.z_H.shape  # [B, seq_len + task_emb_len, hidden]
    shape_L = carry.inner_carry.z_L.shape

    # Broadcast the 1-D init buffer across batch and sequence, then clone so
    # downstream in-place ops (or the ACT recurrence's write-back) don't alias
    # the shared buffer.
    z_H = inner.H_init.to(device=device, dtype=dtype).expand(shape_H).clone()
    z_L = inner.L_init.to(device=device, dtype=dtype).expand(shape_L).clone()

    # pass_idx==0 is the deterministic anchor so K=1 reproduces single-shot eval.
    if pass_idx > 0 and sigma > 0.0:
        # CPU generator: CUDA torch.Generator manual_seed behaviour differs
        # across drivers; CPU is portable and the noise is one-shot per pass.
        g = torch.Generator(device="cpu").manual_seed(pass_idx)
        z_H = z_H + (torch.randn(shape_H, generator=g) * sigma).to(device=device, dtype=dtype)
        z_L = z_L + (torch.randn(shape_L, generator=g) * sigma).to(device=device, dtype=dtype)

    new_current_data = {k: v.clone() for k, v in batch.items()}
    # halted=False signals "carry state is already populated, don't reset_carry"
    halted = torch.zeros(B, dtype=torch.bool, device=device)

    return replace(
        carry,
        inner_carry=InnerCarry(z_H=z_H, z_L=z_L),
        halted=halted,
        current_data=new_current_data,
    )


def _trm_single_pass(
    model,
    batch: dict,
    pass_idx: int,
    sigma: float,
) -> torch.Tensor:
    """One full ACT rollout with seeded latent-init perturbation. Returns preds [B, L]."""
    carry = _trm_perturbed_carry(model, batch, pass_idx, sigma)
    for _ in range(model.config.halt_max_steps):
        carry, outputs = model(carry=carry, batch=batch)
    return outputs["logits"].argmax(-1)


def run_k_vote_trm(
    model,
    loss_head,  # kept in signature for API symmetry; ACT inference uses model directly
    loader,
    k_values: list[int],
    device: str,
    task_id: int = 0,
    latent_sigma: float = 0.02,
    output_dir: str = "experiments",
) -> list[dict]:
    """Majority-vote over K stochastic TRM rollouts; one entry per K."""
    del task_id, loss_head  # task_id is baked into the collate; loss_head unused at inference
    model.eval()
    results = []

    for k in k_values:
        carbon = CarbonTracker(f"k_vote_trm_k{k}", output_dir=output_dir)
        carbon.start()
        t0 = time.perf_counter()

        cell_correct = cell_total = puz_correct = puz_total = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"TRM K-vote (K={k})"):
                batch = {name: v.to(device) for name, v in batch.items()}
                labels = batch["labels"]
                # Passes kept independent per K — reusing predictions across K
                # would conflate latency/energy numbers the caller compares.
                preds_stack = torch.stack(
                    [_trm_single_pass(model, batch, i, latent_sigma) for i in range(k)],
                    dim=0,
                )
                voted = _majority_vote(preds_stack)

                cc, ct, pc, pt = _score_batch(voted, labels)
                cell_correct += cc
                cell_total += ct
                puz_correct += pc
                puz_total += pt

        elapsed = time.perf_counter() - t0
        emissions = carbon.stop()
        kwh = float(emissions.get("energy_kwh", 0.0))

        results.append({
            "k": k,
            "puzzle_acc": puz_correct / max(1, puz_total),
            "cell_acc": cell_correct / max(1, cell_total),
            "mean_latency_ms": (elapsed * 1000.0) / max(1, puz_total),
            "kwh_per_puzzle": kwh / max(1, puz_total),
            "n_puzzles": puz_total,
        })

    return results


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _llm_forward_logits(model, inputs: torch.Tensor) -> torch.Tensor:
    """Return next-token logits aligned to label positions [B, L-1, V].

    Mirrors trainer_llm.evaluate: shift by one (logits[:-1] predicts labels[1:]).
    Handles BaselineLLM (returns HF output object) and DistilledLLM (tensor).
    """
    out = model(inputs) if not hasattr(model, "model") else model(input_ids=inputs)
    logits = out.logits if hasattr(out, "logits") else out
    # Only shift for causal LMs (BaselineLLM). DistilledLLM is non-causal and
    # returns positionally-aligned logits already — detect via presence of
    # HF .logits attr on the raw forward output.
    if hasattr(out, "logits"):
        return logits[:, :-1, :]
    return logits


def _llm_sample_preds_k(
    logits: torch.Tensor, temperature: float, k: int, seed: int = 0,
) -> torch.Tensor:
    """Draw K temperature-sampled predictions per position in one pass.

    Returns [K, B, L]. Softmax runs once per batch (not K times) and
    multinomial stays on-device with num_samples=k, eliminating the
    ~768 MB GPU->CPU copy that dominated the old per-pass loop.
    """
    B, L, V = logits.shape
    if temperature <= 0:
        # Argmax is deterministic — duplicate across K for shape consistency
        # so the caller's mode-vote reduces to the same value.
        preds = logits.argmax(-1)
        return preds.unsqueeze(0).expand(k, -1, -1).contiguous()
    scaled = logits / temperature
    probs = F.softmax(scaled.float(), dim=-1).reshape(-1, V)  # [B*L, V]
    # Match the generator device to the tensor so we avoid a CPU round-trip;
    # seed is per-batch (not per-pass) because reproducibility within a batch
    # is what matters for the K-vote comparison.
    gen_device = probs.device if probs.is_cuda else torch.device("cpu")
    g = torch.Generator(device=gen_device).manual_seed(seed)
    idx = torch.multinomial(probs, num_samples=k, replacement=True, generator=g)  # [B*L, K]
    return idx.T.reshape(k, B, L).contiguous()


def run_k_vote_llm(
    model,
    loader,
    k_values: list[int],
    device: str,
    temperature: float = 0.7,
    output_dir: str = "experiments",
) -> list[dict]:
    """Majority-vote over K temperature-sampled LLM predictions."""
    model.eval()
    results = []

    for k in k_values:
        carbon = CarbonTracker(f"k_vote_llm_k{k}", output_dir=output_dir)
        carbon.start()
        t0 = time.perf_counter()

        cell_correct = cell_total = puz_correct = puz_total = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"LLM K-vote (K={k})"):
                # Loader may be tuple-style (trainer_llm) or dict-style (official).
                if isinstance(batch, dict):
                    inputs = batch["inputs"].to(device)
                    raw_labels = batch["labels"].to(device)
                    # Align labels to shifted-logit space when the LM is causal.
                    # We compute both spaces below and pick whichever matches logits.
                else:
                    inputs, raw_labels = batch
                    inputs = inputs.to(device)
                    raw_labels = raw_labels.to(device)

                logits = _llm_forward_logits(model, inputs)
                # Align labels. Shifted logits -> shifted labels. Also normalize
                # the dataset ignore sentinel (0) to -100 so _score_batch is
                # consistent with the TRM path.
                if logits.shape[1] == inputs.shape[1] - 1:
                    labels = raw_labels[:, 1:]
                else:
                    labels = raw_labels
                labels = torch.where(labels == 0, torch.full_like(labels, IGNORE_LABEL_ID), labels)

                # Single softmax + single multinomial(num_samples=k); the old
                # per-pass list-comp redid softmax and a 768 MB GPU->CPU copy K
                # times per batch, which dominated wall-clock (~2 s × K / batch).
                preds_stack = _llm_sample_preds_k(logits, temperature, k)
                # K=1 with temp>0 is stochastic; that's the intended baseline —
                # the paper reports K=1 vs K>1 under the same sampling regime.
                voted = _majority_vote(preds_stack)

                cc, ct, pc, pt = _score_batch(voted, labels)
                cell_correct += cc
                cell_total += ct
                puz_correct += pc
                puz_total += pt

        elapsed = time.perf_counter() - t0
        emissions = carbon.stop()
        kwh = float(emissions.get("energy_kwh", 0.0))

        results.append({
            "k": k,
            "puzzle_acc": puz_correct / max(1, puz_total),
            "cell_acc": cell_correct / max(1, cell_total),
            "mean_latency_ms": (elapsed * 1000.0) / max(1, puz_total),
            "kwh_per_puzzle": kwh / max(1, puz_total),
            "n_puzzles": puz_total,
        })

    return results
