"""Forensic harness: reproduce the 0.789 -> 0.11 puzzle_acc crash on
TRMOfficial maze after one optimizer step from the HF-init weights, and
pinpoint the layers whose deltas dominate the update.

Pipeline:
  1. Build model + HF-remapped init via OfficialTRMTrainer._load_init_weights
     (imported -- no reimplementation). Snapshot state_dict -> snap_0.pt.
  2. Run a quick test-set eval (evaluate_official). Expect ~0.789.
  3. Pull ONE batch from the train loader, do forward/backward/step, logging
     per-param-group pre-clip grad norms and top params by |grad|/total.
  4. Snapshot state_dict -> snap_1.pt. Re-run eval. Expect collapse.
  5. Rank layers by L2(delta) between snap_0 and snap_1. Cross-reference
     with missing keys from the HF load to test the "random-init dominates"
     hypothesis.

Usage:
    python scripts/forensic_maze_corruption.py
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import torch
from torch.utils.data import DataLoader

from src.data.collate import official_collate_fn
from src.data.maze_dataset import MazeDataset
from src.evaluation.evaluate import evaluate_official
from src.models.losses_official import ACTLossHead
from src.models.trm_official import TRMOfficial
from src.training.trainer_official import OfficialTRMTrainer
from src.utils.config import load_config
from src.utils.seed import set_seed

CONFIG_PATH = "configs/trm_official_maze.yaml"
INIT_WEIGHTS = "hf_checkpoints/Maze-Hard/remapped_for_local.pt"


def _resolve_forensic_out_dir() -> str:
    """Pick a non-OneDrive output directory for the snap_*.pt tensors.

    Forensic snapshots are ~27 MB each. Writing them into the repo's
    results/ directory would bloat the user's OneDrive quota (the repo is
    typically cloned inside OneDrive for multi-device sync). Priority:
      1. $TRM_FORENSIC_OUT_DIR if explicitly set
      2. $TRM_WORK_DIR / forensic (follows the same convention as training)
      3. C:/ml-trm-work/forensic on Windows
      4. ~/ml-trm-work/forensic elsewhere
      5. results/forensic as last-resort fallback (works everywhere)
    The chosen dir is created eagerly; the run.log stays in the repo's
    results/forensic/ so the small text trail remains discoverable without
    needing to know where the binary tensors landed.
    """
    explicit = os.environ.get("TRM_FORENSIC_OUT_DIR")
    if explicit:
        out = explicit
    else:
        work = os.environ.get("TRM_WORK_DIR")
        if work:
            out = os.path.join(work, "forensic")
        elif os.name == "nt":
            out = os.path.join("C:/", "ml-trm-work", "forensic")
        elif os.path.expanduser("~"):
            out = os.path.join(os.path.expanduser("~"), "ml-trm-work", "forensic")
        else:
            out = "results/forensic"
    if "onedrive" in out.lower():
        # Never silently fall back into OneDrive — that defeats the whole
        # point of this picker. Force the repo-local fallback and shout.
        print(f"[forensic] WARNING: candidate out dir looks OneDrive-ish: {out}")
        print(f"[forensic] falling back to results/forensic (legacy behaviour)")
        out = "results/forensic"
    os.makedirs(out, exist_ok=True)
    return out


OUT_DIR = _resolve_forensic_out_dir()


def _build_model(config):
    mc = {
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
    return TRMOfficial(mc)


def _loaders(config):
    collate_fn = official_collate_fn(config.training.task_id)
    train_ds = MazeDataset(config.data.data_dir, "train", mask_non_path=config.data.mask_non_path)
    test_ds = MazeDataset(config.data.data_dir, "test", mask_non_path=config.data.mask_non_path)
    # num_workers=0 so the single-batch pull is deterministic and cheap.
    train_loader = DataLoader(
        train_ds, batch_size=config.training.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        test_ds, batch_size=config.training.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )
    return train_loader, val_loader


def _eval(model, val_loader, config, tag: str) -> dict:
    # evaluate_official handles device + bf16 cast internally. No EMA.
    model.eval()
    t0 = time.time()
    res = evaluate_official(model, val_loader, config, ema=None)
    dt = time.time() - t0
    print(f"[eval:{tag}] cell={res['cell_accuracy']:.4f} "
          f"puzzle={res['puzzle_accuracy']:.4f} "
          f"avg_steps={res.get('avg_act_steps', 0):.2f} ({dt:.1f}s)")
    return res


def _named_state_dict_snapshot(model) -> dict:
    # Clone to CPU-fp32 so deltas are computed in fp32 even though the live
    # model runs bf16 (bf16 subtraction would lose ~3 decimal digits of
    # precision and make the ranking noisy for small-delta layers).
    return {k: v.detach().to("cpu", torch.float32).clone() for k, v in model.state_dict().items()}


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    config = load_config(CONFIG_PATH)
    set_seed(42)

    # Forensic runs should not pollute the shared wandb project with diagnostic
    # rows that look like real training. OfficialTRMTrainer.__init__ checks
    # config.training.use_wandb before calling init_wandb, so disabling it
    # here is sufficient. Same for auto_continue (forensic is a single-step
    # probe, not a resumable run).
    config.training.use_wandb = False
    config.training.auto_continue = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} forward_dtype={config.model.forward_dtype}")
    print(f"[env] snap_*.pt output dir: {os.path.abspath(OUT_DIR)}")
    print(f"[env]   (override via TRM_FORENSIC_OUT_DIR or TRM_WORK_DIR)")

    model = _build_model(config)
    loss_head = ACTLossHead(model)
    train_loader, val_loader = _loaders(config)

    # --- STEP 1: build trainer (triggers _load_init_weights) ---
    trainer = OfficialTRMTrainer(
        model, loss_head, train_loader, val_loader, config,
        resume_checkpoint="", init_weights=INIT_WEIGHTS,
    )

    # Keys that stayed random after the partial state_dict load -- the
    # hypothesis under test. We recover this via a second dry-run load call
    # to collect strict=False diagnostics without mutating state.
    ckpt = torch.load(INIT_WEIGHTS, map_location="cpu", weights_only=False)
    init_state = ckpt.get("model_state_dict", ckpt)
    missing_keys = [k for k in model.state_dict().keys() if k not in init_state]
    print(f"[hf_load] model has {len(list(model.state_dict().keys()))} keys; "
          f"{len(missing_keys)} missing from HF (randomly initialised).")
    for k in missing_keys:
        print(f"  MISSING: {k}")

    snap_0 = _named_state_dict_snapshot(model)
    snap_0_path = os.path.join(OUT_DIR, "snap_0_after_load.pt")
    torch.save(snap_0, snap_0_path)
    print(f"[snap] snap_0_after_load.pt saved -> {snap_0_path}")

    # --- STEP 2: baseline eval at snap_0 ---
    eval_0 = _eval(model, val_loader, config, "snap_0")

    # --- STEP 3: ONE optimizer step on one batch ---
    model.train()
    loss_head.train()
    print(f"[train] model.training={model.training} loss_head.training={loss_head.training}")

    batch = next(iter(train_loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    print(f"[train] batch sizes: {[(k, tuple(v.shape), v.dtype) for k, v in batch.items()]}")

    carry = loss_head.initial_carry(batch)
    # Single carry-forward step (matches trainer loop's inner call). The trainer
    # actually does multiple inner ACT steps per batch -- we only need one to
    # reproduce the update that already does the damage on step 1.
    carry, loss, metrics, _outputs, all_halted = loss_head(
        return_keys=(), carry=carry, batch=batch,
    )
    print(f"[train] loss={float(loss):.4f} "
          f"lm={float(metrics.get('lm_loss', 0)):.4f} "
          f"q_halt={float(metrics.get('q_halt_loss', 0)):.4f} "
          f"q_cont={float(metrics.get('q_continue_loss', 0)):.4f} "
          f"all_halted={bool(all_halted)}")

    loss.backward()

    # Per-param-group pre-clip grad norm (group 0 = non-task-emb, group 1 = task_emb)
    for gi, group in enumerate(trainer.optimizer.param_groups):
        gn = torch.norm(
            torch.stack([p.grad.detach().float().norm(2)
                         for p in group["params"] if p.grad is not None])
        )
        print(f"[grad] param_group[{gi}] (lr={group['lr']:g}) pre-clip ||g||2 = {float(gn):.4f}")

    # Total grad norm + top contributors (by |g| / total) BEFORE clipping
    per_param_norms = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        per_param_norms.append((name, float(p.grad.detach().float().norm(2))))
    total_pre_clip = (sum(n * n for _, n in per_param_norms)) ** 0.5
    per_param_norms.sort(key=lambda t: t[1], reverse=True)
    print(f"[grad] total pre-clip ||g||2 = {total_pre_clip:.4f} "
          f"(max_grad_norm={config.training.max_grad_norm})")
    print(f"[grad] top params with |g|/total > 0.1:")
    for name, n in per_param_norms[:15]:
        ratio = n / max(1e-12, total_pre_clip)
        flag = " <-- RANDOM-INIT" if name in missing_keys else ""
        if ratio > 0.1:
            print(f"  {ratio*100:5.1f}% |g|={n:8.2f} {name}{flag}")

    # Apply clip (matches trainer behaviour)
    clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
    print(f"[grad] post clip_grad_norm_ return = {float(clipped):.4f}")

    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    # NOTE: we do NOT update EMA or scheduler here -- evaluate_official is
    # being called with ema=None, so the shadow wouldn't be applied anyway.

    # --- STEP 4: snapshot after step ---
    snap_1 = _named_state_dict_snapshot(model)
    snap_1_path = os.path.join(OUT_DIR, "snap_1_after_one_step.pt")
    torch.save(snap_1, snap_1_path)
    print(f"[snap] snap_1_after_one_step.pt saved -> {snap_1_path}")

    # --- STEP 5: eval at snap_1 ---
    eval_1 = _eval(model, val_loader, config, "snap_1")

    # --- STEP 6: rank layers by L2 delta ---
    deltas: list[tuple[str, float, float]] = []  # (name, ||delta||, ||w0||)
    for name, w0 in snap_0.items():
        w1 = snap_1.get(name)
        if w1 is None:
            continue
        d = (w1 - w0).norm(2).item()
        w0n = w0.norm(2).item()
        deltas.append((name, d, w0n))
    deltas.sort(key=lambda t: t[1], reverse=True)

    print("\n[delta] top 15 layers by ||w1 - w0||2 (fp32):")
    print(f"  {'rank':>4}  {'||delta||':>10}  {'||w0||':>10}  {'rel %':>6}  name")
    for i, (name, d, w0n) in enumerate(deltas[:15]):
        rel = (d / max(1e-12, w0n)) * 100.0
        flag = " <-- RANDOM-INIT" if name in missing_keys else ""
        print(f"  {i+1:>4}  {d:>10.4f}  {w0n:>10.4f}  {rel:>5.2f}  {name}{flag}")

    # --- Summary ---
    print("\n=== SUMMARY ===")
    print(f"  eval @ snap_0 (post-HF-load):    puzzle={eval_0['puzzle_accuracy']:.4f} "
          f"cell={eval_0['cell_accuracy']:.4f}")
    print(f"  eval @ snap_1 (post-one-step):   puzzle={eval_1['puzzle_accuracy']:.4f} "
          f"cell={eval_1['cell_accuracy']:.4f}")
    delta_puzzle = eval_0['puzzle_accuracy'] - eval_1['puzzle_accuracy']
    print(f"  puzzle_acc drop:                 {delta_puzzle:+.4f}")
    top_k_missing = [n for n, _, _ in deltas[:5] if n in missing_keys]
    print(f"  top-5 delta layers that were RANDOM-INIT at load: "
          f"{len(top_k_missing)}/5 -> {top_k_missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
