import os
from enum import Enum
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env from repo root on module import (idempotent — safe if called twice).
# Any machine-local secrets/paths in .env become available via os.getenv BEFORE
# wandb.init() or huggingface_hub auth run.
load_dotenv()


class ModelType(str, Enum):
    TRM_SUDOKU = "trm_sudoku"
    TRM_MAZE = "trm_maze"
    TRM_OFFICIAL_SUDOKU = "trm_official_sudoku"
    TRM_OFFICIAL_MAZE = "trm_official_maze"
    LLM_FINETUNE = "llm_finetune"
    LLM_DISTILL = "llm_distill"


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


class TrainingConfig(BaseModel):
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1.0
    warmup_steps: int = 2000
    batch_size: int = 32
    grad_accum_steps: int = 24
    epochs: int = 5000
    ema_decay: float = 0.999
    n_latent: int = 6
    T_deep: int = 3
    N_sup: int = 16
    act_threshold: float = 0.5
    max_grad_norm: float = 1.0

    # Distillation-specific
    distill_alpha: float = 0.7
    distill_temperature: float = 4.0

    # Auto-continue: keep training in increments when epochs finish successfully
    auto_continue: bool = False
    epoch_increment: int = 0  # 0 = same as initial epochs

    # Logging
    use_wandb: bool = False
    wandb_project: str = "trm-coursework"
    wandb_entity: str = ""  # empty = use your default wandb user/team
    use_weave: bool = True  # Weave traces for wandb.ai/<entity>/<project>/weave/monitors
    log_interval: int = 50
    save_interval: int = 500
    # How often (epochs) to run the full validation pass. 0 = fall back to
    # log_interval (legacy behavior where logging and eval were fused). Split
    # them when eval is expensive — e.g. log_interval=5 for cheap train-metric
    # printing, eval_interval=50 to run the full test split only 10 times per
    # 500-epoch run instead of 100 times.
    eval_interval: int = 0

    # HuggingFace Hub checkpoint sync (empty string = disabled)
    hf_repo_id: str = ""

    # Rolling local crash-recovery backup (e.g. external drive)
    rolling_checkpoint_dir: str = ""        # empty = disabled
    rolling_checkpoint_max: int = 3         # keep N most recent
    rolling_checkpoint_interval: int = 100  # save every N epochs

    # Milestone snapshots at fixed fractions of training (never rotated)
    milestone_checkpoints: bool = False
    milestone_fractions: list[float] = Field(
        default_factory=lambda: [0.10, 0.25, 0.50, 0.75]
    )

    # wandb artifact upload for best.pt on every new best
    wandb_best_artifact: bool = False

    # Official TRM optimizer
    optimizer: str = "adamw"  # "adamw" or "adam_atan2"
    task_emb_lr: float = 0.01
    task_emb_weight_decay: float = 0.1

    # ACT loss weighting (applied to q_halt_loss + q_continue_loss)
    # Paper uses 0.5 (from-scratch). Drop to 0.01 when fine-tuning from a
    # pretrained checkpoint to prevent Q-loss hijacking the backbone.
    q_loss_weight: float = 0.5

    # Task ID for collate
    task_id: int = 0  # 0=sudoku, 1=maze

    # Early stopping (LLMTrainer only): halt when the monitored metric has not
    # improved for `early_stop_patience` epochs. Useful for LLM baselines that
    # plateau well before the full epoch budget — e.g. cell_acc flatlines after
    # ~50 epochs on a 500-epoch run. Disabled when patience=0.
    # metric: "val_cell_acc" | "val_puzzle_acc" | "train_loss"
    # mode:   "max" (higher is better) | "min" (lower is better)
    early_stop_patience: int = 0
    early_stop_metric: str = "val_cell_acc"
    early_stop_mode: str = "max"
    early_stop_min_delta: float = 1e-4


class DataConfig(BaseModel):
    dataset: str = "sudoku"
    data_dir: str = "data/sudoku-extreme-full"
    num_workers: int = 4
    subsample_size: Optional[int] = None
    # Maze-only: when True (paper behavior), CE loss ignores non-path cells,
    # which creates a reward-hacking attractor where a constant-`o` predictor
    # scores 100% on puzzle_acc without solving any maze. Set False to grade
    # all 900 cells during training — the model must correctly output walls,
    # open cells, S, G, AND the path. Fixes the degenerate-optimum issue seen
    # on maze-seed0.
    mask_non_path: bool = True


class ExperimentConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "models"
    experiment_dir: str = "experiments"


def load_config(path: str) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentConfig(**raw)
    _apply_env_overrides(cfg)
    return cfg


def _apply_env_overrides(cfg: ExperimentConfig) -> None:
    """Override blank config fields from environment variables.

    Semantics: an explicit YAML value always wins over env. Only fields that
    are empty-string in YAML get overridden. This lets per-model configs
    (e.g. llm_qwen.yaml) set their own wandb_project without being silently
    clobbered by a blanket TRM_WANDB_PROJECT env var, while machine-specific
    fields (entity, HF repo, rolling dir) default to "" in YAML and get
    filled from .env on each machine.
    """
    # Wandb
    if not cfg.training.wandb_entity:
        cfg.training.wandb_entity = os.getenv("TRM_WANDB_ENTITY", "")
    if os.getenv("TRM_WANDB_PROJECT"):
        cfg.training.wandb_project = os.environ["TRM_WANDB_PROJECT"]

    # HuggingFace Hub
    if not cfg.training.hf_repo_id:
        cfg.training.hf_repo_id = os.getenv("TRM_HF_REPO_ID", "")

    # Rolling checkpoint dir
    if not cfg.training.rolling_checkpoint_dir:
        cfg.training.rolling_checkpoint_dir = os.getenv("TRM_ROLLING_CHECKPOINT_DIR", "")

    # Data / checkpoint paths
    if os.getenv("TRM_DATA_DIR"):
        cfg.data.data_dir = os.environ["TRM_DATA_DIR"]
    if os.getenv("TRM_CHECKPOINT_DIR"):
        cfg.checkpoint_dir = os.environ["TRM_CHECKPOINT_DIR"]
    if os.getenv("TRM_EXPERIMENT_DIR"):
        cfg.experiment_dir = os.environ["TRM_EXPERIMENT_DIR"]

    # Multi-machine safety: warn if training outputs land inside a OneDrive-
    # synced path. OneDrive rewrites files under your feet during upload,
    # which corrupts mid-write checkpoints and creates rename conflicts when
    # two machines touch the same file. Training-output dirs should live on
    # local non-synced storage; set TRM_CHECKPOINT_DIR / TRM_EXPERIMENT_DIR
    # to a local path (e.g. C:/ml-trm-work/<task>-seed<N>) on each machine.
    for label, path in (
        ("checkpoint_dir", cfg.checkpoint_dir),
        ("experiment_dir", cfg.experiment_dir),
    ):
        if "onedrive" in path.lower():
            print(
                f"[config] WARNING: {label}='{path}' looks like a OneDrive "
                f"path. Parallel runs on shared OneDrive will corrupt "
                f"checkpoints. Set TRM_{label.upper()} to a local path."
            )
