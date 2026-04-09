from enum import Enum
from typing import Optional

import yaml
from pydantic import BaseModel


class ModelType(str, Enum):
    TRM_SUDOKU = "trm_sudoku"
    TRM_MAZE = "trm_maze"
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
    log_interval: int = 50
    save_interval: int = 500

    # HuggingFace Hub checkpoint sync (empty string = disabled)
    hf_repo_id: str = ""


class DataConfig(BaseModel):
    dataset: str = "sudoku"
    data_dir: str = "data/sudoku-extreme-full"
    num_workers: int = 4
    subsample_size: Optional[int] = None


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
    return ExperimentConfig(**raw)
