import os

from argdantic import ArgParser
from pydantic import BaseModel

from src.utils.config import ExperimentConfig, ModelType, load_config
from src.utils.gpu_config import apply_gpu_overrides
from src.utils.seed import set_seed


cli = ArgParser()


class RunConfig(BaseModel):
    config: str = "configs/trm_sudoku.yaml"
    mode: str = "train"  # train | eval | distill
    checkpoint: str = ""
    resume: str = ""  # path to checkpoint to resume training from
    seed: int = 42


@cli.command(singleton=True)
def main(cfg: RunConfig):
    config = load_config(cfg.config)
    config.seed = cfg.seed
    apply_gpu_overrides(config)
    set_seed(cfg.seed)

    if cfg.mode == "train":
        _run_train(config, cfg.resume)
    elif cfg.mode == "eval":
        _run_eval(config, cfg.checkpoint)
    elif cfg.mode == "distill":
        _run_distill(config, cfg.checkpoint)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


def _run_train(config: ExperimentConfig, resume: str = "") -> None:
    increment = config.training.epoch_increment or config.training.epochs

    while True:
        _run_train_once(config, resume)

        if not config.training.auto_continue:
            break

        # Bump epochs and resume from latest checkpoint
        latest = os.path.join(config.checkpoint_dir, "latest.pt")
        if not os.path.isfile(latest):
            print("auto_continue: latest.pt not found, stopping")
            break

        config.training.epochs += increment
        resume = latest
        print(f"\n=== AUTO-CONTINUE: extending to {config.training.epochs} epochs ===\n")


def _run_train_once(config: ExperimentConfig, resume: str = "") -> None:
    model_type = config.model.model_type

    if model_type == ModelType.TRM_SUDOKU:
        from src.data.sudoku_dataset import get_sudoku_loaders
        from src.models.trm_sudoku import TRMSudoku
        from src.training.trainer_trm import TRMTrainer

        model = TRMSudoku(
            vocab_size=config.model.vocab_size,
            seq_len=config.model.seq_len,
            d_model=config.model.d_model,
            ff_hidden=config.model.ff_hidden,
            num_classes=config.model.num_classes,
        )
        print(f"TRM-Sudoku params: {model.param_count():,}")

        train_loader, val_loader = get_sudoku_loaders(
            config.data.data_dir,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
        )
        trainer = TRMTrainer(model, train_loader, val_loader, config, resume_checkpoint=resume)
        trainer.train()

    elif model_type == ModelType.TRM_MAZE:
        from src.data.maze_dataset import get_maze_loaders
        from src.models.trm_sudoku import TRMMaze
        from src.training.trainer_trm import TRMTrainer

        model = TRMMaze(
            vocab_size=config.model.vocab_size,
            seq_len=config.model.seq_len,
            d_model=config.model.d_model,
            ff_hidden=config.model.ff_hidden,
            num_classes=config.model.num_classes,
        )
        print(f"TRM-Maze params: {model.param_count():,}")

        train_loader, val_loader = get_maze_loaders(
            config.data.data_dir,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
        )
        trainer = TRMTrainer(model, train_loader, val_loader, config, resume_checkpoint=resume)
        trainer.train()

    elif model_type == ModelType.LLM_FINETUNE:
        from src.data.sudoku_dataset import get_sudoku_loaders
        from src.models.baseline_llm import BaselineLLM
        from src.training.trainer_llm import LLMTrainer

        model = BaselineLLM(
            model_name=config.model.llm_name,
            lora_r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            use_qlora=config.model.use_qlora,
        )
        print(f"LLM trainable params: {model.trainable_param_count():,} / {model.total_param_count():,}")

        train_loader, val_loader = get_sudoku_loaders(
            config.data.data_dir,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
        )
        trainer = LLMTrainer(model, train_loader, val_loader, config)
        trainer.train()

    else:
        raise ValueError(f"Use --mode distill for {model_type}")


def _run_eval(config: ExperimentConfig, checkpoint_path: str) -> None:
    if not checkpoint_path:
        raise ValueError("--checkpoint required for eval mode")

    from src.evaluation.evaluate import load_and_evaluate, save_results

    model_type = config.model.model_type
    if model_type == ModelType.TRM_SUDOKU:
        from src.data.sudoku_dataset import get_sudoku_loaders
        _, test_loader = get_sudoku_loaders(
            config.data.data_dir,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
        )
    elif model_type == ModelType.TRM_MAZE:
        from src.data.maze_dataset import get_maze_loaders
        _, test_loader = get_maze_loaders(
            config.data.data_dir,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
        )
    else:
        from src.data.sudoku_dataset import get_sudoku_loaders
        _, test_loader = get_sudoku_loaders(
            config.data.data_dir,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
        )

    results = load_and_evaluate(checkpoint_path, test_loader, config)
    save_results(results, "results", config.model.model_type.value)

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def _run_distill(config: ExperimentConfig, teacher_checkpoint: str) -> None:
    if not teacher_checkpoint:
        raise ValueError("--checkpoint required (path to trained teacher)")

    import torch

    from src.data.sudoku_dataset import get_sudoku_loaders
    from src.models.distilled_llm import DistilledLLM
    from src.training.trainer_distill import DistillationTrainer

    # Load teacher
    teacher_ckpt = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher = DistilledLLM(
        vocab_size=config.model.vocab_size,
        seq_len=config.model.seq_len,
    )
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])

    # Student
    student = DistilledLLM(
        vocab_size=config.model.vocab_size,
        seq_len=config.model.seq_len,
        d_model=config.model.distill_d_model,
        n_layers=config.model.distill_n_layers,
        ff_hidden=config.model.distill_ff_hidden,
        n_heads=config.model.distill_n_heads,
    )
    print(f"Student params: {student.param_count():,}")

    train_loader, val_loader = get_sudoku_loaders(
        config.data.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
    )

    trainer = DistillationTrainer(teacher, student, train_loader, val_loader, config)
    trainer.train()


if __name__ == "__main__":
    cli()
