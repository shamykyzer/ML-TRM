import os
import time

from dotenv import load_dotenv

# Load .env before anything else — wandb/huggingface/weave all read their
# auth from env vars at import time. Explicit call here is defensive
# redundancy (config.py also calls it on import) so the dependency is
# visible at the entry point.
load_dotenv()

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
    # Initialize model weights from a file (partial state_dict OK), but start
    # training from scratch — optimizer/step/epoch all reset to zero. Use this
    # for transfer learning from a pretrained checkpoint. Mutually exclusive
    # with --resume (which restores the full training state).
    init_weights: str = ""
    # -1 (default) = inherit from YAML's `seed:` field (reproducible).
    # Pass an explicit non-negative int to override for one run.
    # To opt into a fresh wall-clock seed per run, set `seed: -1` in the YAML.
    seed: int = -1
    # -1 (default) = inherit YAML's `training.epochs`. Pass a non-negative
    # int to override for one run. Used by scripts/run_seed.sh --dry-run to
    # run a 5-epoch pipeline smoke test without editing the YAML.
    epochs: int = -1


@cli.command(singleton=True)
def main(cfg: RunConfig):
    config = load_config(cfg.config)
    # Seed priority: CLI --seed > YAML config.seed > wall-clock fallback.
    # YAML default is 42 (see ExperimentConfig in src/utils/config.py), so
    # running without --seed gives deterministic, reproducible runs. Only
    # YAML seed < 0 triggers the legacy wall-clock behavior.
    if cfg.seed < 0:
        cfg.seed = config.seed if config.seed >= 0 else int(time.time()) % (2**31)
    config.seed = cfg.seed
    if cfg.epochs >= 0:
        config.training.epochs = cfg.epochs
        print(f"[Epochs] overridden to {cfg.epochs}")
    apply_gpu_overrides(config)
    print(f"[Seed] {cfg.seed}")
    set_seed(cfg.seed)

    if cfg.mode == "train":
        if cfg.resume and cfg.init_weights:
            raise ValueError(
                "--resume and --init-weights are mutually exclusive. "
                "--resume restores full training state (optimizer, step, epoch); "
                "--init-weights only loads model weights and starts fresh."
            )
        _run_train(config, cfg.resume, cfg.init_weights)
    elif cfg.mode == "eval":
        _run_eval(config, cfg.checkpoint)
    elif cfg.mode == "distill":
        _run_distill(config, cfg.checkpoint)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


def _run_train(config: ExperimentConfig, resume: str = "", init_weights: str = "") -> None:
    increment = config.training.epoch_increment or config.training.epochs

    first_iteration = True
    while True:
        # init_weights only applies to the first iteration. Auto-continue
        # rounds after that resume from latest.pt (full training state), so
        # we do NOT re-apply init_weights on each continuation.
        _run_train_once(config, resume, init_weights if first_iteration else "")
        first_iteration = False

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


def _run_train_once(config: ExperimentConfig, resume: str = "", init_weights: str = "") -> None:
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
        loss_head = ACTLossHead(model, q_loss_weight=config.training.q_loss_weight)
        print(f"TRM-Official params: {model.param_count():,}")

        collate_fn = official_collate_fn(config.training.task_id)

        if config.data.dataset == "maze":
            from src.data.maze_dataset import MazeDataset
            from torch.utils.data import DataLoader

            train_ds = MazeDataset(
                config.data.data_dir, "train",
                mask_non_path=config.data.mask_non_path,
            )
            test_ds = MazeDataset(
                config.data.data_dir, "test",
                mask_non_path=config.data.mask_non_path,
            )
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
            init_weights=init_weights,
        )
        trainer.train()

    elif model_type == ModelType.LLM_FINETUNE:
        from src.models.baseline_llm import BaselineLLM
        from src.training.trainer_llm import LLMTrainer

        model = BaselineLLM(
            model_name=config.model.llm_name,
            lora_r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            use_qlora=config.model.use_qlora,
            use_gradient_checkpointing=config.model.use_gradient_checkpointing,
        )
        print(f"LLM trainable params: {model.trainable_param_count():,} / {model.total_param_count():,}")

        # Pick the loader matching the dataset declared in the YAML. Both
        # sudoku and maze datasets emit (input_ids, labels) with label=0
        # marking ignore positions; trainer_llm handles the HF -100 remap.
        if config.data.dataset == "maze":
            from src.data.maze_dataset import get_maze_loaders
            train_loader, val_loader = get_maze_loaders(
                config.data.data_dir,
                batch_size=config.training.batch_size,
                num_workers=config.data.num_workers,
            )
        else:
            from src.data.sudoku_dataset import get_sudoku_loaders
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

    from src.models.distilled_llm import DistilledLLM
    from src.training.trainer_distill import DistillationTrainer

    teacher_ckpt = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher_cfg = teacher_ckpt.get("config", {})
    teacher_model_type = teacher_cfg.get("model", {}).get("model_type", "")

    # Detect teacher type from its saved config so we can distill from EITHER:
    #   • a fine-tuned BaselineLLM (the headline pipeline — proposal calls this
    #     out as "compact student model trained to approximate the fine-tuned
    #     LLM"), OR
    #   • a previously-trained DistilledLLM (self-distillation, smaller-still).
    if teacher_model_type == ModelType.LLM_FINETUNE.value:
        from src.models.baseline_llm import BaselineLLM
        teacher_model_cfg = teacher_cfg.get("model", {})
        teacher = BaselineLLM(
            model_name=teacher_model_cfg.get("llm_name", config.model.llm_name),
            lora_r=teacher_model_cfg.get("lora_r", config.model.lora_r),
            lora_alpha=teacher_model_cfg.get("lora_alpha", config.model.lora_alpha),
            use_qlora=teacher_model_cfg.get("use_qlora", config.model.use_qlora),
            use_gradient_checkpointing=teacher_model_cfg.get(
                "use_gradient_checkpointing", config.model.use_gradient_checkpointing
            ),
        )
        # .to(device) must precede load: QLoRA bnb Linear4bit quant buffers
        # (absmax/quant_map/quant_state) only materialize once weights land on CUDA.
        teacher_device = config.device if torch.cuda.is_available() else "cpu"
        teacher.to(teacher_device)
        teacher.load_state_dict(teacher_ckpt["model_state_dict"])
        teacher_kind = "baseline_llm"
    else:
        teacher = DistilledLLM(
            vocab_size=config.model.vocab_size,
            seq_len=config.model.seq_len,
        )
        teacher.load_state_dict(teacher_ckpt["model_state_dict"])
        teacher_kind = "distilled_llm"

    student = DistilledLLM(
        vocab_size=config.model.vocab_size,
        seq_len=config.model.seq_len,
        d_model=config.model.distill_d_model,
        n_layers=config.model.distill_n_layers,
        ff_hidden=config.model.distill_ff_hidden,
        n_heads=config.model.distill_n_heads,
    )
    print(f"Teacher: {teacher_kind} | Student params: {student.param_count():,}")

    if config.data.dataset == "maze":
        from src.data.maze_dataset import get_maze_loaders
        train_loader, val_loader = get_maze_loaders(
            config.data.data_dir,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
        )
    else:
        from src.data.sudoku_dataset import get_sudoku_loaders
        train_loader, val_loader = get_sudoku_loaders(
            config.data.data_dir,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
        )

    trainer = DistillationTrainer(
        teacher, student, train_loader, val_loader, config,
        teacher_kind=teacher_kind,
    )
    trainer.train()


if __name__ == "__main__":
    cli()
