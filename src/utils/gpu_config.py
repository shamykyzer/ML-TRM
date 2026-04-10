"""Auto-detect GPU and apply optimal training config overrides."""
import platform
import torch

from src.utils.config import ExperimentConfig


# Optimal settings per GPU for TRM training.
# batch_size = max that fits in VRAM with 16 deep supervision steps + AMP
# The paper uses effective batch_size=768, but for small datasets we skip
# grad_accum entirely -- deep supervision already does 16 optimizer steps/batch.
# num_workers = 0 on Windows (multiprocessing DataLoader crashes), 4 on Linux
GPU_PROFILES = {
    # 8GB VRAM
    "RTX 3070": {
        "vram_gb": 8,
        "sudoku": {"batch_size": 32},  # N_sup=16 needs smaller batch on 8GB
        "maze":   {"batch_size": 8},
    },
    "RTX 4060": {
        "vram_gb": 8,
        "sudoku": {"batch_size": 48},  # N_sup=8 fits comfortably on 8GB
        "maze":   {"batch_size": 8},
    },
    # 12GB VRAM
    "RTX 4070": {
        "vram_gb": 12,
        "sudoku": {"batch_size": 128},
        "maze":   {"batch_size": 16},
    },
    "RTX 5070": {
        "vram_gb": 12,
        "sudoku": {"batch_size": 64},   # N_sup=16 needs smaller batch on 12GB
        "maze":   {"batch_size": 16},
    },
    # 16GB VRAM
    "RTX 5090": {
        "vram_gb": 32,
        "sudoku": {"batch_size": 256},
        "maze":   {"batch_size": 64},
    },
    # 48GB VRAM (paper used L40S)
    "L40S": {
        "vram_gb": 48,
        "sudoku": {"batch_size": 768},
        "maze":   {"batch_size": 128},
    },
    # Fallback by VRAM size
    "default_8gb": {
        "vram_gb": 8,
        "sudoku": {"batch_size": 32},
        "maze":   {"batch_size": 4},
    },
    "default_12gb": {
        "vram_gb": 12,
        "sudoku": {"batch_size": 128},
        "maze":   {"batch_size": 16},
    },
    "default_16gb+": {
        "vram_gb": 16,
        "sudoku": {"batch_size": 256},
        "maze":   {"batch_size": 32},
    },
}


def detect_gpu() -> dict:
    """Detect GPU and return optimal training config."""
    if not torch.cuda.is_available():
        print("[GPU Config] No CUDA GPU detected. Using CPU defaults.")
        return GPU_PROFILES["default_8gb"]

    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)

    print(f"[GPU Config] Detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    # Match by name
    for profile_name, profile in GPU_PROFILES.items():
        if profile_name in gpu_name:
            print(f"[GPU Config] Using profile: {profile_name}")
            return profile

    # Fallback by VRAM size
    if vram_gb >= 16:
        print(f"[GPU Config] Unknown GPU, using 16GB+ defaults")
        return GPU_PROFILES["default_16gb+"]
    elif vram_gb >= 10:
        print(f"[GPU Config] Unknown GPU, using 12GB defaults")
        return GPU_PROFILES["default_12gb"]
    else:
        print(f"[GPU Config] Unknown GPU, using 8GB defaults")
        return GPU_PROFILES["default_8gb"]


def get_num_workers() -> int:
    """Return optimal num_workers for the current OS."""
    if platform.system() == "Windows":
        return 0  # Windows multiprocessing DataLoader crashes
    return 4


def apply_gpu_overrides(config: ExperimentConfig) -> None:
    """Apply GPU-optimal batch_size and num_workers to an ExperimentConfig in-place."""
    gpu_profile = detect_gpu()
    task = config.data.dataset if hasattr(config.data, "dataset") else "sudoku"
    task_profile = gpu_profile.get(task, gpu_profile.get("sudoku", {}))

    if "batch_size" in task_profile:
        config.training.batch_size = task_profile["batch_size"]

    config.data.num_workers = get_num_workers()

    print(f"[GPU Config] batch_size={config.training.batch_size}, "
          f"num_workers={config.data.num_workers}")


if __name__ == "__main__":
    # Manual GPU-detection debugging: `python -m src.utils.gpu_config`
    profile = detect_gpu()
    print(f"\nProfile: {profile}")
    print(f"num_workers (this OS): {get_num_workers()}")
