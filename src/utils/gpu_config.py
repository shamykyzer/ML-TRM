"""Auto-detect GPU and apply optimal training config overrides."""
import os
import platform
import torch
import yaml


# Optimal settings per GPU for TRM Sudoku (seq_len=81, d_model=512, 6.4M params)
# batch_size = max that fits in VRAM with 16 deep supervision steps + AMP
# num_workers = 0 on Windows (multiprocessing DataLoader crashes), 4 on Linux
GPU_PROFILES = {
    "RTX 3070": {
        "vram_gb": 8,
        "sudoku": {"batch_size": 48, "grad_accum_steps": 16},  # effective 768
        "maze":   {"batch_size": 8,  "grad_accum_steps": 96},  # effective 768
    },
    "RTX 4060": {
        "vram_gb": 8,
        "sudoku": {"batch_size": 48, "grad_accum_steps": 16},  # effective 768
        "maze":   {"batch_size": 8,  "grad_accum_steps": 96},  # effective 768
    },
    "RTX 4070": {
        "vram_gb": 12,
        "sudoku": {"batch_size": 128, "grad_accum_steps": 6},  # effective 768
        "maze":   {"batch_size": 16,  "grad_accum_steps": 48}, # effective 768
    },
    "RTX 5070": {
        "vram_gb": 12,
        "sudoku": {"batch_size": 128, "grad_accum_steps": 6},  # effective 768
        "maze":   {"batch_size": 16,  "grad_accum_steps": 48}, # effective 768
    },
    # Fallback for unknown GPUs
    "default_8gb": {
        "vram_gb": 8,
        "sudoku": {"batch_size": 32, "grad_accum_steps": 24},
        "maze":   {"batch_size": 4,  "grad_accum_steps": 192},
    },
    "default_12gb": {
        "vram_gb": 12,
        "sudoku": {"batch_size": 128, "grad_accum_steps": 6},
        "maze":   {"batch_size": 16,  "grad_accum_steps": 48},
    },
    "default_16gb+": {
        "vram_gb": 16,
        "sudoku": {"batch_size": 256, "grad_accum_steps": 3},
        "maze":   {"batch_size": 32,  "grad_accum_steps": 24},
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


def auto_tune_config(config_path: str) -> str:
    """Read a YAML config, apply GPU-optimal overrides, write to a temp config, return path."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    gpu_profile = detect_gpu()
    task = config.get("data", {}).get("dataset", "sudoku")
    task_profile = gpu_profile.get(task, gpu_profile.get("sudoku", {}))

    # Apply overrides
    if "training" in config:
        config["training"]["batch_size"] = task_profile.get("batch_size", config["training"]["batch_size"])
        config["training"]["grad_accum_steps"] = task_profile.get("grad_accum_steps", config["training"].get("grad_accum_steps", 1))

    if "data" in config:
        config["data"]["num_workers"] = get_num_workers()

    effective_batch = config["training"]["batch_size"] * config["training"]["grad_accum_steps"]
    print(f"[GPU Config] batch_size={config['training']['batch_size']}, "
          f"grad_accum={config['training']['grad_accum_steps']}, "
          f"effective_batch={effective_batch}, "
          f"num_workers={config['data']['num_workers']}")

    # Write tuned config to temp file
    tuned_path = config_path.replace(".yaml", "_tuned.yaml")
    with open(tuned_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return tuned_path


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/trm_sudoku.yaml"
    tuned = auto_tune_config(path)
    print(f"\nTuned config written to: {tuned}")
