import logging
import os
import warnings

# Suppress noisy pynvml/codecarbon GPU warnings on WSL
logging.getLogger("codecarbon").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Failed to retrieve gpu.*")
warnings.filterwarnings("ignore", message=".*NVMLError.*")

from codecarbon import EmissionsTracker


class CarbonTracker:
    """Wrapper around CodeCarbon for energy/CO2 tracking."""

    def __init__(self, experiment_name: str, output_dir: str = "experiments"):
        os.makedirs(output_dir, exist_ok=True)
        self.tracker = EmissionsTracker(
            project_name=experiment_name,
            output_dir=output_dir,
            log_level="error",
            gpu_ids=None,  # Disable GPU tracking on WSL (NVML not supported)
        )
        self._emissions = None

    def start(self) -> None:
        self.tracker.start()

    def stop(self) -> dict:
        self._emissions = self.tracker.stop()
        scheduler = getattr(self.tracker, "_scheduler", None)
        duration = getattr(scheduler, "duration", 0) if scheduler is not None else 0
        total_energy = getattr(self.tracker, "_total_energy", None)
        energy_kwh = getattr(total_energy, "kWh", 0) if total_energy is not None else 0
        return {
            "emissions_kg": self._emissions,
            "duration_s": duration,
            "energy_kwh": energy_kwh,
        }
