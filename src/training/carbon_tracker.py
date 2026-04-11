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

    def flush(self) -> dict:
        """Return current cumulative emissions without stopping the tracker.

        Called from the training loop at each log_interval so wandb gets a
        live carbon curve instead of a single end-of-run scalar. CodeCarbon's
        EmissionsTracker.flush() returns the cumulative kg CO2eq and updates
        _total_energy in place, so both fields are safe to read afterwards.
        Returns zeros if flush hasn't populated state yet (very early in run).
        """
        try:
            emissions_kg = self.tracker.flush() or 0.0
        except Exception:
            emissions_kg = 0.0
        total_energy = getattr(self.tracker, "_total_energy", None)
        energy_kwh = getattr(total_energy, "kWh", 0) if total_energy is not None else 0
        return {
            "emissions_kg": float(emissions_kg),
            "energy_kwh": float(energy_kwh),
        }

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
