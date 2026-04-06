import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy that tracks a smoothed version of weights.
    Use apply_shadow() before evaluation, restore() after.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        self.backup = {}

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param, alpha=1.0 - self.decay)

    def apply_shadow(self) -> None:
        """Copy EMA weights into model for evaluation."""
        self.backup = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict: dict) -> None:
        for name, tensor in state_dict.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor)
