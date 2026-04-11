import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy that tracks a smoothed version of weights.
    Use apply_shadow() before evaluation, restore() after.

    Precision: the shadow is stored in **float32 regardless of the model's
    forward dtype**. This is load-bearing when the model runs in bfloat16
    (OfficialTRMTrainer casts with ``model.to(dtype=bfloat16)``). The EMA
    update ``shadow = 0.999*shadow + 0.001*param`` is a ~0.1% delta, but
    bfloat16's 7-bit mantissa can't resolve fractional changes smaller than
    ~0.8% of the current value — so a bf16 shadow is effectively frozen
    after construction (each update rounds to a no-op, and ``0.999`` itself
    rounds to ``1.0`` in bf16). Stored-in-fp32 fixes this: the math runs at
    fp32 precision and ``apply_shadow``/``restore`` use ``Tensor.copy_``
    which handles the fp32↔bf16 cast transparently.

    Upstream SamsungSAILMontreal/TinyRecursiveModels gets away with an
    inherit-param-dtype shadow because their model is fp32 end-to-end —
    they never cast to bf16. This trainer opted into native bf16 for
    throughput, so the EMA has to compensate here.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {
            name: param.detach().clone().to(torch.float32)
            for name, param in model.named_parameters()
        }
        self.backup = {}

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            # Cast param to fp32 before mixing — the shadow is fp32, but
            # param is bf16 in the native-bf16 training path, and mixing
            # dtypes in a single fused op would either error or silently
            # downcast the result. Explicit cast keeps the math in fp32.
            self.shadow[name].mul_(self.decay).add_(
                param.to(torch.float32), alpha=1.0 - self.decay
            )

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
