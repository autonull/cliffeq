import torch
from torch import nn
from typing import Optional

class EnergyFunction(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        self.use_spectral_norm = use_spectral_norm

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def apply_sn(self):
        if not self.use_spectral_norm:
            return
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.utils.spectral_norm(module)

    def get_max_singular_value(self) -> Optional[float]:
        max_sigma = 0.0
        found = False
        for module in self.modules():
            if hasattr(module, "weight_orig"):
                with torch.no_grad():
                    u = getattr(module, "weight_u")
                    v = getattr(module, "weight_v")
                    w_orig = getattr(module, "weight_orig")
                    sigma = torch.dot(u, torch.mv(w_orig, v))
                    max_sigma = max(max_sigma, abs(sigma.item()))
                    found = True
        return max_sigma if found else None
