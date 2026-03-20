import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction

class DummyEnergy(EnergyFunction):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__(use_spectral_norm=use_spectral_norm)
        self.lin = nn.Linear(4, 4, bias=False)
        self.apply_sn()
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.lin(state))

def test_f2_sn():
    energy = DummyEnergy(use_spectral_norm=True)
    assert hasattr(energy.lin, "weight_orig")
    sigma = energy.get_max_singular_value()
    assert sigma is not None
    assert sigma > 0.0
    print(f"test_f2_sn passed")

if __name__ == "__main__":
    test_f2_sn()
