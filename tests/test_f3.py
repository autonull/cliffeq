import torch
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import LinearDot, GeomProduct, Riemannian
from cliffordlayers.signature import CliffordSignature

class SimpleNormEnergy(EnergyFunction):
    def __init__(self, g):
        super().__init__()
        self.g = g
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(state ** 2, dim=-1)

def test_f3_rules():
    g = torch.tensor([1.0, 1.0])
    energy = SimpleNormEnergy(g)
    x = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]], requires_grad=True)
    alpha = 0.1
    rule = LinearDot()
    x_new = rule.step(x, energy, alpha)
    assert torch.allclose(x_new, torch.tensor([[[0.9, 0.9, 0.0, 0.0]]]))
    rule = GeomProduct(g, normalize=True)
    x_new = rule.step(x, energy, alpha)
    assert energy(x_new).sum() < energy(x).sum()
    print("test_f3_rules passed")

if __name__ == "__main__":
    test_f3_rules()
