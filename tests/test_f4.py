import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import LinearDot
from cliffeq.training.ep_engine import EPEngine

class SimpleLinearEnergy(EnergyFunction):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum((state - self.W) ** 2, dim=-1)

def test_f4_ep():
    energy = SimpleLinearEnergy()
    rule = LinearDot()
    engine = EPEngine(energy, rule, n_free=10, n_clamped=5, beta=1.0, dt=0.1)
    x_init = torch.zeros((1, 1, 4))
    x_free = engine.free_phase(x_init)
    target = torch.tensor([[2.0]])
    def loss_fn(state, target):
        return 0.5 * torch.sum((state[..., 0] - target) ** 2)
    x_clamped = engine.clamped_phase(x_free, target, loss_fn)
    original_W = energy.W.clone()
    engine.parameter_update(x_free, x_clamped)
    optimizer = torch.optim.SGD(energy.parameters(), lr=1.0)
    optimizer.step()
    assert energy.W[0, 0] > original_W[0, 0]
    print("test_f4_ep passed")

if __name__ == "__main__":
    test_f4_ep()
