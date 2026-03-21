import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import DynamicsRule

class CDEngine:
    """
    Contrastive Divergence (CD) Engine for Clifford Energy-Based Models.
    Uses Clifford-Langevin dynamics: x_{t+1} = x_t − α ∂E/∂x + ε
    where ε is Clifford-valued noise.
    """
    def __init__(
        self,
        energy_fn: EnergyFunction,
        n_steps: int = 10,
        alpha: float = 0.1,
        noise_std: float = 0.05
    ):
        self.energy_fn = energy_fn
        self.n_steps = n_steps
        self.alpha = alpha
        self.noise_std = noise_std

    def sample_langevin(self, x_init: torch.Tensor) -> torch.Tensor:
        x = x_init.detach().clone()
        for _ in range(self.n_steps):
            x.requires_grad_(True)
            E = self.energy_fn(x).sum()
            grad = torch.autograd.grad(E, x)[0]

            # Clifford-Langevin update
            noise = torch.randn_like(x) * self.noise_std
            x = (x - self.alpha * grad + noise).detach()
        return x

    def train_step(self, x_pos: torch.Tensor, optimizer: torch.optim.Optimizer):
        # Generate negative samples starting from positive data (CD-k)
        x_neg = self.sample_langevin(x_pos)

        optimizer.zero_grad()
        # Loss: E(x_pos) - E(x_neg)
        loss = self.energy_fn(x_pos).mean() - self.energy_fn(x_neg).mean()
        loss.backward()
        optimizer.step()

        return loss.item()
