import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import DynamicsRule
from typing import Optional, Callable

class CHLEngine:
    """
    Contrastive Hebbian Learning (CHL) Engine.
    Positive phase: relax to find equilibrium with target clamped (or positive data).
    Negative phase: relax to find equilibrium without target (or with negative data).
    ΔW ∝ [∂E(x_pos)/∂W − ∂E(x_neg)/∂W]
    """
    def __init__(
        self,
        energy_fn: EnergyFunction,
        dynamics_rule: DynamicsRule,
        n_pos: int,
        n_neg: int,
        dt: float
    ):
        self.energy_fn = energy_fn
        self.dynamics_rule = dynamics_rule
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.dt = dt

    def relax(self, x_init: torch.Tensor, n_steps: int, energy_fn: EnergyFunction) -> torch.Tensor:
        x = x_init.detach().clone()
        for _ in range(n_steps):
            x = self.dynamics_rule.step(x, energy_fn, self.dt)
        return x

    def positive_phase(self, x_init: torch.Tensor, target: torch.Tensor, loss_fn: Callable, beta: float) -> torch.Tensor:
        class ClampedEnergy(EnergyFunction):
            def __init__(self, base_energy, target, loss_fn, beta):
                super().__init__()
                self.base_energy = base_energy
                self.target = target
                self.loss_fn = loss_fn
                self.beta = beta
            def forward(self, state: torch.Tensor) -> torch.Tensor:
                return self.base_energy(state).sum() + self.beta * self.loss_fn(state, self.target)

        clamped_energy = ClampedEnergy(self.energy_fn, target, loss_fn, beta)
        return self.relax(x_init, self.n_pos, clamped_energy)

    def negative_phase(self, x_init: torch.Tensor) -> torch.Tensor:
        return self.relax(x_init, self.n_neg, self.energy_fn)

    def parameter_update(self, x_pos: torch.Tensor, x_neg: torch.Tensor):
        self.energy_fn.zero_grad()
        E_pos = self.energy_fn(x_pos).sum()
        E_pos.backward()
        grads_pos = {name: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                      for name, p in self.energy_fn.named_parameters()}

        self.energy_fn.zero_grad()
        E_neg = self.energy_fn(x_neg).sum()
        E_neg.backward()

        with torch.no_grad():
            for name, p in self.energy_fn.named_parameters():
                grad_neg = p.grad if p.grad is not None else torch.zeros_like(p)
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                # Note: CHL update direction.
                # EP uses (clamped - free) / beta.
                # CHL uses (pos - neg)
                p.grad.copy_(grads_pos[name] - grad_neg)
        return
