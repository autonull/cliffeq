import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import DynamicsRule
from typing import Optional, Callable

class EPEngine:
    def __init__(
        self,
        energy_fn: EnergyFunction,
        dynamics_rule: DynamicsRule,
        n_free: int,
        n_clamped: int,
        beta: float,
        dt: float,
        use_spectral_norm: bool = False
    ):
        self.energy_fn = energy_fn
        self.dynamics_rule = dynamics_rule
        self.n_free = n_free
        self.n_clamped = n_clamped
        self.beta = beta
        self.dt = dt
        self.use_spectral_norm = use_spectral_norm

    def free_phase(self, x_init: torch.Tensor) -> torch.Tensor:
        x = x_init.detach().clone()
        for _ in range(self.n_free):
            x = self.dynamics_rule.step(x, self.energy_fn, self.dt)
        return x

    def clamped_phase(self, x_free: torch.Tensor, target: torch.Tensor, loss_fn: Callable) -> torch.Tensor:
        x = x_free.detach().clone()

        class ClampedEnergy(EnergyFunction):
            def __init__(self, base_energy, target, loss_fn, beta):
                super().__init__()
                self.base_energy = base_energy
                self.target = target
                self.loss_fn = loss_fn
                self.beta = beta
                if hasattr(base_energy, 'g'):
                    self.g = base_energy.g

            def forward(self, state: torch.Tensor) -> torch.Tensor:
                return self.base_energy(state).sum() + self.beta * self.loss_fn(state, self.target)

        clamped_energy = ClampedEnergy(self.energy_fn, target, loss_fn, self.beta)

        for _ in range(self.n_clamped):
            x = self.dynamics_rule.step(x, clamped_energy, self.dt)
        return x

    def parameter_update(self, x_free: torch.Tensor, x_clamped: torch.Tensor):
        self.energy_fn.zero_grad()
        E_free = self.energy_fn(x_free).sum()
        E_free.backward()
        grads_free = {name: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                      for name, p in self.energy_fn.named_parameters()}

        self.energy_fn.zero_grad()
        E_clamped = self.energy_fn(x_clamped).sum()
        E_clamped.backward()

        with torch.no_grad():
            for name, p in self.energy_fn.named_parameters():
                grad_clamped = p.grad if p.grad is not None else torch.zeros_like(p)
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.copy_((grad_clamped - grads_free[name]) / self.beta)
        return
