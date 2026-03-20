import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import DynamicsRule
from cliffeq.training.ep_engine import EPEngine

class EPModel(nn.Module):
    """
    General Equilibrium Propagation Model wrapper.
    energy_fn: The energy function to minimize.
    dynamics_rule: The rule to update the state.
    n_free: Number of steps in the free phase.
    n_clamped: Number of steps in the clamped phase.
    beta: Nudge strength in the clamped phase.
    dt: Step size for the dynamics rule.
    """
    def __init__(self, energy_fn: EnergyFunction, dynamics_rule: DynamicsRule, n_free: int, n_clamped: int, beta: float, dt: float):
        super().__init__()
        self.energy_fn = energy_fn
        self.engine = EPEngine(energy_fn, dynamics_rule, n_free, n_clamped, beta, dt)

    def forward(self, x: torch.Tensor, h_init: torch.Tensor = None) -> torch.Tensor:
        """
        Run the free phase to find the equilibrium state.
        x: Input to the energy function.
        h_init: Initial state for the free phase (defaults to zeros).
        """
        if hasattr(self.energy_fn, 'set_input'):
            self.energy_fn.set_input(x)

        if h_init is None:
            # Determine state shape from energy_fn if possible
            # Fallback to (B, hidden_dim, comp)
            B = x.shape[0]
            hidden_dim = getattr(self.energy_fn, 'hidden_dim', 64)
            comp = 1 if not hasattr(self.energy_fn, 'sig') else self.energy_fn.sig.n_blades
            h_init = torch.zeros((B, hidden_dim, comp), device=x.device)

        h_free = self.engine.free_phase(h_init)
        if hasattr(self.energy_fn, 'get_output'):
            return self.energy_fn.get_output(h_free)
        return h_free

    def train_step(self, x: torch.Tensor, target: torch.Tensor, optimizer: torch.optim.Optimizer, loss_fn: callable = None):
        """
        Perform one step of EP training.
        """
        if hasattr(self.energy_fn, 'set_input'):
            self.energy_fn.set_input(x)

        B = x.shape[0]
        hidden_dim = getattr(self.energy_fn, 'hidden_dim', 64)
        comp = 1 if not hasattr(self.energy_fn, 'sig') else self.energy_fn.sig.n_blades
        h_init = torch.zeros((B, hidden_dim, comp), device=x.device)

        h_free = self.engine.free_phase(h_init)

        if loss_fn is None:
            def ep_loss_fn(h, target):
                out = self.energy_fn.get_output(h)
                return 0.5 * torch.sum((out - target) ** 2)
            loss_fn = ep_loss_fn

        h_clamped = self.engine.clamped_phase(h_free, target, loss_fn)
        self.engine.parameter_update(h_free, h_clamped)
        optimizer.step()
        optimizer.zero_grad()
        return h_free
