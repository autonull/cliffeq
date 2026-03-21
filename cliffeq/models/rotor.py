import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import DynamicsRule
from cliffeq.training.ep_engine import EPEngine
from cliffordlayers.signature import CliffordSignature

class RotorRiemannianDynamics(DynamicsRule):
    """
    Riemannian gradient descent on the unit sphere S^3 (unit quaternions).
    """
    def __init__(self, g: torch.Tensor):
        self.g = g
        self.sig = CliffordSignature(g)
        if self.sig.dim != 3:
            raise ValueError("Rotor states are defined for Cl(3,0)")

    def step(self, q: torch.Tensor, energy_fn: EnergyFunction, alpha: float) -> torch.Tensor:
        # q: (B, nodes, I) where I=8, but only even part is used.
        # Actually, let's assume q is (B, nodes, 4) for quaternions [1, e12, e13, e23]
        q_in = q.detach().requires_grad_(True)
        with torch.enable_grad():
            E = energy_fn(q_in).sum()
            grad = torch.autograd.grad(E, q_in)[0]

        # Projection: grad = grad - (grad \cdot q) * q
        dot = torch.sum(grad * q_in, dim=-1, keepdim=True)
        proj_grad = grad - dot * q_in

        # Update
        q_new = q_in - alpha * proj_grad

        # Retraction (normalization)
        norm = torch.norm(q_new, dim=-1, keepdim=True)
        q_new = q_new / (norm + 1e-8)

        return q_new.detach()

class RotorEPModel(nn.Module):
    def __init__(
        self,
        energy_fn: EnergyFunction,
        n_free: int,
        n_clamped: int,
        beta: float,
        dt: float
    ):
        super().__init__()
        self.energy_fn = energy_fn
        # Cl(3,0) signature
        g = torch.tensor([1.0, 1.0, 1.0])
        self.dynamics_rule = RotorRiemannianDynamics(g)
        self.engine = EPEngine(energy_fn, self.dynamics_rule, n_free, n_clamped, beta, dt)

    def forward(self, x: torch.Tensor, q_init: torch.Tensor = None) -> torch.Tensor:
        if hasattr(self.energy_fn, 'set_input'):
            self.energy_fn.set_input(x)

        if q_init is None:
            B = x.shape[0]
            nodes = getattr(self.energy_fn, 'nodes', 1)
            # Initialize as identity rotors [1, 0, 0, 0]
            q_init = torch.zeros((B, nodes, 4), device=x.device)
            q_init[..., 0] = 1.0

        q_free = self.engine.free_phase(q_init)
        if hasattr(self.energy_fn, 'get_output'):
            return self.energy_fn.get_output(q_free)
        return q_free

    def train_step(self, x: torch.Tensor, target: torch.Tensor, optimizer: torch.optim.Optimizer, loss_fn: callable = None):
        if hasattr(self.energy_fn, 'set_input'):
            self.energy_fn.set_input(x)

        B = x.shape[0]
        nodes = getattr(self.energy_fn, 'nodes', 1)
        q_init = torch.zeros((B, nodes, 4), device=x.device)
        q_init[..., 0] = 1.0

        q_free = self.engine.free_phase(q_init)

        if loss_fn is None:
            def ep_loss_fn(q, target):
                out = self.energy_fn.get_output(q)
                # target should be a rotor too?
                return 0.5 * torch.sum((out - target) ** 2)
            loss_fn = ep_loss_fn

        q_clamped = self.engine.clamped_phase(q_free, target, loss_fn)
        self.engine.parameter_update(q_free, q_clamped)
        optimizer.step()
        optimizer.zero_grad()
        return q_free
