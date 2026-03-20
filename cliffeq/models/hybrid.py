import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import DynamicsRule
from cliffeq.training.ep_engine import EPEngine
from cliffeq.algebra.utils import embed_vector, scalar_part
from cliffordlayers.signature import CliffordSignature

class CliffordEPBottleneck(nn.Module):
    """
    A Clifford-EP bottleneck layer that can be inserted into standard architectures.
    Input: scalar feature tensor (batch, d)
    1. Project to multivectors (batch, d/comp, comp)
    2. Run EP iterations to reach geometric equilibrium
    3. Extract scalar component (batch, d/comp)
    Output: scalar feature tensor (batch, d/comp)
    """
    def __init__(
        self,
        energy_fn: EnergyFunction,
        dynamics_rule: DynamicsRule,
        n_free: int = 10,
        dt: float = 0.1,
        comp: int = 4 # Default to Cl(2,0)
    ):
        super().__init__()
        self.energy_fn = energy_fn
        self.dynamics_rule = dynamics_rule
        self.engine = EPEngine(energy_fn, dynamics_rule, n_free, n_clamped=0, beta=0.0, dt=dt)
        self.comp = comp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        # Project to multivectors.
        # Here we just reshape and treat the last dimension as the Clifford component.
        # Ensure D is divisible by comp
        if D % self.comp != 0:
            # Pad if necessary
            pad = self.comp - (D % self.comp)
            x = torch.cat([x, torch.zeros(B, pad, device=x.device)], dim=-1)
            D = x.shape[1]

        nodes = D // self.comp
        h_init = x.view(B, nodes, self.comp)

        if hasattr(self.energy_fn, 'set_input'):
            self.energy_fn.set_input(h_init)

        h_free = self.engine.free_phase(h_init)

        # Extract scalar part
        return h_free[..., 0]
