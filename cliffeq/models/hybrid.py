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

from cliffeq.algebra.utils import geometric_product
import torch.nn.functional as F

class CliffordBPBottleneck(nn.Module):
    """
    Clifford bottleneck trained with standard backprop.
    Uses learnable Clifford weights in a Feedforward manner.
    """
    def __init__(self, in_features, clifford_dim, sig_g):
        super().__init__()
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.in_features = in_features
        self.clifford_dim = clifford_dim
        self.n_blades = self.sig.n_blades

        self.fc_in = nn.Linear(in_features, clifford_dim * self.n_blades)
        # Learnable Clifford weight matrix (Nout, Nin, I)
        self.W_clif = nn.Parameter(torch.randn(clifford_dim, clifford_dim, self.n_blades) * 0.01)
        self.fc_out = nn.Linear(clifford_dim * self.n_blades, in_features)

    def forward(self, x):
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Project to Clifford multivectors
        h = self.fc_in(x_flat).view(-1, self.clifford_dim, self.n_blades)

        # Apply Clifford linear layer (geometric product with weights)
        h = geometric_product(h, self.W_clif, self.g)
        h = F.relu(h)

        # Project back to scalar
        out = self.fc_out(h.reshape(-1, self.clifford_dim * self.n_blades))
        return out.reshape(*shape)
