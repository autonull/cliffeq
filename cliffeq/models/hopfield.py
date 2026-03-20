import torch
from torch import nn
from cliffeq.energy.zoo import HopfieldEnergy
from cliffeq.dynamics.rules import DynamicsRule
from cliffeq.training.ep_engine import EPEngine

class CliffordHopfieldNetwork(nn.Module):
    def __init__(
        self,
        n_patterns: int,
        nodes: int,
        sig_g: torch.Tensor,
        dynamics_rule: DynamicsRule,
        beta: float = 1.0,
        n_free: int = 50,
        dt: float = 0.1
    ):
        super().__init__()
        self.energy_fn = HopfieldEnergy(n_patterns, nodes, sig_g, beta)
        self.dynamics_rule = dynamics_rule
        self.engine = EPEngine(self.energy_fn, dynamics_rule, n_free, n_clamped=0, beta=0.0, dt=dt)
        self.nodes = nodes
        self.n_blades = self.energy_fn.sig.n_blades

    def forward(self, x_init: torch.Tensor) -> torch.Tensor:
        """
        Run the network to retrieve a pattern.
        x_init: (B, nodes, I)
        """
        return self.engine.free_phase(x_init)

    def retrieve_with_patterns(self, x_init: torch.Tensor):
        """
        Manual retrieval using softmax over similarities (one-step update).
        x_init: (B, nodes, I)
        """
        # (B, nodes, I), (M, nodes, I) -> (B, M)
        from cliffeq.algebra.utils import get_blade_signs, reverse
        signs = get_blade_signs(self.energy_fn.sig, x_init.device)
        patterns_rev = reverse(self.energy_fn.patterns, self.energy_fn.sig)

        sim = torch.einsum("bni,mni,i->bm", x_init, patterns_rev, signs)
        attn = torch.softmax(self.energy_fn.beta * sim, dim=-1)
        # patterns: (M, nodes, I)
        return torch.einsum("bm,mni->bni", attn, self.energy_fn.patterns)
