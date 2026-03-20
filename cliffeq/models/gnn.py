import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.dynamics.rules import DynamicsRule
from cliffeq.training.ep_engine import EPEngine
from cliffeq.algebra.utils import geometric_product, get_blade_signs, reverse
from cliffordlayers.signature import CliffordSignature
try:
    from torch_geometric.nn import MessagePassing
except ImportError:
    class MessagePassing: pass

class LocalGraphEnergy(EnergyFunction):
    """
    Energy function for a graph: E = Σ_{ij} scalar(x̃_i W_ij x_j)
    """
    def __init__(self, nodes, sig_g, use_spectral_norm: bool = False):
        super().__init__(use_spectral_norm=use_spectral_norm)
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        # For simplicity, shared W for all edges if edge_index is provided later
        self.W = nn.Parameter(torch.randn(self.sig.n_blades) * 0.1)
        self.edge_index = None
        self.apply_sn()

    def set_graph(self, edge_index):
        self.edge_index = edge_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, I)
        B, N, I = x.shape
        row, col = self.edge_index

        # x_i: (B, E, I), x_j: (B, E, I)
        x_i = x[:, row, :]
        x_j = x[:, col, :]

        # scalar(x̃_i W x_j)
        # We can compute geometric_product(x_i_rev, W * x_j) and take scalar part
        signs = get_blade_signs(self.sig, x.device)
        x_i_rev = reverse(x_i, self.sig)

        # Elementwise geometric product of (W, x_j)
        # Since W is (I,), we expand it
        W_exp = self.W.view(1, 1, I).expand(B, x_j.shape[1], I)
        W_xj = geometric_product(W_exp, x_j, self.g)

        # scalar part of (x_i_rev * W_xj)
        E_edges = torch.einsum("bei,bei,i->be", x_i_rev, W_xj, signs)
        return E_edges.sum(dim=-1)

class GeometricEquilibriumGNN(nn.Module):
    def __init__(
        self,
        nodes: int,
        sig_g: torch.Tensor,
        dynamics_rule: DynamicsRule,
        n_free: int,
        n_clamped: int,
        beta: float,
        dt: float
    ):
        super().__init__()
        self.energy_fn = LocalGraphEnergy(nodes, sig_g)
        self.dynamics_rule = dynamics_rule
        self.engine = EPEngine(self.energy_fn, dynamics_rule, n_free, n_clamped, beta, dt)
        self.n_blades = self.energy_fn.sig.n_blades

    def forward(self, x_init: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.energy_fn.set_graph(edge_index)
        return self.engine.free_phase(x_init)

    def train_step(self, x_init, edge_index, target, optimizer, loss_fn=None):
        self.energy_fn.set_graph(edge_index)
        h_free = self.engine.free_phase(x_init)

        if loss_fn is None:
            def ep_loss_fn(h, target):
                return 0.5 * torch.sum((h - target) ** 2)
            loss_fn = ep_loss_fn

        h_clamped = self.engine.clamped_phase(h_free, target, loss_fn)
        self.engine.parameter_update(h_free, h_clamped)
        optimizer.step()
        optimizer.zero_grad()
        return h_free
