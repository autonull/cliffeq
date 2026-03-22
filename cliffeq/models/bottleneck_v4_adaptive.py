import torch
from torch import nn
import torch.nn.functional as F
from cliffeq.algebra.utils import geometric_product, reverse, scalar_part
from cliffordlayers.signature import CliffordSignature

class CliffordAdaptiveBottleneck(nn.Module):
    """
    Adaptive Clifford Bottleneck.
    Learns per-blade significance weights to modulate the multivector bottleneck.
    """
    def __init__(self, in_features, clifford_dim, sig_g):
        super().__init__()
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.in_features = in_features
        self.clifford_dim = clifford_dim

        # Projections
        self.fc_in = nn.Linear(in_features, clifford_dim * self.sig.n_blades)
        self.fc_out = nn.Linear(clifford_dim * self.sig.n_blades, in_features)

        # Adaptive significance weights per head and blade
        self.blade_weights = nn.Parameter(torch.ones(clifford_dim, self.sig.n_blades))

    def forward(self, x):
        # x: (B, ..., in_features)
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Project to Clifford space
        h = self.fc_in(x_flat)
        h = h.view(-1, self.clifford_dim, self.sig.n_blades)

        # Modulate by adaptive weights
        h = h * self.blade_weights

        # Flatten and project back
        h_flat = h.reshape(-1, self.clifford_dim * self.sig.n_blades)
        out = self.fc_out(h_flat)

        return out.reshape(*shape)
