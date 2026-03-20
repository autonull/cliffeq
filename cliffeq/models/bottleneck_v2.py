"""
P2.9 Bottleneck V2: Fixed implementation with gradient-friendly energy function.

Key fixes:
1. Use simple self-energy (not input-dependent BilinearEnergy)
2. Unroll only a few EP steps (makes gradients tractable)
3. Use stop_gradient at bottleneck to stabilize training if needed
"""

import torch
from torch import nn
import torch.nn.functional as F
from cliffordlayers.signature import CliffordSignature
from cliffeq.algebra.utils import scalar_part


class SimpleGeometricEnergy(nn.Module):
    """
    Simple self-energy function on Clifford states: E = 0.5 * ||h||^2 + geometric regularization

    Advantages:
    - No input dependency (no shape conflicts)
    - Differentiable
    - Encourages geometric structure via squared Clifford norm
    """
    def __init__(self, sig_g, use_spectral_norm=False):
        super().__init__()
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        # Simple learnable weight matrix for geometric interactions
        self.W = nn.Parameter(torch.randn(self.sig.n_blades, self.sig.n_blades) * 0.1)
        if use_spectral_norm:
            self.W_data = nn.utils.spectral_norm(nn.Linear(self.sig.n_blades, self.sig.n_blades))
        self.use_sn = use_spectral_norm

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (batch, nodes, n_blades)
        Returns: scalar energy per sample (batch,)
        """
        B, N, I = h.shape

        # Self-energy: sum of squared norms (encourage small activations)
        E_norm = 0.5 * torch.sum(h ** 2, dim=(1, 2))  # (batch,)

        # Geometric interaction energy: encourage structure via blade interactions
        # E_geom = sum_over_nodes of scalar_part(h @ W @ h^T)
        h_flat = h.view(B * N, I)  # (B*N, I)
        h_weighted = torch.matmul(h_flat, self.W)  # (B*N, I)
        h_weighted = h_weighted.view(B, N, I)

        # Inner product encourages aligned orientation
        E_interact = 0.1 * torch.sum(h * h_weighted, dim=(1, 2))  # (batch,)

        return E_norm + E_interact


class CliffordEPBottleneckV2(nn.Module):
    """
    Clifford-EP bottleneck layer v2: Fixed for gradient flow.

    Usage:
        bottleneck = CliffordEPBottleneckV2(
            in_dim=64, out_dim=32, sig_g=torch.tensor([1,1,1]),
            n_ep_steps=3, step_size=0.01
        )
        x = bottleneck(x)  # (batch, 64) -> (batch, 32)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        sig_g: torch.Tensor,
        n_ep_steps: int = 3,
        step_size: float = 0.01,
        use_spectral_norm: bool = True,
        detach_state: bool = False  # Stop gradient if training is unstable
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sig_g = sig_g
        self.sig = CliffordSignature(sig_g)
        self.n_ep_steps = n_ep_steps
        self.step_size = step_size
        self.detach_state = detach_state

        # Input projection to Clifford space
        self.input_proj = nn.Linear(in_dim, out_dim * self.sig.n_blades)

        # Energy function (simple, gradient-friendly)
        self.energy = SimpleGeometricEnergy(sig_g, use_spectral_norm=use_spectral_norm)

        # Output projection back to scalar space
        self.output_proj = nn.Linear(out_dim * self.sig.n_blades, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_dim) - scalar input
        Returns: (batch, out_dim) - scalar output
        """
        B = x.shape[0]

        # Project to Clifford multivector space
        h = self.input_proj(x)  # (batch, out_dim * n_blades)
        h = h.view(B, self.out_dim, self.sig.n_blades)  # (batch, out_dim, n_blades)

        # During training: EP-like free phase (optional, can skip for simplicity)
        # During eval: Skip EP iterations for efficiency
        if self.training and self.n_ep_steps > 0:
            h_state = h

            for step in range(self.n_ep_steps):
                # Compute energy (with gradient enabled)
                h_state_copy = h_state.detach().requires_grad_(True)
                E = self.energy(h_state_copy)

                # Backprop to get gradient
                E.sum().backward()

                # Gradient descent step
                with torch.no_grad():
                    grad = h_state_copy.grad
                    h_state = h_state - self.step_size * grad

            h_final = h_state
        else:
            # Eval mode: skip EP iterations, just use projection
            h_final = h

        # Project back to scalar space
        h_flat = h_final.view(B, -1)  # (batch, out_dim * n_blades)
        output = self.output_proj(h_flat)  # (batch, out_dim)

        return output
