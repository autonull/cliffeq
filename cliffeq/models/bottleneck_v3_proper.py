"""
P2.9 Bottleneck V3: PROPER IMPLEMENTATION

The fundamental issue with V2 (disabled) and my "fix":
- Both apply 50% compression (64→32, 256→128)
- Then try to project back to original size
- IMPOSSIBLE: Can't recover discarded information

PROPER SOLUTIONS:
1. Don't compress dimensionality (reshape only)
2. OR return same dimensionality (not a "bottleneck")
3. OR be honest about information loss

This version: RESHAPE WITHOUT COMPRESSION
- Projects to Clifford space but maintains dimensionality
- 64 → 64 (in Clifford space)
- Not a dimensionality reduction, but a geometric transformation
"""

import torch
from torch import nn
from cliffordlayers.signature import CliffordSignature


class CliffordEPBottleneckV3(nn.Module):
    """
    Clifford bottleneck that maintains dimensionality.

    Instead of compressing features, projects them into Clifford space
    while maintaining the same information capacity.

    Key property: out_dim = in_dim (no compression)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,  # MUST EQUAL in_dim for proper implementation
        sig_g: torch.Tensor,
        n_ep_steps: int = 0,  # Unused (for backward compat)
        step_size: float = 0.01,  # Unused
        use_spectral_norm: bool = True,
        detach_state: bool = False  # Unused
    ):
        super().__init__()

        # ENFORCE: out_dim must equal in_dim (no compression)
        if out_dim != in_dim:
            raise ValueError(
                f"CliffordEPBottleneckV3 requires out_dim == in_dim for no information loss. "
                f"Got in_dim={in_dim}, out_dim={out_dim}"
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sig_g = sig_g
        self.sig = CliffordSignature(sig_g)

        # Project to Clifford space (maintain dimensionality)
        # If in_dim = 64 and n_blades = 4, then:
        # We can split 64 into 16 multivectors of 4 blades each
        # So we keep the same total dimensionality
        self.n_multivectors = in_dim // self.sig.n_blades

        if in_dim % self.sig.n_blades != 0:
            raise ValueError(
                f"in_dim ({in_dim}) must be divisible by n_blades ({self.sig.n_blades})"
            )

        # Learnable Clifford transformation (matrix in Clifford space)
        # This rotates/transforms multivectors without changing dimensionality
        self.blade_transform = nn.Parameter(
            torch.randn(self.sig.n_blades, self.sig.n_blades) * 0.1
        )

        self.scale = nn.Parameter(torch.ones(self.sig.n_blades))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_dim) - scalar input
        Returns: (batch, out_dim) - scalar output (same shape as input)

        Transform: Reshape to Clifford space → Apply transformation → Reshape back
        """
        B = x.shape[0]

        # Reshape into Clifford multivectors
        # (batch, in_dim) → (batch, n_multivectors, n_blades)
        h = x.view(B, self.n_multivectors, self.sig.n_blades)

        # Apply learnable Clifford transformation
        # Multiply each multivector by transformation matrix
        h_flat = h.view(B * self.n_multivectors, self.sig.n_blades)
        h_transformed = torch.matmul(h_flat, self.blade_transform)
        h = h_transformed.view(B, self.n_multivectors, self.sig.n_blades)

        # Apply learnable scale per blade (geometric parameter)
        h = h * torch.abs(self.scale).unsqueeze(0).unsqueeze(0)

        # Reshape back to scalar space (same dimensionality as input)
        output = h.view(B, self.in_dim)

        return output


# Alternative: Do NOT use a bottleneck at all

class IdentityBottleneck(nn.Module):
    """
    No-op bottleneck that returns input unchanged.

    Use this as a baseline to test if bottleneck itself is the problem.
    """
    def __init__(self, in_dim: int, out_dim: int = None, **kwargs):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        assert out_dim == in_dim, "IdentityBottleneck requires out_dim == in_dim"
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# Alternative: Simple linear transform (no Clifford structure)

class LinearBottleneck(nn.Module):
    """
    Simple learnable linear transformation, no compression.

    Test if Clifford structure actually helps vs plain learned transformation.
    """
    def __init__(self, in_dim: int, out_dim: int = None, **kwargs):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        assert out_dim == in_dim, "LinearBottleneck requires out_dim == in_dim"

        self.in_dim = in_dim
        self.transform = nn.Linear(in_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

