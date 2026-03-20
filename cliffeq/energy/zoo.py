import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.algebra.utils import geometric_product, scalar_part, clifford_norm_sq
from cliffordlayers.signature import CliffordSignature

class NormEnergy(EnergyFunction):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__(use_spectral_norm=use_spectral_norm)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # E = ||x||^2 = scalar(x̃x)
        # For simplicity, we use squared Euclidean norm of the components
        return 0.5 * torch.sum(state**2, dim=tuple(range(1, state.ndim)))

class BilinearEnergy(EnergyFunction):
    def __init__(self, in_nodes, hidden_nodes, sig_g, use_spectral_norm: bool = False):
        super().__init__(use_spectral_norm=use_spectral_norm)
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.W = nn.Parameter(torch.randn(hidden_nodes, in_nodes, self.sig.n_blades) * 0.1)
        self.input_x = None
        self.apply_sn()

    def set_input(self, x):
        self.input_x = x

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # E = 0.5 * ||h||^2 - scalar(h̃ W x)
        E_norm = 0.5 * torch.sum(h**2, dim=(-1, -2))
        Wx = geometric_product(self.input_x, self.W, self.g)
        E_int = torch.sum(h * Wx, dim=(-1, -2))
        return E_norm - E_int

class GraphEnergy(EnergyFunction):
    def __init__(self, nodes, sig_g, use_spectral_norm: bool = False):
        super().__init__(use_spectral_norm=use_spectral_norm)
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.W = nn.Parameter(torch.randn(nodes, nodes, self.sig.n_blades) * 0.1)
        self.apply_sn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # E = Σ_{ij} scalar(x̃_i W_ij x_j)
        # x: (B, N, I)
        Wx = geometric_product(x, self.W, self.g) # (B, N, I)
        return torch.sum(x * Wx, dim=(-1, -2))

class GradeWeightedEnergy(EnergyFunction):
    def __init__(self, nodes, sig_g, lambdas=None, use_spectral_norm: bool = False):
        super().__init__(use_spectral_norm=use_spectral_norm)
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        if lambdas is None:
            lambdas = torch.ones(self.sig.dim + 1)
        self.register_buffer("lambdas", lambdas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # E = Σ_k λ_k ||<x>_k||^2
        if self.sig.dim == 3:
            blade_grades = [0, 1, 1, 1, 2, 2, 2, 3]
        elif self.sig.dim == 2:
            blade_grades = [0, 1, 1, 2]
        else:
            blade_grades = [0, 1]

        E = 0.0
        for i, g in enumerate(blade_grades):
            E += self.lambdas[g] * torch.sum(x[..., i]**2, dim=-1)
        return E

class HopfieldEnergy(EnergyFunction):
    def __init__(self, n_patterns, nodes, sig_g, beta=1.0, use_spectral_norm: bool = False):
        super().__init__(use_spectral_norm=use_spectral_norm)
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.patterns = nn.Parameter(torch.randn(n_patterns, nodes, self.sig.n_blades) * 0.1)
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # E = -log Σ_m exp(β · scalar(ξ̃_m x))
        # x: (B, N, I), patterns: (M, N, I)
        # ξ̃_m x product: we want scalar part
        # scalar(ξ̃_m x) = Σ_i signs_i * reverse(ξ)_mi * x_i
        from cliffeq.algebra.utils import get_blade_signs, reverse
        signs = get_blade_signs(self.sig, x.device)
        patterns_rev = reverse(self.patterns, self.sig)

        # (B, N, I), (M, N, I) -> (B, M)
        sim = torch.einsum("bni,mni,i->bm", x, patterns_rev, signs)
        return -torch.logsumexp(self.beta * sim, dim=-1)
