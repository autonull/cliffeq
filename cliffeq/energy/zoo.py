import torch
from torch import nn
from cliffeq.energy.base import EnergyFunction
from cliffeq.algebra.utils import geometric_product, scalar_part, clifford_norm_sq, reverse, get_blade_signs
from cliffordlayers.signature import CliffordSignature

class NormEnergy(EnergyFunction):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__(use_spectral_norm=use_spectral_norm)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # E = 0.5 * ||x||^2
        return 0.5 * torch.sum(state**2, dim=tuple(range(1, state.ndim)))

class BilinearEnergy(EnergyFunction):
    def __init__(self, in_nodes, hidden_nodes, sig_g, use_spectral_norm: bool = False):
        super().__init__(use_spectral_norm=use_spectral_norm)
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.hidden_dim = hidden_nodes

        from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear
        g_val = sig_g.tolist() if isinstance(sig_g, torch.Tensor) else sig_g
        # CliffordLinear bias=False is currently bugged in the library, use bias=True and zero it
        self.W_layer = CliffordLinear(g_val, in_nodes, hidden_nodes, bias=True)
        self.W_out_layer = CliffordLinear(g_val, hidden_nodes, 1, bias=True)
        with torch.no_grad():
            self.W_layer.bias.zero_()
            self.W_out_layer.bias.zero_()

        self.input_x = None
        self.apply_sn()

    def set_input(self, x):
        self.input_x = x

    def get_output(self, h):
        # rho(h) = clamp(h, 0, 1)
        rho_h = torch.clamp(h, 0, 1)
        out_mv = self.W_out_layer(rho_h)
        return out_mv[..., 0]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # E = 0.5 * ||h||^2 - scalar(rev(rho(h)) * W * x)
        rho_h = torch.clamp(h, 0, 1)
        E_norm = 0.5 * torch.sum(h**2, dim=(-1, -2))
        Wx = self.W_layer(self.input_x)

        # scalar(rev(A) * B) = sum_i signs_i * A_i * B_i
        signs = get_blade_signs(self.sig, h.device)
        E_int = torch.einsum("bni,bni,i->b", rho_h, Wx, signs)
        return E_norm - E_int

class GraphEnergy(EnergyFunction):
    def __init__(self, nodes, sig_g, use_spectral_norm: bool = False):
        super().__init__(use_spectral_norm=use_spectral_norm)
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.W = nn.Parameter(torch.randn(nodes, nodes, self.sig.n_blades) * 0.1)
        self.apply_sn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # E = 0.5 * ||x||^2 + Σ_{ij} scalar(rev(x_i) * W_ij * x_j)
        E_norm = 0.5 * torch.sum(x**2, dim=(-1, -2))
        Wx = geometric_product(x, self.W, self.g) # (B, N, I)
        signs = get_blade_signs(self.sig, x.device)
        # scalar(rev(x) * Wx)
        E_int = torch.einsum("bni,bni,i->b", x, Wx, signs)
        return E_norm + E_int

class GradeWeightedEnergy(EnergyFunction):
    def __init__(self, sig_g, lambdas=None, use_spectral_norm: bool = False):
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
        elif self.sig.dim == 1:
            blade_grades = [0, 1]
        else:
            blade_grades = [0] * x.shape[-1]

        E = 0.0
        for i, g in enumerate(blade_grades):
            E += self.lambdas[g] * torch.sum(x[..., i]**2, dim=tuple(range(1, x.ndim-1)))
        return E

class HopfieldEnergy(EnergyFunction):
    def __init__(self, n_patterns, nodes, sig_g, beta=1.0, use_spectral_norm: bool = False):
        super().__init__(use_spectral_norm=use_spectral_norm)
        self.register_buffer("g", sig_g)
        self.sig = CliffordSignature(sig_g)
        self.patterns = nn.Parameter(torch.randn(n_patterns, nodes, self.sig.n_blades) * 0.1)
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # E = 0.5 * ||x||^2 - (1/beta) * log Σ_m exp(β · scalar(rev(ξ_m) x))
        E_norm = 0.5 * torch.sum(x**2, dim=(-1, -2))

        signs = get_blade_signs(self.sig, x.device)
        # scalar(rev(patterns) * x) = sum_i signs_i * patterns_i * x_i
        # patterns: (M, N, I), x: (B, N, I) -> (B, M)
        sim = torch.einsum("mni,bni,i->bm", self.patterns, x, signs)
        E_hop = - (1.0 / self.beta) * torch.logsumexp(self.beta * sim, dim=-1)
        return E_norm + E_hop
