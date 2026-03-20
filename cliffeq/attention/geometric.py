import torch
from torch import nn
import torch.nn.functional as F
from cliffeq.algebra.utils import geometric_product, scalar_part, bivector_part
from cliffordlayers.signature import CliffordSignature

class CliffordAttention(nn.Module):
    def __init__(self, n_heads: int, clifford_dim: int, sig_g: torch.Tensor, use_orientation_bias: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.clifford_dim = clifford_dim
        self.sig_g = sig_g
        self.sig = CliffordSignature(sig_g)
        self.use_orientation_bias = use_orientation_bias
        self.n_blades = self.sig.n_blades
        self.q_proj = nn.Linear(clifford_dim * self.n_blades, clifford_dim * self.n_blades)
        self.k_proj = nn.Linear(clifford_dim * self.n_blades, clifford_dim * self.n_blades)
        self.v_proj = nn.Linear(clifford_dim * self.n_blades, clifford_dim * self.n_blades)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x = x.view(B, L, self.n_heads, self.clifford_dim, self.n_blades)
        q = self.q_proj(x.view(B, L, self.n_heads, -1)).view(B, L, self.n_heads, self.clifford_dim, self.n_blades)
        k = self.k_proj(x.view(B, L, self.n_heads, -1)).view(B, L, self.n_heads, self.clifford_dim, self.n_blades)
        v = self.v_proj(x.view(B, L, self.n_heads, -1)).view(B, L, self.n_heads, self.clifford_dim, self.n_blades)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        g = self.sig_g
        if self.sig.dim == 3:
            signs = torch.tensor([1.0, g[0], g[1], g[2], g[0]*g[1], g[0]*g[2], g[1]*g[2], g[0]*g[1]*g[2]], device=x.device)
        elif self.sig.dim == 2:
            signs = torch.tensor([1.0, g[0], g[1], g[0]*g[1]], device=x.device)
        else:
            signs = torch.tensor([1.0, g[0]], device=x.device)
        q_w = q * signs
        logits = torch.einsum("bhldi,bhmdi->bhlm", q_w, k) / (self.clifford_dim * self.n_blades)**0.5
        attn = F.softmax(logits, dim=-1)
        out = torch.einsum("bhlm,bhmdi->bhldi", attn, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return out
